require 'torch'
require 'nn'
require 'image'
require 'optim'
require 'loadcaffe'
require 'artistic_video_core'

local flowFile = require 'flowFileLoader'

--------------------------------------------------------------------------------

local cmd = torch.CmdLine()

-- Basic options
cmd:option('-style_image', 'example/seated-nude.jpg',
           'Style target image')
cmd:option('-style_blend_weights', 'nil')
cmd:option('-content_pattern', 'example/marple8_%02d.ppm',
           'Content target pattern')
cmd:option('-num_images', 0, 'Number of content images. Set 0 for autodetect.')
cmd:option('-start_number', 1, 'Frame index to start with')
cmd:option('-gpu', 0, 'Zero-indexed ID of the GPU to use; for CPU mode set -gpu = -1')
cmd:option('-number_format', '%d', 'Number format of the output images.')

-- Flow options
cmd:option('-forwardFlow_pattern', 'example/deepflow/forward_[%d]_{%d}.flo',
           'Flow file pattern. [.] will be replaced with the "from"-index, {.} with the "to"-index.')
cmd:option('-backwardFlow_pattern', 'example/deepflow/backward_[%d]_{%d}.flo',
           'Flow file pattern. [.] will be replaced with the "from"-index, {.} with the "to"-index.')
cmd:option('-forwardFlow_weight_pattern', 'example/deepflow/reliable_[%d]_{%d}.pgm',
           'Flow file pattern. [.] will be replaced with the "from"-index, {.} with the "to"-index.')
cmd:option('-backwardFlow_weight_pattern', 'example/deepflow/reliable_[%d]_{%d}.pgm',
           'Flow file pattern. [.] will be replaced with the "from"-index, {.} with the "to"-index.')

-- Multi-pass options
cmd:option('-blendWeight', 1.0, '')
cmd:option('-blendWeight_lastPass', 0.0, '')
cmd:option('-use_temporalLoss_after', 8, '')
cmd:option('-num_passes', 15, 'Number of passes')
cmd:option('-continue_with_pass', 1, '')

-- Optimization options
cmd:option('-content_weight', 5e0)
cmd:option('-style_weight', 1e2)
cmd:option('-temporal_weight', 5e2)
cmd:option('-tv_weight', 1e-3)
cmd:option('-temporal_loss_criterion', 'mse', 'mse|smoothl1')
cmd:option('-num_iterations', 100, 'Number of iterations per pass')
cmd:option('-tol_loss_relative', 0, 'stop if relative change of the loss function is below this value')
cmd:option('-tol_loss_relative_interval', 100, 'interval between two function comparisons')
cmd:option('-normalize_gradients', false)
cmd:option('-init', 'random', 'random|image|prevWarped')
cmd:option('-optimizer', 'lbfgs', 'lbfgs|adam')
cmd:option('-learning_rate', 1e1)

-- Output options
cmd:option('-print_iter', 50)
cmd:option('-save_iter', 0)
cmd:option('-output_image', 'out.png')
cmd:option('-output_folder', '')
cmd:option('-save_init', false, 'Whether the initialization image should be saved (for debugging purposes).')

-- Other options
cmd:option('-style_scale', 1.0)
cmd:option('-pooling', 'max', 'max|avg')
cmd:option('-proto_file', 'models/VGG_ILSVRC_19_layers_deploy.prototxt')
cmd:option('-model_file', 'models/VGG_ILSVRC_19_layers.caffemodel')
cmd:option('-backend', 'nn', 'nn|cudnn|clnn')
cmd:option('-cudnn_autotune', false)
cmd:option('-seed', -1)

cmd:option('-content_layers', 'relu4_2', 'layers for content')
cmd:option('-style_layers', 'relu1_1,relu2_1,relu3_1,relu4_1,relu5_1', 'layers for style')

cmd:option('-args', '', 'Arguments in a file, one argument per line')

function nn.SpatialConvolutionMM:accGradParameters()
  -- nop.  not needed by our net
end

local function main(params)
  if params.gpu >= 0 then
    if params.backend ~= 'clnn' then
      require 'cutorch'
      require 'cunn'
      cutorch.setDevice(params.gpu + 1)
    else
      require 'clnn'
      require 'cltorch'
      cltorch.setDevice(params.gpu + 1)
    end
  else
    params.backend = 'nn'
  end

  if params.backend == 'cudnn' then
    require 'cudnn'
    if params.cudnn_autotune then 
      cudnn.benchmark = true
    end
    cudnn.SpatialConvolution.accGradParameters = nn.SpatialConvolutionMM.accGradParameters -- ie: nop
  end
  
  local loadcaffe_backend = params.backend
  if params.backend == 'clnn' then loadcaffe_backend = 'nn' end
  local cnn = loadcaffe.load(params.proto_file, params.model_file, loadcaffe_backend):float()
  cnn = MaybePutOnGPU(cnn, params)
  
  local num_images = params.num_images
  if num_images == 0 then
    num_images = calcNumberOfContentImages(params)
    print("Detected " .. num_images .. " content images.")
  end
  local end_image_idx = num_images + params.start_number - 1

  local style_images_caffe = getStyleImages(params)
  
  -- Set up the network, inserting style and content loss modules
  local net, style_losses, losses_indices, losses_type = buildNet(cnn, params, style_images_caffe)

  -- We don't need the base CNN anymore, so clean it up to save memory.
  cnn = nil
  for i=1,#net.modules do
    local module = net.modules[i]
    if torch.type(module) == 'nn.SpatialConvolutionMM' then
        -- remote these, not used, but uses gpu memory
        module.gradWeight = nil
        module.gradBias = nil
    end
  end
  collectgarbage()

  local img = nil
  
  -- Initialize the image
  if params.seed >= 0 then
      torch.manualSeed(params.seed)
  end
  local content_size = image.load(string.format(params.content_pattern, params.start_number), 3):size()
  local randImg = torch.randn(content_size):mul(0.001)
  
  local usePrev = params.init == 'prev' or params.init == 'prevWarped'
  local needFlow = params.init == 'prevWarped' or params.prevPlusFlow_layers ~= ''
  
  for run=params.continue_with_pass, params.num_passes do

    local flag = run % 2
    local start = (flag == 0) and end_image_idx or params.start_number
    local endp = (flag == 0) and params.start_number or end_image_idx
    local incr = (flag == 0) and -1 or 1
  
    for frameIdx=start,endp, incr do

      local content_image_caffe = getContentImage(frameIdx, params)
      local content_losses, prevPlusFlow_losses = {}, {}
      local additional_layers = 0
      local num_iterations = params.num_iterations

      -- Previous and following frame warped
      local prevImageWarped, nextImageWarped = nil, nil
      -- The warped frame which will be used for temporal consistency.
      local imageWarped = nil
      
      -- Find out if we are forward or backward pass, and set "imageWarped" accordingly.
      if frameIdx > params.start_number then
        prevImageWarped = readPrevImageWarped(frameIdx, params, run - (1 - flag), false)
      end
      if run > 1 and frameIdx < end_image_idx then
        nextImageWarped = readNextImageWarped(frameIdx, params, run - flag, false)
      end
      if flag == 1 then imageWarped = prevImageWarped end
      if flag == 0 then imageWarped = nextImageWarped end
      
      local temporalLossEnabled = run >= params.use_temporalLoss_after and imageWarped ~= nil

      -- add layers for this iteration
      for i=1, #losses_indices do
        if losses_type[i] == 'content'  then
          local content_loss = getContentLossModuleForLayer(net,
            losses_indices[i] + additional_layers,
            content_image_caffe, params)
          net:insert(content_loss, losses_indices[i] + additional_layers)
          additional_layers = additional_layers + 1
          table.insert(content_losses, content_loss)
        elseif temporalLossEnabled then
          imageWarped = preprocess(imageWarped):float()
          imageWarped = MaybePutOnGPU(imageWarped, params)
          local flowWeights = nil
          if losses_type[i] == 'prevPlusFlowWeighted' then
            local weightsFileName = nil
            if flag == 1 then
              weightsFileName = getFormatedFlowFileName(params.backwardFlow_weight_pattern, frameIdx-1, frameIdx)
            else
              weightsFileName = getFormatedFlowFileName(params.forwardFlow_weight_pattern, frameIdx+1, frameIdx)
            end
            print(string.format('Reading flowWeights file "%s".', weightsFileName))
            flowWeights = image.load(weightsFileName):float()
            flowWeights = flowWeights:expand(3, flowWeights:size(2), flowWeights:size(3))
            flowWeights = MaybePutOnGPU(flowWeights, params)
          end
          local loss_module = getWeightedContentLossModuleForLayer(net,
            losses_indices[i] + additional_layers, imageWarped,
            params, flowWeights)
          net:insert(loss_module, losses_indices[i] + additional_layers)
          table.insert(prevPlusFlow_losses, loss_module)
          additional_layers = additional_layers + 1
        end
      end

      if run == 1 then
        -- For the first run, process the frames independently
        if frameIdx == params.start_number or params.init == 'random' then
          img = randImg:clone():float()
        elseif init == 'image' then
          img = content_image:clone():float()
        elseif params.init == 'prevWarped' then
          local prevImageWarpedWithPad = readPrevImageWarped(frameIdx, params, run - (1 - flag), true)
          img = preprocess(prevImageWarpedWithPad):float()
        else
          print('Unknown initialization method.')
          os.exit()
        end
      else
        -- For subsequent runs, blend neighboring frames into the current frame
        img = image.load(build_OutFilename(params, frameIdx, run - 1), 3)
        -- Make sure to correctly normalize the result
        local divisor = torch.zeros(content_image_caffe:size())
        divisor:add(1)
        if frameIdx > params.start_number then
          local weightsFileName = getFormatedFlowFileName(params.backwardFlow_weight_pattern, frameIdx-1, frameIdx)
          print(string.format('Reading flowWeights file "%s".', weightsFileName))
          local prevImageWeights = image.load(weightsFileName)
          prevImageWeights = prevImageWeights:expand(3, prevImageWeights:size(2), prevImageWeights:size(3))
          prevImageWeights:mul(flag == 1 and params.blendWeight or params.blendWeight_lastPass)
          img:add(torch.cmul(prevImageWarped, prevImageWeights))
          divisor:add(prevImageWeights)
        end
        if frameIdx < end_image_idx then
          local weightsFileName = getFormatedFlowFileName(params.forwardFlow_weight_pattern, frameIdx+1, frameIdx)
          print(string.format('Reading flowWeights file "%s".', weightsFileName))
          local nextImageWeights = image.load(weightsFileName)
          nextImageWeights = nextImageWeights:expand(3, nextImageWeights:size(2), nextImageWeights:size(3))
          nextImageWeights:mul(flag == 0 and params.blendWeight or params.blendWeight_lastPass)
          img:add(torch.cmul(nextImageWarped, nextImageWeights))
          divisor:add(nextImageWeights)
        end
        img:cdiv(divisor)
        img = preprocess(img):float()
      end
      
      img = MaybePutOnGPU(img, params)

      if params.save_init then
        save_image(img, params.output_folder .. string.format(
          'init-' .. params.number_format .. '_%d.png', frameIdx, run))
      end

      -- Run the optimization for some iterations, save the result to disk
      runOptimization(params, net, content_losses, style_losses, prevPlusFlow_losses,
          img, frameIdx, run, num_iterations)

      -- Remove this iteration's content and temporal layers
      for i=#losses_indices, 1, -1 do
        if temporalLossEnabled or losses_type[i] == 'content' then
          additional_layers = additional_layers - 1
          net:remove(losses_indices[i] + additional_layers)
        end
      end
      
      assert(additional_layers == 0)
      
    end
    
  end

end

-- warp previous frame.
-- Disocclusions at the borders will be filled with the VGG mean pixel, if pad_mean_pixel is true.
function readPrevImageWarped(idx, params, run, pad_mean_pixel)
  local flowFileName = getFormatedFlowFileName(params.backwardFlow_pattern, idx-1, idx)
  print(string.format('Reading backward flow file "%s".', flowFileName))
  local flow = flowFile.load(flowFileName)
  local prevImg = image.load(build_OutFilename(params, idx-1, run), 3)
  local result = nil
  if pad_mean_pixel then
    local mean_pixel = torch.DoubleTensor({123.68/256.0, 116.779/256.0, 103.939/256.0})
    result = image.warp(prevImg, flow, 'bilinear', true, 'pad', -1)
    for x=1, result:size(2) do
      for y=1, result:size(3) do
        if result[1][x][y] == -1 and result[2][x][y] == -1 and result[3][x][y] == -1 then
          result[1][x][y] = mean_pixel[1]
          result[2][x][y] = mean_pixel[2]
          result[3][x][y] = mean_pixel[3]
        end
      end
    end
  else
    result = image.warp(prevImg, flow)
  end
  return result
end

-- warp following frame.
-- Disocclusions at the borders will be filled with the VGG mean pixel, if pad_mean_pixel is true.
function readNextImageWarped(idx, params, run, pad_mean_pixel)
  local flowFileName = getFormatedFlowFileName(params.forwardFlow_pattern, idx+1, idx)
  print(string.format('Reading forward flow file "%s".', flowFileName))
  local flow = flowFile.load(flowFileName)
  local nextImg = image.load(build_OutFilename(params, idx+1, run), 3)
  if pad_mean_pixel then
    local mean_pixel = torch.DoubleTensor({123.68/256.0, 116.779/256.0, 103.939/256.0})
    result = image.warp(nextImg, flow, 'bilinear', true, 'pad', -1)
    for x=1, result:size(2) do
      for y=1, result:size(3) do
        if result[1][x][y] == -1 and result[2][x][y] == -1 and result[3][x][y] == -1 then
          result[1][x][y] = mean_pixel[1]
          result[2][x][y] = mean_pixel[2]
          result[3][x][y] = mean_pixel[3]
        end
      end
    end
  else
    result = image.warp(nextImg, flow)
  end
  return result
end

local tmpParams = cmd:parse(arg)
local params = nil
local file = io.open(tmpParams.args, 'r')

if tmpParams.args == '' or file == nil  then
  params = cmd:parse(arg)
else
  local args = {}
  io.input(file)
  local argPos = 1
  while true do
    local line = io.read()
    if line == nil then break end
    if line:sub(0, 1) == '-' then
      local splits = str_split(line, " ", 2)
      args[argPos] = splits[1]
      args[argPos + 1] = splits[2]
      argPos = argPos + 2
    end
  end
  for i=1, #arg do
    args[argPos] = arg[i]
    argPos = argPos + 1
  end
  params = cmd:parse(args)
  io.close(file)
end

main(params)
