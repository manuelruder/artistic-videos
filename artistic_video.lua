require 'torch'
require 'nn'
require 'image'
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
cmd:option('-continue_with', 1, 'Continue with the given frame index.')
cmd:option('-gpu', 0, 'Zero-indexed ID of the GPU to use; for CPU mode set -gpu = -1')
cmd:option('-number_format', '%d', 'Number format of the output images.')

--Flow options
cmd:option('-flow_pattern', 'example/deepflow/backward_[%d]_{%d}.flo',
           'Optical flow files pattern')
cmd:option('-flowWeight_pattern', 'example/deepflow/reliable_[%d]_{%d}.pgm',
           'Optical flow weight files pattern.')
cmd:option('-flow_relative_indices', '1', 'Use flow from the given indices.')
cmd:option('-use_flow_every', -1, 'Uses flow from the given index and every multiple of that; -1 to to disable.')
cmd:option('-invert_flowWeights', 0, 'Invert flow weights given by flowWeight_pattern.')

-- Optimization options
cmd:option('-content_weight', 5e0)
cmd:option('-style_weight', 1e2)
cmd:option('-temporal_weight', 1e3)
cmd:option('-tv_weight', 1e-3)
cmd:option('-temporal_loss_criterion', 'mse', 'mse|smoothl1')
cmd:option('-num_iterations', '2000,1000',
           'Can be set separately for the first and for subsequent iterations, separated by comma, or one value for all.')
cmd:option('-tol_loss_relative', 0.0001, 'Stop if relative change of the loss function is below this value')
cmd:option('-tol_loss_relative_interval', 50, 'Interval between two loss comparisons')
cmd:option('-normalize_gradients', false)
cmd:option('-init', 'random,prevWarped', 'random|image,random|image|prev|prevWarped')
cmd:option('-optimizer', 'lbfgs', 'lbfgs|adam')
cmd:option('-learning_rate', 1e1)

-- Output options
cmd:option('-print_iter', 100)
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

-- Advanced options (changing them is usually not required)
cmd:option('-combine_flowWeights_method', 'closestFirst',
           'Which long-term weighting scheme to use: normalize or closestFirst. Deafult and recommended: closestFirst')

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

  local style_images_caffe = getStyleImages(params)

  -- Set up the network, inserting style losses. Content and temporal loss will be inserted in each iteration.
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

  -- There can be different setting for the first frame and for subsequent frames.
  local num_iterations_split = params.num_iterations:split(",")
  local numIters_first, numIters_subseq = num_iterations_split[1], num_iterations_split[2] or num_iterations_split[1]
  local init_split = params.init:split(",")
  local init_first, init_subseq = init_split[1], init_split[2] or init_split[1]
  
  local firstImg = nil
  local flow_relative_indices_split = params.flow_relative_indices:split(",")

  local num_images = params.num_images
  if num_images == 0 then
    num_images = calcNumberOfContentImages(params)
    print("Detected " .. num_images .. " content images.")
  end

  -- Iterate over all frames in the video sequence
  for frameIdx=params.start_number + params.continue_with - 1, params.start_number + num_images - 1 do

    -- Set seed
    if params.seed >= 0 then
      torch.manualSeed(params.seed)
    end

    local content_image = getContentImage(frameIdx, params)
    if content_image == nil then
      print("No more frames.")
      do return end
    end
    local content_losses, temporal_losses = {}, {}
    local additional_layers = 0
    local num_iterations = frameIdx == params.start_number and tonumber(numIters_first) or tonumber(numIters_subseq)
    local init = frameIdx == params.start_number and init_first or init_subseq
    -- stores previous image indices used for the temporal constraint
    local J = {}
    -- stores previous image(s) warped
    local imgsWarped = {}
    
    -- Calculate from which indices we need a warped image
    if frameIdx > params.start_number and params.temporal_weight ~= 0 then
      for i=1, #flow_relative_indices_split do
        local prevIndex = frameIdx - tonumber(flow_relative_indices_split[i])
        if prevIndex >= params.start_number then 
          table.insert(J, frameIdx - tonumber(flow_relative_indices_split[i]))
        end
      end
      if params.use_flow_every > 0 then
        for prevIndex=frameIdx - params.use_flow_every, params.start_number, -1 * params.use_flow_every do
          if not tabl_contains(J, prevIndex) then
            table.insert(J, prevIndex)
          end
        end
      end
      -- Sort table descending, usefull to compute the long-term weights
      table.sort(J, function(a,b) return a>b end)
      -- Read the optical flow(s) and warp the previous image(s)
      for j=1, #J do
        local prevIndex = J[j]
        local flowFileName = getFormatedFlowFileName(params.flow_pattern, math.abs(prevIndex), math.abs(frameIdx))
        print(string.format('Reading flow file "%s".', flowFileName))
        local flow = flowFile.load(flowFileName)
        local fileName = build_OutFilename(params, math.abs(prevIndex - params.start_number + 1), -1)
        local imgWarped = warpImage(image.load(fileName, 3), flow)
        imgWarped = preprocess(imgWarped):float()
        imgWarped = MaybePutOnGPU(imgWarped, params)
        table.insert(imgsWarped, imgWarped)
      end
    end

    -- Add content and temporal loss for this iteration. Style loss is already included in the net.
    for i=1, #losses_indices do
      if losses_type[i] == 'content'  then
        local loss_module = getContentLossModuleForLayer(net,
          losses_indices[i] + additional_layers, content_image, params)
        net:insert(loss_module, losses_indices[i] + additional_layers)
        table.insert(content_losses, loss_module)
        additional_layers = additional_layers + 1
      elseif losses_type[i] == 'prevPlusFlow' and frameIdx > params.start_number then
        for j=1, #J do
          local loss_module = getWeightedContentLossModuleForLayer(net,
            losses_indices[i] + additional_layers, imgsWarped[j],
            params, nil)
          net:insert(loss_module, losses_indices[i] + additional_layers)
          table.insert(temporal_losses, loss_module)
          additional_layers = additional_layers + 1
        end
      elseif losses_type[i] == 'prevPlusFlowWeighted' and frameIdx > params.start_number then
        local flowWeightsTabl = {}
        -- Read all flow weights
        for j=1, #J do
          local weightsFileName = getFormatedFlowFileName(params.flowWeight_pattern, J[j], math.abs(frameIdx))
          print(string.format('Reading flowWeights file "%s".', weightsFileName))
          table.insert(flowWeightsTabl, image.load(weightsFileName):float())
        end
        -- Preprocess flow weights, calculate long-term weights
        processFlowWeights(flowWeightsTabl, params.combine_flowWeights_method, params.invert_flowWeights)
        -- Create loss modules, one for each previous frame warped
        for j=1, #J do
          local flowWeights = flowWeightsTabl[j]
          flowWeights = flowWeights:expand(3, flowWeights:size(2), flowWeights:size(3))
          flowWeights = MaybePutOnGPU(flowWeights, params)
          local loss_module = getWeightedContentLossModuleForLayer(net,
            losses_indices[i] + additional_layers, imgsWarped[j],
            params, flowWeights)
          net:insert(loss_module, losses_indices[i] + additional_layers)
          table.insert(temporal_losses, loss_module)
          additional_layers = additional_layers + 1
        end
      end
    end

    -- Initialization
    local img = nil
    if init == 'random' then
      img = torch.randn(content_image:size()):float():mul(0.001)
    elseif init == 'image' then
      img = content_image:clone():float()
    elseif init == 'prevWarped' and frameIdx > params.start_number then
      local flowFileName = getFormatedFlowFileName(params.flow_pattern, math.abs(frameIdx - 1), math.abs(frameIdx))
      print(string.format('Reading flow file "%s".', flowFileName))
      local flow = flowFile.load(flowFileName)
      local fileName = build_OutFilename(params, math.abs(frameIdx - params.start_number), -1)
      img = warpImage(image.load(fileName, 3), flow)
      img = preprocess(img):float()
    elseif init == 'prev' and frameIdx > params.start_number then
      local fileName = build_OutFilename(params, math.abs(frameIdx - params.start_number), -1)
      img = image.load(fileName, 3)
      img = preprocess(img):float()
    elseif init == 'first' then
      img = firstImg:clone():float()
    else
      print('ERROR: Invalid initialization method.')
      os.exit()
    end
    img = MaybePutOnGPU(img, params)
    if params.save_init then
      save_image(img,
        string.format('%sinit-' .. params.number_format .. '.png',
          params.output_folder, math.abs(frameIdx - params.start_number + 1)))
    end

    -- Run the optimization to stylize the image, save the result to disk
    runOptimization(params, net, content_losses, style_losses, temporal_losses, img, frameIdx, -1, num_iterations)

    if frameIdx == params.start_number then
      firstImg = img:clone():float()
    end
    
    -- Remove this iteration's content and temporal layers
    for i=#losses_indices, 1, -1 do
      if frameIdx > params.start_number or losses_type[i] == 'content' then
        if losses_type[i] == 'prevPlusFlowWeighted' or losses_type[i] == 'prevPlusFlow' then
          for j=1, #J do
            additional_layers = additional_layers - 1
            net:remove(losses_indices[i] + additional_layers)
          end
        else
          additional_layers = additional_layers - 1
          net:remove(losses_indices[i] + additional_layers)
        end
      end
    end
    
    -- Ensure that all layer have been removed correctly
    assert(additional_layers == 0)
    
  end
end

-- warp a given image according to the given optical flow.
-- Disocclusions at the borders will be filled with the VGG mean pixel.
function warpImage(img, flow)
  local mean_pixel = torch.DoubleTensor({123.68/256.0, 116.779/256.0, 103.939/256.0})
  result = image.warp(img, flow, 'bilinear', true, 'pad', -1)
  for x=1, result:size(2) do
    for y=1, result:size(3) do
      if result[1][x][y] == -1 and result[2][x][y] == -1 and result[3][x][y] == -1 then
        result[1][x][y] = mean_pixel[1]
        result[2][x][y] = mean_pixel[2]
        result[3][x][y] = mean_pixel[3]
      end
    end
  end
  return result
end

-- Creates long-term flow weights
function processFlowWeights(flowWeightsTabl, method, invert)
  if invert == 1 then
    for j=1, #flowWeightsTabl do
      flowWeightsTabl[j]:apply(function(x) return 1 - x end)
    end
  end
  if method == 'normalize' then
    -- Normalize so that the weights sum up to max 1
    local sum = tabl_sum(flowWeightsTabl)
    sum:cmax(1)
    for j=1, #flowWeightsTabl do
      flowWeightsTabl[j]:cdiv(sum)
    end
  elseif method == 'closestFirst' then
    -- Take the closest previous frame(s).
    for j=2, #flowWeightsTabl do
      for k=1, j-1 do
        flowWeightsTabl[j]:add(-1, flowWeightsTabl[j-k])
      end
      flowWeightsTabl[j]:cmax(0)
    end
  end
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
