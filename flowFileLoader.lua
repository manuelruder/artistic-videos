require 'torch'
require 'image'

--[[
  Reads a flow field from a binary flow file.
  
   bytes   contents
    0-3     tag: "PIEH" in ASCII, which in little endian happens to be the float 202021.25
            (just a sanity check that floats are represented correctly)
    4-7     width as an integer
    8-11    height as an integer
    12-end  data (width*height*2*4 bytes total)
--]]
local function flowFileLoader_load(fileName)
  local flowFile = torch.DiskFile(fileName, 'r')
  flowFile:binary()
  flowFile:readFloat()
  local W = flowFile:readInt()
  local H = flowFile:readInt()
  -- image.warp needs 2xHxW, and also expects (y, x) for some reason...
  local flow = torch.Tensor(2, H, W)
  for y=1, H do
    for x=1, W do
      flow[2][y][x] = flowFile:readFloat()
      flow[1][y][x] = flowFile:readFloat()
    end
  end
  flowFile:close()
  return flow
end

return {
  load = flowFileLoader_load
}