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
  local raw_flow = torch.data(flow)
  local elems_in_dim = H * W
  local storage = flowFile:readFloat(2 * elems_in_dim)
  for y=0, H - 1 do
    for x=0, W - 1 do
      local shift = y * W + x
      raw_flow[elems_in_dim + shift] = storage[2 * shift + 1]
      raw_flow[shift] = storage[2 * shift + 2]
    end
  end
  flowFile:close()
  return flow
end

return {
  load = flowFileLoader_load
}
