-- functions for alignet
require 'stn'

function bilinearSampler()
  -- Network which performs bilinear sampling in batches
  local spanet=nn.Sequential()
  local parl = nn.ParallelTable()
  -- branch1 - IMAGE to interpolate
  local inputImNode = nn.Sequential()
  inputImNode:add(nn.Identity())
  inputImNode:add(nn.Transpose({2,3},{3,4})) -- converts IM BDWH to BHWD (D should equal 1.. but doesnt work)
  -- branch2 - the interpolation field
  local gtWarp = nn.Sequential()
  gtWarp:add(nn.Transpose({2,3},{3,4}))
  -- concatenate Warp and Image
  parl:add(inputImNode)
  parl:add(gtWarp)
  -- Apply the Bilinear lookup, then convert back to BDWH
  spanet:add(parl)
  spanet:add(nn.BilinearSamplerBHWD())
  spanet:add(nn.Transpose({3,4},{2,3}))
  spanet = spanet:cuda()
  collectgarbage()
  return spanet
end

function interpField(B, sz)
  -- create mini network to generate interpolation grid
  local fullGrid = nn.Sequential()
  fullGrid:add(nn.AffineGridGeneratorBHWD(sz[1],sz[2]))
  fullGrid = fullGrid:cuda()
  -- create a tensor of B x identity (uniform) interpolation grids
  local I = torch.CudaTensor(B, 2,3):fill(0)
  I[{{},{1},{1}}] = torch.CudaTensor(B,1,1):fill(1)
  I[{{},{2},{2}}] = torch.CudaTensor(B,1,1):fill(1)
  local fieldI = fullGrid:forward(I)
  fieldI = fieldI:transpose(4,2):transpose(3,4)
  collectgarbage();
  return fieldI
end