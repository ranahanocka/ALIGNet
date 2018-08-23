paths.dofile('alignet.lua')
require 'Cumsum'

local outSz = 12


function getEstWarpC(model)
  local nodes

  nodes = model:findModules('nn.View')

  correct_node = nodes[#nodes]
  return correct_node.output
end

function createModel(nGPU)
  local model = nn.Sequential()
  local parl2 = nn.ParallelTable()

  --------------------------------------
  -- branch1 - just Image 1 (source)
  --------------------------------------
  local sourceIm = nn.Sequential()
  sourceIm:add(nn.Identity())

  --------------------------------------
  -- branch2 - the estimated warp field
  --------------------------------------
  local warp = nn.Sequential()
  local parl = nn.ParallelTable() -- takes INPUTS and parallelizes them

  -- warpF branch generates warp field (INPUT 1)
  local warpF = nn.Sequential()
  warpF:add(nn.SpatialMaxPooling(2,2,2,2))
  warpF:add(nn.SpatialConvolution(2,20,5,5)) -- smaller number of channels
  warpF:add(nn.ReLU(true))
  warpF:add(nn.SpatialMaxPooling(2,2,2,2))
  warpF:add(nn.SpatialConvolution(20,20,5,5))
  warpF:add(nn.ReLU(true)) -- 100 x 20 x 2 x 2
  warpF:add(nn.SpatialMaxPooling(2,2,2,2))
  warpF:add(nn.SpatialConvolution(20,20,2,2))
  warpF:add(nn.ReLU(true))
  if opt.csz < outSz/2 then
    warpF:add(nn.SpatialMaxPooling(2,2,2,2))
    outSz = 6
  end
  local fSize = outSz - opt.csz + 1  
  warpF:add(nn.SpatialConvolution(20,20,fSize,fSize))
  warpF:add(nn.ReLU(true))
  warpF = warpF:cuda()


  warpF:add(nn.View(20*opt.csz*opt.csz))
  warpF:add(nn.Linear(20*opt.csz*opt.csz,20))
  warpF:add(nn.ReLU(true))
--  initialize the output layer so it gives the identity transform
  local outLayer = nn.Linear(20,2*opt.csz*opt.csz)

  -- 
  local Y, X = meshgrid(torch.linspace(-1,1,opt.csz), torch.linspace(-1,1,opt.csz))
  local bias = torch.cat(X:view(opt.csz*opt.csz), Y:view(opt.csz*opt.csz))
  
  local del, beta, dNet
  if opt.delCage then
    local par4 = nn.ParallelTable()  
    local cum_mono = nn.Sequential()
    cum_mono:add(nn.View(2,opt.csz))
    cum_mono:add(nn.Replicate(opt.csz,2,1))
    cum_mono:add(nn.SplitTable(2))
    par4:add(nn.Transpose({2,3}))
    par4:add(nn.Identity())
    cum_mono:add(par4)
    cum_mono:add(nn.JoinTable(2))
    cum_mono:add(nn.View(2,opt.csz,opt.csz))

    ---- compute delta and beta
    del = 2/(opt.csz-1) -- spacing in the grid
    beta = -1 - del -- grid offset

    bias = torch.Tensor(2*opt.csz*opt.csz):fill(del) -- new bias if using delta cage...
    dNet = nn.Sequential()
    
    -- force grid values to be strictly positive (i.e., axial monotonicity)
    if opt.range_fct == "abs" then
      dNet:add(nn.Abs())
    elseif opt.range_fct == "relu" then
      dNet:add(nn.ReLU())
    elseif opt.range_fct == "none" then
      dNet:add(nn.Identity())
    else
      error("invalid range fct")
    end

    dNet:add(nn.View(2,opt.csz,opt.csz))
    

    -- add l1-penalty on warp field...
    -- subtract delta -> zeros cage (apply l1-penalty) -> add back delta
    dNet:add(nn.AddConstant(-del))
    dNet:add(nn.L1Penalty(opt.cage_reg))
    dNet:add(nn.AddConstant(del))
    dNet:add(nn.Cumsum())

    -- adds beta (starting point) to matrix of deltas 
    -- beta can be learnable / unlearnable
    if opt.learn_beta then
      local beta_bias = nn.Add(1,true)
      beta_bias.bias = torch.Tensor(1):fill(beta)
      dNet:add(beta_bias)
    else  
      dNet:add(nn.AddConstant(beta))
    end

  end



  outLayer.weight:fill(0)
  outLayer.bias:copy(bias)
  warpF:add(outLayer)


  if opt.delCage then warpF:add(dNet) end

  -- generate the grids
  warpF:add(nn.View(2,opt.csz,opt.csz))

  -- apply bilinear upsample (INPUT 2)
  -- Takes in constant bilinear sampler...
  local fieldI = nn.Sequential()
  fieldI:add(nn.Identity())
  parl:add(warpF)
  parl:add(fieldI)  
  warp:add(parl)
  local fullWarp = bilinearSampler()
  warp:add(fullWarp)

  --------------------------------------
  -- Compute estimaed target image from warp field
  --------------------------------------
  parl2:add(sourceIm)
  parl2:add(warp)
  -- Apply the Bilinear lookup, then convert back to BDWH
  model:add(parl2)
  -- 
  local sampler = bilinearSampler()
  model:add(sampler)

  return model
end
