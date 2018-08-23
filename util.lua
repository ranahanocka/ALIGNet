require 'cunn'
local ffi=require 'ffi'

function makeDataParallel(model, nGPU)   
  if nGPU > 1 then
    print('converting module to nn.DataParallelTable')
    assert(nGPU <= cutorch.getDeviceCount(), 'number of GPUs less than nGPU specified')
    if opt.backend == 'cudnn' and opt.cudnnAutotune == 1 then
      local gpu_table = torch.range(1, nGPU):totable()
      local dpt = nn.DataParallelTable(1, true):add(model, gpu_table):threads(function() require 'cudnn'
          cudnn.benchmark = true  end)
        dpt.gradInput = nil
        model = dpt:cuda()
      else
        local model_single = model
        model = nn.DataParallelTable(1)
        for i=1, nGPU do
          cutorch.setDevice(i)
          model:add(model_single:clone():cuda(), i)
        end
        cutorch.setDevice(opt.GPU)
      end
    else
      if (opt.backend == 'cudnn' and opt.cudnnAutotune == 1) then
        require 'cudnn'
        cudnn.benchmark = true
      end
    end

    return model
  end

  local function cleanDPT(module)
    -- This assumes this DPT was created by the function above: all the
    -- module.modules are clones of the same network on different GPUs
    -- hence we only need to keep one when saving the model to the disk.
    local newDPT = nn.DataParallelTable(1)
    cutorch.setDevice(opt.GPU)
    newDPT:add(module:get(1), opt.GPU)
    return newDPT
  end

  function saveDataParallel(filename, model)
    if torch.type(model) == 'nn.DataParallelTable' then
      torch.save(filename, cleanDPT(model))
    elseif torch.type(model) == 'nn.Sequential' then
      local temp_model = nn.Sequential()
      for i, module in ipairs(model.modules) do
        if torch.type(module) == 'nn.DataParallelTable' then
          temp_model:add(cleanDPT(module))
        else
          temp_model:add(module)
        end
      end
      torch.save(filename, temp_model)
    else
      error('This saving function only works with Sequential or DataParallelTable modules.')
    end
  end

  function loadDataParallel(filename, nGPU)
    if opt.backend == 'cudnn' then
      require 'cudnn'
    end
    local model = torch.load(filename)
    if torch.type(model) == 'nn.DataParallelTable' then
      return makeDataParallel(model:get(1):float(), nGPU)
    elseif torch.type(model) == 'nn.Sequential' then
      for i,module in ipairs(model.modules) do
        if torch.type(module) == 'nn.DataParallelTable' then
          model.modules[i] = makeDataParallel(module:get(1):float(), nGPU)
        end
      end
      return model
    else
      error('The loaded model is not a Sequential or DataParallelTable module.')
    end
  end

-- Algorithmic utilities
  function meshgrid(x,y)
    local xx = torch.repeatTensor(x, y:size(1),1)
    local yy = torch.repeatTensor(y:view(-1,1), 1, x:size(1))
    return xx, yy
  end

  function ind2sub(sz,ii)
    x = torch.Tensor(1):fill(ii-1)
    r = 1 + torch.fmod(x,sz[2])[1]
    c = 1 + torch.floor((ii-1)/sz[2])
    return {r, c}
  end
  
function upsamplewarp(warpC, sz)
  local fieldI = interpField(warpC:size(1), sz)
  local fullWarp = bilinearSampler()
  local warpF = fullWarp:forward({warpC, fieldI})
  return warpF
end

function computeEstTarget(warpC, sil_S)
  local sz = {sil_S:size(3), sil_S:size(4)}
  local warpF = upsamplewarp(warpC, sz)
  local lookUp = bilinearSampler()
  local EstTarget = lookUp:forward({sil_S, warpF})
  return EstTarget
end

-- compute IOU
function computeIOU(predTarget,gtTarget)
  local B = predTarget:size(1)
  -- look at IOU of target and est target
  local t = gtTarget[{{},{1}}]:gt(0.5):double()
  local et = predTarget[{{},{1}}]:gt(0.5):double()
  local t_et_sum = t + et
  local intrsct_v = torch.sum(t_et_sum:eq(2):double():view(-1,B),1)
  local un_v = torch.sum(t_et_sum:gt(0):double():view(-1,B),1)
  local iou = torch.cdiv(intrsct_v:double(),un_v:double()):sum() / B
  return iou
end

function getGroupNames(HDF5file, matchStr)
  local handle = io.popen(string.format('h5ls -r %s', HDF5file))
  local h5List, dataList = {},{}
  while true do
    local line = handle:read()
    if line == nil then break end
    table.insert(h5List, string.split(line, " ")[1])
    if string.match(line, matchStr) then table.insert(dataList, string.split(line, " ")[1]) end
  end
  handle:close()
  return dataList
end

function readtxtfile(file)
  local lines = {}
  for line in io.lines(file) do 
    lines[#lines + 1] = line
  end
  return lines
end