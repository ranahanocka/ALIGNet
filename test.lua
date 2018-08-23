----------------------------------------------------------------------
-- Testing routine
--
-- Rana Hanocka
----------------------------------------------------------------------

-- if we run eval only mode, write to train logger file the model name 
if opt.evalOnly and opt.retrain ~= 'none' then
  trainLogger:add{opt.retrain}
end

testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
testLogger:setNames{'avg loss','avg IOU %', 'avg penalty'}

local batchNumber
local loss_set
local avg_iou
local avg_id_pen
local timer = torch.Timer()

function test()
  print('==> doing epoch on validation data:')
  print("==> online epoch # " .. epoch)

  batchNumber = 0
  cutorch.synchronize()
  timer:reset()

  model:evaluate()

  loss_set = 0
  avg_iou = 0
  avg_id_pen = 0
  for i = 1, math.ceil(nTest/opt.batchSize) do -- nTest is set in data.lua
    donkeys:addjob(
      -- work to be done by donkey thread
      function()
        local inputs, fieldI, sourceImgs, targetImgs = testLoader:sample(opt.batchSize)
        return inputs, fieldI, sourceImgs, targetImgs
      end,
      -- callback that is run in the main thread once the work is done
      testBatch
    )
  end
  

  donkeys:synchronize()
  cutorch.synchronize()


  loss_set = loss_set / nTest -- because loss is calculated per batch
  avg_iou = avg_iou*100
  testLogger:add{loss_set, avg_iou, avg_id_pen}
  print(string.format('Epoch: [%d][TESTING SUMMARY] Total Time(s): %.2f \t'
      .. 'average IOU (per batch): %.2f \t ',
      epoch, timer:time().real, avg_iou))

  print('\n')
  collectgarbage()

end -- of test()
-----------------------------------------------------------------------------
-----------------------------------------------------------------------------
local inputs = torch.CudaTensor()
local fieldI = torch.CudaTensor()
local sourceImgs = torch.CudaTensor()
local targetImgs = torch.CudaTensor()

function testBatch(inputsCPU, fieldICPU, sourceImgsCPU, targetImgsCPU)
  batchNumber = batchNumber + opt.batchSize

  inputs:resize(inputsCPU:size()):copy(inputsCPU)
  fieldI:resize(fieldICPU:size()):copy(fieldICPU)
  sourceImgs:resize(sourceImgsCPU:size()):copy(sourceImgsCPU)
  targetImgs:resize(targetImgsCPU:size()):copy(targetImgsCPU)
  
  if opt.swap then sourceImgs, targetImgs = targetImgs, sourceImgs end

  local outputs = model:forward({sourceImgs, {inputs, fieldI}})
  
  -- but we want to know the loss w.r.t the original outputs
  local outputs_o = outputs[{{},{1}}]:clone()
  outputs_o = torch.repeatTensor(outputs_o, 1,2,1,1,1) -- TO-DO : need to change for 2D and 3D
  
  local err = criterion:forward(outputs_o, targetImgs)
  cutorch.synchronize()

  loss_set = loss_set + err
  
  -- compute IOU
  local iou = computeIOU(outputs,targetImgs)
  avg_iou = avg_iou + iou*(inputs:size(1)/nTest)

  ---- warp error
  local estwarpC = getEstWarpC(model)

  print(('Epoch: Testing [%d][%d/%d]\tTime %.3f IOU %.4f'):format(epoch, batchNumber, 
        nTest, timer:time().real, iou))

  -- save current test
  local visualPath = paths.concat(opt.save, 'visuals')
  local sample = torch.uniform(1,inputs:size(1))
  local str_prefix = string.format('epoch%02d',epoch)
  
  -- save out entire set if epoch high enough // makes sure also an image...
  if epoch%25 == 1 and estwarpC:dim() == 4 then
    saveWarpedSet(inputs, sourceImgs, targetImgs, outputs, batchNumber-opt.batchSize)
  end
  collectgarbage()
end
