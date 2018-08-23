----------------------------------------------------------------------
-- Training routine
--
-- Rana Hanocka
----------------------------------------------------------------------

require 'torch'   -- torch
require 'optim'   -- an optimization package, for online and batch methods

----------------------------------------------------------------------
--[[
   1. Setup SGD optimization state and learning rate schedule
   2. Create loggers.
   3. train - this function handles the high-level training loop,
              i.e. load data, train model, save model and state to disk
   4. trainBatch - Used by train() to train a single batch after the data is loaded.
]]--

-- Setup a reused optimization state (for sgd). If needed, reload it from disk
print(sys.COLORS.red ..  '==> configuring optimizer')
local optimState = {
  learningRate = opt.LR,
  learningRateDecay = opt.learningRateDecay,
  momentum = opt.momentum, -- not needed in ADAM..
  weightDecay = opt.weightDecay,
  beta1 = opt.beta1,
  beta2 = opt.beta2,
  epsilon = opt.epsilon
}

if opt.optimState ~= 'none' then
  assert(paths.filep(opt.optimState), 'File not found: ' .. opt.optimState)
  print('Loading optimState from file: ' .. opt.optimState)
  optimState = torch.load(opt.optimState)
end

-- 2. Create loggers.
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
--trainLogger:display(false)
local batchNumber
local loss_epoch

-- 3. train - this function handles the high-level training loop,
--            i.e. load data, train model, save model and state to disk
function train()
  print('==> doing epoch on training data:')
  print("==> online epoch # " .. epoch)

  batchNumber = 0
  cutorch.synchronize()

  -- set the dropouts to training mode
  model:training()

  local tm = torch.Timer()
  loss_epoch = 0
  for i=1,opt.epochSize do
    -- queue jobs to data-workers
    donkeys:addjob(
      -- the job callback (runs in data-worker thread)
      function()
        local inputs, fieldI, sourceImgs, targetImgs = trainLoader:sample(opt.batchSize)
        return inputs, fieldI, sourceImgs, targetImgs
      end,
      -- the end callback (runs in the main thread)
      trainBatch
    )
  end

  donkeys:synchronize()
  cutorch.synchronize()


  loss_epoch = loss_epoch / (opt.epochSize * opt.batchSize)

  trainLogger:add{
    ['avg loss (train set)'] = loss_epoch
  }
  print(string.format('Epoch: [%d][TRAINING SUMMARY] Total Time(s): %.2f\t'
      .. 'average loss (per batch): %.2f \t ',
      epoch, tm:time().real, loss_epoch))
  print('\n')

  collectgarbage()
  -- save model
  -- clear the intermediate states in the model before saving to disk
  -- this saves lots of disk space
  model:clearState()
   saveDataParallel(paths.concat(opt.save, 'model_' .. epoch .. '.t7'), model) -- defined in util.lua
   torch.save(paths.concat(opt.save, 'optimState_' .. epoch .. '.t7'), optimState)
end -- of train()

-------------------------------------------------------------------------------------------
-- GPU inputs (preallocate)
local inputs = torch.CudaTensor()
local fieldI = torch.CudaTensor()
local sourceImgs = torch.CudaTensor()
local targetImgs = torch.CudaTensor()

local timer = torch.Timer()
local dataTimer = torch.Timer()

local parameters, gradParameters = model:getParameters()

-- 4. trainBatch - Used by train() to train a single batch after the data is loaded.
function trainBatch(inputsCPU, fieldICPU, sourceImgsCPU, targetImgsCPU)
  cutorch.synchronize()
  collectgarbage()
  local dataLoadingTime = dataTimer:time().real
  timer:reset()

  -- transfer over to GPU
  inputs:resize(inputsCPU:size()):copy(inputsCPU)
  fieldI:resize(fieldICPU:size()):copy(fieldICPU)
  sourceImgs:resize(sourceImgsCPU:size()):copy(sourceImgsCPU)
  targetImgs:resize(targetImgsCPU:size()):copy(targetImgsCPU)
  
  if opt.swap then sourceImgs, targetImgs = targetImgs, sourceImgs end -- needed for 3D (compute forward warping)

  local err, outputs
  feval = function(x)
    model:zeroGradParameters()
    outputs = model:forward({sourceImgs, {inputs, fieldI}})
    err = criterion:forward(outputs, targetImgs)
    local gradOutputs = criterion:backward(outputs, targetImgs)
    model:backward({sourceImgs, {inputs, fieldI}}, gradOutputs)
    return err, gradParameters
  end
  optim.adam(feval, parameters, optimState)

  cutorch.synchronize()
  batchNumber = batchNumber + 1
  loss_epoch = loss_epoch + err

  -- Print information
  print(('Epoch: [%d][%d/%d]\tTime %.3f Err %.4f LR %.0e DataLoadingTime %.3f'):format(
      epoch, batchNumber, opt.epochSize, timer:time().real, err,
      optimState.learningRate, dataLoadingTime))

  dataTimer:reset()
end
