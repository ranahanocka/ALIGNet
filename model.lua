require 'nn'
require 'cunn'
require 'optim'

--[[
   1. Create Model
   2. Create Criterion
   3. Convert model to CUDA
]]--

-- 1. Create Network
-- 1.1 If preloading option is set, preload weights from existing models appropriately
if opt.retrain ~= 'none' then
   assert(paths.filep(opt.retrain), 'File not found: ' .. opt.retrain)
   print('Loading model from file: ' .. opt.retrain);
   paths.dofile('models/' .. opt.netType .. '.lua')
   model = loadDataParallel(opt.retrain, opt.nGPU) -- defined in util.lua
else
   paths.dofile('models/' .. opt.netType .. '.lua')
   print('=> Creating model from file: models/' .. opt.netType .. '.lua')
   model = createModel(opt.nGPU) -- for the model creation code, check the models/ folder
   if opt.backend == 'cudnn' then
      require 'cudnn'
      cudnn.convert(model, cudnn)
   elseif opt.backend == 'cunn' then
      require 'cunn'
      model = model:cuda()
   elseif opt.backend ~= 'nn' then
      error'Unsupported backend'
   end
end

-- 2. Create Criterion
criterion = nn.SmoothL1Criterion()

print('=> Model')
print(model)

print('=> Criterion')
print(criterion)

-- 3. Convert model to CUDA
-- model is converted to CUDA in the init script itself
criterion:cuda()

collectgarbage()
