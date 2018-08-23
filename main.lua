----------------------------------------------------------------------
-- train cnn to compute non-rigid deformation field
--
-- Author: Rana Hanocka
----------------------------------------------------------------------
--------------------------------------
-- DEPENDENCIES
--------------------------------------
require 'torch'
require 'cutorch'
require 'paths'
require 'xlua'
require 'optim'
require 'nn'


--------------------------------------
-- CONFIGURATION PARAMS
--------------------------------------
torch.setdefaulttensortype('torch.FloatTensor')
local opts = paths.dofile('opts.lua')
opt = opts.parse(arg)

print(sys.COLORS.red ..  '==> load config params')
print(opt)
torch.manualSeed(opt.manualSeed)

--------------------------------------
-- READ IN THE NETWORK ARCHITECURE
--------------------------------------
paths.dofile('util.lua')
paths.dofile('model.lua')

cutorch.setDevice(opt.GPU) -- by default, use GPU 1

print('Saving everything to: ' .. opt.save)
os.execute('mkdir -p ' .. opt.save)

--------------------------------------
-- DEFINE THE DATA, TRAIN, AND TEST ROUTINES
--------------------------------------
paths.dofile('data.lua')
paths.dofile('train.lua')
paths.dofile('test.lua')
paths.dofile('utils/visual_routines.lua') -- functions for visualization

epoch = opt.epochNumber
for i=1,opt.nEpochs do
  train()
  test()
  epoch = epoch + 1
  collectgarbage()
end