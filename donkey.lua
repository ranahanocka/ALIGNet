paths.dofile(opt.dataLoader .. '.lua')
paths.dofile('util.lua')

-- This file contains the data-loading logic and details.
-- It is run by each data-loader thread.
------------------------------------------

-- a cache file of the mean/std
local meanstdCache = paths.concat(opt.data, 'meanstdCache.t7')

-- Check for existence of opt.data
if not os.execute('cd ' .. opt.data) then
    error(("could not chdir to '%s'"):format(opt.data))
end

--------------------------------------------------------------------------------
--[[
   Section 1: Create a train data loader (trainLoader)
--]]

 print('Creating train metadata')
 trainLoader = dataLoader{
  dataPath =  paths.concat(opt.data),
  inputDataKey = opt.inputDataKey,
  labelDataKey = opt.labelDataKey,
  mean = mean,
  std = std
}
collectgarbage()

-- End of train loader section
--------------------------------------------------------------------------------
--[[
   Section 2: Create a test data loader (testLoader)
--]]

print('Creating test metadata')
 testLoader = dataLoader{
  dataPath =  paths.concat(opt.data),
  inputDataKey = opt.inputDataKey,
  labelDataKey = opt.labelDataKey,
  mean = mean,
  std = std,
  test = true
}
collectgarbage()
-- End of test loader section


-- channel-wise mean and std. Calculate or load them from disk
local mean,std
-- Estimate the per-channel mean/std (so that the loaders can normalize appropriately)
if paths.filep(meanstdCache) then
   local meanstd = torch.load(meanstdCache)
   mean = meanstd.mean
   std = meanstd.std
   print('Loaded mean and std from cache.')
else
   local tm = torch.Timer()
   local nSamples = 10000
   print('Estimating the mean (per-channel, shared for all pixels) over ' .. nSamples .. ' randomly sampled training images')
   -- preallocate a table with same dimensions as input data
   meanEstimate = {}
   for ii=1,trainLoader._dim2 do meanEstimate[ii] = 0 end
   for i=1,nSamples do
      local img = trainLoader:sample(1)[1]
      for j=1,trainLoader._dim2 do
         meanEstimate[j] = meanEstimate[j] + img[j]:mean()
      end
   end
   for j=1,trainLoader._dim2 do
      meanEstimate[j] = meanEstimate[j] / nSamples
   end
   mean = meanEstimate

   print('Estimating the std (per-channel, shared for all pixels) over ' .. nSamples .. ' randomly sampled training images')
   stdEstimate = {}
   for ii=1,trainLoader._dim2 do stdEstimate[ii] = 0 end
   for i=1,nSamples do
      local img = trainLoader:sample(1)[1]
      for j=1,trainLoader._dim2 do
         stdEstimate[j] = stdEstimate[j] + img[j]:std()
      end
   end
   for j=1,trainLoader._dim2 do
      stdEstimate[j] = stdEstimate[j] / nSamples
   end
   std = stdEstimate

   local cache = {}
   cache.mean = mean
   cache.std = std
   torch.save(meanstdCache, cache)
   print('Time to estimate:', tm:time().real)
end

-- dump the new mean/std into loaders
testLoader.mean = mean
testLoader.std = std

trainLoader.mean = mean
trainLoader.std = std