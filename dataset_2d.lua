require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')
require 'hdf5'
local ffi = require 'ffi'
local argcheck = require 'argcheck'
paths.dofile('models/alignet.lua')
require 'image'


local dataset = torch.class('dataLoader')

local initcheck = argcheck{
  pack=true,
  help=[[
     A dataset class for hdf5 files]],

  {name="dataPath",
    type="string",
    help="path of the rendered HDF5 data"},

  {name="inputDataKey",
    type="string",
    help="key name of the input data"},

  {name="labelDataKey",
    type="string",
    help="key name of the label (warp) data"},

  {name="mean",
    type="table",
    help="table of mean data values",
    opt = true},

  {name="std",
    type="table",
    help="table of std data values",
    opt = true},
  
  {name="test",
    type="boolean",
    default=false,
    help="boolean indicating whether test set"},
}

function dataset:__init(...)

  -- argcheck
  local args =  initcheck(...)
  print(args)
  for k,v in pairs(args) do self[k] = v end
  
  -- construct list of train/test files
  if self.test then
    self.data_file = paths.concat(self.dataPath,'testfiles.txt')
  else
    self.data_file = paths.concat(self.dataPath,'trainfiles.txt')
  end
  if not paths.filep(self.data_file) then
    error("the file", self.data_file, "doesnt exist!")
  end
  self._data_list = readtxtfile(self.data_file)
  self._nSamples = #self._data_list
  
  -- constructors
  self:set_imsz()
  self._dim2 = 2
  

  -- initializing mean, std
  if not self.mean then 
    print('no mean assigned, setting to 0')
    local mean = {}
    for i=1,self._dim2 do mean[i] = 0 end
    self.mean = mean
  end
  if not self.std then
    print('no std assigned, setting to 1')
    local std = {}
    for i=1,self._dim2 do std[i] = 1 end
    self.std = std
  end
  
  if self.test then
    self.testcounter = 1
    self:compute_test_permutations()
  end
end

--------------------------------------------------------------
------------------------ MAIN ROUTINES -----------------------
--------------------------------------------------------------

-- main wrapper to randomly sample data -- used during train / test
function dataset:sample(quantity)
  
  local sidx, tidx
  if self.test then
    sidx, tidx = self:testidx(quantity)
  else
    sidx, tidx = self:randidx(quantity)
  end
  
  -- get the data at these indexes
  local data = self:get_data(sidx, tidx)
  if not self.test then
    local SRng = {torch.normal(1,0.1), torch.normal(1,0.1)}
    local TRng = {0, 0}
    data[{{},{1},{},{}}] = self:aug_affine_warp(data[{{},{1},{},{}}], {0, 0}, TRng, SRng) -- changed to 0/0 ...
    data[{{},{2},{},{}}] = self:aug_affine_warp(data[{{},{2},{},{}}], {0, 0}, TRng, SRng)
  end
  
  -- partial data on the fly
  data, targetImgs = self:partialData_train(data)
  local targetImgs = torch.repeatTensor(targetImgs, 1, 2, 1, 1)
  
  -- get sourceImgs before normalizing by mu/std
  local sourceImgs = data[{{},{1},{},{}}]:clone()
  if self.test then
    sourceImgs = self:addtexture(sourceImgs)
  else
    sourceImgs = torch.repeatTensor(sourceImgs, 1, 2, 1, 1)
  end
  
  -- normalize by mean and std
  data = self:muStdNorm(data)
  
  -- create sampling grid to upsample cage
  local fieldI = interpField(data:size(1), {self._imsz[1], self._imsz[2]})
  
  collectgarbage()
  return data, fieldI, sourceImgs, targetImgs
end

function dataset:testidx(quantity)
  local nextcounter = math.min(self.testcounter+ quantity-1,self._nTest)
  local sidx = self.sidx[{{self.testcounter, nextcounter}}]
  local tidx = self.tidx[{{self.testcounter, nextcounter}}]
  if nextcounter == self._nTest then
    self.testcounter = 1
  else
    self.testcounter = nextcounter
  end
  return sidx, tidx
end

function dataset:randidx(quantity)
  -- generate random source/target idxs
  local sidx, tidx = torch.Tensor(), torch.Tensor()
  while sidx:numel() < quantity do
    -- jumble samples for source & target
    sidx = torch.cat(sidx,torch.randperm(self._nSamples))
    tidx = torch.cat(tidx,torch.randperm(self._nSamples))
    -- truncate up to quantity
    sidx = sidx[{{1,math.min(sidx:numel(), quantity)}}]
    tidx = tidx[{{1,math.min(tidx:numel(), quantity)}}]
  end
  return sidx, tidx
end

function dataset:compute_test_permutations()
      -- compute all permutations of 2 for entire test set (all images source to target, etc)
    local sids, tids= {}, {}
    local xx,yy = meshgrid(torch.range(1,#self._data_list), torch.range(1,#self._data_list))
    local NN = xx:numel()
    local sids_ = yy:view(NN,1)
    local tids_ = xx:view(NN,1)
    -- only keep pairs which are not the same
    for i = 1,NN do
      if tids_[i][1] ~= sids_[i][1] then
        table.insert(sids,sids_[i][1])
        table.insert(tids,tids_[i][1])
      end
    end
    self.sidx = torch.Tensor(sids)
    self.tidx = torch.Tensor(tids)
    self._nTest = self.sidx:numel()
end

function dataset:size()
  return self._nTest
end


--------------------------------------------------------------
---------------------- MAIN HELPER ROUTINES ------------------
--------------------------------------------------------------

function dataset:get_data(sids, tids)
  local B = sids:numel()
  -- allocate data size
  local data = torch.zeros(B, self._dim2, self._imsz[1], self._imsz[2])
  for i = 1,B do
    -- get source sil
    local src_datalist = self.s_data_list or self._data_list
    local sim = self:get_sil(sids[i], src_datalist)

    -- get a target sil
    local tgt_datalist = self.t_data_list or self._data_list
    local tim = self:get_sil(tids[i], tgt_datalist)

    -- drop into tensors for saving
    data[{{i},{1},{},{}}] = sim
    data[{{i},{2},{},{}}] = tim
  end
  collectgarbage()
  return data
end

function dataset:get_sil(id, data_list)
  local HDF5file = paths.concat(self.dataPath, data_list[id])
  local myFile = hdf5.open(HDF5file,'r')
  local sil = myFile:read(self.inputDataKey):all()
  myFile:close()
  sil = sil:transpose(1,2)
  return sil
end

function dataset:muStdNorm(data)
  for i=1,data:size(2) do -- channels
    data[{{},{i},{},{}}]:add(-self.mean[i])
    data[{{},{i},{},{}}]:div(self.std[i])
  end
  return data
end

--------------------------------------------------------------
--------------------- DATA ROUTINES --------------------------
--------------------------------------------------------------
function dataset:get_texture(sil_S) -- to-do: update for several diff source
  local txtimage_
    if self.s_data_list then
      local hdf5file = paths.concat(self.dataPath, self.s_data_list[1])
      local myFile = hdf5.open(hdf5file,'r')
      txtimage_ = myFile:read('/txtr/view1'):all()
      txtimage_ = txtimage_:transpose(2,3)
      txtimage_ = txtimage_:float()
      myFile:close()
    else
      txtimage_ = image.load('utils/checker.png')
    end
    local txtimage = torch.repeatTensor(txtimage_,sil_S:size(1),1,1,1)
    return txtimage
end

function dataset:addtexture(sil_S)
  -- read in checker image and make it size of 2nd slot...
  local checkerim = self:get_texture(sil_S)
  local tsz = {checkerim:size(3),checkerim:size(4)}
  if tsz[1] ~= self._imsz[1] or tsz[2] ~= self._imsz[2] then
      checkerim = sil_S:clone()
      checkerim = torch.repeatTensor(checkerim, 1,3,1,1)
  end
  local ch = sil_S:size(2)
  local text_ch = checkerim:size(2)
  local sourceImgs = sil_S:clone()
  sourceImgs = torch.repeatTensor(sourceImgs, 1, text_ch+1, 1, 1)
  sourceImgs[{{},{2,text_ch+1},{},{}}] = torch.cmul(sourceImgs[{{},{2,text_ch+1},{},{}}], checkerim)
  return sourceImgs
end

function dataset:create_T(B,RRng, TRng, SRng)
  local T = torch.CudaTensor(B, 2,3):zero()
  local Sx = SRng[1] + (SRng[2]-SRng[1])*torch.rand(B)
  local Sy = SRng[1] + (SRng[2]-SRng[1])*torch.rand(B)
  local R = RRng[1] + (RRng[2]-RRng[1])*torch.rand(B)
  local r11 = torch.cmul(Sx,torch.cos(R))
  local r21 = torch.cmul(Sy,torch.sin(R))
  local r22 = torch.cmul(Sy,torch.cos(R))
  local r12 = torch.cmul(Sx,-torch.sin(R))
  local tx = torch.cmul(Sx, TRng[1] + (TRng[2]-TRng[1])*torch.rand(B))
  local ty = torch.cmul(Sy, TRng[1] + (TRng[2]-TRng[1])*torch.rand(B))
  T[{{},{1},{1}}] = torch.ones(B,1,1):cmul(Sx):cuda()
  T[{{},{2},{2}}] = torch.ones(B,1,1):cmul(Sy):cuda()
  T[{{},{1},{3}}] = torch.ones(B,1,1):cmul(tx):cuda()
  T[{{},{2},{3}}] = torch.ones(B,1,1):cmul(ty):cuda()
  T[{{},{1},{1}}] = r11:cuda()
  T[{{},{2},{1}}] = r21:cuda()
  T[{{},{2},{2}}] = r22:cuda()
  T[{{},{1},{2}}] = r12:cuda()
  
  return T
end

-- affine warp images
function dataset:aug_affine_warp(inputs, RRng, TRng, SRng)
  -- Sx and Sy are torch tensors that vectors of length batchSize
  -- i.e., torch.ones(batchSize) does not perform any augmentation 
  
  local B = inputs:size(1)
  local csz = inputs:size(inputs:dim())
  
  -- generate affine full warp field
  local smallGrid = nn.Sequential()
  smallGrid:add(nn.AffineGridGeneratorBHWD(csz,csz))
  smallGrid = smallGrid:cuda()

  local T = self:create_T(B,RRng, TRng, SRng)

  local warpC = smallGrid:forward(T)
  warpC = warpC:transpose(4,2):transpose(3,4)

  -- warp images with it
  local SIL = torch.Tensor(inputs,1,2,1,1):cuda()
  local lookup = bilinearSampler()
  local SIL_warped = lookup:forward({SIL, warpC})
  SIL_warped = SIL_warped:float()
  SIL_warped = SIL_warped[{{},{1},{},{}}]

  collectgarbage()
  return SIL_warped
end

function dataset:partialData_train(inputs)
  local B = inputs:size(1)
  -- create targetImgs
  local targetImgs = inputs[{{},{2},{},{}}]:clone()
  local partialTargetImgs = inputs[{{},{2},{},{}}]
  -- uniform range  of half win size
  local pWinSz = {30, 40}
  -- vector of window (of hole) sizes 
  local WIN = ((pWinSz[2] - pWinSz[1])*torch.rand(B,1) + pWinSz[1]):round()

  local numHoles = 1
  local x,y = meshgrid(torch.range(1,self._imsz[1]), torch.range(1,self._imsz[2]))

  for i = 1,B do
    local partialImg = partialTargetImgs[{{i},{1},{},{}}]
    for j =1,numHoles do
      local win = WIN[i][1]
      local Y = y[partialImg:squeeze():gt(0)]
      local X = x[partialImg:squeeze():gt(0)]
      local ii = torch.random(1,Y:size(1))
      local XY = {Y[ii], X[ii]}
      local xR = torch.Tensor({XY[2] - win, XY[2] + win}):clamp(1,self._imsz[1])
      local yR = torch.Tensor({XY[1] - win, XY[1] + win}):clamp(1,self._imsz[2])
      partialImg[{{},{},{xR[1],xR[2]},{yR[1],yR[2]}}] = 0
    end
  end
  collectgarbage()
  return inputs, targetImgs
end



--------------------------------------------------------------
------------------------ CONSTRUCTORS ------------------------
--------------------------------------------------------------
function dataset:set_imsz()
  local HDF5file = paths.concat(self.dataPath, self._data_list[1])
  local myFile = hdf5.open(HDF5file,'r')
  local imageSize = myFile:read(self.inputDataKey):dataspaceSize()
  myFile:close()
  self._imsz = imageSize
end


return dataset
