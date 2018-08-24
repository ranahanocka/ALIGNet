local BilinearSamplerVolumetric, parent = torch.class('nn.BilinearSamplerVolumetric', 'nn.Module')

--[[
   BilinearSamplerBHWD() :
   BilinearSamplerBHWD:updateOutput({inputImages, grids})
   BilinearSamplerBHWD:updateGradInput({inputImages, grids}, gradOutput)

   BilinearSamplerBHWD will perform bilinear sampling of the input images according to the
   normalized coordinates provided in the grid. Output will be of same size as the grids, 
   with as many features as the input images.

   - inputImages has to be in BDHWD layout

   - grids have to be in BDHWD layout, with dim(D)=4
   - grids contains, for each sample (first dim), the normalized coordinates of the output wrt the input sample
      - Z, Y, X coordinate
      - normalized coordinates : (-1,-1, -1) points to front top left, (-1, -1,1) points to front top right
      - if the normalized coordinates fall outside of the image, then output will be filled with zeros
]]

function BilinearSamplerVolumetric:__init()
  parent.__init(self)
  self.gradInput={}
end

function BilinearSamplerVolumetric:check(input, gradOutput)
  local inputImages = input[1]
	local grids = input[2]

  assert(inputImages:isContiguous(), 'Input images have to be contiguous')
  assert(inputImages:nDimension()==5)
  assert(grids:nDimension()==5)
  assert(inputImages:size(1)==grids:size(1)) -- batch
  assert(grids:size(5)==4) -- coordinates
   
  if gradOutput then
    assert(grids:size(1)==gradOutput:size(1)) --batchsize
    assert(grids:size(2)==gradOutput:size(2)) --depth == dist
    assert(grids:size(3)==gradOutput:size(3)) --height
    assert(grids:size(4)==gradOutput:size(4)) --width
  end
end

local function addOuterDim(t)
  local sizes = t:size()
  local newsizes = torch.LongStorage(sizes:size()+1)
  newsizes[1]=1
  for i=1,sizes:size() do
    newsizes[i+1]=sizes[i]
  end
  return t:view(newsizes)
end

function BilinearSamplerVolumetric:updateOutput(input)
  local _inputImages = input[1]
  local _grids = input[2]
   --print("D")
   --print(_grids)
   
  local inputImages, grids
  if _inputImages:nDimension()==4 then  --image:size(4) = channel, image:size(1)=depth, image:size(2) = height, image:size(3)=width
    inputImages = addOuterDim(_inputImages)
    grids = addOuterDim(_grids)
  else
    inputImages = _inputImages
    grids = _grids
  end

  local input = {inputImages, grids}

  self:check(input)

  self.output:resize(inputImages:size(1), grids:size(2), grids:size(3), grids:size(4), inputImages:size(5))
  inputImages.nn.BilinearSamplerVolumetric_updateOutput(self, inputImages, grids, self.output)

  if _inputImages:nDimension()==4 then
    self.output=self.output:select(1,1)
  end
	
  return self.output
end

function BilinearSamplerVolumetric:updateGradInput(_input, _gradOutput)
  local _inputImages = _input[1]
	local _grids = _input[2]

  local inputImages, grids, gradOutput
  if _inputImages:nDimension()==4 then
    inputImages = addOuterDim(_inputImages)
    grids = addOuterDim(_grids)
    gradOutput = addOuterDim(_gradOutput)
  else
    inputImages = _inputImages
    grids = _grids
    gradOutput = _gradOutput
  end

  local input = {inputImages, grids}

  self:check(input, gradOutput)
	for i=1,#input do
	  self.gradInput[i] = self.gradInput[i] or input[1].new()
    self.gradInput[i]:resizeAs(input[i]):zero()
  end

  local gradInputImages = self.gradInput[1]
  local gradGrids = self.gradInput[2]

  inputImages.nn.BilinearSamplerVolumetric_updateGradInput(self, inputImages, grids, gradInputImages, gradGrids, gradOutput)

  if _gradOutput:nDimension()==4 then
    self.gradInput[1]=self.gradInput[1]:select(1,1)
    self.gradInput[2]=self.gradInput[2]:select(1,1)
  end
   
  return self.gradInput
end
