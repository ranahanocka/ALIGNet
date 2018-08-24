
local PGG, parent = torch.class('nn.VolumetricGridGenerator', 'nn.Module')

function PGG:__init(depth,height,width)
   parent.__init(self)
   assert(depth > 1)
   assert(height > 1)
   assert(width > 1)
   self.depth = depth
   self.height = height
   self.width = width
   self.baseGrid = torch.Tensor(depth, height, width, 4)
   
   for k=1,self.depth do
     for i=1,self.height do
       for j=1,self.width do
          self.baseGrid[k][i][j][1] = (-1 + (k-1)/(self.depth-1) * 2)
          self.baseGrid[k][i][j][2] = (-1 + (i-1)/(self.height-1) * 2)
          self.baseGrid[k][i][j][3] = (-1 + (j-1)/(self.width-1) * 2)
          self.baseGrid[k][i][j][4] = 1
       end
     end
   end
   self.batchGrid = torch.Tensor(1, depth, height, width, 4):copy(self.baseGrid)
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

function PGG:updateOutput(_transformMatrix)
   local transformMatrix
   if _transformMatrix:nDimension()==2 then
      transformMatrix = addOuterDim(_transformMatrix)
   else
      transformMatrix = _transformMatrix
   end
   assert(transformMatrix:nDimension()==3
          and transformMatrix:size(2)==4
          and transformMatrix:size(3)==4
          , 'please input affine transform matrices (bx4x4)')
   local batchsize = transformMatrix:size(1)
   
   if self.batchGrid:size(1) ~= batchsize then
      self.batchGrid:resize(batchsize, self.depth, self.height, self.width, 4)
      for i=1,batchsize do
         self.batchGrid:select(1,i):copy(self.baseGrid)
      end
   end

   self.output:resize(batchsize, self.depth, self.height, self.width, 4)
   local flattenedBatchGrid = self.batchGrid:view(batchsize, self.depth*self.width*self.height, 4)
   local flattenedOutput = self.output:view(batchsize, self.depth*self.width*self.height, 4)
   torch.bmm(flattenedOutput, flattenedBatchGrid, transformMatrix:transpose(2,3)) 
   if _transformMatrix:nDimension()==2 then
      self.output = self.output:select(1,1)
   end
   return self.output
end

function PGG:updateGradInput(_transformMatrix, _gradGrid)
   local transformMatrix, gradGrid
   if _transformMatrix:nDimension()==2 then
      transformMatrix = addOuterDim(_transformMatrix)
      gradGrid = addOuterDim(_gradGrid)
   else
      transformMatrix = _transformMatrix
      gradGrid = _gradGrid
   end

   local batchsize = transformMatrix:size(1)

   local flattenedGradGrid = gradGrid:view(batchsize, self.depth*self.width*self.height, 4)
   local flattenedBatchGrid = self.batchGrid:view(batchsize, self.depth*self.width*self.height, 4)
   self.gradInput:resizeAs(transformMatrix):zero()
   self.gradInput:baddbmm(flattenedGradGrid:transpose(2,3), flattenedBatchGrid) ---????
   -- torch.baddbmm doesn't work on cudatensors for some reason

   if _transformMatrix:nDimension()==2 then
      self.gradInput = self.gradInput:select(1,1)
   end

   return self.gradInput
end
