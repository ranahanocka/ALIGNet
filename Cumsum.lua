local Cumsum, parent = torch.class('nn.Cumsum','nn.Module')

--[[
    The input is always b*2*h*w. the first channel is y axis and the second one is x axis

]]

function Cumsum:__init()
    parent.__init(self)
end
    
function Cumsum:updateOutput(input)
    
    local yxz=input:split(1,2)
    self.output:resizeAs(input)
    
    for i = 1,input:size(2) do
      self.output[{{},i}]:cumsum(yxz[i]:squeeze(), i+1)   --sum along each dimension
    end 
    return self.output

end

function Cumsum:updateGradInput(input, gradOutput)
    
   if not gradOutput:isContiguous() then
      self._gradOutput = self._gradOutput or gradOutput.new()
      self._gradOutput:resizeAs(gradOutput):copy(gradOutput)
      gradOutput = self._gradOutput
   end
   
  dim = {}
  for i = 1,input:nDimension()-2 do
    dim[i] = input:size(i+2)
  end

  

   local yxGrad = gradOutput:split(1,2)
   self.gradInput:resizeAs(input)
   
  for i = 1,input:size(2) do
     local Grad = yxGrad[i]:squeeze()
     local GradFlip = Grad:index(i+1,torch.linspace(dim[i],1,dim[i]):long())
     local GradFlipCum = GradFlip:cumsum(i+1)
     local GradCum = GradFlipCum:index(i+1,torch.linspace(dim[i],1,dim[i]):long())
     self.gradInput[{{},i}]:copy(GradCum)
  end

   return self.gradInput
end

function Cumsum:clearState()
   nn.utils.clear(self, '_gradOutput')
   return parent.clearState(self)
end