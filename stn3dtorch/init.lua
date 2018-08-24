require 'nn'
local withCuda = pcall(require, 'cutorch')

require 'libvolstn'
if withCuda then
   require 'libcuvolstn'
end

require('volstn.VolumetricGridGenerator')
require('volstn.BilinearSamplerVolumetric')

return nn
