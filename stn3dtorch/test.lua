require 'torch'
require 'nn'
require 'cunn'
--require 'cudnn'
require 'volstn'
require 'nngraph'
require 'optim'
require 'image'



opt={}
opt.vox_size = 64

function create_projector(opt)
  local grid_stream = nn.VolumetricGridGenerator(opt.vox_size, opt.vox_size, opt.vox_size)
  local input_stream = nn.Transpose({2,4},{4,5})
  local projector = nn.Sequential()
  projector:add(nn.ParallelTable():add(input_stream):add(grid_stream))
  projector:add(nn.BilinearSamplerVolumetric())
  projector:add(nn.Transpose({4,5}, {2,4}))
  -- B x c x Dim1 x Dim2 x Dim3
  --projector:add(nn.Max(4)) 
  return projector
end

local specify_pers_transformation = function(phi, theta, azi)
  local T = torch.Tensor(4, 4):zero()
  local K = torch.Tensor(4, 4):eye(4)
  local E = torch.Tensor(4, 4):eye(4)

  local sin_phi = math.sin(phi*math.pi/180.0)
  local cos_phi = math.cos(phi*math.pi/180.0)

  local sin_azi = math.sin(azi*math.pi/180.0)
  local cos_azi = math.cos(azi*math.pi/180.0)
  
  local sin_theta = math.sin((-theta)*math.pi/180.0)
  local cos_theta = math.cos((-theta)*math.pi/180.0)



  -- rotation axis -- z
  R_the = torch.Tensor(3, 3):zero()
  R_the[1][1] = cos_theta
  R_the[3][3] = cos_theta
  R_the[1][3] = -sin_theta
  R_the[3][1] = sin_theta
  R_the[2][2] = 1

  -- rotation axis -- x
  R_ele = torch.Tensor(3, 3):zero()
  R_ele[1][1] = cos_phi
  R_ele[1][2] = sin_phi
  R_ele[2][1] = -sin_phi
  R_ele[2][2] = cos_phi
  R_ele[3][3] = 1


  -- rotation axis -- y
  R_azi = torch.Tensor(3, 3):zero()
  R_azi[2][2] = cos_azi
  R_azi[2][3] = sin_azi
  R_azi[3][2] = -sin_azi
  R_azi[3][3] = cos_azi
  R_azi[1][1] = 1


  R_comb = R_the * R_ele * R_azi

  local colR = torch.Tensor(3,1):zero()
  
  colR[1][1] = 0
  colR = R_comb * colR
  E[{{1,3}, {1,3}}] = R_comb:clone()
  E[{{1,3}, {4}}] = -colR:clone()
  
  T = E 

  return T
end

vol = torch.randn(64,64,64):ge(0.5):float()


projector = create_projector(opt)
trans = specify_pers_transformation(0, 0, 90)
projector:cuda()

vol_batch = vol:view(1,1, opt.vox_size, opt.vox_size, opt.vox_size):cuda()
trans_batch = trans:view(1, 4, 4):cuda()
output = projector:forward({vol_batch, trans_batch})
print(output:size())
--torch.save('output.t7',{vol=output[1][1]:float()})

