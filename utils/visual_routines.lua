require 'image'

function imNorm(im, maxVal)
  im = im - im:min()
  local imMax = im:max()
  im = im:div(imMax)
  im = im:mul(maxVal)
  return im
end

function whiteBkg(im, mask)
  local thr = (mask:max()-mask:min())*0.9
  mask = torch.repeatTensor(mask, im:size(1),1,1)
  im[mask:lt(thr)]=1
  return im
end

local function thr_mask(im)
  local thr = (im:max()-im:min())*0.2
  im2 = im:clone()
  im2 = im2:fill(0)
  im2[im:gt(thr)]=1
  return im2
end

function full_partTarget(partTarg, fullTarg)

  local m_p = thr_mask(partTarg)
  local m_f = thr_mask(fullTarg)
  local m_pf = m_p + m_f
  -- make 3d
  local fullpartTarg = m_p:clone()
  -- invert the colors
  fullpartTarg[m_p:lt(0.5)] = 1
  fullpartTarg[m_p:ge(0.5)] = 0
  
  fullpartTarg = torch.repeatTensor(fullpartTarg, 3,1,1)  
  -- - adds pink in missing region
--  fullpartTarg[1][m_pf:eq(1)] = 255/255 -- ch1 = red
--  fullpartTarg[2][m_pf:eq(1)] = 240/255 -- ch2 = green
--  fullpartTarg[3][m_pf:eq(1)] = 228/255 -- ch1 = blue
  
  return fullpartTarg
end


function saveWarpedImgs(inputs, source, target, estimate, str_prefix, sample, savePath)
  if not os.execute('cd ' .. savePath) then
    os.execute('mkdir -p ' .. savePath)
  end

  local sz = {inputs:size(3), inputs:size(4)}
  local sourceImg = source[{{sample},{2,source:size(2)},{}}]:squeeze(1)
  local targetIn = inputs[sample][2]
  local targetImg = target[sample][1]
  local estimatedTarget = estimate[{{sample},{2,source:size(2)},{}}]:squeeze(1)
  
  sourceImg = whiteBkg(sourceImg, source[{{sample},{1},{}}]:squeeze(1))
  estimatedTarget = whiteBkg(estimatedTarget, estimate[{{sample},{1},{}}]:squeeze(1))
  local targetImg2 = full_partTarget(targetIn, targetImg)

  image.save(paths.concat(savePath, string.format('%s_source.png', str_prefix)), sourceImg:float())
  image.save(paths.concat(savePath, string.format('%s_targetPart.png', str_prefix)), targetImg2:float())
  image.save(paths.concat(savePath, string.format('%s_targetPred.png', str_prefix)), estimatedTarget:float())
end

function saveWarpedSet(inputs, source, target, estimate, batchNumber)

-- create qual visuals sub directory
  local savePath = paths.concat(opt.save, 'qual')
  local Source = imNorm(source, 1)
  local Estimate = imNorm(estimate, 1)
-- iterate over all samples
  for i = 1,inputs:size(1) do
    local sample = i + batchNumber
    local str_prefix = string.format('sample%03d',sample)
    saveWarpedImgs(inputs, Source, target, Estimate, str_prefix, i, savePath)
  end
  collectgarbage()
end