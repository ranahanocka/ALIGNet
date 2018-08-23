--------------------------------------
-- DEFINE CONFIGURATION PARAMS
--------------------------------------

local M = { }

function M.parse(arg)
    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text('Torch-7 nonrigid deformation script')
    cmd:text()
    cmd:text('Options:')
    ------------ General options --------------------
    cmd:option('-data', '/mnt/data/silhouettes/rendered/vase/h5files', 'folder with: hdf5 data, trainfiles.txt, testfiles.txt')
    cmd:option('-save', 'none', 'where to save the checkpoint information (if none, saves where data is)')
    cmd:option('-manualSeed',         5829, 'Manually set RNG seed')
    cmd:option('-GPU',                1, 'Default preferred GPU')
    cmd:option('-nGPU',               1, 'Number of GPUs to use by default')
    cmd:option('-backend',     'cunn',     'Options: cudnn | nn')
    ------------- Data options ------------------------
    cmd:option('-nDonkeys',        2, 'number of data loading threads')
    cmd:option('-inputDataKey',    '/dep/view1',    'the key name for silhouette data')
    cmd:option('-labelDataKey',    'labels',    'the key name for label data')
    cmd:option('-partialFlag',     1, 'flag for training on partial data')
    cmd:option('-dataLoader',      'dataset_2d', 'dataset_2d | dataset_3d')
    cmd:option('-augmentAffine',   false, 'flag to augment source with affine transform')
    ------------- Training options --------------------
    cmd:option('-nEpochs',          500,    'Number of total epochs to run')
    cmd:option('-epochSize',        1000, 'Number of batches per epoch')
    cmd:option('-epochNumber',      1,     'Manual epoch number (useful on restarts)')
    cmd:option('-batchSize',        100,   'mini-batch size (1 = pure stochastic)')
    ---------- Optimization options ----------------------
    cmd:option('-LR',    		          0.001, 'learning rate')
    cmd:option('-momentum',           0.9,   'momentum')
    cmd:option('-weightDecay',        0,  'weight decay')
    cmd:option('-learningRateDecay',  0,  'learning rate decay')
    cmd:option('-beta1',              0.9,  'beta1 for ADAM')
    cmd:option('-beta2',              0.999,  'beta2 for ADAM')
    cmd:option('-epsilon',            1e-8,  'episolon for ADAM')
    ---------- Model options ----------------------------------
    cmd:option('-netType',        'alignet_2d', 'Options: ')
    cmd:option('-csz',             8, 'square grid size (8x8) Options: sizes 2 through 12')
    cmd:option('-delCage',         true, 'flag to compute differential grid values instead of absolute')
    cmd:option('-cage_reg',        1e-5, 'cage regularization weight (adds TV identity regularization ')
    cmd:option('-learn_beta',      true, 'cage regularization weight (adds l1 regularization to (zerod) grid weight)')
    cmd:option('-retrain',        'none', 'provide path to model to retrain with')
    cmd:option('-optimState',     'none', 'provide path to an optimState to reload from')
    cmd:option('-range_fct',      'abs', ' abs | relu | none function which maps output of CNN to grid codomain')-- delete
    ------------ Evaluation options --------------------
    cmd:option('-demo',           'none',     'pass two image names seperated by whitespace')
    cmd:text()

    local opt = cmd:parse(arg or {})
    -- add commandline specified options
    if opt.save == 'none' then
      opt.save = paths.concat(paths.dirname(opt.data), 'checkpoint',
                              cmd:string(opt.netType, opt,
                                         {netType=true, retrain=true, optimState=true, cache=true, data=true,
                                           inputDataKey=true, labelDataKey=true,GPU=true}))
      -- add date/time
      opt.save = paths.concat(opt.save, '' .. os.date():gsub(' ',''))
    end
      
    
    if opt.demo == 'none' then
      opt.demo = false
    else
      local demofiles = {}
      for s in opt.demo:gmatch("%S+") do table.insert(demofiles, s) end
      opt.demo = demofiles
      if table.getn(opt.demo) ~= 2 then error("demo expects two filenames seperated by whitespace! e.g., th main.lua -demo 'file1.png file2.png'") end
    end

    return opt
end

return M
