-- Mostly adapted from char-rnn by Andrej Karpathy
require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'
require 'util.misc'
--require 'util.LookupTableMaskZero'
require 'util.MaskedLoss'
local BatchLoader = require 'util.BatchLoader'
local model_utils = require 'util.model_utils'
local LSTM = require 'model.LSTM'

cmd = torch.CmdLine()
cmd:text('Options')
-- data
cmd:option('-data_dir','data/ptb','data directory. Should contain the file input.txt with input data')
-- model params
cmd:option('-rnn_size', 150, 'size of LSTM internal state')
cmd:option('-word_vec_size', 100, 'size of word embeddings')
cmd:option('-num_layers', 2, 'number of layers in the LSTM')
-- optimization
cmd:option('-learning_rate',2e-3,'learning rate')
cmd:option('-decay_rate',0.95,'decay rate for rmsprop')
cmd:option('-dropout',0.5,'dropout to use just before classifier. 0 = no dropout')
cmd:option('-seq_length',20,'maximum sentence length')
cmd:option('-batch_size',10,'number of sequences to train on in parallel')
cmd:option('-max_epochs',30,'number of full passes through the training data')
cmd:option('-grad_clip',5,'clip gradients at')
-- bookkeeping
cmd:option('-seed',123,'torch manual random number generator seed')
cmd:option('-print_every',1,'how many steps/minibatches between printing out the loss')
cmd:option('-eval_val_every',1000,'every how many iterations should we evaluate on validation data?')
cmd:option('-checkpoint_dir', 'cv', 'output directory where checkpoints get written')
cmd:option('-savefile','lstm','filename to autosave the checkpont to. Will be inside checkpoint_dir/')
-- GPU/CPU
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:text()

-- parse input params
opt = cmd:parse(arg)
torch.manualSeed(opt.seed)

if opt.gpuid >= 0 then
    print('using CUDA on GPU ' .. opt.gpuid .. '...')
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(opt.gpuid + 1) -- lua index starts from 1
end
-- create the data loader class
local loader = BatchLoader.create(opt.data_dir, opt.batch_size, opt.seq_length)
opt.vocab_size = loader.vocab_size  -- the number of distinct words
print('vocab size: ' .. opt.vocab_size)
opt.seq_length = loader.max_sentence_l
print('sequence length: ' .. opt.seq_length)
-- make sure output directory exists
if not path.exists(opt.checkpoint_dir) then lfs.mkdir(opt.checkpoint_dir) end

-- define the model: prototypes for one timestep, then clone them in time
protos = {}
print('creating an LSTM with ' .. opt.num_layers .. ' layers')
protos.rnn = LSTM.lstm(opt.vocab_size, opt.rnn_size, opt.num_layers, opt.dropout, opt.word_vec_size)
-- the initial state of the cell/hidden states
init_state = {}
for L=1,opt.num_layers do
    local h_init = torch.zeros(opt.batch_size, opt.rnn_size)
    if opt.gpuid >=0 then h_init = h_init:cuda() end
    table.insert(init_state, h_init:clone())
    table.insert(init_state, h_init:clone())
end
state_predict_index = #init_state -- index of blob to make prediction from
-- classifier on top
protos.softmax = nn.Sequential():add(nn.Linear(opt.rnn_size, opt.vocab_size)):add(nn.LogSoftMax())
-- training criterion (negative log likelihood)
protos.criterion = nn.MaskedLoss()

-- ship the model to the GPU if desired
if opt.gpuid >= 0 then
    for k,v in pairs(protos) do v:cuda() end
end

-- put the above things into one flattened parameters tensor
params, grad_params = model_utils.combine_all_parameters(protos.rnn, protos.softmax)
params:uniform(-0.05, 0.05)
print('number of parameters in the model: ' .. params:nElement())
-- make a bunch of clones after flattening, as that reallocates memory
clones = {}
for name,proto in pairs(protos) do
    print('cloning ' .. name)
    clones[name] = model_utils.clone_many_times(proto, opt.seq_length, not proto.parameters)
end

-- evaluate the loss over an entire split
function eval_split(split_index, max_batches)
    print('evaluating loss over split index ' .. split_index)
    local n = loader.split_sizes[split_index]
    if max_batches ~= nil then n = math.min(max_batches, n) end
    loader:reset_batch_pointer(split_index) -- move batch iteration pointer for this split to front
    local loss = 0
    local rnn_state = {[0] = init_state}
    local count_token = 0
    for i = 1,n do -- iterate over batches in the split
        -- fetch a batch
        local x, y = loader:next_batch(split_index)
        if opt.gpuid >= 0 then -- ship the input arrays to GPU
            -- have to convert to float because integers can't be cuda()'d
            x = x:float():cuda()
            y = y:float():cuda()
        end
        -- forward pass
        for t=1,opt.seq_length do
            clones.rnn[t]:evaluate() -- for dropout proper functioning
            rnn_state[t] = clones.rnn[t]:forward{x[{{}, t}], unpack(rnn_state[t-1])}
            if type(rnn_state[t]) ~= 'table' then rnn_state[t] = {rnn_state[t]} end
            local prediction = clones.softmax[t]:forward(rnn_state[t][state_predict_index])
            local result = clones.criterion[t]:forward({prediction, y[{{}, t}]})[1]
            loss = loss + result[1]
            count_token = count_token + result[3]
        end
        -- carry over lstm state
        rnn_state[0] = rnn_state[#rnn_state]
        print('evaluating' .. i .. '/' .. n .. '...')
    end

    loss = loss / count_token
    local perp = torch.exp(loss)    
    return perp
end

-- do fwd/bwd and return loss, grad_params
local init_state_global = clone_list(init_state)
function feval(x)
    if x ~= params then
        params:copy(x)
    end
    grad_params:zero()

    ------------------ get minibatch -------------------
    local x, y = loader:next_batch(1)
    if opt.gpuid >= 0 then -- ship the input arrays to GPU
        -- have to convert to float because integers can't be cuda()'d
        x = x:float():cuda()
        y = y:float():cuda()
    end
    ------------------- forward pass -------------------
    local rnn_state = {[0] = init_state_global}
    local predictions = {}           -- softmax outputs
    local loss = 0
    for t=1,opt.seq_length do
        clones.rnn[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)
        rnn_state[t] = clones.rnn[t]:forward{x[{{}, t}], unpack(rnn_state[t-1])}
        -- the following line is needed because nngraph tries to be clever
        if type(rnn_state[t]) ~= 'table' then rnn_state[t] = {rnn_state[t]} end
        predictions[t] = clones.softmax[t]:forward(rnn_state[t][state_predict_index])
        loss = loss + clones.criterion[t]:forward({predictions[t], y[{{}, t}]})[1]
    end
    loss = loss / opt.seq_length
    ------------------ backward pass -------------------
    -- initialize gradient at time t to be zeros (there's no influence from future)
    local drnn_state = {[opt.seq_length] = clone_list(init_state, true)} -- true also zeros the clones
    for t=opt.seq_length,1,-1 do
        -- backprop through loss, and softmax/linear
        local doutput_t = clones.criterion[t]:backward({predictions[t], y[{{}, t}]})
        drnn_state[t][state_predict_index] = clones.softmax[t]:backward(rnn_state[t][state_predict_index], doutput_t)
        -- backprop through LSTM timestep
        local drnn_statet_passin = drnn_state[t]
        -- we have to be careful with nngraph again
        if #(rnn_state[t]) == 1 then drnn_statet_passin = drnn_state[t][1] end
        local dlst = clones.rnn[t]:backward({x[{{}, t}], unpack(rnn_state[t-1])}, drnn_statet_passin)
        drnn_state[t-1] = {}
        for k,v in pairs(dlst) do
            if k > 1 then
                -- note we do k-1 because first item is dembeddings, and then follow the 
                -- derivatives of the state, starting at index 2. I know...
                drnn_state[t-1][k-1] = v
            end
        end
    end
    ------------------------ misc ----------------------
    -- transfer final state to initial state (BPTT)
    init_state_global = rnn_state[#rnn_state] -- NOTE: I don't think this needs to be a clone, right?
    -- clip gradient element-wise
    grad_params:clamp(-opt.grad_clip, opt.grad_clip)
    return loss, grad_params
end

-- start optimization here
train_losses = {}
val_losses = {} -- actually the perp
local optim_state = {learningRate = opt.learning_rate, alpha = opt.decay_rate}
local iterations = opt.max_epochs * loader.split_sizes[1]
local iterations_per_epoch = loader.split_sizes[1]
local loss0 = nil
for i = 1, iterations do
    local epoch = i / loader.split_sizes[1]

    local timer = torch.Timer()
    local _, loss = optim.rmsprop(feval, params, optim_state)
    local time = timer:time().real

    local train_loss = loss[1] -- the loss is inside a list, pop it
    train_losses[i] = train_loss

    -- every now and then or on last iteration
    if i % opt.eval_val_every == 0 or i == iterations then
        -- evaluate loss on validation data
        local val_loss = eval_split(2) -- 2 = validation
        val_losses[i] = val_loss

        local savefile = string.format('%s/lm_%s_epoch%.2f_%.4f.t7', opt.checkpoint_dir, opt.savefile, epoch, val_loss)
        print('saving checkpoint to ' .. savefile)
        print ('val loss ' .. val_loss)
        local checkpoint = {}
        checkpoint.protos = protos
        checkpoint.opt = opt
        checkpoint.train_losses = train_losses
        checkpoint.val_loss = val_loss
        checkpoint.val_losses = val_losses
        checkpoint.i = i
        checkpoint.epoch = epoch
        checkpoint.vocab = loader.vocab_mapping
        torch.save(savefile, checkpoint)
    end

    if i % opt.print_every == 0 then
        print(string.format("%d/%d (epoch %.3f), train_loss = %6.8f, grad/param norm = %6.4e, time/batch = %.2fs", i, iterations, epoch, train_loss, grad_params:norm() / params:norm(), time))
    end
   
    if i % 10 == 0 then collectgarbage() end

    -- handle early stopping if things are going really bad
    if loss0 == nil then loss0 = loss[1] end
    if loss[1] > loss0 * 3 then
        print('loss is exploding, aborting.')
        break -- halt
    end
end
local test_loss = eval_split(3)
print (test_loss)
