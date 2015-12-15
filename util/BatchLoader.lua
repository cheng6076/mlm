-- Loader for standard encoder-decoder (one input to one output)

local BatchLoader = {}
local stringx = require('pl.stringx')
BatchLoader.__index = BatchLoader

function BatchLoader.create(data_dir, batch_size, max_sentence_l)
    local self = {}
    setmetatable(self, BatchLoader)

    local train_file = path.join(data_dir, 'train.txt')
    local valid_file = path.join(data_dir, 'valid.txt')
    local test_file = path.join(data_dir, 'test.txt')
    local input_files = {train_file, valid_file, test_file}
    local vocab_file = path.join(data_dir, 'vocab.t7')
    local tensor_file = path.join(data_dir, 'data.t7')

    -- construct a tensor with all the data
    if not (path.exists(vocab_file) or path.exists(tensor_file)) then
        print('one-time setup: preprocessing input train/valid/test files in dir: ' .. data_dir)
        BatchLoader.text_to_tensor(input_files, vocab_file, tensor_file, max_sentence_l)
    end

    print('loading data files...')
    local all_data = torch.load(tensor_file) -- train, valid, test tensors
    local vocab_mapping = torch.load(vocab_file)
    self.idx2word, self.word2idx = table.unpack(vocab_mapping)
    self.vocab_size = #self.idx2word  --including the SOS and EOS tokens
    print(string.format('Word vocab size: %d', #self.idx2word))
    -- create sentence-word mappings
    self.max_sentence_l = all_data[1]:size(2)
    -- cut off the end for train/valid sets so that it divides evenly
    self.batch_size = batch_size
    self.split_sizes = {}
    self.all_batches = {}
    print('reshaping tensors...')  
    local x_batches, y_batches, nbatches
    for split, data in ipairs(all_data) do
       local len = data:size(1)
       if len % (batch_size) ~= 0 then
          data = data:sub(1, batch_size * math.floor(len / batch_size))
       end
       local ydata = data:clone()
       ydata:sub(1,-2):copy(data:sub(2,-1))
       ydata[-1]:copy(data[1])
       for i=1,data:size(1) do
         for j=1, data:size(2) do
           if data[i][j]== 0 then data[i][j]= 2 end
         end
       end
       x_batches = data:split(batch_size,1)
       y_batches = ydata:split(batch_size,1)
       nbatches = #x_batches	   
       self.split_sizes[split] = nbatches
       assert(#x_batches == #y_batches)
       self.all_batches[split] = {x_batches, y_batches}
    end
    self.batch_idx = {0,0,0}
    print(string.format('data load done. Number of batches in train: %d, val: %d, test: %d', self.split_sizes[1], self.split_sizes[2], self.split_sizes[3]))
    collectgarbage()
    return self
end

function BatchLoader:reset_batch_pointer(split_idx, batch_idx)
    batch_idx = batch_idx or 0
    self.batch_idx[split_idx] = batch_idx
end

function BatchLoader:next_batch(split_idx)
    -- split_idx is integer: 1 = train, 2 = val, 3 = test
    self.batch_idx[split_idx] = self.batch_idx[split_idx] + 1
    if self.batch_idx[split_idx] > self.split_sizes[split_idx] then
        self.batch_idx[split_idx] = 1 -- cycle around to beginning
    end
    -- pull out the correct next batch
    local idx = self.batch_idx[split_idx]
    return self.all_batches[split_idx][1][idx], self.all_batches[split_idx][2][idx]
end

function BatchLoader.text_to_tensor(input_files, out_vocabfile, out_tensorfile, max_sentence_l)
    print('Processing text into tensors...')
    local f
    local tokens_ZERO = 'ZERO'
    local tokens_START = 'START'
    local tokens_END = 'END'
    local output_tensors = {} -- output tensors for train/val/test
    local vocab_count = {} -- vocab count 
    local max_sentence_l_tmp = 0 -- max sentence length
    local idx2word = {tokens_START, tokens_END} -- SOS and EOS token
    local word2idx = {};  word2idx[tokens_START] = 1; word2idx[tokens_END] = 2
    local split_counts = {}

    -- first go through train/valid/test to get max sentence length
    -- also counts the number of sentences
    for	split = 1,3 do -- split = 1 (train), 2 (val), or 3 (test)
       f = io.open(input_files[split], 'r')       
       local scounts = 0
       for line in f:lines() do
          scounts = scounts + 1
          local wcounts = 0
          for word in line:gmatch'([^%s]+)' do
	     wcounts = wcounts + 1
          end
          max_sentence_l_tmp = math.max(max_sentence_l_tmp, wcounts)
       end
       f:close()
       split_counts[split] = scounts  --the number of sentences in each split
    end
      
    print('After first pass of data, max sentence length is: ' .. max_sentence_l_tmp)
    print(string.format('Token count: train %d, val %d, test %d', 
    			split_counts[1], split_counts[2], split_counts[3]))

    -- if actual max sentence length is less than the limit, use that
    max_sentence_l = math.min(max_sentence_l_tmp, max_sentence_l)
   
    for	split = 1,3 do -- split = 1 (train), 2 (val), or 3 (test)     
       -- Preallocate the tensors we will need.
       -- Watch out the second one needs a lot of RAM.
       output_tensors[split] = torch.zeros(split_counts[split], max_sentence_l + 2):long() -- we use 2 additional slot for start and end tokens
       -- process each file in split
       f = io.open(input_files[split], 'r')
       local sentence_num = 0
       for line in f:lines() do
          sentence_num = sentence_num + 1
          -- append the start token
          output_tensors[split][sentence_num][1] = word2idx[tokens_START]
          -- append tokens in the sentence
          local word_num = 1
          for rword in line:gmatch'([^%s]+)' do
             word_num = word_num + 1
             if word_num == max_sentence_l + 2 then break end -- leave the last token to EOS
             if word2idx[rword]==nil then
                idx2word[#idx2word + 1] = rword 
                word2idx[rword] = #idx2word
             end
             output_tensors[split][sentence_num][word_num] = word2idx[rword]
          end
          -- append the end token
          output_tensors[split][sentence_num][word_num] = word2idx[tokens_END]
       end
    end
    print "done"
    -- save output preprocessed files
    print('saving ' .. out_vocabfile)
    torch.save(out_vocabfile, {idx2word, word2idx})
    print('saving ' .. out_tensorfile)
    torch.save(out_tensorfile, output_tensors)
end

return BatchLoader

