require 'torch'
require 'hdf5'

local utils = require 'util.utils'

local DataLoader = torch.class('DataLoader')


function DataLoader:__init(kwargs)
  local h5_file = utils.get_kwarg(kwargs, 'input_h5')
  self.batch_size = utils.get_kwarg(kwargs, 'batch_size')
  self.seq_length = utils.get_kwarg(kwargs, 'seq_length')
  local N, T = self.batch_size, self.seq_length

  -- Just slurp all the data into memory
  local splits = {}
  local f = hdf5.open(h5_file, 'r')
  splits.train = f:read('/train'):all()
  splits.val = f:read('/val'):all()
  splits.test = f:read('/test'):all()

  self.x_splits = {}
  self.y_splits = {}
  self.split_sizes = {}
  for split, v in pairs(splits) do

     local len = v:nElement()
     local n_pairs = len / (self.seq_length * 2 + 1)
     local n_batches = math.floor(n_pairs / self.batch_size)

     -- prepare containers
     local vx = torch.ByteTensor(self.batch_size * n_batches * self.seq_length)
     local vy = torch.ByteTensor(self.batch_size * n_batches * self.seq_length)

     -- read/write pointers
     local r_pos = 1
     local w_pos = 1

     for tune = 1, n_batches * self.batch_size do
	vx:sub(w_pos,
	       w_pos + self.seq_length - 1):copy(v:sub(r_pos,
						       r_pos + self.seq_length - 1))
	vy:sub(w_pos,
	       w_pos + self.seq_length - 1):copy(v:sub(r_pos + self.seq_length + 1,
						       r_pos + 2 * self.seq_length))
	r_pos = r_pos + 2 * self.seq_length + 1
	w_pos = w_pos + self.seq_length
     end

     self.x_splits[split] = vx:view(-1, N, T):contiguous()
     self.y_splits[split] = vy:view(-1, N, T):contiguous()
     self.split_sizes[split] = n_batches
  end

  self.split_idxs = {train=1, val=1, test=1}
end


function DataLoader:nextBatch(split)
  local idx = self.split_idxs[split]
  assert(idx, 'invalid split ' .. split)
  local x = self.x_splits[split][idx]
  local y = self.y_splits[split][idx]
  if idx == self.split_sizes[split] then
    self.split_idxs[split] = 1
  else
    self.split_idxs[split] = idx + 1
  end
  return x, y
end

