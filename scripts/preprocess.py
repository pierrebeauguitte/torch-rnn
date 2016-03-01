# -*- coding: utf-8 -*-

import argparse, json, os
import numpy as np
import h5py
import codecs
import random

parser = argparse.ArgumentParser()
parser.add_argument('--input_txt', default='data/tiny-shakespeare.txt')
parser.add_argument('--output_h5', default='data/tiny-shakespeare.h5')
parser.add_argument('--output_json', default='data/tiny-shakespeare.json')
parser.add_argument('--val_frac', type=float, default=0.1)
parser.add_argument('--test_frac', type=float, default=0.1)
parser.add_argument('--quiet', action='store_true')
parser.add_argument('--encoding', default='utf-8')
args = parser.parse_args()


if __name__ == '__main__':
  if args.encoding == 'bytes': args.encoding = None

  # First go the file once to see how big it is and to build the vocab
  token_to_idx = {}
  total_size = 0
  total_line = 0
  line_size = 0
  with codecs.open(args.input_txt, 'r', args.encoding) as f:
    for line in f:
      line_size = len(line) - 1
      total_size += len(line) - 1
      total_line += 1
      for char in line[:-1]:
        if char not in token_to_idx:
          token_to_idx[char] = len(token_to_idx) + 1
  print token_to_idx

  # Now we can figure out the split sizes
  val_size = int(args.val_frac * total_size)
  test_size = int(args.test_frac * total_size)
  train_size = total_size - val_size - test_size
 
  val_size_line = int(args.val_frac * total_line)
  test_size_line = int(args.test_frac * total_line)
  train_size_line = total_line - val_size_line - test_size_line

  if not args.quiet:
    print 'Total vocabulary size: %d' % len(token_to_idx)
    print 'Total tokens in file: %d' % total_size
    print '  Training size: %d' % train_size
    print '  Val size: %d' % val_size
    print '  Test size: %d' % test_size
    print 'Total lines in file: %d' % total_line
    print '  Training size_line: %d' % train_size_line
    print '  Val size_line: %d' % val_size_line
    print '  Test size_line: %d' % test_size_line

  # Choose the datatype based on the vocabulary size
  dtype = np.uint8
  if len(token_to_idx) > 255:
    dtype = np.uint32
  if not args.quiet:
    print 'Using dtype ', dtype

  # Just load data into memory ... we'll have to do something more clever
  # for huge datasets but this should be fine for now
  train = np.zeros(train_size_line * line_size , dtype=dtype)
  val = np.zeros(val_size_line * line_size, dtype=dtype)
  test = np.zeros(test_size_line * line_size, dtype=dtype)
  splits = [train, val, test]

  # Go through the file again and write data to numpy arrays
  split_idx = 0
  cur_idx = [0, 0, 0]
  with codecs.open(args.input_txt, 'r', args.encoding) as f:
    for line in f:
      while True:
        split_idx = random.randint(0,2)
        if cur_idx[split_idx] < splits[split_idx].size:
          break
      for char in line[:-1]:
        splits[split_idx][cur_idx[split_idx]] = token_to_idx[char]
        cur_idx[split_idx] += 1
      full = True
      for i in range(0,3):
        if cur_idx[i] < splits[i].size:
          full = False
          break
      if full:
        break

  # Write data to HDF5 file
  with h5py.File(args.output_h5, 'w') as f:
    f.create_dataset('train', data=train)
    f.create_dataset('val', data=val)
    f.create_dataset('test', data=test)

  # For 'bytes' encoding, replace non-ascii characters so the json dump
  # doesn't crash
  if args.encoding is None:
    new_token_to_idx = {}
    for token, idx in token_to_idx.iteritems():
      if ord(token) > 127:
        new_token_to_idx['[%d]' % ord(token)] = idx
      else:
        new_token_to_idx[token] = idx
    token_to_idx = new_token_to_idx

  # Dump a JSON file for the vocab
  json_data = {
    'token_to_idx': token_to_idx,
    'idx_to_token': {v: k for k, v in token_to_idx.iteritems()},
  }
  with open(args.output_json, 'w') as f:
    json.dump(json_data, f)
