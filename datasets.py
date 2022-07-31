import copy
import random
import numpy as np
import torch


class DataInput(object):

    def __init__(self, config, raw_data, bundle_map):

        self.config = config
        self.batch_size = config.batch_size
        self.raw_data = raw_data
        self.epoch_size = len(raw_data) // self.batch_size
        if self.epoch_size * self.batch_size < len(raw_data):
            self.epoch_size += 1
        self.i = 0
        self.bundle_map = bundle_map
        self.pad = config.pad

        random.shuffle(self.raw_data)

    def __iter__(self):
        return self

    def __len__(self):
        return self.epoch_size

    def __next__(self):
        if self.i == self.epoch_size:
            raise StopIteration

        start = self.i * self.batch_size
        end = min((self.i + 1) * self.batch_size, len(self.raw_data))

        max_sl_i, max_sl_j, h_sl = 0, 0, 0
        h, i, j = [], [], []
        for t in self.raw_data[start:end]:
            h.append([])
            # h储存的是bundle序列包含的item的序列
            for tt in t[1]: h[-1].extend(self.bundle_map[tt])

            # i储存的是pos bundle包含的item
            i.append([self.config.go_symbol] + list(self.bundle_map[t[2]]) + [self.config.eof_symbol])
            # j储存的是neg bundle包含的item
            j.append([self.config.go_symbol] + list(self.bundle_map[t[3]]) + [self.config.eof_symbol])

            # 分别计算pos、neg和item序列的最大item长度
            max_sl_i = max(max_sl_i, len(i[-1]))
            max_sl_j = max(max_sl_j, len(j[-1]))
            h_sl = max(h_sl, len(h[-1]))

        # dec_out_i = copy.deepcopy(i)
        # dec_out_j = copy.deepcopy(j)

        # 针对max length添加pad
        for k in range(end - start):
            # 给解码器的输出加结束符
            # dec_out_i[k].append(self.config.eof_symbol)
            # dec_out_j[k].append(self.config.eof_symbol)
            # i[k].append(self.config.eof_symbol)
            # j[k].append(self.config.eof_symbol)

            while len(i[k]) < max_sl_i:
                i[k].append(self.pad)
                # dec_out_i[k].append(self.pad)
            while len(j[k]) < max_sl_j:
                j[k].append(self.pad)
                # dec_out_j[k].append(self.pad)
            while len(h[k]) < h_sl:
                h[k].append(self.pad)

        h = torch.LongTensor(np.array(h, np.int32))
        i = torch.LongTensor(np.array(i, np.int32))
        j = torch.LongTensor(np.array(j, np.int32))

        # dec_out_i = torch.LongTensor(np.array(dec_out_i, np.int32))
        # dec_out_j = torch.LongTensor(np.array(dec_out_j, np.int32))

        self.i += 1
        return self.i, (h, i, j)


class TestDataInput(object):

    def __init__(self, config, raw_data, bundle_map):
        self.batch_size = config.batch_size
        self.raw_data = raw_data
        self.epoch_size = len(raw_data) // self.batch_size
        if self.epoch_size * self.batch_size < len(raw_data):
            self.epoch_size += 1
        self.i = 0
        self.bundle_map = bundle_map
        self.pad = config.pad

    def __iter__(self):
        return self

    def __len__(self):
        return self.epoch_size

    def __next__(self):
        if self.i == self.epoch_size:
            raise StopIteration

        start = self.i * self.batch_size
        end = min((self.i + 1) * self.batch_size, len(self.raw_data))

        max_sl_i, h_sl = 0, 0
        h, i = [], []
        for t in self.raw_data[start:end]:
            h.append([])
            # h储存的是bundle序列包含的item的序列
            for tt in t[1]: h[-1].extend(self.bundle_map[tt])

            # i储存的是pos bundle包含的item
            i.append(list(self.bundle_map[t[2]]))

            # 分别计算pos、neg和item序列的最大item长度
            max_sl_i = max(max_sl_i, len(i[-1]))
            h_sl = max(h_sl, len(h[-1]))

        # 针对max length添加pad
        for k in range(end - start):
            while len(i[k]) < max_sl_i: i[k].append(self.pad)
            while len(h[k]) < h_sl: h[k].append(self.pad)

        h = torch.LongTensor(np.array(h, np.int32))
        i = torch.LongTensor(np.array(i, np.int32))
        self.i += 1
        return self.i, (h, i)
