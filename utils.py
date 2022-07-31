import pickle
import random
import time

import numpy as np
import torch

from build_graph import construct_global_graph


def load_data(config):
    flag = config.flag
    assert flag in ['clo', 'ele']
    source_path = 'data/%s/bundle_%s_all.pkl' % (flag, flag)
    save_path = 'model/%s' % flag

    with open(source_path, 'rb') as f:
        train_set = pickle.load(f)
        test_set = pickle.load(f)
        cate_list = pickle.load(f)
        bundle_map = pickle.load(f)
        (user_count, item_count, cate_count, bundle_count, bundle_rank, _) = pickle.load(f)
        gen_groundtruth_data = pickle.load(f)

    cate_list = np.concatenate([cate_list, [cate_count, cate_count, cate_count]])
    cate_list = torch.LongTensor(cate_list).to(config.device)

    train_seq = []
    for data in test_set:
        tmp = []
        for bundle in data[1]:
            tmp += bundle_map[bundle]
        train_seq.append(tmp)

    adj, neighbor = construct_global_graph(train_seq, 2, 12, item_count + 1)

    config.set_count(item_count, cate_count, cate_list)

    return train_set, test_set, gen_groundtruth_data, bundle_map , adj


def init_seed(seed=None):
    if seed is None:
        seed = int(time.time() * 1000 // 1000)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
