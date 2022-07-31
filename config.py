import torch


class Config(object):
    """配置参数"""

    def __init__(self):
        self.flag = 'clo'  # dataset name
        # self.flag = 'ele'  # dataset name

        self.item_count = None  # item node size
        self.cate_count = None  # item cate size
        self.cate_list = None  # cate_list
        self.go_symbol = None  # item start of symbol
        self.eof_symbol = None  # item end of symbol
        self.pad = None  # pad symbol

        self.beam_width = 50
        self.n_best = 5
        self.max_dec_steps = 200

        self.teacher_forcing_ratio = 0.5
        self.clip = 5
        # self.reg_rate = 0
        self.reg_rate = 0.0025
        self.n_steps = 2
        self.bidirectional = False
        self.num_layers = 2

        self.enc_hidden = 128
        self.dec_hidden = 128
        self.hidden = 128  # Embedding Size
        self.dropout = 0.0
        self.filter_sizes = [1, 2, 4, 8, 12, 16, 32, 64]
        # self.filter_sizes = (1, 1)
        self.num_filters = 12

        self.seed = 2022
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 100
        self.epoch = 100
        self.lr = 1e-3
        self.decay = 1e-5
        self.log_interval = 300

    def set_count(self, item_count, cate_count, cate_list):
        self.item_count = item_count
        self.cate_count = cate_count
        self.cate_list = cate_list

        self.go_symbol = item_count
        self.eof_symbol = item_count + 2
        self.pad = item_count + 1
