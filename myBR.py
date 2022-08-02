# -*- coding: utf-8 -*-
"""Seq2Seq(Attention)-Torch

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1eObkehym2HauZo-NBYi39aAsWE1ujExk

# 3 - Neural Machine Translation by Jointly Learning to Align and Translate

In this third notebook on sequence-to-sequence models using PyTorch and TorchText, we'll be implementing the model from [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473). This model achives our best perplexity yet, ~27 compared to ~34 for the previous model.

## Introduction

As a reminder, here is the general encoder-decoder model:

![](https://github.com/bentrevett/pytorch-seq2seq/blob/master/assets/seq2seq1.png?raw=1)

In the previous model, our architecture was set-up in a way to reduce "information compression" by explicitly passing the context vector, $z$, to the decoder at every time-step and by passing both the context vector and embedded input word, $d(y_t)$, along with the hidden state, $s_t$, to the linear layer, $f$, to make a prediction.

![](https://github.com/bentrevett/pytorch-seq2seq/blob/master/assets/seq2seq7.png?raw=1)

Even though we have reduced some of this compression, our context vector still needs to contain all of the information about the source sentence. The model implemented in this notebook avoids this compression by allowing the decoder to look at the entire source sentence (via its hidden states) at each decoding step! How does it do this? It uses *attention*. 

Attention works by first, calculating an attention vector, $a$, that is the length of the source sentence. The attention vector has the property that each element is between 0 and 1, and the entire vector sums to 1. We then calculate a weighted sum of our source sentence hidden states, $H$, to get a weighted source vector, $w$. 

$$w = \sum_{i}a_ih_i$$

We calculate a new weighted source vector every time-step when decoding, using it as input to our decoder RNN as well as the linear layer to make a prediction. We'll explain how to do all of this during the tutorial.

## Preparing Data

Again, the preparation is similar to last time.

First we import all the required modules.
"""
from tqdm import tqdm

from BR import config
from BR.Dense import FA_Linear
from BR.config import Config
from BR.datasets import DataInput, TestDataInput
from BR.utils import load_data, init_seed
from beam import beam_search_decoding
from metrics import print_all_metrics

'''
refer: https://github.com/bentrevett/pytorch-seq2seq/blob/master/3%20-%20Neural%20Machine%20Translation%20by%20Jointly%20Learning%20to%20Align%20and%20Translate.ipynb
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

import spacy
import numpy as np

import random
import math
import time

"""Set the random seeds for reproducability."""

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

"""Load the German and English spaCy models."""

# ! python -m spacy download de
# spacy_de = spacy.load('de_core_news_sm')
# spacy_en = spacy.load('en_core_web_sm')

"""We create the tokenizers."""

# def tokenize_de(text):
#     Tokenizes German text from a string into a list of strings
# return [tok.text for tok in spacy_de.tokenizer(text)]


# def tokenize_en(text):
#     Tokenizes English text from a string into a list of strings
# return [tok.text for tok in spacy_en.tokenizer(text)]


"""The fields remain the same as before."""

# SRC = Field(tokenize=tokenize_de,
#             init_token='<sos>',
#             eos_token='<eos>',
#             lower=True)
#
# TRG = Field(tokenize=tokenize_en,
#             init_token='<sos>',
#             eos_token='<eos>',
#             lower=True)
#
# """Load the data."""
#
# train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(SRC, TRG))
#
# """Build the vocabulary."""
#
# SRC.build_vocab(train_data, min_freq=2)
# TRG.build_vocab(train_data, min_freq=2)
#
# """Define the device."""
#
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""Create the iterators."""

BATCH_SIZE = 128
# train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
#     (train_data, valid_data, test_data),
#     batch_size=BATCH_SIZE,
#     device=device)

"""## Building the Seq2Seq Model

### Encoder

First, we'll build the encoder. Similar to the previous model, we only use a single layer GRU, however we now use a *bidirectional RNN*. With a bidirectional RNN, we have two RNNs in each layer. A *forward RNN* going over the embedded sentence from left to right (shown below in green), and a *backward RNN* going over the embedded sentence from right to left (teal). All we need to do in code is set `bidirectional = True` and then pass the embedded sentence to the RNN as before. 

![](https://github.com/bentrevett/pytorch-seq2seq/blob/master/assets/seq2seq8.png?raw=1)

We now have:

$$\begin{align*}
h_t^\rightarrow &= \text{EncoderGRU}^\rightarrow(e(x_t^\rightarrow),h_{t-1}^\rightarrow)\\
h_t^\leftarrow &= \text{EncoderGRU}^\leftarrow(e(x_t^\leftarrow),h_{t-1}^\leftarrow)
\end{align*}$$

Where $x_0^\rightarrow = \text{<sos>}, x_1^\rightarrow = \text{guten}$ and $x_0^\leftarrow = \text{<eos>}, x_1^\leftarrow = \text{morgen}$.

As before, we only pass an input (`embedded`) to the RNN, which tells PyTorch to initialize both the forward and backward initial hidden states ($h_0^\rightarrow$ and $h_0^\leftarrow$, respectively) to a tensor of all zeros. We'll also get two context vectors, one from the forward RNN after it has seen the final word in the sentence, $z^\rightarrow=h_T^\rightarrow$, and one from the backward RNN after it has seen the first word in the sentence, $z^\leftarrow=h_T^\leftarrow$.

The RNN returns `outputs` and `hidden`. 

`outputs` is of size **[src len, batch size, hid dim * num directions]** where the first `hid_dim` elements in the third axis are the hidden states from the top layer forward RNN, and the last `hid_dim` elements are hidden states from the top layer backward RNN. We can think of the third axis as being the forward and backward hidden states concatenated together other, i.e. $h_1 = [h_1^\rightarrow; h_{T}^\leftarrow]$, $h_2 = [h_2^\rightarrow; h_{T-1}^\leftarrow]$ and we can denote all encoder hidden states (forward and backwards concatenated together) as $H=\{ h_1, h_2, ..., h_T\}$.

`hidden` is of size **[n layers * num directions, batch size, hid dim]**, where **[-2, :, :]** gives the top layer forward RNN hidden state after the final time-step (i.e. after it has seen the last word in the sentence) and **[-1, :, :]** gives the top layer backward RNN hidden state after the final time-step (i.e. after it has seen the first word in the sentence).

As the decoder is not bidirectional, it only needs a single context vector, $z$, to use as its initial hidden state, $s_0$, and we currently have two, a forward and a backward one ($z^\rightarrow=h_T^\rightarrow$ and $z^\leftarrow=h_T^\leftarrow$, respectively). We solve this by concatenating the two context vectors together, passing them through a linear layer, $g$, and applying the $\tanh$ activation function. 

$$z=\tanh(g(h_T^\rightarrow, h_T^\leftarrow)) = \tanh(g(z^\rightarrow, z^\leftarrow)) = s_0$$

**Note**: this is actually a deviation from the paper. Instead, they feed only the first backward RNN hidden state through a linear layer to get the context vector/decoder initial hidden state. This doesn't seem to make sense to me, so we have changed it.

As we want our model to look back over the whole of the source sentence we return `outputs`, the stacked forward and backward hidden states for every token in the source sentence. We also return `hidden`, which acts as our initial hidden state in the decoder.
"""


class Encoder(nn.Module):
    def __init__(self, config, embedding, cate_embedding, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
        self.config = config
        self.embedding = embedding
        self.cate_embedding = cate_embedding
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def get_embedding(self, inputs):
        item_embedding = self.embedding(inputs)
        cate_inputs = self.config.cate_list[inputs]
        cate_embedding = self.cate_embedding(cate_inputs)
        outputs = torch.cat([item_embedding, cate_embedding], -1)
        return outputs

    def forward(self, src):
        '''
        src = [src_len, batch_size]
        '''
        src = src.transpose(0, 1)  # src = [batch_size, src_len]
        embedded = self.dropout(self.get_embedding(src)).transpose(0, 1)  # embedded = [src_len, batch_size, emb_dim]

        # enc_output = [src_len, batch_size, hid_dim * num_directions]
        # enc_hidden = [n_layers * num_directions, batch_size, hid_dim]
        enc_output, enc_hidden = self.rnn(embedded)  # if h_0 is not give, it will be set 0 acquiescently

        # enc_hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        # enc_output are always from the last layer

        # enc_hidden [-2, :, : ] is the last of the forwards RNN 
        # enc_hidden [-1, :, : ] is the last of the backwards RNN

        # initial decoder hidden is final hidden state of the forwards and backwards 
        # encoder RNNs fed through a linear layer
        # s = [batch_size, dec_hid_dim]
        s = torch.tanh(self.fc(torch.cat((enc_hidden[-2, :, :], enc_hidden[-1, :, :]), dim=1)))

        return enc_output, s


"""### Attention

Next up is the attention layer. This will take in the previous hidden state of the decoder, $s_{t-1}$, and all of the stacked forward and backward hidden states from the encoder, $H$. The layer will output an attention vector, $a_t$, that is the length of the source sentence, each element is between 0 and 1 and the entire vector sums to 1.

Intuitively, this layer takes what we have decoded so far, $s_{t-1}$, and all of what we have encoded, $H$, to produce a vector, $a_t$, that represents which words in the source sentence we should pay the most attention to in order to correctly predict the next word to decode, $\hat{y}_{t+1}$. 

First, we calculate the *energy* between the previous decoder hidden state and the encoder hidden states. As our encoder hidden states are a sequence of $T$ tensors, and our previous decoder hidden state is a single tensor, the first thing we do is `repeat` the previous decoder hidden state $T$ times. We then calculate the energy, $E_t$, between them by concatenating them together and passing them through a linear layer (`attn`) and a $\tanh$ activation function. 

$$E_t = \tanh(\text{attn}(s_{t-1}, H))$$ 

This can be thought of as calculating how well each encoder hidden state "matches" the previous decoder hidden state.

We currently have a **[dec hid dim, src len]** tensor for each example in the batch. We want this to be **[src len]** for each example in the batch as the attention should be over the length of the source sentence. This is achieved by multiplying the `energy` by a **[1, dec hid dim]** tensor, $v$.

$$\hat{a}_t = v E_t$$

We can think of $v$ as the weights for a weighted sum of the energy across all encoder hidden states. These weights tell us how much we should attend to each token in the source sequence. The parameters of $v$ are initialized randomly, but learned with the rest of the model via backpropagation. Note how $v$ is not dependent on time, and the same $v$ is used for each time-step of the decoding. We implement $v$ as a linear layer without a bias.

Finally, we ensure the attention vector fits the constraints of having all elements between 0 and 1 and the vector summing to 1 by passing it through a $\text{softmax}$ layer.

$$a_t = \text{softmax}(\hat{a_t})$$

This gives us the attention over the source sentence!

Graphically, this looks something like below. This is for calculating the very first attention vector, where $s_{t-1} = s_0 = z$. The green/teal blocks represent the hidden states from both the forward and backward RNNs, and the attention computation is all done within the pink block.

![](https://github.com/bentrevett/pytorch-seq2seq/blob/master/assets/seq2seq9.png?raw=1)
"""


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim, bias=False)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, s, enc_output):
        # s = [batch_size, dec_hid_dim]
        # enc_output = [src_len, batch_size, enc_hid_dim * 2]

        batch_size = enc_output.shape[1]
        src_len = enc_output.shape[0]

        # repeat decoder hidden state src_len times
        # s = [batch_size, src_len, dec_hid_dim]
        # enc_output = [batch_size, src_len, enc_hid_dim * 2]
        s = s.unsqueeze(1).repeat(1, src_len, 1)
        enc_output = enc_output.transpose(0, 1)

        # energy = [batch_size, src_len, dec_hid_dim]
        energy = torch.tanh(self.attn(torch.cat((s, enc_output), dim=2)))

        # attention = [batch_size, src_len]
        attention = self.v(energy).squeeze(2)

        return F.softmax(attention, dim=1)


"""### Decoder

Next up is the decoder. 

The decoder contains the attention layer, `attention`, which takes the previous hidden state, $s_{t-1}$, all of the encoder hidden states, $H$, and returns the attention vector, $a_t$.

We then use this attention vector to create a weighted source vector, $w_t$, denoted by `weighted`, which is a weighted sum of the encoder hidden states, $H$, using $a_t$ as the weights.

$$w_t = a_t H$$

The embedded input word, $d(y_t)$, the weighted source vector, $w_t$, and the previous decoder hidden state, $s_{t-1}$, are then all passed into the decoder RNN, with $d(y_t)$ and $w_t$ being concatenated together.

$$s_t = \text{DecoderGRU}(d(y_t), w_t, s_{t-1})$$

We then pass $d(y_t)$, $w_t$ and $s_t$ through the linear layer, $f$, to make a prediction of the next word in the target sentence, $\hat{y}_{t+1}$. This is done by concatenating them all together.

$$\hat{y}_{t+1} = f(d(y_t), w_t, s_t)$$

The image below shows decoding the first word in an example translation.

![](https://github.com/bentrevett/pytorch-seq2seq/blob/master/assets/seq2seq10.png?raw=1)

The green/teal blocks show the forward/backward encoder RNNs which output $H$, the red block shows the context vector, $z = h_T = \tanh(g(h^\rightarrow_T,h^\leftarrow_T)) = \tanh(g(z^\rightarrow, z^\leftarrow)) = s_0$, the blue block shows the decoder RNN which outputs $s_t$, the purple block shows the linear layer, $f$, which outputs $\hat{y}_{t+1}$ and the orange block shows the calculation of the weighted sum over $H$ by $a_t$ and outputs $w_t$. Not shown is the calculation of $a_t$.
"""


class Decoder(nn.Module):
    def __init__(self, config, embedding, cate_embedding, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout,
                 attention):
        super().__init__()
        self.config = config
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = embedding
        self.cate_embedding = cate_embedding
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, config.hidden)

        self.FA_Linear = FA_Linear(config)

        self.dropout = nn.Dropout(dropout)

    def get_embedding(self, inputs):
        item_embedding = self.embedding(inputs)
        cate_inputs = self.config.cate_list[inputs]
        cate_embedding = self.cate_embedding(cate_inputs)
        outputs = torch.cat([item_embedding, cate_embedding], -1)
        return outputs

    def forward(self, dec_input, s, enc_output):
        # dec_input = [batch_size]
        # s = [batch_size, dec_hid_dim]
        # enc_output = [src_len, batch_size, enc_hid_dim * 2]

        dec_input = dec_input.unsqueeze(1)  # dec_input = [batch_size, 1]

        embedded = self.dropout(self.get_embedding(dec_input)).transpose(0, 1)  # embedded = [1, batch_size, emb_dim]

        # a = [batch_size, 1, src_len]  
        a = self.attention(s, enc_output).unsqueeze(1)

        # enc_output = [batch_size, src_len, enc_hid_dim * 2]
        enc_output = enc_output.transpose(0, 1)

        # c = [1, batch_size, enc_hid_dim * 2]
        c = torch.bmm(a, enc_output).transpose(0, 1)

        # rnn_input = [1, batch_size, (enc_hid_dim * 2) + emb_dim]
        rnn_input = torch.cat((embedded, c), dim=2)

        # dec_output = [src_len(=1), batch_size, dec_hid_dim]
        # dec_hidden = [n_layers * num_directions, batch_size, dec_hid_dim]
        dec_output, dec_hidden = self.rnn(rnn_input, s.unsqueeze(0))

        # embedded = [batch_size, emb_dim]
        # dec_output = [batch_size, dec_hid_dim]
        # c = [batch_size, enc_hid_dim * 2]
        embedded = embedded.squeeze(0)
        dec_output = dec_output.squeeze(0)
        c = c.squeeze(0)

        # pred = [batch_size, output_dim]
        pred = self.fc_out(torch.cat((dec_output, c, embedded), dim=1))
        pred = F.relu(pred)

        pred = self.FA_Linear(pred)

        return pred, dec_hidden.squeeze(0)


"""### Seq2Seq

This is the first model where we don't have to have the encoder RNN and decoder RNN have the same hidden dimensions, however the encoder has to be bidirectional. This requirement can be removed by changing all occurences of `enc_dim * 2` to `enc_dim * 2 if encoder_is_bidirectional else enc_dim`. 

This seq2seq encapsulator is similar to the last two. The only difference is that the `encoder` returns both the final hidden state (which is the final hidden state from both the forward and backward encoder RNNs passed through a linear layer) to be used as the initial hidden state for the decoder, as well as every hidden state (which are the forward and backward hidden states stacked on top of each other). We also need to ensure that `hidden` and `encoder_outputs` are passed to the decoder. 

Briefly going over all of the steps:
- the `outputs` tensor is created to hold all predictions, $\hat{Y}$
- the source sequence, $X$, is fed into the encoder to receive $z$ and $H$
- the initial decoder hidden state is set to be the `context` vector, $s_0 = z = h_T$
- we use a batch of `<sos>` tokens as the first `input`, $y_1$
- we then decode within a loop:
  - inserting the input token $y_t$, previous hidden state, $s_{t-1}$, and all encoder outputs, $H$, into the decoder
  - receiving a prediction, $\hat{y}_{t+1}$, and a new hidden state, $s_t$
  - we then decide if we are going to teacher force or not, setting the next input as appropriate
"""


class Seq2Seq(nn.Module):
    def __init__(self, config, attn, device):
        super().__init__()
        self.embedding = nn.Embedding(config.item_count + 3, config.hidden // 2)
        self.cate_embedding = nn.Embedding(config.cate_count + 1, config.hidden // 2)
        self.encoder = Encoder(config, self.embedding, self.cate_embedding, config.item_count + 3, config.enc_hidden,
                               config.enc_hidden * 2,
                               config.dec_hidden * 2, config.dropout)
        self.decoder = Decoder(config, self.embedding, self.cate_embedding, config.item_count + 3, config.dec_hidden,
                               config.enc_hidden * 2,
                               config.dec_hidden * 2, config.dropout, attn)
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = [src_len, batch_size]
        # trg = [trg_len, batch_size]
        # teacher_forcing_ratio is probability to use teacher forcing

        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # enc_output is all hidden states of the input sequence, back and forwards
        # s is the final forward and backward hidden states, passed through a linear layer
        enc_output, s = self.encoder(src)

        if self.training:
            l2_loss = torch.sum(s ** 2) / 2

        # first input to the decoder is the <sos> tokens
        dec_input = trg[0, :]

        for t in range(1, trg_len):
            # insert dec_input token embedding, previous hidden state and all encoder hidden states
            # receive output tensor (predictions) and new hidden state
            dec_output, s = self.decoder(dec_input, s, enc_output)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = dec_output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = dec_output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            dec_input = trg[t] if teacher_force else top1

        if self.training:
            return outputs, l2_loss
        else:
            return outputs


"""## Training the Seq2Seq Model

The rest of this tutorial is very similar to the previous one.

We initialise our parameters, encoder, decoder and seq2seq model (placing it on the GPU if we have one).
"""
config = Config()
train_set, val_set, test_set, bundle_map, adj = load_data(config)
# INPUT_DIM = len(SRC.vocab)
# OUTPUT_DIM = len(TRG.vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
ENC_HID_DIM = 512
DEC_HID_DIM = 512
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
# attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
attn = Attention(config.enc_hidden * 2, config.dec_hidden * 2)
# enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
# enc =
# dec =
# dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

model = Seq2Seq(config, attn, device).to(device)
TRG_PAD_IDX = config.pad
criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

"""We then create the training loop..."""


def train(model, train, bundle_map, optimizer, criterion):
    model.train()
    epoch_loss = 0

    for i, uij in tqdm(DataInput(config, train, bundle_map)):
        inputs, trg, neg = uij
        inputs = inputs.transpose(0, 1).contiguous().cuda()
        trg = trg.transpose(0, 1).contiguous().cuda()
        pred, l2_loss = model(inputs, trg)
        pred_dim = pred.shape[-1]

        trg = trg[1:].view(-1)
        pred = pred[1:].view(-1, pred_dim)

        loss = criterion(pred, trg) + config.reg_rate * l2_loss
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.clip)  # 进行梯度裁剪，防止梯度爆炸。clip：梯度阈值
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / (i + 1)

    # for i, batch in enumerate(iterator):
    #     src = batch.src
    #     trg = batch.trg # trg = [trg_len, batch_size]
    #
    #     # pred = [trg_len, batch_size, pred_dim]
    #     pred = model(src, trg)
    #
    #     pred_dim = pred.shape[-1]
    #
    #     # trg = [(trg len - 1) * batch size]
    #     # pred = [(trg len - 1) * batch size, pred_dim]
    #     trg = trg[1:].view(-1)
    #     pred = pred[1:].view(-1, pred_dim)
    #
    #     loss = criterion(pred, trg)
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #     epoch_loss += loss.item()

    # return epoch_loss / len(iterator)


"""...and the evaluation loop, remembering to set the model to `eval` mode and turn off teaching forcing."""


def evaluate(model, val, bundle_map, criterion):
    model.eval()
    auc = 0.0
    with torch.no_grad():
        for i, uij in tqdm(DataInput(config, val, bundle_map)):
            inputs, trg, neg = uij
            inputs = inputs.transpose(0, 1).contiguous().cuda()
            trg = trg.transpose(0, 1).contiguous().cuda()
            neg = neg.transpose(0, 1).contiguous().cuda()

            pos_output = model(inputs, trg, teacher_forcing_ratio=0)
            pos_output = pos_output[1:].transpose(0, 1)
            neg_output = model(inputs, neg, teacher_forcing_ratio=0)
            neg_output = neg_output[1:].transpose(0, 1)

            # pos_output = pos_output[1:].view(-1, pos_output.shape[-1])
            # pos = pos.view(-1)
            for pos_logit, neg_logit, target, negative in zip(pos_output, neg_output, trg[1:].transpose(0, 1),
                                                              neg[1:].transpose(0, 1)):
                pos_loss = criterion(pos_logit, target)
                neg_loss = criterion(neg_logit, negative)
                # 计算batch内的每一项的损失
                if pos_loss < neg_loss:
                    auc += 1
    auc /= len(val)

    return auc


def test(model,test, bundle_map):
    model.eval()
    res = []
    with torch.no_grad():
        for i, uij in tqdm(TestDataInput(config, test, bundle_map)):
            inputs, trg = uij

            inputs = inputs.transpose(0, 1).contiguous().cuda()

            enc_outs, h = model.encoder(inputs)

            # decoded_seqs: (bs, T)
            # start_time = time.time()

            # [batch_size, beam_width, max_dec_steps]
            decoded_seqs = beam_search_decoding(decoder=model.decoder,
                                                enc_outs=enc_outs,
                                                enc_last_h=h,
                                                beam_width=config.beam_width,
                                                n_best=config.n_best,
                                                sos_token=config.go_symbol,
                                                eos_token=config.eof_symbol,
                                                max_dec_steps=config.max_dec_steps,
                                                device=config.device)
            # end_time = time.time()
            # print(f'for loop beam search time: {end_time - start_time:.3f}')

            def save_n_best(decoded_seq):
                for i in range(len(decoded_seq)):
                    res.append((trg[i], decoded_seq[i]))
            save_n_best(decoded_seqs)

        pre_10, div_10 = print_all_metrics(config.flag, res, config.topk1)
        pre_5, div_5 = print_all_metrics(config.flag, res, config.topk2)
        return pre_10, div_10, pre_5, div_5


"""Finally, define a timing function."""


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


"""Then, we train our model, saving the parameters that give us the best validation loss."""

best_auc = 0.0

for epoch in range(10):
    start_time = time.time()
    init_seed(2022)
    train_loss = train(model, train_set, bundle_map, optimizer, criterion)
    auc = evaluate(model, val_set, bundle_map, criterion)
    pre_10, div_10, pre_5, div_5 = test(model, test_set, bundle_map)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if auc > best_auc:
        best_auc = auc
        torch.save(model.state_dict(), 'model_[auc:{}]_[pre.pt')

    print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f}\t Val. Auc: {auc:.3f}\tTrain PPL: {math.exp(train_loss):7.3f}')
    print('%s\tP@%d: %.4f%%\tDiv: %.4f\tP@%d: %.4f%%\tDiv: %.4f'
          % (config.flag, config.topk1, pre_10, div_10, config.topk2, pre_5, div_5))

"""Finally, we test the model on the test set using these "best" parameters."""

# model.load_state_dict(torch.load('tut3-model.pt'))

# test_loss = evaluate(model, val_set, bundle_map, criterion)
#
# print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

"""We've improved on the previous model, but this came at the cost of doubling the training time.

In the next notebook, we'll be using the same architecture but using a few tricks that are applicable to all RNN architectures - packed padded sequences and masking. We'll also implement code which will allow us to look at what words in the input the RNN is paying attention to when decoding the output.
"""
