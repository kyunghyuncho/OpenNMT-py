import torch
import torch.nn as nn
from torch.autograd import Variable
import onmt.modules
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack

import numpy


class Encoder(nn.Module):

    def __init__(self, opt, dicts):
        self.layers = opt.layers
        self.num_directions = 2 if opt.brnn else 1
        assert opt.rnn_size % self.num_directions == 0
        self.hidden_size = opt.rnn_size // self.num_directions
        input_size = opt.word_vec_size

        super(Encoder, self).__init__()
        self.word_lut = nn.Embedding(dicts.size(),
                                     opt.word_vec_size,
                                     padding_idx=onmt.Constants.PAD)
        self.rnn = nn.LSTM(input_size, self.hidden_size,
                           num_layers=opt.layers,
                           dropout=opt.dropout,
                           bidirectional=opt.brnn)

    def load_pretrained_vectors(self, opt):
        if opt.pre_word_vecs_enc is not None:
            pretrained = torch.load(opt.pre_word_vecs_enc)
            self.word_lut.weight.data.copy_(pretrained)

    def forward(self, input, hidden=None):
        if isinstance(input, tuple):
            # Lengths data is wrapped inside a Variable.
            lengths = input[1].data.view(-1).tolist()
            emb = pack(self.word_lut(input[0]), lengths)
        else:
            emb = self.word_lut(input)
        outputs, hidden_t = self.rnn(emb, hidden)
        if isinstance(input, tuple):
            outputs = unpack(outputs)[0]
        return hidden_t, outputs

class GNN(nn.Module):

    def __init__(self, opt):
        super(GNN, self).__init__()

        #self.num_directions = 2 if opt.brnn else 1
        self.hidden_size = opt.rnn_size 
        self.iter = opt.iter

        # for adjacency
        self.trans_a1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.trans_a2 = nn.Linear(self.hidden_size, 1, bias=True)
        # for message passing
        self.trans_m = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        # for adaptive damping
        self.trans_g = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

        self.sigm = nn.Sigmoid()

    def adj(self, input, mask=None):
        # first construct adjacency matrices
        seqs = input.transpose(0, 1)
        seqsA = self.trans_a1(seqs.contiguous().view(-1, self.hidden_size))
        seqsB = self.trans_a1(seqs.contiguous().view(-1, self.hidden_size))
        seqsA = seqsA.view(input.size(1), input.size(0), self.hidden_size)
        seqsB = seqsB.view(input.size(1), input.size(0), self.hidden_size)
        seqsA = seqsA.unsqueeze(1).expand(seqsA.size(0), seqsA.size(1), seqsA.size(1), self.hidden_size)
        seqsB = seqsB.unsqueeze(2).expand(seqsB.size(0), seqsB.size(1), seqsB.size(1), self.hidden_size)
        seqs = torch.tanh(seqsA + seqsB).view(seqsA.size(0) * (seqsA.size(1) ** 2), -1)
        scores = self.trans_a2(seqs)
        scores = scores.view(input.size(1),input.size(0),input.size(0))

        scores.masked_fill_(mask.unsqueeze(2).expand_as(scores), -1e+8)
        scores = scores - scores.max(2)[0].expand_as(scores)
        A = torch.exp(scores)

        if mask is not None:
            Z = (A * (1-mask).float().unsqueeze(2).expand_as(A)).sum(2)
            Z.masked_fill_(mask.unsqueeze(2), 1.)
            A = A / Z.expand_as(A)
            A.masked_fill_(mask.unsqueeze(2).expand_as(A), 0.)
        else:
            A = A / A.sum(2).expand_as(A)

        return A

    def forward(self, input, mask=None):
        if mask is not None:
            mask = Variable(mask)

        adjs = self.adj(input, mask)

        # apply GCN self.iter-many times
        hid = input.transpose(0,1)
        for ii in xrange(numpy.minimum(self.iter,input.size(0))):
            hid0 = torch.bmm(adjs, hid)
            hid_ = self.trans_m(hid0.view(-1,self.hidden_size)).view(input.size(1), input.size(0), -1)
            hid_ = torch.tanh(hid_)
            gate_ = self.trans_g(hid0.view(-1,self.hidden_size)).view(input.size(1), input.size(0), -1)
            gate_ = self.sigm(gate_)
            #hid_ = hid_.clamp(min=0.)
            if mask is not None:
                hid_.masked_fill_(mask.unsqueeze(2).expand_as(hid_), 0.)
            # residual 
            hid = gate_ * hid_ + (1. - gate_) * hid

        # should put time first
        return hid.transpose(1,0)

class StackedLSTM(nn.Module):
    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, input, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input, (h_1, c_1)


class Decoder(nn.Module):

    def __init__(self, opt, dicts):
        self.layers = opt.layers
        self.input_feed = opt.input_feed
        input_size = opt.word_vec_size
        if self.input_feed:
            input_size += opt.rnn_size

        super(Decoder, self).__init__()
        self.word_lut = nn.Embedding(dicts.size(),
                                     opt.word_vec_size,
                                     padding_idx=onmt.Constants.PAD)
        self.rnn = StackedLSTM(opt.layers, input_size,
                               opt.rnn_size, opt.dropout)
        self.attn = onmt.modules.GlobalAttention(opt.rnn_size)
        self.dropout = nn.Dropout(opt.dropout)

        self.hidden_size = opt.rnn_size

        self.normalized_output = opt.normalized_output
        if opt.normalized_output:
            if 'run_rate' in opt.__dict__:
                self.run_rate = opt.run_rate
            else:
                self.run_rate = 0.9
            self.register_buffer('hid_mean', torch.zeros(opt.rnn_size))
            self.gpus = opt.gpus
            self.dropout_rate = opt.dropout

    def load_pretrained_vectors(self, opt):
        if opt.pre_word_vecs_dec is not None:
            pretrained = torch.load(opt.pre_word_vecs_dec)
            self.word_lut.weight.data.copy_(pretrained)

    def forward(self, input, hidden, context, init_output):
        emb = self.word_lut(input)

        # n.b. you can increase performance if you compute W_ih * x for all
        # iterations in parallel, but that's only possible if
        # self.input_feed=False
        outputs = []
        output = init_output
        for emb_t in emb.split(1):
            emb_t = emb_t.squeeze(0)
            if self.input_feed:
                emb_t = torch.cat([emb_t, output], 1)

            output, hidden = self.rnn(emb_t, hidden)
            output, attn = self.attn(output, context.t())
            output = self.dropout(output)
            outputs += [output]

        if self.normalized_output:
            if self.training:
                osum = 0.
                for output in outputs:
                    osum += output.mean(0)
                osum /= len(outputs)
                self.hid_mean = self.run_rate * self.hid_mean + (1.-self.run_rate) * osum.data

            if not self.training:
                # because dropout multiplies the activation by self.dropout_rate during training
                hid_mean_ = Variable(-self.hid_mean * (1. - self.dropout_rate), requires_grad=False)
            else:
                hid_mean_ = Variable(-self.hid_mean, requires_grad=False)
            if len(self.gpus) >= 1:
                hid_mean_ = hid_mean_.cuda()

            for output in outputs:
                output.add_(hid_mean_.expand_as(output))

        outputs = torch.stack(outputs)
        return outputs, hidden, attn


class NMTModel(nn.Module):

    def __init__(self, encoder, decoder, gnn=None):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.gnn = gnn

    def make_init_decoder_output(self, context):
        batch_size = context.size(1)
        h_size = (batch_size, self.decoder.hidden_size)
        return Variable(context.data.new(*h_size).zero_(), requires_grad=False)

    def _fix_enc_hidden(self, h):
        #  the encoder hidden is  (layers*directions) x batch x dim
        #  we need to convert it to layers x batch x (directions*dim)
        if self.encoder.num_directions == 2:
            return h.view(h.size(0) // 2, 2, h.size(1), h.size(2)) \
                    .transpose(1, 2).contiguous() \
                    .view(h.size(0) // 2, h.size(1), h.size(2) * 2)
        else:
            return h

    def forward(self, input):
        src = input[0]
        tgt = input[1][:-1]  # exclude last target from inputs
        enc_hidden, context = self.encoder(src)

        if self.gnn is not None:
            context = self.gnn(context, mask=None)

        init_output = self.make_init_decoder_output(context)

        enc_hidden = (self._fix_enc_hidden(enc_hidden[0]),
                      self._fix_enc_hidden(enc_hidden[1]))

        out, dec_hidden, _attn = self.decoder(tgt, enc_hidden,
                                              context, init_output)

        return out




