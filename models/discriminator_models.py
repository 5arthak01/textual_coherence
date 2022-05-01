import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


def _sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = Variable(seq_range_expand)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = sequence_length.unsqueeze(1).expand_as(seq_range_expand)
    return seq_range_expand < seq_length_expand


def compute_loss(logit, target, length):
    logit_flat = logit.view(-1, logit.size(-1))
    target_flat = target.view(-1)
    losses_flat = F.cross_entropy(logit_flat, target_flat, reduction="none")
    losses = losses_flat.view(*target.size())
    mask = _sequence_mask(length, target.size(1))
    losses = losses * mask.float()
    loss = losses.sum() / length.float().sum()
    return loss


class MLP(nn.Module):
    def __init__(self, input_dims, n_hiddens, n_class, dropout, use_bn, scorer=None):
        super(MLP, self).__init__()
        assert isinstance(input_dims, int), "Invalid type for input_dims!"
        self.input_dims = input_dims
        current_dims = input_dims
        layers = OrderedDict()

        if isinstance(n_hiddens, int):
            n_hiddens = [n_hiddens]
        else:
            n_hiddens = list(n_hiddens)

        for i, n_hidden in enumerate(n_hiddens):
            l_i = i + 1
            layers["fc{}".format(l_i)] = nn.Linear(current_dims, n_hidden)
            layers["relu{}".format(l_i)] = nn.ReLU()
            layers["drop{}".format(l_i)] = nn.Dropout(dropout)
            if use_bn:
                layers["bn{}".format(l_i)] = nn.BatchNorm1d(n_hidden)
            current_dims = n_hidden
        layers["out"] = nn.Linear(current_dims, n_class)
        if scorer == "sigmoid":
            layers["sigmoid"] = nn.Sigmoid()
        elif scorer == "tanh":
            layers["tanh"] = nn.Tanh()

        self.model = nn.Sequential(layers)

    def forward(self, input):
        return self.model.forward(input)


class MLP_Discriminator(nn.Module):
    def __init__(self, embed_dim, hparams, use_cuda):
        super(MLP_Discriminator, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_state = hparams["hidden_state"]
        self.hidden_layers = hparams["hidden_layers"]
        self.hidden_dropout = hparams["hidden_dropout"]
        self.input_dropout = hparams["input_dropout"]
        self.use_bn = hparams["use_bn"]
        self.bidirectional = hparams["bidirectional"]
        self.use_cuda = use_cuda
        try:
            self.scorer = hparams["scorer"]
        except KeyError:
            self.scorer = None

        self.mlp = MLP(
            embed_dim * 5,
            [self.hidden_state] * self.hidden_layers,
            1,
            self.hidden_dropout,
            self.use_bn,
            self.scorer,
        )
        self.dropout = nn.Dropout(self.input_dropout)
        if self.bidirectional:
            self.backward_mlp = MLP(
                embed_dim * 5,
                [self.hidden_state] * self.hidden_layers,
                1,
                self.hidden_dropout,
                self.use_bn,
                self.scorer,
            )
            self.backward_dropout = nn.Dropout(self.input_dropout)

    def forward(self, s1, s2):
        inputs = torch.cat([s1, s2, s1 - s2, s1 * s2, torch.abs(s1 - s2)], -1)
        scores = self.mlp(self.dropout(inputs))
        if self.bidirectional:
            backward_inputs = torch.cat(
                [s2, s1, s2 - s1, s1 * s2, torch.abs(s1 - s2)], -1
            )
            backward_scores = self.backward_mlp(self.backward_dropout(backward_inputs))
            scores = (scores + backward_scores) / 2
        return scores
