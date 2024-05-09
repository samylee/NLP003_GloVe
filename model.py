import torch
import torch.nn as nn


class GloVe(nn.Module):
    def __init__(self, vocab_size, vector_size):
        super(GloVe, self).__init__()
        # center words weight and biase
        self.c_weight = nn.Embedding(len(vocab_size), vector_size,
                                     _weight=torch.randn(len(vocab_size),
                                                         vector_size,
                                                         dtype=torch.float,
                                                         requires_grad=True)/100)

        self.c_biase = nn.Embedding(len(vocab_size), 1, _weight=torch.randn(len(vocab_size),
                                                                            1, dtype=torch.float,
                                                                            requires_grad=True)/100)

        # surround words weight and biase
        self.s_weight = nn.Embedding(len(vocab_size), vector_size,
                                     _weight=torch.randn(len(vocab_size),
                                                         vector_size, dtype=torch.float,
                                                         requires_grad=True)/100)

        self.s_biase = nn.Embedding(len(vocab_size), 1,
                                    _weight=torch.randn(len(vocab_size),
                                                        1, dtype=torch.float,
                                                        requires_grad=True)/100)

    def forward(self, c, s):
        c_w = self.c_weight(c)
        c_b = self.c_biase(c)
        s_w = self.s_weight(s)
        s_b = self.s_biase(s)
        return torch.sum(c_w.mul(s_w), 1, keepdim=True) + c_b + s_b