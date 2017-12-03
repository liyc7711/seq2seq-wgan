import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Discriminator(nn.Module):
    """
        Convolutional Neural Networks for Sentence Classification
        https://arxiv.org/abs/1408.5882
    """
    def __init__(self, vocab_size, embed_size, n_classes, dropout=0.5):
        super(Discriminator, self).__init__()
        print("Building Conv classifier model...")
        c_out = 100
        kernels = [3, 4, 5]
        self.static = False

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.convs = nn.ModuleList([nn.Conv2d(1, c_out, (k, embed_size))
                                   for k in kernels])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernels) * c_out, n_classes)

    def forward(self, x):
        x = self.embed(x.t())   # [b, i] -> [b, i, e]
        if self.static:
            x = Variable(x)
        x = x.unsqueeze(1)  # [b, c_in, i, e]
        #  [(b, c_out, i), ...] * len(kernels)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        #  [(b, c_out), ...] * len(kernels)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)  # (b, len(kernels) * c_out)
        logit = self.fc(x)   # (b, o)
        return logit
