import torch
from torch import nn


def dot_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, measure=False, margin=1.0, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        if measure == 'dot':
            self.sim = dot_sim

        self.max_violation = max_violation

    def forward(self, im, s):
        scores = self.sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        cost_s = (self.margin + scores - d1).clamp(min=0)
        cost_im = (self.margin + scores - d2).clamp(min=0)


        mask1 = scores.eq(d1).to(device)
        mask2 = mask1.t()
        cost_s = cost_s.masked_fill_(mask1, 0)
        cost_im = cost_im.masked_fill_(mask2, 0)

        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()
