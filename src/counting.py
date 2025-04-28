import torch
import torch.nn as nn
from torch.autograd import Variable

class Counter(nn.Module):
    """ Counting module as proposed in [1].
    Count the number of objects from a set of bounding boxes and a set of scores for each bounding box.
    This produces (self.objects + 1) number of count features.
    [1]: Yan Zhang, Jonathon Hare, Adam Prügel-Bennett: Learning to Count Objects in Natural Images for Visual Question Answering.
    https://openreview.net/forum?id=B12Js_yRb
    """
    def __init__(self, objects, already_sigmoided=False):
        super().__init__()
        self.objects = objects
        self.already_sigmoided = already_sigmoided
        self.f = nn.ModuleList([PiecewiseLin(16) for _ in range(8)])
        self.count_activation = None

    def forward(self, boxes, attention):
        boxes, attention = self.filter_most_important(self.objects, boxes, attention)
        if not self.already_sigmoided:
            attention = torch.sigmoid(attention)

        relevancy = self.outer_product(attention)
        distance = 1 - self.iou(boxes, boxes)
        score = self.f[0](relevancy) * self.f[1](distance)

        dedup_score = self.f[3](relevancy) * self.f[4](distance)
        dedup_per_entry, dedup_per_row = self.deduplicate(dedup_score, attention)
        score = score / dedup_per_entry

        correction = self.f[0](attention * attention) / dedup_per_row
        score = score.sum(dim=2).sum(dim=1, keepdim=True) + correction.sum(dim=1, keepdim=True)
        score = (score + 1e-20).sqrt()
        one_hot = self.to_one_hot(score)

        att_conf = (self.f[5](attention) - 0.5).abs()
        dist_conf = (self.f[6](distance) - 0.5).abs()
        conf = self.f[7](att_conf.mean(dim=1, keepdim=True) + dist_conf.mean(dim=2).mean(dim=1, keepdim=True))

        return one_hot * conf

    def deduplicate(self, dedup_score, att):
        att_diff = self.outer_diff(att)
        score_diff = self.outer_diff(dedup_score)
        sim = self.f[2](1 - score_diff).prod(dim=1) * self.f[2](1 - att_diff)
        row_sims = sim.sum(dim=2)
        all_sims = self.outer_product(row_sims)
        return all_sims, row_sims

    def to_one_hot(self, scores):
        scores = scores.clamp(min=0, max=self.objects)
        i = scores.long().data
        f = scores.frac()
        target_l = scores.data.new(i.size(0), self.objects + 1).fill_(0)
        target_r = scores.data.new(i.size(0), self.objects + 1).fill_(0)
        target_l.scatter_(dim=1, index=i.clamp(max=self.objects), value=1)
        target_r.scatter_(dim=1, index=(i + 1).clamp(max=self.objects), value=1)
        return (1 - f) * Variable(target_l) + f * Variable(target_r)

    def filter_most_important(self, n, boxes, attention):
        attention, idx = attention.topk(n, dim=1, sorted=False)
        idx = idx.unsqueeze(dim=1).expand(boxes.size(0), boxes.size(1), idx.size(1))
        boxes = boxes.gather(2, idx)
        return boxes, attention

    def outer(self, x):
        size = tuple(x.size()) + (x.size()[-1],)
        a = x.unsqueeze(dim=-1).expand(*size)
        b = x.unsqueeze(dim=-2).expand(*size)
        return a, b

    def outer_product(self, x):
        a, b = self.outer(x)
        return a * b

    def outer_diff(self, x):
        a, b = self.outer(x)
        return (a - b).abs()

    def iou(self, a, b):
        inter = self.intersection(a, b)
        area_a = self.area(a).unsqueeze(2).expand_as(inter)
        area_b = self.area(b).unsqueeze(1).expand_as(inter)
        return inter / (area_a + area_b - inter + 1e-12)

    def area(self, box):
        x = (box[:, 2, :] - box[:, 0, :]).clamp(min=0)
        y = (box[:, 3, :] - box[:, 1, :]).clamp(min=0)
        return x * y

    def intersection(self, a, b):
        size = (a.size(0), 2, a.size(2), b.size(2))
        min_point = torch.max(
            a[:, :2, :].unsqueeze(dim=3).expand(*size),
            b[:, :2, :].unsqueeze(dim=2).expand(*size),
        )
        max_point = torch.min(
            a[:, 2:, :].unsqueeze(dim=3).expand(*size),
            b[:, 2:, :].unsqueeze(dim=2).expand(*size),
        )
        inter = (max_point - min_point).clamp(min=0)
        area = inter[:, 0, :, :] * inter[:, 1, :, :]
        return area

class PiecewiseLin(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.weight = nn.Parameter(torch.ones(n + 1))

        self.weight.data[0] = 0

    def forward(self, x):
        w = self.weight.abs()
        w = w / w.sum()
        w = w.view([self.n + 1] + [1] * x.dim())
        csum = w.cumsum(dim=0)
        csum = csum.expand((self.n + 1,) + tuple(x.size()))
        w = w.expand_as(csum)
        y = self.n * x.unsqueeze(0)
        idx = Variable(y.long().data)
        f = y.frac()
        x = csum.gather(0, idx.clamp(max=self.n))
        x = x + f * w.gather(0, (idx + 1).clamp(max=self.n))
        return x.squeeze(0)
