#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import torch
import numpy as np
from torch import nn


class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1)
        den = torch.sum(predict, dim=1) + torch.sum(target, dim=1) + self.smooth

        dice_score = 2*num / den
        loss_avg = 1 - dice_score.mean()

        return loss_avg


class DiceLoss4BraTS(nn.Module):
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss4BraTS, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict %s & target %s shape do not match' % (predict.shape, target.shape)
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = torch.sigmoid(predict)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss/(target.shape[1]-1 if self.ignore_index!=None else target.shape[1])


class BCELoss4BraTS(nn.Module):
    def __init__(self, ignore_index=None, **kwargs):
        super(BCELoss4BraTS, self).__init__()
        self.kwargs = kwargs
        self.ignore_index = ignore_index
        self.criterion = nn.BCEWithLogitsLoss()

    def weighted_BCE_cross_entropy(self, output, target, weights = None):
        if weights is not None:
            assert len(weights) == 2
            output = torch.clamp(output, min=1e-7, max=1-1e-7)
            bce = weights[1] * (target * torch.log(output)) + \
                  weights[0] * ((1-target) * torch.log((1-output)))
        else:
            output = torch.clamp(output, min=1e-3, max=1 - 1e-3)
            bce = target * torch.log(output) + (1-target) * torch.log((1-output))
        return torch.neg(torch.mean(bce))

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        total_loss = 0
        for i in range(target.shape[1]):
            if i != self.ignore_index:
                bce_loss = self.criterion(predict[:, i], target[:, i])
                total_loss += bce_loss

        return total_loss.mean()


class BCELossBoud(nn.Module):
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(BCELossBoud, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index
        self.criterion = nn.BCEWithLogitsLoss()

    def weighted_BCE_cross_entropy(self, output, target, weights = None):
        if weights is not None:
            assert len(weights) == 2
            output = torch.clamp(output, min=1e-3, max=1-1e-3)
            bce = weights[1] * (target * torch.log(output)) + \
                  weights[0] * ((1-target) * torch.log((1-output)))
        else:
            output = torch.clamp(output, min=1e-3, max=1 - 1e-3)
            bce = target * torch.log(output) + (1-target) * torch.log((1-output))
        return torch.neg(torch.mean(bce))

    def forward(self, predict, target):

        bs, category, depth, width, heigt = target.shape
        bce_loss = []
        for i in range(predict.shape[1]):
            pred_i = predict[:,i]
            targ_i = target[:,i]
            tt = np.log(depth * width * heigt / (target[:, i].cpu().data.numpy().sum()+1))
            bce_i = self.weighted_BCE_cross_entropy(pred_i, targ_i, weights=[1, tt])
            bce_loss.append(bce_i)

        bce_loss = torch.stack(bce_loss)
        total_loss = bce_loss.mean()
        return total_loss


class MultipleOutputLoss2(nn.Module):
    def __init__(self, loss, weight_factors=None):
        """
        use this if you have several outputs and ground truth (both list of same len) and the loss should be computed
        between them (x[0] and y[0], x[1] and y[1] etc)
        :param loss:
        :param weight_factors:
        """
        super(MultipleOutputLoss2, self).__init__()
        self.weight_factors = weight_factors
        self.loss = loss

    def forward(self, x, y):
        assert isinstance(x, (tuple, list)), "x must be either tuple or list"
        assert isinstance(y, (tuple, list)), "y must be either tuple or list"
        # if self.weight_factors is None:
        #     weights = [1] * len(x)
        # else:
        #     weights = self.weight_factors

        # l = weights[0] * self.loss(x[0], y[0])
        # for i in range(1, len(x)):
        #     if weights[i] != 0:
        #         l += weights[i] * self.loss(x[i], y[i])
        l = self.loss(x[0], y[0])
        return l
