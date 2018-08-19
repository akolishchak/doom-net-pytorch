#
# focal_loss.py, doom-net
#
# Created by Andrey Kolishchak on 03/11/18.
#
import torch
import torch.nn as nn
import torch.nn.functional as F
from device import device


class FocalLoss(nn.Module):
    def __init__(self, alfa=1, gamma=2):
        super().__init__()
        self.alfa = alfa
        self.gamma = gamma
        self.epsilon = 0 if int(gamma) == gamma and gamma != 0 else 1e-5
        pass

    def forward(self, input, target):
        target = target.view(target.shape[0], 1, *target.shape[1:])
        target_one_hot = torch.zeros(*input.shape, device=device)
        target_one_hot = target_one_hot.scatter_(1, target, 1.0)
        logp = input * target_one_hot
        p = logp.exp()+self.epsilon
        output = -(1-p).pow(self.gamma) * logp
        output = output.sum(dim=1).mean()
        return output


def test():
    loss_nll = nn.NLLLoss2d()
    loss_focal = FocalLoss(gamma=0)
    target = torch.Tensor(2, 1, 5).random_(3).long()

    data = torch.rand(2, 3, 1, 5)
    input1 = torch.Tensor(data, requires_grad=True)
    loss1 = loss_nll(F.log_softmax(input1), target)
    loss1.backward()
    print(loss1)
    print(input1.grad)

    input2 = torch.Tensor(data, requires_grad=True)
    loss2 = loss_focal(F.log_softmax(input2), target)
    loss2.backward()
    print(loss2)
    print(input2.grad)


#test()
