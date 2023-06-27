import torch
import torch.nn as nn
import torch.nn.functional as F

class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        elif mode == 'dice':
            return self.dice_loss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w, d = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss = loss.sum()/n

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss = loss.sum() / n

        return loss

def Dice_loss(logit, target):
    N = target.size(0)
    smooth = 1

    input_obj = logit[:,0,:,:,:]
    input_flat = input_obj.view(N, -1)
    target_flat = target[:,0,:,:,:].view(N, -1)
    # target_flat = target.view(N, -1)

    intersection = input_flat * target_flat
    # print(torch.sum(target_flat))
    # print(torch.sum(input_flat))
    # print(torch.sum(intersection))
    loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
    # print(loss)
    loss = 1 - loss.sum() / N

    return loss

def SR_loss(logit, image, target):
    input_obj = logit
    loss_mat = F.mse_loss(input_obj, image, reduction='none')
    mask = (target * 4 + 1) * 0.2 #
    loss_mat = torch.mul(loss_mat, mask)
    return loss_mat.mean()

def Dice_loss_part(logit, target, loss_map):
    N = target.size(0)
    smooth = 1

    input_obj = torch.mul(logit[:,0,:,:,:], loss_map[:,0,:,:,:])
    input_flat = input_obj.view(N, -1)
    target_flat = torch.mul(target[:,0,:,:,:], loss_map[:,0,:,:]).view(N, -1)
    # target_flat = target.view(N, -1)

    intersection = input_flat * target_flat
    # print(torch.sum(target_flat))
    # print(torch.sum(input_flat))
    # print(torch.sum(intersection))
    loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
    loss = 1 - loss.sum() / N

    return loss

if __name__ == "__main__":
    # loss = SegmentationLosses(cuda=True)
    # a = torch.rand(1, 3, 7, 7).cuda()
    # b = torch.rand(1, 7, 7).cuda()
    # print(loss.CrossEntropyLoss(a, b).item())
    # print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    # print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())
    a = torch.rand(2,1,7,7).cuda()-1
    print(torch.min(a))
    b = torch.rand(2,1,7,7).cuda()
    print(torch.nn.MSELoss()(a,b).type())




