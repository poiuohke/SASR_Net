import numpy as np


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)
        self.gt_image_all = []
        self.pre_image_all = []

    def Dice(self):
        N = len(self.gt_image_all)
        smooth = 1
        dice = 0
        for i in range(N):
            input_obj = self.pre_image_all[i]
            input_flat = np.reshape(input_obj, (-1))
            target_flat = np.reshape(self.gt_image_all[i],(-1))
            # target_flat = target.view(N, -1)

            intersection = input_flat * target_flat
            # print(torch.sum(target_flat))
            # print(torch.sum(input_flat))
            # print(torch.sum(intersection))
            coef = 2 * (np.sum(intersection)  + smooth) / (np.sum(input_flat) + np.sum(target_flat) + smooth)
            dice = dice + coef
        dice = dice / N

        return dice

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        self.pre_image_all.append(pre_image)
        self.gt_image_all.append(gt_image)
        self.gt_image = gt_image
        self.pre_image = pre_image
        assert gt_image.shape == pre_image.shape
        pre_image = (pre_image>0.5).astype(np.uint8)
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)



def FWIou(gt_image, pre_image):
    mask = (gt_image >= 0) & (gt_image < 2)
    label = 2 * gt_image[mask].astype('int') + pre_image[mask]
    count = np.bincount(label, minlength=2 ** 2)
    confusion_matrix = count.reshape(2, 2)

    freq = np.sum(confusion_matrix, axis=1) / np.sum(confusion_matrix)
    iu = np.diag(confusion_matrix) / (
                np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
                np.diag(confusion_matrix))

    FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
    return FWIoU

