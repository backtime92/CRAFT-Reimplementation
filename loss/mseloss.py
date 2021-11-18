import numpy as np
import torch
import torch.nn as nn


class Maploss(nn.Module):
    def __init__(self, use_gpu = True):

        super(Maploss,self).__init__()

    def single_image_loss(self, pre_loss, loss_label):
        batch_size = pre_loss.shape[0]
        # sum_loss = torch.mean(pre_loss.view(-1))*0
        # pre_loss = pre_loss.view(batch_size, -1)
        # loss_label = loss_label.view(batch_size, -1)

        positive_pixel = (loss_label > 0.1).float()
        positive_pixel_number = torch.sum(positive_pixel)
        positive_loss_region = pre_loss * positive_pixel
        positive_loss = torch.sum(positive_loss_region) / positive_pixel_number

        negative_pixel = (loss_label <= 0.1).float()
        negative_pixel_number = torch.sum(negative_pixel)

        if negative_pixel_number < 3*positive_pixel_number:
            negative_loss_region = pre_loss * negative_pixel
            negative_loss = torch.sum(negative_loss_region) / negative_pixel_number
        else:
            negative_loss_region = pre_loss * negative_pixel
            negative_loss = torch.sum(torch.topk(negative_loss_region.view(-1), int(3*positive_pixel_number))[0]) / (positive_pixel_number*3)

        # negative_loss_region = pre_loss * negative_pixel
        # negative_loss = torch.sum(negative_loss_region) / negative_pixel_number


        total_loss = positive_loss + negative_loss
        return total_loss

    def forward(self, region_scores_label, affinity_socres_label, region_scores_pre, affinity_scores_pre, mask):
        loss_fn = torch.nn.MSELoss(reduce=False, size_average=False)

        assert region_scores_label.size() == region_scores_pre.size() and affinity_socres_label.size() == affinity_scores_pre.size()
        loss1 = loss_fn(region_scores_pre, region_scores_label)
        loss2 = loss_fn(affinity_scores_pre, affinity_socres_label)
        loss_region = torch.mul(loss1, mask)
        loss_affinity = torch.mul(loss2, mask)

        char_loss = self.single_image_loss(loss_region, region_scores_label)
        affi_loss = self.single_image_loss(loss_affinity, affinity_socres_label)
        return char_loss + affi_loss