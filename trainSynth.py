import os
import torch
import torch.optim as optim
import cv2
import time

from data.dataset import SynthTextDataLoader

from craft import CRAFT
from loss.mseloss import Maploss
from torch.autograd import Variable


def adjust_learning_rate(optimizer, gamma, step, lr):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = lr * (gamma ** step)
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return param_group['lr']



if __name__ == "__main__":
    synthData_dir = {"synthtext":"/data/CRAFT-Reimplementation/dataset/SynthText"}
    target_size = 768
    batch_size = 16
    num_workers = 8
    lr = 1e-4
    training_lr = 1e-4
    weight_decay = 5e-4
    gamma = 0.8
    whole_training_step = 100000


    synthDataLoader = SynthTextDataLoader(target_size, synthData_dir)
    train_loader = torch.utils.data.DataLoader(synthDataLoader,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=num_workers,
                                               drop_last=True,
                                               pin_memory=True)

    craft = CRAFT()
    craft = torch.nn.DataParallel(craft).cuda()
    craft.load_state_dict(torch.load("/data/CRAFT-Reimplementation/dataset/weights_7000.pth"))
    optimizer = optim.Adam(craft.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = Maploss()

    update_lr_rate_step = 2

    train_step = 40000
    loss_value = 0
    batch_time = 0
    while train_step < whole_training_step:
        for index, (image, region_image, affinity_image, confidence_mask, confidences) in enumerate(train_loader):
            start_time = time.time()
            craft.train()
            if train_step>0 and train_step%20000==0:
                training_lr = adjust_learning_rate(optimizer, gamma, update_lr_rate_step, lr)
                update_lr_rate_step += 1

            images = Variable(image).cuda()
            region_image_label = Variable(region_image).cuda()
            affinity_image_label = Variable(affinity_image).cuda()
            confidence_mask_label = Variable(confidence_mask).cuda()

            output, _ = craft(images)

            out1 = output[:, :, :, 0]
            out2 = output[:, :, :, 1]
            loss = criterion(region_image_label, affinity_image_label, out1, out2, confidence_mask_label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            end_time = time.time()
            loss_value += loss.item()
            batch_time += (end_time - start_time)
            if train_step > 0 and train_step%5==0:
                mean_loss = loss_value / 5
                loss_value = 0
                display_batch_time = time.time()
                avg_batch_time = batch_time/5
                batch_time = 0
                print("{}, training_step: {}|{}, learning rate: {:.8f}, training_loss: {:.5f}, avg_batch_time: {:.5f}".format(time.strftime('%Y-%m-%d:%H:%M:%S',time.localtime(time.time())), train_step, whole_training_step, training_lr, mean_loss, avg_batch_time))

            train_step += 1

            if index % 1000 == 0 and index != 0:
                print('Saving state, index:', index)
                torch.save(craft.state_dict(),
                           '/data/CRAFT-Reimplementation/dataset/weights_' + repr(index) + '.pth')
                # test('/data/CRAFT-pytorch/synweights/synweights_' + repr(index) + '.pth')
                #test('/data/CRAFT-pytorch/craft_mlt_25k.pth')
                # getresult()

