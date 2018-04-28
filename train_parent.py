# Package Includes
from __future__ import division

import os
import socket
import timeit
from datetime import datetime
from tensorboardX import SummaryWriter

# PyTorch includes
import torch
from torch.autograd import Variable
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

# Custom includes
from util import visualize as viz
from dataloaders import davis_2016 as db
from dataloaders import custom_transforms as tr
import networks.i3d_osvos as vos
from layers.osvos_layers import class_balanced_cross_entropy_loss
from mypath import Path

# Select which GPU, -1 if CPU
gpu_id = 0
print('Using GPU: {} '.format(gpu_id))

# Setting of parameters
# Parameters in p are used for the name of the model
p = {
    'trainBatch': 1,  # Number of Images in each mini-batch
}

# # Setting other parameters
resume_epoch = 0  # Default is 0, change if want to resume
nEpochs = 240  # Number of epochs for training (500.000/2079)
useTest = True  # See evolution of the test set when training?
testBatch = 1  # Testing Batch
nTestInterval = 5  # Run on test set every nTestInterval epochs
db_root_dir = Path.db_root_dir()
vis_net = 0  # Visualize the network?
snapshot = 40  # Store a model every snapshot epochs
nAveGrad = 10
train_rgb = True
save_dir = Path.save_root_dir()
if not os.path.exists(save_dir):
    os.makedirs(os.path.join(save_dir))

# Network definition
#siddhanj: here number of classes do not matter, as that just helps choose what kind of i3d video we want to use
modelName = 'parent'
if resume_epoch == 0:
    if train_rgb:
        netRGB = vos.I3D(num_classes=400, modality='rgb')
    else:
        netFlow = vos.I3D(num_classes=400, modality='flow')

'''
# REMOVING FUNCTIONALITY FOR RESUMING TRAINING FOR NOW
else:
    net = vo.OSVOS(pretrained=0)
    print("Updating weights from: {}".format(
        os.path.join(save_dir, modelName + '_epoch-' + str(resume_epoch - 1) + '.pth')))
    net.load_state_dict(
        torch.load(os.path.join(save_dir, modelName + '_epoch-' + str(resume_epoch - 1) + '.pth'),
                   map_location=lambda storage, loc: storage))
                   
'''

# Logging into Tensorboard
log_dir = os.path.join(save_dir, 'runs', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
writer = SummaryWriter(log_dir=log_dir, comment='-parent')
y = netRGB.forward(Variable(torch.randn(1, 3, 480, 854)))
writer.add_graph(netRGB, y[-1])

# Visualize the network
if vis_net:
    x = torch.randn(1, 3, 480, 854)
    x = Variable(x)
    y = netRGB.forward(x)
    g = viz.make_dot(y, netRGB.state_dict())
    g.view()

if gpu_id >= 0:
    torch.cuda.set_device(device=gpu_id)
    netRGB.cuda()
    netFlow.cuda()


# Use the following optimizer
lr = 1e-8
wd = 0.0002
_momentum = 0.9
optimizer = optim.SGD(netRGB.parameters(), lr, momentum=_momentum,
                                            weight_decay=wd)


# Preparation of the data loaders
# Define augmentation transformations as a composition
composed_transforms = transforms.Compose([tr.RandomHorizontalFlip(),
                                          tr.ScaleNRotate(rots=(-30, 30), scales=(.75, 1.25)),
                                          tr.ToTensor()])
# Training dataset and its iterator
db_train = db.DAVIS2016(train=True, inputRes=None, db_root_dir=db_root_dir, transform=composed_transforms)
trainloader = DataLoader(db_train, batch_size=p['trainBatch'], shuffle=True, num_workers=2)

# Testing dataset and its iterator
db_test = db.DAVIS2016(train=False, db_root_dir=db_root_dir, transform=tr.ToTensor())
testloader = DataLoader(db_test, batch_size=testBatch, shuffle=False, num_workers=2)

num_img_tr = len(trainloader)
num_img_ts = len(testloader)
running_loss_tr = [0] * 5
running_loss_ts = [0] * 5
loss_tr = []
loss_ts = []
aveGrad = 0

print("Training Network")
# Main Training and Testing Loop
for epoch in range(resume_epoch, nEpochs):
    start_time = timeit.default_timer()
    # One training epoch
    for ii, sample_batched in enumerate(trainloader):

        inputs, gts = sample_batched['image'], sample_batched['gt']

        # Forward-Backward of the mini-batch
        inputs, gts = Variable(inputs), Variable(gts)
        if gpu_id >= 0:
            inputs, gts = inputs.cuda(), gts.cuda()

        outputs = netRGB.forward(inputs)

        # Compute the losses, side outputs and fuse

        losses = [0] * len(outputs)
        for i in range(0, len(outputs)):
            losses[i] = class_balanced_cross_entropy_loss(outputs[i], gts, size_average=False)
            running_loss_tr[i] += losses[i].data[0]
        loss = (1 - epoch / nEpochs)*sum(losses[:-1]) + losses[-1]

        # Print stuff
        if ii % num_img_tr == num_img_tr - 1:
            running_loss_tr = [x / num_img_tr for x in running_loss_tr]
            loss_tr.append(running_loss_tr[-1])
            writer.add_scalar('data/total_loss_epoch', running_loss_tr[-1], epoch)
            print('[Epoch: %d, numImages: %5d]' % (epoch, ii + 1))
            for l in range(0, len(running_loss_tr)):
                print('Loss %d: %f' % (l, running_loss_tr[l]))
                running_loss_tr[l] = 0

            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time))

        # Backward the averaged gradient
        loss /= nAveGrad
        loss.backward()
        aveGrad += 1

        # Update the weights once in nAveGrad forward passes
        if aveGrad % nAveGrad == 0:
            writer.add_scalar('data/total_loss_iter', loss.data[0], ii + num_img_tr * epoch)
            optimizer.step()
            optimizer.zero_grad()
            aveGrad = 0

    # Save the model
    if (epoch % snapshot) == snapshot - 1 and epoch != 0:
        torch.save(netRGB.state_dict(), os.path.join(save_dir, modelName + '_epoch-' + str(epoch) + '.pth'))


    '''
    Siddhant: Not doing this right now. Will come back to this, once we have test code ready?
    # One testing epoch
    if useTest and epoch % nTestInterval == (nTestInterval - 1):
        for ii, sample_batched in enumerate(testloader):
            inputs, gts = sample_batched['image'], sample_batched['gt']

            # Forward pass of the mini-batch
            inputs, gts = Variable(inputs, volatile=True), Variable(gts, volatile=True)
            if gpu_id >= 0:
                inputs, gts = inputs.cuda(), gts.cuda()

            outputs = net.forward(inputs)

            # Compute the losses, side outputs and fuse
            losses = [0] * len(outputs)
            for i in range(0, len(outputs)):
                losses[i] = class_balanced_cross_entropy_loss(outputs[i], gts, size_average=False)
                running_loss_ts[i] += losses[i].data[0]
            loss = (1 - epoch / nEpochs) * sum(losses[:-1]) + losses[-1]

            # Print stuff
            if ii % num_img_ts == num_img_ts - 1:
                running_loss_ts = [x / num_img_ts for x in running_loss_ts]
                loss_ts.append(running_loss_ts[-1])

                print('[Epoch: %d, numImages: %5d]' % (epoch, ii + 1))
                writer.add_scalar('data/test_loss_epoch', running_loss_ts[-1], epoch)
                for l in range(0, len(running_loss_ts)):
                    print('***Testing *** Loss %d: %f' % (l, running_loss_ts[l]))
                    running_loss_ts[l] = 0
    '''
writer.close()