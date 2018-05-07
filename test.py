# Package Includes
from __future__ import division

import os
import socket
import timeit
from datetime import datetime
from tensorboardX import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import cv2
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
from logger import Logger
from dataloaders import helpers

# Select which GPU, -1 if CPU
gpu_id = 0
print('Using GPU: {} '.format(gpu_id))

# Setting of parameters
# Parameters in p are used for the name of the model
p = {
    'trainBatch': 1,  # Number of Images in each mini-batch
}

seqname = 'parkour'

# # Setting other parameters
resume_epoch = 0  # Default is 0, change if want to resume
nEpochs = 150  # Number of epochs for training (500.000/2079)
useTest = True  # See evolution of the test set when training?
testBatch = 1  # Testing Batch
nTestInterval = 5  # Run on test set every nTestInterval epochs
db_root_dir = Path.db_root_dir()
vis_net = 0  # Visualize the network?
snapshot = 50  # Store a model every snapshot epochs
nAveGrad = 1
train_rgb = True
save_dir = Path.save_root_dir()
if not os.path.exists(save_dir):
    os.makedirs(os.path.join(save_dir))

# Network definition
# siddhanj: here number of classes do not matter, as that just helps choose what kind of i3d video we want to use
modelName = 'test'

if train_rgb:
    netRGB = vos.I3D(num_classes=400, modality='rgb')
else:
    netFlow = vos.I3D(num_classes=400, modality='flow')

netRGB.load_state_dict(torch.load('models/online_epoch-199.pth'),False)

tboardLogger = Logger('../logs/tensorboardLogs', 'test')

if gpu_id >= 0:
   # torch.cuda.set_device(device=gpu_id)
    if train_rgb:
	#model = torch.nn.DataParallel(model).cuda()
        netRGB = torch.nn.DataParallel(netRGB,device_ids=[0, 1]).cuda()
        netRGB.eval()
    else:
        netFlow.cuda()
        netFlow.eval()

# Preparation of the data loaders
# Define augmentation transformations as a composition
# composed_transforms = transforms.Compose([tr.RandomHorizontalFlip(),
#                                           tr.ScaleNRotate(rots=(-30, 30), scales=(.75, 1.25)),
#                                           tr.ToTensor()])

composed_transforms = transforms.Compose([tr.ToTensor(),tr.VideoResize([480,848])])

#composed_transforms = transforms.Compose([tr.ToTensor()])

# composed_transforms = transforms.Compose([tr.ToTensor()])

# Testing dataset and its iterator
db_test = db.DAVIS2016(train=False, train_online=False, db_root_dir=db_root_dir, transform=composed_transforms, seq_name=seqname)
testloader = DataLoader(db_test, batch_size=testBatch, shuffle=False, num_workers=2)

def getHeatMapFrom2DArray(inArray):
    inArray_ = (inArray - np.min(inArray)) / (np.max(inArray) - np.min(inArray))
    cm = plt.get_cmap('jet')
    retValue = cm(inArray_)
    return retValue


print("Testing Network")
# Testing for one epoch
for ii, sample_batched in enumerate(testloader):

    inputs, gts = sample_batched['image'], sample_batched['gt']
    outputs = [np.zeros([1,1,inputs.shape[1],inputs.shape[3],inputs.shape[4]])]*5

    numRows = inputs.shape[3]
    numCols = inputs.shape[4]
    outputs_counter = np.zeros([numRows,numCols])
    rowCtr = 0
    colCtr = 0

    while rowCtr + 224 < numRows:
        while colCtr + 224 < numCols:
            inputs_crop = inputs[:,:,:,rowCtr:rowCtr+224,colCtr:colCtr+224]
            gts_crop = gts[:,:,:,rowCtr:rowCtr+224,colCtr:colCtr+224]
            inputs_crop, gts_crop = Variable(inputs_crop, volatile=True), Variable(gts_crop, volatile=True)

            inputs_crop = torch.transpose(inputs_crop, 1, 2)
            gts_crop = torch.transpose(gts_crop, 1, 2)
            if gpu_id >= 0:
                inputs_crop, gts_crop = inputs_crop.cuda(), gts_crop.cuda()

            outputs_crop = netRGB.forward(inputs_crop)
            if gpu_id >=0:
                outputs_crop = [outputs_crop[i].data.cpu().numpy() for i in range(5)]
            else:
                outputs_crop = [outputs_crop[i].data.numpy() for i in range(5)]

            outputs[0][:, :, :, rowCtr:rowCtr + 224, colCtr:colCtr + 224] += outputs_crop[0]
            outputs[1][:, :, :, rowCtr:rowCtr + 224, colCtr:colCtr + 224] += outputs_crop[1]
            outputs[2][:, :, :, rowCtr:rowCtr + 224, colCtr:colCtr + 224] += outputs_crop[2]
            outputs[3][:, :, :, rowCtr:rowCtr + 224, colCtr:colCtr + 224] += outputs_crop[3]
            outputs[4][:, :, :, rowCtr:rowCtr + 224, colCtr:colCtr + 224] += outputs_crop[4]
            outputs_counter[rowCtr:rowCtr+224,colCtr:colCtr+224] +=1
            colCtr+=1
        rowCtr+=1

    outputs = outputs/outputs_counter

    outputs = [Variable(outputs[i])for i in range(5)]

    if gpu_id>=0:
        outputs = [outputs[i].cuda for i in range(5)]

    images_list = []
    number_frames = inputs.shape[2]
    logging_frames = np.arange(number_frames)
    inputs_ = torch.transpose(inputs, 1, 2)
    outputs_ = torch.transpose(outputs[-1], 1, 2)
    if gpu_id >= 0:
        all_inputs = inputs_.data.cpu().numpy()[0, logging_frames, :, :, :]
        all_outputs = outputs_.data.cpu().numpy()[0, logging_frames, :, :, :]
    else:
        all_inputs = inputs_.data.numpy()[0, logging_frames, :, :, :]
        all_outputs = outputs_.data.numpy()[0, logging_frames, :, :, :]
    for imageIndex in range(number_frames):
        inputImage = all_inputs[imageIndex, :, :, :]
        inputImage = np.transpose(inputImage, (1, 2, 0))
        #inputImage = inputImage[:, :, ::-1]
        images_list.append(inputImage)
        mask = all_outputs[imageIndex, 0, :, :]
        heatMap = getHeatMapFrom2DArray(mask)
        images_list.append(heatMap)
        mask = 1 / (1 + np.exp(-mask))
        print('max: ' + str(np.max(mask)) + 'min: ' + str(np.min(mask)))
        mask_ = np.greater(mask, 0.5).astype(np.float32)
        #mask_ = np.greater(mask, 0).astype(np.float32)
        overlayedImage = helpers.overlay_mask(inputImage, mask_)
        overlayedImage = overlayedImage * 255
	fileName = 'results_test/' + str(seqname) + '_' + str(imageIndex) + '.jpg'
	cv2.imwrite(fileName, overlayedImage)
	images_list.append(overlayedImage)
    tboardLogger.image_summary('image_test', images_list, 1)


