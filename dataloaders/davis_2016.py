from __future__ import division

import os
import numpy as np
import cv2
from scipy.misc import imresize

from dataloaders.helpers import *
from torch.utils.data import Dataset


class DAVIS2016(Dataset):
    """DAVIS 2016 dataset constructed using the PyTorch built-in functionalities"""

    def __init__(self, train=True,
                 train_online=False,
                 inputRes=None,
                 db_root_dir='/media/eec/external/Databases/Segmentation/DAVIS-2016',
                 year = '2016',
                 transform=None,
                 meanval=(104.00699, 116.66877, 122.67892),
                 seq_name=None):
        """Loads image to label pairs for tool pose estimation
        db_root_dir: dataset directory with subfolders "JPEGImages" and "Annotations"
        """
        self.train = train
        self.inputRes = inputRes
        self.db_root_dir = db_root_dir
        self.transform = transform
        self.meanval = meanval
        self.seq_name = seq_name
        self.train_online = train_online
        if self.train:
            fname = 'train'
        else:
            fname = 'val'

        #Siddhanj: When there is a sequence name given, we want to load just that one sequence, else, we want to load all sequences
        #Maybe for online training, we will write a separate data loader? For now focus is on just loading the entire sequence

        if self.seq_name is None:

            # Initialize the original DAVIS splits for training the parent network
            with open(os.path.join(db_root_dir,'ImageSets',year, fname + '.txt')) as f:
                seqs = f.readlines()
                img_list = []
                labels = []
                for seq in seqs:
                    images = np.sort(os.listdir(os.path.join(db_root_dir, 'JPEGImages/480p/', seq.strip())))
                    images_path = list(map(lambda x: os.path.join('JPEGImages/480p/', seq.strip(), x), images))
                    img_list.extend(images_path)
                    lab = np.sort(os.listdir(os.path.join(db_root_dir, 'Annotations/480p/', seq.strip())))
                    lab_path = list(map(lambda x: os.path.join('Annotations/480p/', seq.strip(), x), lab))
                    labels.extend(lab_path)
        else:
            seqs = [seq_name]
            # Initialize the per sequence images for online training
            names_img = np.sort(os.listdir(os.path.join(db_root_dir, 'JPEGImages/480p/', str(seq_name))))
            img_list = list(map(lambda x: os.path.join('JPEGImages/480p/', str(seq_name), x), names_img))
            name_label = np.sort(os.listdir(os.path.join(db_root_dir, 'Annotations/480p/', str(seq_name))))
            labels = [os.path.join('Annotations/480p/', str(seq_name), name_label[0])]
            labels.extend([None]*(len(names_img)-1))
            # if self.train:
            #     img_list = [img_list[0]]
            #     labels = [labels[0]]

        assert (len(labels) == len(img_list))

        self.img_list = img_list
        self.labels = labels
        self.seqs = seqs
        print('Done initializing ' + fname + ' Dataset')

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):

        if self.train_online:
            img,gt = self.make_img_gt_pair_train_online()
        else:
            img, gt = self.make_img_gt_pair(idx)

        sample = {'image': img, 'gt': gt}

        if self.seq_name is not None:
            fname = os.path.join(self.seq_name, "%05d" % idx)
            sample['fname'] = fname

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def make_img_gt_pair_train_online(self):
        curr_seq_name = self.seqs[0]
        names_img = np.sort(os.listdir(os.path.join(self.db_root_dir, 'JPEGImages/480p/', str(curr_seq_name.strip()))))
        seq_img_list = list(map(lambda x: os.path.join('JPEGImages/480p/', str(curr_seq_name.strip()), x), names_img))

        names_label = np.sort(
            os.listdir(os.path.join(self.db_root_dir, 'Annotations/480p/', str(curr_seq_name.strip()))))

        seq_labels = list(map(lambda x: os.path.join('Annotations/480p/', str(curr_seq_name.strip()), x), names_label))

        imgSize = self.get_img_size()

        totalNumOfFrames = len(seq_img_list)
        if totalNumOfFrames % 4 != 0:
            totalNumOfFrames = totalNumOfFrames + (4 - totalNumOfFrames % 4)

        imgs = np.zeros([totalNumOfFrames, 3, imgSize[0], imgSize[1]], dtype=np.float32)
        gts = np.zeros([totalNumOfFrames, 1, imgSize[0], imgSize[1]], dtype=np.float32)

        for ctr in range(totalNumOfFrames):

            if ctr >= 1:
                img_name = seq_img_list[0]
                label_name = seq_labels[0]
            else:
                img_name = seq_img_list[ctr]
                label_name = seq_labels[ctr]

            c_img = cv2.imread(os.path.join(self.db_root_dir, img_name))
            # c_img = np.subtract(c_img, np.array(self.meanval, dtype=np.float32))
            c_label = cv2.imread(os.path.join(self.db_root_dir, label_name), 0)
            c_img = np.transpose(c_img, (2, 0, 1))

            # siddhanj:checkpoint c_img needs to be transposed
            if self.inputRes is not None:
                c_img = imresize(c_img, self.inputRes)
                c_label = imresize(c_label, self.inputRes, interp='nearest')

            imgs[ctr, :, :, :] = c_img
            gts[ctr, :, :, :] = c_label

        imgs = np.array(imgs, dtype=np.float32)

        gts = np.array(gts, dtype=np.float32)
        gts = gts / np.max([gts.max(), 1e-8])

        return imgs, gts

    def make_img_gt_pair(self, idx):
        """
        Make the images-ground-truth pair
        """

        curr_seq_name = self.seqs[idx]
        names_img = np.sort(os.listdir(os.path.join(self.db_root_dir, 'JPEGImages/480p/', str(curr_seq_name.strip()))))
        seq_img_list = list(map(lambda x: os.path.join('JPEGImages/480p/', str(curr_seq_name.strip()), x), names_img))



        names_label = np.sort(os.listdir(os.path.join(self.db_root_dir, 'Annotations/480p/', str(curr_seq_name.strip()))))

        seq_labels = list(map(lambda x: os.path.join('Annotations/480p/', str(curr_seq_name.strip()), x), names_label))


        imgSize = self.get_img_size()

        totalNumOfFrames = len(seq_img_list)
        if totalNumOfFrames%4!=0:
            totalNumOfFrames = totalNumOfFrames + (4-totalNumOfFrames%4)

        imgs = np.zeros([totalNumOfFrames,3,imgSize[0],imgSize[1]],dtype=np.float32)
        gts = np.zeros([totalNumOfFrames,1,imgSize[0],imgSize[1]],dtype=np.float32)

        for ctr in range(totalNumOfFrames):

            if ctr >= len(seq_img_list):
                img_name = seq_img_list[len(seq_img_list)-1]
                label_name = seq_labels[len(seq_img_list)-1]
            else:
                img_name = seq_img_list[ctr]
                label_name = seq_labels[ctr]

            c_img = cv2.imread(os.path.join(self.db_root_dir, img_name))
            #c_img = np.subtract(c_img, np.array(self.meanval, dtype=np.float32))
            c_label = cv2.imread(os.path.join(self.db_root_dir, label_name), 0)
            c_img = np.transpose(c_img, (2, 0, 1))

            #siddhanj:checkpoint c_img needs to be transposed
            if self.inputRes is not None:
                c_img = imresize(c_img, self.inputRes)
                c_label = imresize(c_label, self.inputRes, interp='nearest')

            imgs[ctr,:,:,:] = c_img
            gts[ctr,:,:,:] = c_label

        imgs = np.array(imgs, dtype=np.float32)



        gts = np.array(gts, dtype=np.float32)
        gts = gts / np.max([gts.max(), 1e-8])

        return imgs, gts

    def get_img_size(self):
        img = cv2.imread(os.path.join(self.db_root_dir, self.img_list[0]))

        return list(img.shape[:2])


if __name__ == '__main__':
    import custom_transforms as tr
    import torch
    from torchvision import transforms
    from matplotlib import pyplot as plt
    from dataloaders.helpers import *


    #siddhanj: scale messes is it up for somereason. Investigate into this later
    #transforms = transforms.Compose([tr.RandomHorizontalFlip(), tr.Resize(scales=[0.5, 0.8, 1]), tr.ToTensor()])
    transforms = transforms.Compose([tr.VideoResize (sizes=[224, 224]) , tr.ToTensor()])
    #transforms = transforms.Compose([tr.ToTensor()])

    dataset = DAVIS2016(db_root_dir='../Data/DAVIS',
                        train=False, transform=transforms, train_online = True, seq_name='parkour')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)

    for i, data in enumerate(dataloader):
        plt.figure()
        img = data['image'][0,0,:,:,:]
        label = data['gt'][0,0,:,:,:]

        plt.imshow( overlay_mask( tens2image(im_normalize(img)), tens2image(label) ) )

        if i == 10:
            break

    plt.show(block=True)
