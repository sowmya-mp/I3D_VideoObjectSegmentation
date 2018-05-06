import random
import cv2
import numpy as np
import torch
import scipy.misc as spy
import torchvision.transforms
import PIL


class ScaleNRotate(object):
    """Scale (zoom-in, zoom-out) and Rotate the image and the ground truth.
    Args:
        two possibilities:
        1.  rots (tuple): (minimum, maximum) rotation angle
            scales (tuple): (minimum, maximum) scale
        2.  rots [list]: list of fixed possible rotation angles
            scales [list]: list of fixed possible scales
    """

    def __init__(self, rots=(-30, 30), scales=(.75, 1.25)):
        assert (isinstance(rots, type(scales)))
        self.rots = rots
        self.scales = scales

    def __call__(self, sample):

        if type(self.rots) == tuple:
            # Continuous range of scales and rotations
            rot = (self.rots[1] - self.rots[0]) * random.random() - \
                  (self.rots[1] - self.rots[0]) / 2

            sc = (self.scales[1] - self.scales[0]) * random.random() - \
                 (self.scales[1] - self.scales[0]) / 2 + 1
        elif type(self.rots) == list:
            # Fixed range of scales and rotations
            rot = self.rots[random.randint(0, len(self.rots))]
            sc = self.scales[random.randint(0, len(self.scales))]

        for elem in sample.keys():
            if 'fname' in elem:
                continue

            tmp = sample[elem]

            h, w = tmp.shape[:2]
            center = (w / 2, h / 2)
            assert (center != 0)  # Strange behaviour warpAffine
            M = cv2.getRotationMatrix2D(center, rot, sc)

            if ((tmp == 0) | (tmp == 1)).all():
                flagval = cv2.INTER_NEAREST
            else:
                flagval = cv2.INTER_CUBIC
            tmp = cv2.warpAffine(tmp, M, (w, h), flags=flagval)

            sample[elem] = tmp

        return sample


class Resize(object):
    """Randomly resize the image and the ground truth to specified scales.
    Args:
        scales (list): the list of scales
    """

    def __init__(self, scales=[0.5, 0.8, 1]):
        self.scales = scales

    def __call__(self, sample):

        # Fixed range of scales
        sc = self.scales[random.randint(0, len(self.scales) - 1)]

        for elem in sample.keys():
            if 'fname' in elem:
                continue
            tmp = sample[elem]

            if tmp.ndim == 2:
                flagval = cv2.INTER_NEAREST
            else:
                flagval = cv2.INTER_CUBIC

            tmp = cv2.resize(tmp, None, fx=sc, fy=sc, interpolation=flagval)

            sample[elem] = tmp

        return sample


class VideoResize(object):
    """Resizes the set of videos frames and the ground truths to specified pixel values.
    Args:
        size (list): the list of sizes
    """

    def __init__(self, sizes=[224, 224]):
        self.sizes = sizes

    def __call__(self, sample):
        print(sample.keys())
        for elem in sample.keys():
            if 'fname' in elem:
                continue
            tmp = sample[elem]

            flagval = 'bilinear'

            num_frames = tmp.shape[0]
            tmp = np.transpose(tmp, (0, 2, 3, 1))
            res = []
            isGT = tmp.shape[3] == 1
            for frameIndex in range(num_frames):
                if isGT:
                    toAppend = spy.imresize(tmp[frameIndex, :, :, 0], (self.sizes[0], self.sizes[1]), interp=flagval)
                    res.append(np.reshape(toAppend, (self.sizes[0], self.sizes[1], 1)))
                else:
                    res.append(spy.imresize(tmp[frameIndex, :, :, :], (self.sizes[0], self.sizes[1]), interp=flagval))
            tmp = np.array(res, dtype="float32")
            tmp = np.transpose(tmp, (0, 3, 1, 2))

            # siddhant: We are normalizing here because i3d expects normalized image values
            # For Ground truth the imresize function converts 0-1 values to 0-255, so we are converting it back
            tmp = np.array(tmp, dtype=np.float32)
            tmp = tmp / np.max([tmp.max(), 1e-8])
            sample[elem] = tmp

        return sample


class VideoLucidDream(object):
    """Resizes the set of videos frames and the ground truths to specified pixel values.
    Args:
        size (list): the list of sizes
    """

    def __init__(self, sizes=[224, 224]):
        self.sizes = sizes

    def __call__(self, sample):

        num_frames = sample['gt'].shape[0]
	
        isFlip = random.random() <0.5
        isJitter = np.ones(num_frames, dtype=bool)
 #       isRotate = np.ones(num_frames, dtype=bool)
        for i in range(num_frames):
 #           if random.random() < 0.5:
 #               isFlip[i] = False
            if random.random() < 0.5:
                isJitter[i] = False
 #           if random.random() < 0.5:
 #               isRotate[i] = False
        seed = 12345

        input_tmp = sample['image']
        gt_tmp = sample['gt']

        flagval = 'bilinear'

        num_frames = input_tmp.shape[0]
        input_tmp = np.transpose(input_tmp, (0, 2, 3, 1))
        gt_tmp = np.transpose(gt_tmp, (0, 2, 3, 1))
        height = input_tmp.shape[2]
        width = input_tmp.shape[1]
        res_input = []
        res_gt = []
        left = np.random.randint(0, width-1-224)
        top = np.random.randint(0, height-1-224)
        imageToCrop = gt_tmp[0, :, :, 0]
        num_gt_pixels = np.sum(imageToCrop)
        while True:
            cropped_gt = imageToCrop[left:left+224,top:top+224]
            #threshold = np.sum(cropped_gt) / float(gt_pixels)
            if np.sum(cropped_gt) >= 0.3*num_gt_pixels:
                break
            else:
                left = np.random.randint(0, width - 1 - 224)
                top = np.random.randint(0, height - 1 - 224)
        for frameIndex in range(num_frames):
            # GT transformations
            imageToCrop = gt_tmp[frameIndex, :, :, 0]
            toAppend = imageToCrop[left:left+224, top:top+224]
            toAppend = np.reshape(toAppend, (self.sizes[0], self.sizes[1], 1))
            if isFlip:
                toAppend = cv2.flip(toAppend, flipCode=1)
            # if isRotate[frameIndex]:
            #    M = cv2.getRotationMatrix2D((toAppend.shape[0]/2,toAppend.shape[1]/2),90,1)
            #    toAppend = cv2.warpAffine(toAppend,M,(toAppend.shape[0],toAppend.shape[1]))

            res_gt.append(np.reshape(toAppend, (self.sizes[0], self.sizes[1], 1)))

            #Input transformations
            imageToCrop = input_tmp[frameIndex, :, :, :]
            toAppend = imageToCrop[left:left+224, top:top+224]
            if isFlip:
                toAppend = cv2.flip(toAppend, flipCode=1)
            # if isRotate[frameIndex]:
            #    M = cv2.getRotationMatrix2D((toAppend.shape[0]/2,toAppend.shape[1]/2),90,1)
            #    toAppend = cv2.warpAffine(toAppend,M,(toAppend.shape[0],toAppend.shape[1]))
            if isJitter[frameIndex]:
                noise = np.random.randint(0, 5, (toAppend.shape[0], toAppend.shape[1]))  # design jitter/noise here
                zitter = np.zeros_like(toAppend)
                zitter[:, :, 1] = noise

                toAppend = cv2.add(toAppend, zitter)

            res_input.append(toAppend)

        input_tmp = np.array(res_input, dtype="float32")
        input_tmp = np.transpose(input_tmp, (0, 3, 1, 2))

        # siddhant: We are normalizing here because i3d expects normalized image values
        # For Ground truth the imresize function converts 0-1 values to 0-255, so we are converting it back
        input_tmp = np.array(input_tmp, dtype=np.float32)
        input_tmp = input_tmp / np.max([input_tmp.max(), 1e-8])
        sample['image'] = input_tmp

        gt_tmp = np.array(res_gt, dtype="float32")
        gt_tmp = np.transpose(gt_tmp, (0, 3, 1, 2))

        # siddhant: We are normalizing here because i3d expects normalized image values
        # For Ground truth the imresize function converts 0-1 values to 0-255, so we are converting it back
        gt_tmp = np.array(gt_tmp, dtype=np.float32)
        gt_tmp = gt_tmp / np.max([gt_tmp.max(), 1e-8])
        sample['gt'] = gt_tmp

        return sample


class RandomHorizontalFlip(object):
    """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

    def __call__(self, sample):

        if random.random() < 0.5:
            for elem in sample.keys():
                if 'fname' in elem:
                    continue
                tmp = sample[elem]
                tmp = cv2.flip(tmp, flipCode=1)
                sample[elem] = tmp

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        for elem in sample.keys():
            if 'fname' in elem:
                continue
            tmp = sample[elem]

            if tmp.ndim == 2:
                tmp = tmp[:, :, np.newaxis]

                # swap color axis because
                # numpy image: H x W x C
                # torch image: C X H X W

            #            tmp = tmp.transpose((2, 0, 1))
            sample[elem] = torch.from_numpy(tmp)

        return sample
