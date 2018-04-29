import random
import cv2
import numpy as np
import torch
import scipy.misc as spy


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
                  (self.rots[1] - self.rots[0])/2

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
            assert(center != 0)  # Strange behaviour warpAffine
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
                    res.append(np.reshape(toAppend,(self.sizes[0], self.sizes[1],1)))
                else:
                    res.append(spy.imresize(tmp[frameIndex, :, :, :], (self.sizes[0], self.sizes[1]), interp=flagval))
            tmp = np.array(res,dtype="float32")
            tmp = np.transpose(tmp, (0, 3, 1, 2))

            #siddhant: We are normalizing here because i3d expects normalized image values
            # For Ground truth the imresize function converts 0-1 values to 0-255, so we are converting it back
            tmp = np.array(tmp, dtype=np.float32)
            tmp = tmp / np.max([tmp.max(), 1e-8])
            sample[elem] = tmp

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
