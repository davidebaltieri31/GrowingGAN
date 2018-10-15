import torch.utils.data as data
from PIL import Image
import os
import os.path
import torchvision.datasets.folder
import random
from torchvision import transforms
from torch.nn import functional as F

import torch
import cv2 as cv
import numpy as np

class RandomNoise(object):
    def __init__(self, probability):
         self.probabilit = probability
    def __call__(self, img):
        rd = (torch.randn(img.shape) * 0.05) + 1.0
        return img * rd

class GrowingSmallDatasetFolder(data.Dataset):
    def __init__(self, root, init_size = 4, growt_number = 8, images_multiplier = 0, upright = False, do_rgb = False, preload=True):
        self.extensions = torchvision.datasets.folder.IMG_EXTENSIONS

        classes, class_to_idx = torchvision.datasets.folder.find_classes(root)
        samples = torchvision.datasets.folder.make_dataset(root, class_to_idx, self.extensions)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(self.extensions)))
        self.name = root
        self.root = root
        self.loader = torchvision.datasets.folder.default_loader

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples

        self.transforms = []
        self.sizes = []
        self.current_size = 0

        size = init_size

        for i in range(growt_number + 1):
            temp_transforms = []
            temp_transforms.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2,
                                                          hue=0.1) if do_rgb else transforms.Grayscale())
            temp_transforms.append(transforms.Resize(size) if upright else transforms.Resize(int(size * 1.5)))
            if upright:
                temp_transforms.append(transforms.CenterCrop(size))
            else:
                temp_transforms.append(transforms.RandomCrop(size))
                temp_transforms.append(transforms.RandomHorizontalFlip())
                temp_transforms.append(transforms.RandomVerticalFlip())
            temp_transforms.append(transforms.ToTensor())
            temp_transforms.append(RandomNoise(0.5))
            temp_transforms.append(transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0)))
            self.transforms.append(transforms.Compose(temp_transforms))
            self.sizes.append(size)
            size = size * 2

        self.transform = self.transforms[self.current_size]

        self.images = {}

        self.preloaded=False
        if preload:
            self.preloaded=True
            for s, (path, target) in enumerate(samples):
                img = self.loader(path)
                self.images[path] = img

        self.images_multiplier = images_multiplier
        self.images_idx = [i for j in range(self.images_multiplier) for i in range(len(self.samples))]

    def increase_size(self):
        ret = True
        self.current_size = self.current_size + 1
        if self.current_size >= len(self.sizes):
            self.current_size = len(self.sizes)-1
            ret = False
        self.transform = self.transforms[self.current_size]
        return ret

    def decrease_size(self):
        ret = True
        self.current_size = self.current_size - 1
        if self.current_size <0:
            self.current_size = 0
            ret = False
        self.transform = self.transforms[self.current_size]
        return ret

    def set_transitioning(self, value):
        self.is_transitioning = value

    def getitem(self, index):
        path, target = self.samples[index]
        if self.preloaded:
            sample = self.images[path]
        else:
            sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

    def __getitem__(self, index):
        id = self.images_idx[index]
        return self.getitem(id)

    def __len__(self):
        return len(self.images_idx)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

def get_dataset_element(dataset_iterator, is_transitioning, alpha):
    data = next(dataset_iterator)
    sample_big = data[0]
    if is_transitioning:
        sample_small = F.max_pool2d(sample_big, 2, 2)
        sample_small = F.interpolate(sample_small, size=None, scale_factor=2, mode='nearest')
        sample_big = sample_big * alpha + sample_small * (1.0-alpha)
    return sample_big, data[1]

if __name__ == '__main__':
    dataset = GrowingSmallDatasetFolder('D:/Development/Datasets/mnist_png/training', init_size=8, growt_number=1, images_multiplier=1, upright=True, do_rgb=False, preload=False)
    dataset.increase_size()
    dataset_loader = data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=8)
    dataset_iterator = dataset_loader.__iter__()
    total_num_batches = len(dataset) // dataset_loader.batch_size
    cv.namedWindow("AllClasses Sample",cv.WINDOW_NORMAL)
    num = 0
    while (num < total_num_batches):
        num = num + 1
        image, gt = get_dataset_element(dataset_iterator, True, num/100)
        image = image.numpy()
        image = np.transpose(image, (2, 0, 3, 1))
        image = image.reshape(image.shape[0], image.shape[1] * image.shape[2], image.shape[3])
        max = image.max()
        min = image.min()
        if (max - min > 0.0000001):
            image = (image - min) / (max - min)
        cv.imshow("AllClasses Sample", image)
        cv.waitKey(0)

    """for i in range(100):
        dataset.set_alpha(i/100)
        sample, target = dataset.__getitem__(i)
        sample = sample.numpy() + 0.5
        sample = np.transpose(sample, (1, 2, 0))
        cv.namedWindow("test",cv.WINDOW_NORMAL)
        cv.imshow("test",sample)
        cv.waitKey(0)"""


