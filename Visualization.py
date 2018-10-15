import visdom
import torch
import numpy as np
import cv2 as cv
from torch.nn import functional as F

viz = visdom.Visdom()

def visualize_image(image):
    image = np.transpose(image, (2, 0, 3, 1))
    image = image.reshape(image.shape[0], image.shape[1] * image.shape[2], image.shape[3])
    max = image.max()
    min = image.min()
    if (max - min > 0.0000001):
        image = (image - min)/(max-min)
    cv.imshow("AllClasses Sample", image)
    cv.waitKey(1)

def setup_opencv_viz():
    cv.namedWindow("AllClasses Sample", cv.WINDOW_NORMAL)

#Visdom stuff
def create_vis_plot1(_xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1,)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )
def create_vis_plot2(_xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,2)).cpu(),
        Y=torch.zeros((1,2)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot1(iteration, value, window, update_type):
    viz.line(
        X=torch.ones((1, 1)).cpu() * iteration,
        Y=torch.Tensor([value]).unsqueeze(0).cpu(),
        win=window,
        update=update_type
    )

def update_vis_plot2(iteration, value1, value2, window, update_type):
    viz.line(
        X=torch.ones((1, 2)).cpu() * iteration,
        Y=torch.Tensor([value1, value2]).unsqueeze(0).cpu(),
        win=window,
        update=update_type
    )

def create_plot_images(title):
    return viz.images(tensor=np.random.rand(1, 3, 128, 128),
               nrow=1,
               padding=2,
               opts = dict(title=title, caption='0'))

def plot_images(images, rows, window, title, caption):
    min = images.min()
    max = images.max()
    if (max - min > 0.0000001):
        images = (images-min)/(max-min)
    images = F.interpolate(images, size=128, mode='nearest')
    return viz.images(tensor=images,
               nrow=rows,
               padding=2,
               win=window,
               opts = dict(title=title, caption=caption))