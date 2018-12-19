
from collections import OrderedDict
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import json
import pathlib
from math import floor


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    image = Image.open(image)

    width, height = image.size
    size = 256, 256

    # Maintain original aspect ratio - shorter size to be 256.
    if height > width:
        ratio = float(height) / float(width)
        newheight = ratio * size[0]
        image = image.resize((size[0], int(floor(newheight))), Image.ANTIALIAS)
        #print('Original size: width {}, height {}, aspect ratio {}'.format(width, height, ratio))
        #print('     New size: width {}, height {}, aspect ratio {}'.format(size[0], int(floor(newheight)), float(newheight) / float(size[0]) ))
    else:
        ## Calculate for the other case
        ratio = float(width) / float(height)
        newwidth = ratio * size[1]
        image = image.resize((int(floor(newwidth)), size[1]), Image.ANTIALIAS)
        #print('Original size: width {}, height {}, aspect ratio {}'.format(width, height, ratio))
        #print('     New size: width {}, height {}, aspect ratio {}'.format(newwidth, size[1],  float(newwidth) / float(size[1])))


    # The box is a 4-tuple defining the left, upper, right, and lower pixel coordinate.
    image = image.crop((
            image.width //2 - (224/2),
            image.height //2 - (224/2),
            image.width //2 + (224/2),
            image.height //2 + (224/2))
        )

    # Normalise image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    np_image = np.array(image) / 255
    np_image = (np_image - mean) / std
    np_image = np_image.transpose((2,0,1))

    return torch.from_numpy(np_image)