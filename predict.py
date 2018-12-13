"""Train.py docstring.

This module takes an image path as input and transforms the file into a format
readable by the model created in train.py.

The model will return the top_k (3 by default) potential classes and the associated
probabilities. The model will return flower names if cat_to_name.json is passed as
an optional parameter.


Example:
    An example of using this script is below, passing an image file, checkpoint
    file (both required) and enabling GPU inference::

        $ python predict.py image_path checkpoint_file --gpu
"""
import argparse
from PIL import Image

import data_helper, model_helper

parser = argparse.ArgumentParser(
    description='A script to predict a flower type from an image.',
)
parser.add_argument('input', help="Path to image.")
parser.add_argument('checkpoint', help="Path to model checkpoint.")
parser.add_argument('--top_k', action="store", default=3, type=int, help='Specify top k flower probabilities.')
parser.add_argument('--category_names', action="store", default=None, help='Use a mapping of categories to real names.')
parser.add_argument('--gpu', action="store_true", default=False, help='Use GPU for inference.')

arguments = parser.parse_args()
print('Running predict.py with the following arguments {}'.format(arguments))
