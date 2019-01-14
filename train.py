"""Train.py docstring.

This module allows the user to train a neural network to classify images into 1
of 102 potential classes. The model state and hyperparameters are then saved in
a specified directory, or in the root if not.

This implementation allows three architectures, vgg13, vgg16 and alexnet, which
can be selected using the --arch parameter.

Example:
    An example of using this script is below, passing a data folder (required)
    and enabling GPU training (optional but recommended)::

        $ python train.py data_folder --gpu
"""
import argparse
import data_helper, model_helper
import sys

parser = argparse.ArgumentParser(
    description='A script to train a neural network.',
)
parser.add_argument('data_dir', help="Output language")
parser.add_argument('--save_dir', action="store", default='', help='Specify save directory.')
parser.add_argument('--arch', action="store", default='vgg13', help='Specify model architecture.')
parser.add_argument('--learning_rate', action="store", default=0.001, help='Specify training learning rate.')
parser.add_argument('-a', '--hidden_units', nargs='+', default=[4096, 2048], type=int, help='Specify model hidden units.')
parser.add_argument('--epochs', action="store", default=3, type=int, help='Specify number of training epochs.')
parser.add_argument('--gpu', action="store_true", default=False, help='Specify training via GPU.')

arguments = parser.parse_args()

print('Running train.py with the following arguments {}'.format(arguments))
if not arguments.gpu:
    print('Warning: training using the CPU and may take a while. Set argument --gpu if cuda enabled.')
    
datasets, loaders = data_helper.load_transform_data(arguments.data_dir)

model = model_helper.load_pretrained_model(arguments.arch)

if arguments.arch == 'alexnet':
    in_features = model.classifier[1].in_features
else:
    in_features = model.classifier[0].in_features

classifier = model_helper.Network(in_features, arguments.hidden_units, 102)

model.classifier = classifier

criterion, optimizer = model_helper.configure_criterion_optimizer(model, arguments.learning_rate)

model = model_helper.train_model(model, arguments.epochs, criterion, optimizer, loaders['trainloader'], loaders['validloader'], arguments.gpu)

model_helper.checkpoint_model(model, arguments.arch, datasets['trainset'], arguments.save_dir)
