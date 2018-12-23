
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

def checkpoint_model(model, arch, trainset, path):
    # Create the folder directory.
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    # Get model parameters and state
    input_size = model.classifier.hidden_layers[0].in_features
    hidden_size = [each.out_features for each in model.classifier.hidden_layers]
    output_size = 102
    model.class_to_idx = trainset.class_to_idx

    #checkpoint = torch.load(filepath)
    checkpoint = {'architecture': arch,
                  'input_size': input_size,
                  'hidden_size': hidden_size,
                  'output_size': output_size,
                  'state_dict': model.state_dict(),
                  'class_indicies': model.class_to_idx}

    print('Checkpointing parameters {} in directory {}/p2_checkpoint.pth'.format(type(checkpoint), path))

    torch.save(checkpoint, path+'/p2_checkpoint.pth')

def load_checkpoint(path):
    # Load checkpoint
    checkpoint = torch.load(path)
    model = load_pretrained_model(checkpoint['architecture'])
    model.classifier = Network(checkpoint['input_size'], checkpoint['hidden_size'], checkpoint['output_size'], dropout=0.2)
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_indicies']

    return model


def load_pretrained_model(arch):
    # Choice of three pretrained models.
    if arch == 'vgg13':
        model = models.vgg13(pretrained=True)
    elif arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'alexnet':
        model = models.alexnet(pretrained=True)
    else:
        print('Uknown model architecture, please choose from alexnet, vgg13 or vgg16.')

    # Freeze Parameters
    for parameters in model.parameters():
        parameters.requires_grad = False

    return model


def predict(image_tensor, model, gpu, topk, category_names, debug=False):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    if gpu:
        image_tensor = image_tensor.type(torch.cuda.FloatTensor).unsqueeze(0)
        model.to('cuda')
    else:
        image_tensor = image_tensor.type(torch.FloatTensor).unsqueeze(0)
        model.to('cpu')

    model.eval()
    with torch.no_grad():
        result = torch.exp(model.forward(image_tensor))

    # Unpack result probabilities and indicies
    probs, indicies = result.topk(topk)
    probs = probs.cpu().numpy().tolist()[0]
    indicies = indicies.cpu().numpy().tolist()[0]

    reverse_dictionary = {val: key for key, val in model.class_to_idx.items()}
    topk_classes = [reverse_dictionary[index] for index in indicies]
    if category_names is not None:
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)
        class_names = [cat_to_name[c] for c in topk_classes]
        return probs, class_names
    else:
        return probs, topk_classes


class Network(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.2):
        super().__init__()
        # Add the input layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_size[0])])

        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_size[:-1], hidden_size[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])

        self.output = nn.Linear(hidden_size[-1], output_size)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''

        # Forward through each layer in `hidden_layers`, with ReLU activation and dropout
        for linear in self.hidden_layers:
            x = F.relu(linear(x))
            x = self.dropout(x)

        x = self.output(x)

        return F.log_softmax(x, dim=1)


def configure_criterion_optimizer(model, learning_rate):
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    return criterion, optimizer


def validation(model, validloader, criterion):
    test_loss = 0
    accuracy = 0

    for ii, (inputs, labels) in enumerate(validloader):
        #images.resize_(images.shape[0], 784)
        images, labels = inputs.to('cuda'), labels.to('cuda')

        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return test_loss, accuracy


def train_model(model, epochs, criterion, optimizer, trainloader, validloader, gpu_enabled):

    print_every = 40
    steps = 0
    running_loss = 0

    # change to user specified device
    if gpu_enabled:
        model.to('cuda')

    for e in range(epochs):

        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1

            if gpu_enabled:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:

                model.eval()

                with torch.no_grad():
                    test_loss, accuracy = validation(model, validloader, criterion)

                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Training Loss: {:.4f}".format(running_loss/print_every),
                      "Validation Loss: {:.4f}".format(test_loss/len(validloader)),
                      "Validation Accuracy: {:.4f}".format(accuracy/len(validloader)))

                running_loss = 0

                model.train()

    return model
