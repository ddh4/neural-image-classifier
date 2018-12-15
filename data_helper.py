import torch
from torchvision import datasets, transforms

# Load data and return dictionary of datasets
def load_transform_data(data_dir):

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transforms = transforms.Compose([transforms.RandomRotation(degrees=90),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(p=0.2),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                           std=[0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                           std=[0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    trainset = datasets.ImageFolder(train_dir, transform=train_transforms)
    validset = datasets.ImageFolder(valid_dir, transform=test_transforms)
    testset = datasets.ImageFolder(test_dir, transform=test_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(validset, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

    return {'trainset':trainset, 'validset':validset, 'testset':testset},{'trainloader':trainloader, 'validloader':validloader, 'testloader':testloader}
