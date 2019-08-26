import os

import torch
import torchvision
import torchvision.transforms as transforms

from PIL import Image

# set Mean and Std of RGB channels of IMAGENET to use pre-trained VGG net
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# normalize a image with mean, std
normalize = transforms.Normalize(mean=IMAGENET_MEAN,
                                 std=IMAGENET_STD)

# denormalize a output image
denormalize = transforms.Normalize(mean=[-mean/std for mean, std in zip(IMAGENET_MEAN, IMAGENET_STD)],
        std=[1/std for std in IMAGENET_STD])

class ImageFolder(torch.utils.data.Dataset):
    def __init__(self, root_path, transform):
        super(ImageFolder, self).__init__()

        self.file_names = sorted(os.listdir(root_path))
        self.root_path = root_path
        self.transform = transform

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.root_path + self.file_names[index])).convert("RGB")
        return self.transform(image)

def get_transformer(imsize=None, cropsize=None):
    transformer = []
    if imsize:
        transformer.append(transforms.Resize(imsize))
    if cropsize:
        transformer.append(transforms.RandomCrop(cropsize)),
    transformer.append(transforms.ToTensor())
    transformer.append(normalize)
    return transforms.Compose(transformer)

def imsave(tensor, path):
    if tensor.is_cuda:
        tensor = tensor.cpu()
    tensor = torchvision.utils.make_grid(tensor)
    torchvision.utils.save_image(denormalize(tensor).clamp_(0.0, 1.0), path)
    return None

def imload(path, imsize=None, cropsize=None):
    transformer = get_transformer(imsize, cropsize)
    return transformer(Image.open(path).convert("RGB")).unsqueeze(0)

def extract_features(model, x, layer_index):
    features = []
    for i, layer in enumerate(model):
        x = layer(x)
        if i in layer_index:
            features.append(x)
    return features
