import torch

from network import AvatarNet
from utils import imload, imsave


def network_test(args):
    device = torch.device("cuda" if args.cuda_device_no >= 0 else "cpu")

    # load network
    network = AvatarNet(args.layers)
    network.load_state_dict(torch.load(args.model_load_path))
    network = network.to(device)

    # load target images
    content_image = imload(args.test_content_image_path, args.imsize, args.cropsize)
    style_image = imload(args.test_style_image_path, args.imsize, args.cropsize)
    content_image, style_image = content_image.to(device), style_image.to(device)
    
    # stylize image
    with torch.no_grad():
        output_image = network(content_image, style_image, args.train_flag, args.style_strength, args.patch_size, args.patch_stride)

    imsave(output_image.data, args.output_image_path)
    
    return output_image
