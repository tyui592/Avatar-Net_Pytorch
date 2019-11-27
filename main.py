import os
import argparse

from train import network_train
from test import network_test

def build_parser():
    parser = argparse.ArgumentParser()

    # cpu, gpu mode selection
    parser.add_argument('--gpu-no', type=int,
                    help='cpu : -1, gpu : 0 ~ n ', default=0)

    ### arguments for network training
    parser.add_argument('--train', action='store_true',
                    help='Train flag', default=False)

    parser.add_argument('--max-iter', type=int,
                    help='Train iterations', default=40000)

    parser.add_argument('--batch-size', type=int,
                    help='Batch size', default=16)

    parser.add_argument('--lr', type=float,
                    help='Learning rate to optimize network', default=1e-3)

    parser.add_argument('--check-iter', type=int,
                    help='Number of iteration to check training logs', default=100)

    parser.add_argument('--imsize', type=int,
                    help='Size for resize image during training', default=512)

    parser.add_argument('--cropsize', type=int,
                    help='Size for crop image durning training', default=None)

    parser.add_argument('--cencrop', action='store_true',
                    help='Flag for crop the center rigion of the image, default: randomly crop', default=False)

    parser.add_argument('--layers', type=int, nargs='+',
                    help='Layer indices to extract features', default=[1, 6, 11, 20])

    parser.add_argument('--feature-weight', type=float,
                    help='Feautre loss weight', default=0.1)

    parser.add_argument('--tv-weight', type=float,
                    help='Total valiation loss weight', default=1.0)

    parser.add_argument('--content-dir', type=str,
                    help='Content data path to train the network')

    parser.add_argument('--save-path', type=str,
                    help='Save path', default='./trained_models/')

    parser.add_argument('--check-point', type=str,
                    help="Trained model load path")

    parser.add_argument('--content', type=str,
                    help="Test content image path")

    parser.add_argument('--style', type=str, nargs='+',
                    help="Test style image path")
    
    parser.add_argument('--mask', type=str, nargs='+',
                    help="Mask image for masked stylization", default=None)

    parser.add_argument('--style-strength', type=float,
                    help='Content vs style interpolation value: 1(style), 0(content)', default=1.0)

    parser.add_argument('--interpolation-weights', type=float, nargs='+',
                    help='Multi-style interpolation weights', default=None)

    parser.add_argument('--patch-size', type=int,
                    help='Size of patch for swap normalized content and style features',  default=3)

    parser.add_argument('--patch-stride', type=int,
                    help='Size of patch stride for swap normalized content and style features',  default=1)

    return parser

if __name__ == '__main__':
    parser = build_parser()
    args= parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_no)

    if args.train:
        network_train(args)
    else:
        network_test(args)
