import os
import argparse

from train import network_train
from test import network_test

def build_parser():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected !!!')

    parser = argparse.ArgumentParser()

    # cpu, gpu mode selection
    parser.add_argument('--cuda-device-no', type=int,
                    help='cpu : -1, gpu : 0 ~ n ', default=0)

    ### arguments for network training
    parser.add_argument('--train-flag', type=str2bool,
                    help='Train flag', required=True)

    parser.add_argument('--max-iter', type=int,
                    help='Train iterations', default=40000)

    parser.add_argument('--batch-size', type=int,
                    help='Batch size', default=8)

    parser.add_argument('--lr', type=float,
                    help='Learning rate to optimize network', default=1e-3)

    parser.add_argument('--check-iter', type=int,
                    help='Number of iteration to check training logs', default=100)

    parser.add_argument('--imsize', type=int,
                    help='Size for resize image during training', default=512)

    parser.add_argument('--cropsize', type=int,
                    help='Size for crop image durning training', default=None)

    parser.add_argument('--layers', type=int, nargs='+',
                    help='Layer indices to extract features', default=[1, 6, 11, 20])

    parser.add_argument('--feature-weight', type=float,
                    help='Feautre loss weight', default=0.1)

    parser.add_argument('--reconstruction-weight', type=float,
                    help='Image reconstruction loss weight', default=1.0)

    parser.add_argument('--tv-weight', type=float,
                    help='Total valiation loss weight', default=1.0)

    parser.add_argument('--train-data-path', type=str,
                    help='Content data path for training')

    parser.add_argument('--save-path', type=str,
                    help='Save path', default='./trained_models/')

    parser.add_argument('--model-load-path', type=str,
                    help="Trained model load path")

    parser.add_argument('--test-content-image-path', type=str,
                    help="Test content image path")

    parser.add_argument('--test-style-image-path', type=str,
                    help="Test style image path")
    
    parser.add_argument('--output-image-path', type=str,
                    help='Path to save the stylized image', default='stylized.jpg')

    parser.add_argument('--style-strength', type=float,
                    help='Content vs style interpolation value: 1(style), 0(content)', default=1.0)

    parser.add_argument('--patch-size', type=int,
                    help='Size of patch for swap normalized content and style features',  default=3)

    parser.add_argument('--patch-stride', type=int,
                    help='Size of patch stride for swap normalized content and style features',  default=1)
    return parser

if __name__ == '__main__':
    parser = build_parser()
    args= parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device_no)

    if args.train_flag:
        network_train(args)
    else:
        network_test(args)
