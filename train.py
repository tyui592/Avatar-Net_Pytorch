import torch

from network import AvatarNet
from utils import ImageFolder, get_transformer, imsave
from loss import LossCalculator

def network_train(args):
    device = torch.device("cuda" if args.cuda_device_no >= 0 else "cpu")
    torch.save(args, args.save_path+"arguments.pth")

    # get network
    network = AvatarNet(args.layers)
    network = network.to(device)

    # get data set
    data_set = ImageFolder(args.train_data_path, get_transformer(args.imsize, args.cropsize))

    # get loss calculator
    loss_calculator = LossCalculator(device, args.layers, args.feature_weight, args.reconstruction_weight, args.tv_weight)

    # get optimizer
    optimizer = torch.optim.Adam(network.decoders.parameters(), lr=args.lr)


    # training
    for iteration in range(args.max_iter):
        data_loader = torch.utils.data.DataLoader(data_set, batch_size=args.batch_size, shuffle=True)
        image = next(iter(data_loader)).to(device)

        output = network(image, image, train_flag=True)

        total_loss = loss_calculator.calc_total_loss(output, image)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()


        if (iteration + 1) % 1000 == 0:
            loss_calculator.print_loss_seq()
            torch.save(network.state_dict(), args.save_path+"network.pth")
            torch.save(loss_calculator.loss_seq, args.save_path+"loss_seq.pth")
            imsave(output, args.save_path+"training_image.png")

    return network

