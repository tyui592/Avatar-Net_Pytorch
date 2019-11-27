import time
import torch

from network import AvatarNet, Encoder
from utils import ImageFolder, imsave, lastest_arverage_value

def network_train(args):
    # set device
    device = torch.device('cuda' if args.gpu_no >= 0 else 'cpu')

    # get network
    network = AvatarNet(args.layers).to(device)

    # get data set
    data_set = ImageFolder(args.content_dir, args.imsize, args.cropsize, args.cencrop)

    # get loss calculator
    loss_network = Encoder(args.layers).to(device)
    mse_loss = torch.nn.MSELoss(reduction='mean').to(device)
    loss_seq = {'total':[], 'image':[], 'feature':[], 'tv':[]}

    # get optimizer
    for param in network.encoder.parameters():
        param.requires_grad = False
    optimizer = torch.optim.Adam(network.decoder.parameters(), lr=args.lr)

    # training
    for iteration in range(args.max_iter):
        data_loader = torch.utils.data.DataLoader(data_set, batch_size=args.batch_size, shuffle=True)
        input_image = next(iter(data_loader)).to(device)

        output_image = network(input_image, [input_image], train=True)

        # calculate losses
        total_loss = 0
        ## image reconstruction loss
        image_loss = mse_loss(output_image, input_image)
        loss_seq['image'].append(image_loss.item())
        total_loss += image_loss

        ## feature reconstruction loss
        input_features = loss_network(input_image)
        output_features = loss_network(output_image) 
        feature_loss = 0
        for output_feature, input_feature in zip(output_features, input_features):
            feature_loss += mse_loss(output_feature, input_feature)
        loss_seq['feature'].append(feature_loss.item())
        total_loss += feature_loss * args.feature_weight

        ## total variation loss
        tv_loss = calc_tv_loss(output_image)
        loss_seq['tv'].append(tv_loss.item())
        total_loss += tv_loss * args.tv_weight

        loss_seq['total'].append(total_loss.item())

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # print loss log and save network, loss log and output images
        if (iteration + 1) % args.check_iter == 0:
            imsave(torch.cat([input_image, output_image], dim=0), args.save_path+"training_image.png")
            print("%s: Iteration: [%d/%d]\tImage Loss: %2.4f\tFeature Loss: %2.4f\tTV Loss: %2.4f\tTotal: %2.4f"%(time.ctime(),iteration+1, 
                args.max_iter, lastest_arverage_value(loss_seq['image']), lastest_arverage_value(loss_seq['feature']), 
                lastest_arverage_value(loss_seq['tv']), lastest_arverage_value(loss_seq['total'])))
            torch.save({'iteration': iteration+1,
                'state_dict': network.state_dict(),
                'loss_seq': loss_seq},
                args.save_path+'check_point.pth')

    return network

def calc_tv_loss(x):
    tv_loss = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) 
    tv_loss += torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return tv_loss

