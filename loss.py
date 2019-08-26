import torch
import torch.nn as nn
import torchvision

import time

from utils import extract_features

class LossCalculator:
    def __init__(self, device, layers, feature_loss_weight, reconstruction_loss_weight, tv_loss_weight):
        # pre-trained loss network to calculate feature loss
        self.loss_network = torchvision.models.vgg19(pretrained=True).features
        self.loss_network = self.loss_network.to(device)
        
        # layer indices to extract features 
        self.layers = layers
        
        # loss weights
        self.feature_loss_weight = feature_loss_weight
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.tv_loss_weight = tv_loss_weight
        
        self.mse_criterion = nn.MSELoss(reduction='mean')
        
        # loss log
        self.loss_seq = dict()
        self.loss_seq['total_loss'] = []
        self.loss_seq['feature_loss'] = []
        self.loss_seq['reconstruction_loss'] = []
        self.loss_seq['tv_loss'] = []
        
    def calc_total_loss(self, output, target):
        total_loss = 0
        
        # reconstruction loss
        reconstruction_loss = self.mse_criterion(output, target)
        self.loss_seq['reconstruction_loss'].append(reconstruction_loss.item())
        total_loss += reconstruction_loss * self.reconstruction_loss_weight
        
        # feature loss
        output_features = extract_features(self.loss_network, output, self.layers)
        target_features = extract_features(self.loss_network, target, self.layers)
        feature_loss = 0
        for output_feature, target_feature in zip(output_features, target_features):
            feature_loss+= self.mse_criterion(output_feature, target_feature) * 1/len(output_features)    
        self.loss_seq['feature_loss'].append(feature_loss.item())
        total_loss += feature_loss * self.feature_loss_weight
        
        # tv loss
        tv_loss = self.calc_tv_loss(output)
        self.loss_seq['tv_loss'].append(tv_loss.item())
        total_loss += tv_loss * self.tv_loss_weight
        
        # total loss 
        self.loss_seq['total_loss'].append(total_loss.item())
        return total_loss
                
    
    def calc_tv_loss(self, x):
        tv_loss = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) 
        tv_loss += torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
        return tv_loss

    def print_loss_seq(self):
        str_ = '%s: '%time.ctime()
        for key, value in self.loss_seq.items():
            if len(value) > 100:
                length = 100
            else:
                length = 1
            str_ += '%s: %2.4f,\t'%(key, sum(value[-length:])/length)
        print(str_)
