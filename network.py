import torch
import torch.nn as nn

import torchvision
import torchvision.models as models

from style_decorator import StyleDecorator

class AvatarNet(nn.Module):
    def __init__(self, layers):
        super(AvatarNet, self).__init__()
        vgg = models.vgg19(pretrained=True).features
        
        # get network layers
        self.encoders = get_encoder(vgg, layers)
        self.decoders = get_decoder(vgg, layers)

        self.adain = AdaIN()
        self.decorator = StyleDecorator()

    def forward(self, c, s, train_flag=False, style_strength=1.0, patch_size=3, patch_stride=1):

        for encoder in self.encoders:
            c = encoder(c)

        features = []
        for encoder in self.encoders:
            s = encoder(s)
            features.append(s)

        # delete last feature
        del features[-1]

        if not train_flag:
            c = self.decorator(c, s, style_strength, patch_size, patch_stride)

        for decoder in self.decoders:
            c = decoder(c)
            if features:
                c = self.adain(c, features.pop())
        return c

class AdaIN(nn.Module):
    def __init__(self, ):
        super(AdaIN, self).__init__()
        
    def forward(self, x, t, eps=1e-5):
        b, c, h, w = x.size()
        
        x_mean = torch.mean(x.view(b, c, h*w), dim=2, keepdim=True)
        x_std = torch.std(x.view(b, c, h*w), dim=2, keepdim=True)
        
        t_b, t_c, t_h, t_w = t.size()
        t_mean = torch.mean(t.view(t_b, t_c, t_h*t_w), dim=2, keepdim=True)
        t_std = torch.std(t.view(t_b, t_c, t_h*t_w), dim=2, keepdim=True)
        
        x_ = ((x.view(b, c, h*w) - x_mean)/(x_std + eps))*t_std + t_mean
        
        return x_.view(b, c, h, w)

def get_encoder(vgg, layers):
    encoder = nn.ModuleList()
    temp_seq = nn.Sequential()
    for i in range(max(layers)+1):
        temp_seq.add_module(str(i), vgg[i])
        if i in layers:
            encoder.append(temp_seq)
            temp_seq = nn.Sequential()
            
    return encoder

def get_decoder(vgg, layers):
    decoder = nn.ModuleList()
    temp_seq  = nn.Sequential()
    count = 0
    for i in range(max(layers)-1, -1, -1):
        if isinstance(vgg[i], nn.Conv2d):
            out_channels = vgg[i].in_channels
            in_channels = vgg[i].out_channels
            kernel_size = vgg[i].kernel_size

            temp_seq.add_module(str(count), nn.ReflectionPad2d(padding=(1,1,1,1)))
            count += 1
            temp_seq.add_module(str(count), nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size))
            count += 1
            temp_seq.add_module(str(count), nn.ReLU())
            count += 1
        elif isinstance(vgg[i], nn.MaxPool2d):
            temp_seq.add_module(str(count), nn.Upsample(scale_factor=2))
            count += 1

        if i in layers:
            decoder.append(temp_seq)
            temp_seq  = nn.Sequential()

    # append last conv layers without ReLU activation
    decoder.append(temp_seq[:-1])    
    return decoder
