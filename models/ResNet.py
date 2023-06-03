import torch.nn as nn
import torch

def activation_func(activation,is_inplace=False):
    return nn.ModuleDict([['ReLU',  nn.ReLU(inplace=is_inplace)],
                          ['None',  nn.Identity()]])[activation]

def BatchNorm(is_batch_norm,features):
    if is_batch_norm:
        return nn.BatchNorm2d(features)
    else:
        return nn.Identity()

def scaling_layer(is_scaling=True):
    scale_factor = torch.tensor([0.1], dtype=torch.float32)
    if is_scaling:
        return nn.Mul(scale_factor)
    else:
        return  nn.Identity()

def conv_layer(filter_size, padding = 1, is_batch_norm = False, activation_type='ReLU',*args,**kwargs):
    kernel_size, in_c, out_c = filter_size
    return nn.Sequential(nn.Conv2d(in_channels=in_c,out_channels=out_c,kernel_size=kernel_size,padding=padding,bias=False),BatchNorm(is_batch_norm,in_c),
                         activation_func(activation_type))

def ResNetBlock(filter_size):

    return nn.Sequential(conv_layer(filter_size,activation_type='ReLU'),conv_layer(filter_size, activation_type='None'))
    
class ResNetBlocksModule(nn.Module):
    def __init__(self,device, filter_size, num_blocks):
        super().__init__()
        self.device = device
        self.layers = nn.ModuleList([ ResNetBlock(filter_size=filter_size) for _ in range(num_blocks)])
        self.trace = []

    def forward(self, x):
        scale_factor = torch.tensor([0.1], dtype=torch.float32).to(self.device)
        for layer in self.layers:
            
            x =x+ layer(x)*scale_factor
            
        return x

class ResNet(nn.Module):
    def __init__(self,device,in_ch = 2, num_of_resblocks=5):
        super(ResNet, self).__init__()
        self.in_ch=in_ch
        self.num_of_resblocks = num_of_resblocks
        self.device = device
        kernel_size = 3
        padding = 1
        features = 64
        filter1 = [kernel_size,in_ch, features] #map input to size of feature maps
        filter2 = [kernel_size, features, features] #ResNet Blocks
        filter3 = [kernel_size, features, in_ch] #map output channels  to input channels
        self.layer1 = conv_layer(filter_size=filter1,activation_type='None')
        self.layer2 = ResNetBlocksModule(device=self.device,filter_size=filter2,num_blocks=num_of_resblocks)
        self.layer3 =  conv_layer(filter_size=filter2,activation_type='None')
        self.layer4 = conv_layer(filter_size=filter3, activation_type='None')

    def forward(self,input_x):
        l1_out =self.layer1(input_x)
        l2_out = self.layer2(l1_out)
        l3_out = self.layer3(l2_out)
        temp = l3_out+l1_out
        nw_out = self.layer4(temp)
        return nw_out