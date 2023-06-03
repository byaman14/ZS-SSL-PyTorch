import torch.nn as nn
import torch
from models.ResNet import ResNet
import models.data_consistency as ssdu_dc


lamda_start=0.05

class UnrolledNet(nn.Module):
    
    def __init__(self,args, device=torch.device('cuda:0')):
        
        self.args = args
        self.device = device
        super(UnrolledNet, self).__init__()
        self.regularizer = ResNet(self.device,in_ch=2,num_of_resblocks=args.nb_res_blocks) 
        self.lam = nn.Parameter(torch.tensor([lamda_start]))
    
    def forward(self,input_x,trn_mask,loss_mask,sens_maps):
        
        x= input_x
        
        for _ in range(self.args.nb_unroll_blocks):
            x = self.regularizer(x.float())
            rhs = input_x + self.lam * x
            x = ssdu_dc.dc_block(rhs, sens_maps, trn_mask, self.lam,self.args)
        nw_kspace_output = ssdu_dc.SSDU_kspace_transform(x, sens_maps, loss_mask, self.args)

        return x,self.lam, nw_kspace_output