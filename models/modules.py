import torch
import torch.nn as nn

class Dataset(torch.utils.data.Dataset):
    def __init__(self,trn_atb, trn_mask, loss_mask, sens_maps, ref_kspace):
        self.trn_atb = trn_atb
        self.trn_mask = trn_mask
        self.loss_mask = loss_mask
        self.sens_maps = sens_maps
        self.ref_kspace = ref_kspace
        
    def __len__(self):
        return len(self.trn_atb)
        
    def __getitem__(self,idx):
        nw_input = self.trn_atb[idx] 
        trn_mask , loss_mask = self.trn_mask[idx], self.loss_mask[idx]
        sens_maps =  self.sens_maps[idx]
        ref_kspace = self.ref_kspace[idx]

        nw_input = torch.tensor(nw_input, dtype = torch.float64)
        nw_trn_mask = torch.tensor(trn_mask, dtype = torch.complex64)
        nw_loss_mask = torch.tensor(loss_mask, dtype = torch.complex64)
        sens_maps = torch.tensor(sens_maps, dtype = torch.complex64)
        ref_kspace = torch.tensor(ref_kspace, dtype = torch.float64)

        return nw_input, nw_trn_mask, nw_loss_mask, sens_maps, ref_kspace

class Dataset_Inference(torch.utils.data.Dataset):
    def __init__(self,trn_atb, test_mask, loss_mask, sens_maps):
        self.trn_atb = trn_atb
        self.test_mask = test_mask
        self.loss_mask = loss_mask
        self.sens_maps = sens_maps

        
    def __len__(self):
        return len(self.trn_atb)
        
    def __getitem__(self,idx):
        nw_input = self.trn_atb[idx] 
        test_mask , loss_mask = self.test_mask[idx], self.loss_mask[idx]
        sens_maps =  self.sens_maps[idx]

        nw_input = torch.tensor(nw_input, dtype = torch.float64)
        nw_test_mask = torch.tensor(test_mask, dtype = torch.complex64)
        nw_loss_mask = torch.tensor(loss_mask, dtype = torch.complex64)
        sens_maps = torch.tensor(sens_maps, dtype = torch.complex64)

        return nw_input, nw_test_mask, nw_loss_mask, sens_maps,


class MixL1L2Loss(nn.Module):
    def __init__(self, eps=1e-6,scalar=1/2):
        super().__init__()
        #self.mse = nn.MSELoss()
        self.eps = eps
        self.scalar=scalar
    def forward(self, yhat, y):

        loss = self.scalar*(torch.norm(yhat-y) / torch.norm(y)) + self.scalar*(torch.norm(yhat-y,p=1) / torch.norm(y, p=1))
        
        return loss

def train(train_loader, model, loss_fn, optimizer, device = torch.device('cpu')):
    avg_trn_cost = 0
    model.train()
    for ii,batch in enumerate(train_loader):
        nw_input,nw_trn_mask,nw_loss_mask,nw_sens_maps,nw_ref_kspace= batch
        nw_input= nw_input.permute(0,3,1,2)

        nw_input, nw_trn_mask, nw_loss_mask, nw_sens_maps, nw_ref_kspace = \
            nw_input.to(device), nw_trn_mask.to(device), nw_loss_mask.to(device), nw_sens_maps.to(device), nw_ref_kspace.to(device)

        """Forward Path"""
        nw_img_output, lamdas,nw_kspace_output = model(nw_input,nw_trn_mask,nw_loss_mask,nw_sens_maps)
        
        """Loss"""
        trn_loss =loss_fn(nw_kspace_output,nw_ref_kspace)
        
        """Backpropagation"""
        optimizer.zero_grad()
        trn_loss.backward()
        optimizer.step()

        avg_trn_cost += trn_loss.item()/ len(train_loader)
    return avg_trn_cost, lamdas

def validation(val_loader, model, loss_fn, device = torch.device('cpu')):
    avg_val_cost = 0
    model.eval()
    with torch.no_grad():
        for ii,batch in enumerate(val_loader):
            nw_input,nw_trn_mask,nw_loss_mask,nw_sens_maps,nw_ref_kspace= batch
            nw_input= nw_input.permute(0,3,1,2)

            nw_input, nw_trn_mask, nw_loss_mask, nw_sens_maps, nw_ref_kspace = \
                nw_input.to(device), nw_trn_mask.to(device), nw_loss_mask.to(device), nw_sens_maps.to(device), nw_ref_kspace.to(device)

            """Forward Path"""
            nw_img_output, lamdas,nw_kspace_output = model(nw_input,nw_trn_mask,nw_loss_mask,nw_sens_maps)
            
            """Loss"""
            val_loss =loss_fn(nw_kspace_output,nw_ref_kspace)
            
            avg_val_cost += val_loss.item()/ len(val_loader)

    return avg_val_cost

def test(test_loader, model, device = torch.device('cpu')):

    model.eval()
    with torch.no_grad():
        for ii,batch in enumerate(test_loader):
            nw_input,nw_trn_mask,nw_loss_mask,nw_sens_maps= batch
            nw_input= nw_input.permute(0,3,1,2)

            nw_input, nw_trn_mask, nw_loss_mask, nw_sens_maps = \
                nw_input.to(device), nw_trn_mask.to(device), nw_loss_mask.to(device), nw_sens_maps.to(device)

            """ Forward Path """
            nw_img_output, lamdas,nw_kspace_output = model(nw_input,nw_trn_mask,nw_loss_mask,nw_sens_maps)
    
    nw_img_output = nw_img_output.permute(0,2,3,1).squeeze() 

    return nw_img_output
