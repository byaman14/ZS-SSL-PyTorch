import torch
import fastmri

def c2r_HWC(input_data):
    return torch.stack([input_data.real,input_data.imag],dim=-1)
def r2c_HWC(input_data):
    return torch.complex(input_data[...,0],input_data[...,1])

def c2r_CHW(input_data):
    return torch.stack([input_data.real,input_data.imag],dim=0)

def r2c_CHW(input_data):
    return torch.complex(input_data[0,...],input_data[1,...])

def zdot_reduce_sum(input_x, input_y):
    dims = tuple(range(len(input_x.shape)))
    return (torch.conj(input_x)*input_y).sum(dims).real

class data_consistency():
    """
    Data consistency class can be used for:
        -performing E^h*E operation in the paper
        -transforming final network output to kspace
    """

    def __init__(self, sens_maps, mask):
        self.sens_maps = sens_maps
        self.mask = mask

    def EhE_Op(self, img, mu):

        """
        Performs (E^h*E + mu*I) x
        """
        coil_imgs = self.sens_maps * img
        kspace = r2c_HWC(fastmri.fft2c(c2r_HWC(coil_imgs)))
        masked_kspace =kspace * self.mask
        image_space_coil_imgs  = r2c_HWC(fastmri.ifft2c(c2r_HWC(masked_kspace)))
        image_space_comb  = (image_space_coil_imgs*torch.conj(self.sens_maps)).sum(dim=0)  
        ispace  = image_space_comb + mu * img

        return ispace

    def SSDU_kspace(self, img):
        """
        Transforms unrolled network output to k-space
        and selects only loss mask locations(\Lambda) for computing loss
        """
        coil_imgs = self.sens_maps * img
        kspace = r2c_HWC(fastmri.fft2c(c2r_HWC(coil_imgs)))
        masked_kspace = kspace * self.mask

        return masked_kspace

    def Supervised_kspace(self, img):
        """
        Transforms unrolled network output to k-space
        """


        coil_imgs = self.sens_maps * img
        kspace = r2c_HWC(fastmri.fft2c(c2r_HWC(coil_imgs)))

        return kspace


def conjgrad(rhs, sens_maps, mask,mu, args):
    """
    Parameters
    ----------
    input_data : contains tuple of  reg output rhs = E^h*y + mu*z , sens_maps and mask
    rhs = 2x nrow x ncol 
    sens_maps : coil sensitivity maps ncoil x nrow x ncol
    mask : nrow x ncol
    mu : penalty parameter
    Encoder : Object instance for performing encoding matrix operations
    Returns
    --------
    data consistency output, nrow x ncol x 2
    """    
    Encoder = data_consistency(sens_maps, mask)
    rhs = r2c_CHW(rhs)
    mu = mu.type(torch.complex64)
    x = torch.zeros_like(rhs)
    i, r, p = 0, rhs, rhs
    rsnot = zdot_reduce_sum(r,r)
    rsold, rsnew = rsnot, rsnot


    for ii in range(args.CG_Iter):

        Ap = Encoder.EhE_Op(p,mu)
        pAp = zdot_reduce_sum(p,Ap)
        alpha = (rsold / pAp)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = zdot_reduce_sum(r,r)
        beta = (rsnew / rsold)
        rsold = rsnew
        p = beta * p + r

    return c2r_CHW(x)


def dc_block(rhs, sens_maps, mask, mu,args):
    """
    DC block employs conjugate gradient for data consistency,
    """
    cg_recons = []
    for ii in range(args.batchSize):
        cg_recon = conjgrad(rhs[ii],sens_maps[ii],mask[ii],mu, args)
        cg_recons.append(cg_recon.unsqueeze(0))
    dc_block_recons = torch.cat(cg_recons,0)

    return dc_block_recons


def SSDU_kspace_transform(nw_output, sens_maps, mask, args):
    """
    This function transforms unrolled network output to k-space at only unseen locations in training (\Lambda locations)
    """

    all_recons = []
    for ii in range(args.batchSize):
        Encoder = data_consistency(sens_maps[ii], mask[ii])
        temp=r2c_CHW(nw_output[ii])
        nw_output_kspace = Encoder.SSDU_kspace(temp)
        nw_output_kspace = c2r_HWC(nw_output_kspace)
        all_recons.append(nw_output_kspace.unsqueeze(0))

    return torch.cat(all_recons,0)


def Supervised_kspace_transform(nw_output, sens_maps, mask, args):
    """
    This function transforms unrolled network output to k-space
    """

    all_recons = []

    for ii in range(args.batchSize):
        Encoder = data_consistency(sens_maps[ii], mask[ii])
        temp=r2c_CHW(nw_output[ii])
        nw_output_kspace = Encoder.Supervised_kspace(temp)
        nw_output_kspace = c2r_HWC(nw_output_kspace)
        all_recons.append(nw_output_kspace.unsqueeze(0))
    
    return torch.cat(all_recons,0)

def DIP_kspace_transform(nw_output, sens_maps, mask, args):
    """
    This function transforms unrolled network output to k-space at acquired locations
    """

    all_recons = []
    for ii in range(args.batchSize):
        Encoder = data_consistency(sens_maps[ii], mask[ii])
        temp=r2c_CHW(nw_output[ii])
        nw_output_kspace = Encoder.SSDU_kspace(temp)
        nw_output_kspace = c2r_HWC(nw_output_kspace)
        all_recons.append(nw_output_kspace.unsqueeze(0))

    return torch.cat(all_recons,0)