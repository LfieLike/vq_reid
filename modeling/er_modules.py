import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence

from .re_functions import s_i
def to_scalar(arr):
    if type(arr) == list:
        return [x.item() for x in arr]
    else:
        return arr.item()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)



class soft_get_dis(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,inputs,codebook,sig):
        #print("111111111111111111")
        inputs_size = inputs.size()
        codebook_size = codebook.size(1)
        inputs_flatten = inputs.view(-1,codebook_size)
        codebook_sqr = torch.sum(codebook ** 2, dim = 1)
        inputs_sqr = torch.sum(inputs_flatten ** 2 , dim = 1,keepdim=True)
        #print(inputs.shape)
        #print(inputs_sqr.shape)
        #print(codebook_sqr.shape)
        distances = torch.addmm(inputs_sqr + codebook_sqr,inputs_flatten,codebook.t(),alpha = -2.0,beta = 1.0)
        distances = distances*(sig.detach())
        out = F.softmax(distances,dim=1)
        return out
class soft_vq(nn.Module):
    def __init__(self,M,D,L,sig):
        # D: Dimension of latent code E  
        # M: the number of Z divided into z_hat 
        # D/M: Dimension of z_hat
        # L:the number of centers vectors
        super().__init__()
        self.embedding = nn.Embedding(100,8)
        self.sig = torch.tensor(-sig)
        self.get_dis = soft_get_dis()
    def forward(self,x):
        soft_dis = self.get_dis(x,self.embedding.weight.detach(),self.sig)
        soft_q = s_i(soft_dis,self.embedding.weight)
        soft_q = soft_q.view_as(x)
        return soft_q #量化之后的 潜编码
    def get_index(self,input):
        with torch.no_grad:
            soft_dis = self.get_dis(x,self.embedding.weight.detach(),self.sig)
            _,index = torch.max(soft_dis,dim = 1)
            inputs_size = input.size()
            index_flatten = index.view(*inputs_size[:-1])
            return index_flatten

class soft_vq_vae(nn.Module):
    def __init__(self, input_dim, dim, K=512,sig=100):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 4, 2, 1),
            ResBlock(dim),
            ResBlock(dim),
        )
        self.vq = soft_vq(dim,K, dim,sig)
        self.decoder = nn.Sequential(
            ResBlock(dim),
            ResBlock(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, input_dim, 4, 2, 1),
            nn.Tanh()
        )
        self.apply(weights_init)
    def encode(self,x):
        z_x = self.encoder(x)
        c_z = self.vq.get_index(z_x)
    def decode(self,x):
        q_c = self.vq.embedding(x)
        x_hat = self.decoder(q_c)
    def forward(self, x):
        z_e_x = self.encoder(x)
        q_z = self.vq(z_e_x)
        x_hat = self.decoder(q_z)
        #x_hat = x_hat.view_as(x)
        return x_hat, z_e_x, q_z
  
