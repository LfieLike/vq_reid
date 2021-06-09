# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch.nn.functional as F
import torch
from torch import nn
from torch.autograd import Function
from .backbones.resnet import ResNet, BasicBlock, Bottleneck
from .backbones.senet import SENet, SEResNetBottleneck, SEBottleneck, SEResNeXtBottleneck
from .backbones.resnet_ibn_a import resnet50_ibn_a
from .functions import vq_st,vq
from .er_modules import soft_vq
from .stvq import the_vq_st,the_vq
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)
class VectorQuantization(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,inputs, codebook):
        embedding_size = codebook.size(1)
        inputs_size = inputs.size()
        inputs_flatten = inputs.view(-1, embedding_size)

        codebook_sqr = torch.sum(codebook ** 2, dim=1)
        inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)

        # Compute the distances to the codebook
        distances = torch.addmm(codebook_sqr + inputs_sqr,
            inputs_flatten, codebook.t(), alpha=-2.0, beta=1.0)
        distances = distances#*torch.tensor(100) # soft-hard weight
        #print(distances)
        soft_distances = F.softmax(distances,dim=1)
        #print(soft_distances)
        # _, indices_flatten = torch.min(soft_distances, dim=1)
        # indices = indices_flatten.view(*inputs_size[:-1])
        #ctx.mark_non_differentiable(indices)
        return soft_distances
class VQEmbedding(nn.Module):
    def __init__(self, K, D):
        super().__init__()
        self.embedding = nn.Embedding(K, D)
        self.grads = {}
        self.nvq = VectorQuantization()
    def forward(self, z_e_x):
        latents = self.vq(z_e_x_, self.embedding.weight)
        return latents
    def save_grad(self,name):
        def hook(grad):
            self.grads[name] = grad
        return hook
    def straight_through(self, z_e_x):
        #soft-hard
        size = z_e_x.size()
        indices_soft,indices_hard = vq(z_e_x, self.embedding.weight.detach())
        z_q_x_soft= vq_st(indices_soft.detach(), self.embedding.weight)
        z_q_x_hard= vq_st(indices_hard.detach(), self.embedding.weight)
        z_q_x_soft = z_q_x_soft.view_as(z_e_x)
        z_q_x_hard = z_q_x_hard.view_as(z_e_x)
        self.embedding.weight.register_hook(self.save_grad('x'))
        return z_q_x_soft,z_q_x_hard
class Baseline(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice):
        super(Baseline, self).__init__()
        if model_name == 'resnet18':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride, 
                               block=BasicBlock, 
                               layers=[2, 2, 2, 2])
        elif model_name == 'resnet34':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet50':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet101':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck, 
                               layers=[3, 4, 23, 3])
        elif model_name == 'resnet152':
            self.base = ResNet(last_stride=last_stride, 
                               block=Bottleneck,
                               layers=[3, 8, 36, 3])
            
        elif model_name == 'se_resnet50':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 4, 6, 3], 
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride) 
        elif model_name == 'se_resnet101':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 4, 23, 3], 
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnet152':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 8, 36, 3],
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)  
        elif model_name == 'se_resnext50':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 6, 3], 
                              groups=32, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride) 
        elif model_name == 'se_resnext101':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 23, 3], 
                              groups=32, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'senet154':
            self.base = SENet(block=SEBottleneck, 
                              layers=[3, 8, 36, 3],
                              groups=64, 
                              reduction=16,
                              dropout_p=0.2, 
                              last_stride=last_stride)
        elif model_name == 'resnet50_ibn_a':
            self.base = resnet50_ibn_a(last_stride)
        
        
        
        for i in self.base.parameters():
            i.requires_grad = False
        
        
        
        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......')
        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gap = nn.AdaptiveMaxPool2d(1)
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat

        if self.neck == 'no':
            self.classifier = nn.Linear(self.in_planes, self.num_classes)
        elif self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(self.in_planes)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)


        self.vq = VQEmbedding(256,4)



    def forward(self, x):
        # for i in self.base.parameters():
        #     print(i.requires_grad)
        global_feat = self.gap(self.base(x))  # (b, 2048, 1, 1)
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)  # normalize for angular softmax
        q_feat_soft,q_feat_hard = self.vq.straight_through(feat)
        if self.training:
            cls_score_soft = self.classifier(q_feat_soft)
            cls_score_hard = self.classifier(q_feat_hard)
            return cls_score_soft,cls_score_hard, global_feat,q_feat_soft,q_feat_hard  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                return q_feat_hard
            else:
                return q_feat_hard

    def load_param(self, trained_path):
        print("load!!!!!!!!!!!!!!!!!!")
        param_dict = torch.load(trained_path)
        for i in param_dict.state_dict():
            if "vq.embedding.weight" in i:
                continue
            print(i)
            self.state_dict()[i].copy_(param_dict.state_dict()[i])
