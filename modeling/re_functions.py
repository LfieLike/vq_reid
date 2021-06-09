import torch
from torch.autograd import Function

class soft_indexs(Function):# 获取softmax后的向量后，forward采取取max，也就是hard vq,backward 用 torch.mm拟合
    @staticmethod
    def forward(ctx, soft_dis, codebook):
        ctx.save_for_backward(soft_dis,codebook)
        _,index = torch.max(soft_dis,dim = 1)
        index_flatten = index.view(-1)
        #ctx.mark_non_differentiable(index_flatten)
        codes_flatten = torch.index_select(codebook, dim=0,
            index=index_flatten)
        out = codes_flatten.view(soft_dis.shape[0],codebook.shape[1])
        return out
    @staticmethod
    
    def backward(ctx, grad_output):
        # := backward torch.mm(soft_dis,codebook)
        grad_soft_dis,grad_coodbook = None,None
        soft_dis ,codebook = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_soft_dis = torch.mm(grad_output,codebook)
        if ctx.needs_input_grad[1]:
            grad_coodbook = torch.mm(grad_output,soft_dis)
        return grad_soft_dis,grad_coodbook
s_i = soft_indexs.apply
__all__ = [s_i]
