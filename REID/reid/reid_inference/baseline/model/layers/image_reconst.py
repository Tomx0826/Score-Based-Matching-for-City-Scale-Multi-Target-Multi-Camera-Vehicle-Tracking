# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 13:12:21 2022

@author: A50285
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

# MS model
class MSDecoder(nn.Module):
    def __init__(self, feat_dim):
        super(MSDecoder, self).__init__()
        self.decDense=nn.Linear(in_features=feat_dim,out_features=8192)
        self.deConv3=nn.Conv2d(in_channels=8,out_channels=8,kernel_size=[5,5],padding=2,stride=(1,1))
        self.deConv2=nn.Conv2d(in_channels=8,out_channels=32,kernel_size=[5,5],padding=2,stride=(1,1))
        self.deConv1=nn.Conv2d(in_channels=32,out_channels=3,kernel_size=[5,5],padding=2,stride=(1,1))

        self.decDense2=nn.Linear(in_features=feat_dim,out_features=2048)
        self.deConv32=nn.Conv2d(in_channels=8,out_channels=8,kernel_size=[5,5],padding=2,stride=(1,1))
        self.deConv22=nn.Conv2d(in_channels=8,out_channels=32,kernel_size=[5,5],padding=2,stride=(1,1))
        self.deConv12=nn.Conv2d(in_channels=32,out_channels=3,kernel_size=[5,5],padding=2,stride=(1,1))

        self.decDense3=nn.Linear(in_features=feat_dim,out_features=512)
        self.deConv323=nn.Conv2d(in_channels=8,out_channels=8,kernel_size=[5,5],padding=2,stride=(1,1))
        self.deConv223=nn.Conv2d(in_channels=8,out_channels=32,kernel_size=[5,5],padding=2,stride=(1,1))
        self.deConv123=nn.Conv2d(in_channels=32,out_channels=3,kernel_size=[5,5],padding=2,stride=(1,1))

    def decoder_imClip(self,feats):
        x=self.decDense(feats)
        x=torch.reshape(input=x,shape=[list(x.size())[0],8,32,32])
        unp3shape=[2*di for di in list(x.size())[2:]]
        x=F.interpolate(input=x,size=unp3shape)
        x=self.deConv3(x) # [B,8,64,64]
        x=F.relu(x)
        unp2shape=[2*di for di in list(x.size())[2:]]
        x=F.interpolate(input=x,size=unp2shape)
        x=self.deConv2(x)  # [B,32,128,128]
        x=F.relu(x)
        unp1shape=[2*di for di in list(x.size())[2:]]
        x=F.interpolate(input=x,size=unp1shape)
        imClip=self.deConv1(x) # [B,3,256,256]
        return imClip

    def decoder_imClip1(self,feats):
        x=self.decDense2(feats)
        x=torch.reshape(input=x,shape=[list(x.size())[0],8,16,16])
        unp3shape2=[2*di for di in list(x.size())[2:]]
        x=F.interpolate(input=x,size=unp3shape2)
        x=self.deConv32(x) # [B,8,32,32]
        x=F.relu(x)
        unp2shape2=[2*di for di in list(x.size())[2:]]
        x=F.interpolate(input=x,size=unp2shape2)
        x=self.deConv22(x) # [B,32,64,64]
        x=F.relu(x)
        unp1shape2=[2*di for di in list(x.size())[2:]]
        x=F.interpolate(input=x,size=unp1shape2)
        imClip1=self.deConv12(x) # [B,8,128,128]
        return imClip1

    def decoder_imClip2(self,feats):
        x=self.decDense3(feats)
        x=torch.reshape(input=x,shape=[list(x.size())[0],8,8,8])
        unp3shape23=[2*di for di in list(x.size())[1:-1]]
        x=F.interpolate(input=x,size=unp3shape23)
        x=self.deConv323(x) # [B,8,16,16]
        x=F.relu(x)
        unp2shape23=[2*di for di in list(x.size())[2:]]
        x=F.interpolate(input=x,size=unp2shape23)
        x=self.deConv223(x) # [B,32,32,32]
        x=F.relu(x)
        unp1shape23=[2*di for di in list(x.size())[2:]]
        x=F.interpolate(input=x,size=unp1shape23)
        imClip2=self.deConv123(x) # [B,3,64,64]
        return imClip2

    def forward(self, x):
        imClip = self.decoder_imClip(x)
        imClip1 = self.decoder_imClip1(x)
        imClip2 = self.decoder_imClip2(x)
        return imClip,imClip1,imClip2
        
    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)
                

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)

def ms_loss(images,imClip,imClip1, imClip2):
    batch_size = images.size(0)
    
    images1 = nn.functional.interpolate(input=images,size=[128,128])
    images2 = nn.functional.interpolate(input=images,size=[64,64])

    recLoss = F.mse_loss(images, imClip, size_average=False).div(batch_size)
    recLoss1 = F.mse_loss(images1, imClip1, size_average=False).div(batch_size)
    recLoss2 = F.mse_loss(images2, imClip2, size_average=False).div(batch_size)
    
    return recLoss+recLoss1+recLoss2
