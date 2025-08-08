import math
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.cuda
from torch.autograd import Variable
# import basic_net as basic_net
import yaml
import os
from yaml.loader import SafeLoader
import copy
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# import basic_net as basic_net
from torch.nn.modules.utils import _pair
from scipy import ndimage

from nystrom_attention import NystromAttention
from .DeformableAttention2D import DeformCrossAttention2D
from .DeformableAttention1D import DeformCrossAttention1D

from .ClusterMergeNet import ClusterMergeNet

class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=128):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))

        return x
    
class CrossAttLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=128):
        super().__init__()
        self.norm = norm_layer(dim)
        self.multihead_attn = nn.MultiheadAttention(
                embed_dim = 128,                   # embed_dim
                num_heads = 8,                   # num_heads
                dropout = 0.1,                # dropout
                batch_first = True,
            )
    def forward(self, x1, x2):
        # x = x + self.multihead_attn(self.norm(x))
        x, attn_output_weights = self.multihead_attn(self.norm(x1), self.norm(x2), self.norm(x2), attn_mask=None) #[B, L1, D]
        x = x1 + x
        return x

class FusionNet(nn.Module):
    def __init__(self, feature_dim=128):
        super(FusionNet, self).__init__()
        self.fusion_layer = nn.Linear(feature_dim * 2, feature_dim)

    def forward(self, feature1, feature2):
        # Concatenate features along the last dimension
        combined_features = torch.cat((feature1, feature2), dim=-1)
        # Use a linear layer to learn the fusion
        fused_features = self.fusion_layer(combined_features)
        return fused_features

class TransFusionLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=128):
        super().__init__()
        self.norm = norm_layer(dim)
        self.multihead_attn = nn.MultiheadAttention(
                embed_dim = 128,                   # embed_dim
                num_heads = 8,                   # num_heads
                dropout = 0.1,                # dropout
            )
        self.pooler = Pooler(dim)

    def forward(self, x1, x2):
        # x = x + self.multihead_attn(self.norm(x))
        x, attn_output_weights = self.multihead_attn(self.norm(x1), self.norm(x2), self.norm(x2), attn_mask=None) #[L, B, D]

        x = x1 + x #[L,B,D] + [L,B,D] #both are ok for the final result.

        x = self.pooler(self.norm(x.transpose(0,1))) #[B,D]

        x = x.unsqueeze(dim=1) #[B,1,D]

        return x, attn_output_weights

class UniTeacherEncoder(nn.Module):

    def __init__(self, args, norm_layer=nn.LayerNorm, dim=128):
        super().__init__()
        self.norm = norm_layer(dim)
        self.args = args
        self.attn2d_omic1 = DeformCrossAttention2D(
            dim = 128,                   # feature dimensions
            dim_head = 64,               # dimension per head
            heads = 8,                   # attention heads
            dropout = 0.1,                # dropout
            downsample_factor = 4,       # downsample factor (r in paper)
            offset_scale = 4,            # scale of offset, maximum offset
            offset_groups = 8,        # number of offset groups, should be multiple of heads, original = None.
            offset_kernel_size = 6,      # offset kernel size
        )
        self.attn2d_omic2 = DeformCrossAttention2D(
            dim = 128,                   # feature dimensions
            dim_head = 64,               # dimension per head
            heads = 8,                   # attention heads
            dropout = 0.1,                # dropout
            downsample_factor = 4,       # downsample factor (r in paper)
            offset_scale = 4,            # scale of offset, maximum offset
            offset_groups = 8,        # number of offset groups, should be multiple of heads, original = None.
            offset_kernel_size = 6,      # offset kernel size
        )
        self.fusion_layer = FusionNet(feature_dim=128)
        self.transfusion_layer1 = TransFusionLayer(dim=128)
        self.transfusion_layer2 = TransFusionLayer(dim=128)


    def forward(self, x1, x2, attn_dim, return_vgrid=False):
        # x1(1, 1+2500, 512), x2(1, 1+2500, 512)
        # x1 = x1 + self.attn(self.norm(x1))
        x_omic1, attn_omic1 = self.attn2d_omic1(self.norm(x1[0]).transpose(1, 2), self.norm(x2).transpose(1, 2), return_vgrid=False)
        x_omic2, attn_omic2 = self.attn2d_omic2(self.norm(x1[1]).transpose(1, 2), self.norm(x2).transpose(1, 2), return_vgrid=False)
        x_out1 = x1[0] + x_omic1.transpose(1, 2) 
        x_out2 = x1[1] + x_omic2.transpose(1, 2) # x(B, 2500, 128)   
        #---->Fusion
        x = self.fusion_layer(x_out1, x_out2) #[B, 2500, 128] [B,2500,128] --> x(B, 2500, 128)

        #---->Clustering and Merging
        # init token dict
        B, N, _ = x.shape
        device = x.device
        idx_token = torch.arange(N)[None, :].repeat(B, 1).to(device)
        agg_weight = x.new_ones(B, N, 1)
        token_dict = {'x': x,
                    'token_num': N,
                    'idx_token': idx_token,
                    'agg_weight': agg_weight,
                    'omic1': x1[0],
                    'omic2': x1[1]} 
        # print('input query.shape:', self.norm(x).transpose(0, 1).shape) #2500,3,128
        # print('input key value.shape:', self.norm(x1[0][:,0,:].unsqueeze(dim=1)).transpose(0, 1).shape) #1,3,128
        query = self.norm(x).transpose(0, 1) #[L,B,D]
        kv1 = self.norm(x1[0][:,0,:].unsqueeze(dim=1)).transpose(0, 1)
        kv2 = self.norm(x1[1][:,0,:].unsqueeze(dim=1)).transpose(0, 1)
        x_fusion1, _ = self.transfusion_layer1(query, kv1)
        x_fusion2, _ = self.transfusion_layer2(query, kv2)
        # print('x_fusion1.shape:', x_fusion1.shape) #[B,1,D]
        # x = torch.cat((x_fusion1, x_fusion2), dim=1)
        # print('x.shape:', x.shape) #[B,2,128]
        
        # x = token_dict['x']
        # print('k=2 x.shape:', x.shape)
        # print('k=2 attn_path1.shape:', attn_omic1.shape)
        # print('k=2 attn_path2.shape:', attn_omic2.shape)
        return x_fusion1, x_fusion2, attn_omic1, attn_omic2 #[B,2,128],[B,8,2500,144]


class TeacherEncoder(nn.Module):

    def __init__(self, args, norm_layer=nn.LayerNorm, dim=128):
        super().__init__()
        self.norm = norm_layer(dim)
        self.args = args
        self.attn2d_omic1 = DeformCrossAttention2D(
            dim = 128,                   # feature dimensions
            dim_head = 64,               # dimension per head
            heads = 8,                   # attention heads
            dropout = 0.1,                # dropout
            downsample_factor = 4,       # downsample factor (r in paper)
            offset_scale = 4,            # scale of offset, maximum offset
            offset_groups = 8,        # number of offset groups, should be multiple of heads, original = None.
            offset_kernel_size = 6,      # offset kernel size
        )
        self.attn2d_omic2 = DeformCrossAttention2D(
            dim = 128,                   # feature dimensions
            dim_head = 64,               # dimension per head
            heads = 8,                   # attention heads
            dropout = 0.1,                # dropout
            downsample_factor = 4,       # downsample factor (r in paper)
            offset_scale = 4,            # scale of offset, maximum offset
            offset_groups = 8,        # number of offset groups, should be multiple of heads, original = None.
            offset_kernel_size = 6,      # offset kernel size
        )
        self.fusion_layer = FusionNet(feature_dim=128)
        self.transfusion_layer1 = TransFusionLayer(dim=128)
        self.transfusion_layer2 = TransFusionLayer(dim=128)


    def forward(self, x1, x2, attn_dim, return_vgrid=False):
        # x1(1, 1+2500, 512), x2(1, 1+2500, 512)
        # x1 = x1 + self.attn(self.norm(x1))
        x_omic1, attn_omic1 = self.attn2d_omic1(self.norm(x1[0]).transpose(1, 2), self.norm(x2).transpose(1, 2), return_vgrid=False)
        x_omic2, attn_omic2 = self.attn2d_omic2(self.norm(x1[1]).transpose(1, 2), self.norm(x2).transpose(1, 2), return_vgrid=False)
        x_out1 = x1[0] + x_omic1.transpose(1, 2) 
        x_out2 = x1[1] + x_omic2.transpose(1, 2) # x(B, 2500, 128)   
        #---->Fusion
        x = self.fusion_layer(x_out1, x_out2) #[B, 2500, 128] [B,2500,128] --> x(B, 2500, 128)

        #---->Clustering and Merging
        # init token dict
        B, N, _ = x.shape
        device = x.device
        idx_token = torch.arange(N)[None, :].repeat(B, 1).to(device)
        agg_weight = x.new_ones(B, N, 1)
        token_dict = {'x': x,
                    'token_num': N,
                    'idx_token': idx_token,
                    'agg_weight': agg_weight,
                    'omic1': x1[0],
                    'omic2': x1[1]} 
        # print('input query.shape:', self.norm(x).transpose(0, 1).shape) #2500,3,128
        # print('input key value.shape:', self.norm(x1[0][:,0,:].unsqueeze(dim=1)).transpose(0, 1).shape) #1,3,128
        query = self.norm(x).transpose(0, 1) #[L,B,D]
        kv1 = self.norm(x1[0][:,0,:].unsqueeze(dim=1)).transpose(0, 1)
        kv2 = self.norm(x1[1][:,0,:].unsqueeze(dim=1)).transpose(0, 1)
        x_fusion1, _ = self.transfusion_layer1(query, kv1)
        x_fusion2, _ = self.transfusion_layer2(query, kv2)
        # print('x_fusion1.shape:', x_fusion1.shape) #[B,1,D]
        # x = torch.cat((x_fusion1, x_fusion2), dim=1)
        # print('x.shape:', x.shape) #[B,2,128]
        
        # x = token_dict['x']
        # print('k=2 x.shape:', x.shape)
        # print('k=2 attn_path1.shape:', attn_omic1.shape)
        # print('k=2 attn_path2.shape:', attn_omic2.shape)
        return x_fusion1, x_fusion2, attn_omic1, attn_omic2 #[B,2,128],[B,8,2500,144]([4, 8, 2500, 144])

class StudentEncoder(nn.Module):

    def __init__(self, args, norm_layer=nn.LayerNorm, dim=128):
        super().__init__()
        self.norm = norm_layer(dim)
        self.args = args
        self.attn2d = DeformCrossAttention2D(
            dim = dim,                   # feature dimensions
            dim_head = 64,               # dimension per head
            heads = 8,                   # attention heads
            dropout = 0.1,                # dropout
            downsample_factor = 4,       # downsample factor (r in paper)
            offset_scale = 4,            # scale of offset, maximum offset
            offset_groups = 8,        # number of offset groups, should be multiple of heads, original = None.
            offset_kernel_size = 6,      # offset kernel size
        )
        self.cluster_merge = ClusterMergeNet(
            sample_ratio=self.args.path_cluster_num, #0.25; 0.0128
            dim_out=dim,
        )
        # self.cross_att = CrossAttLayer(dim=dim)


    def forward(self, x1, x2, attn_dim, return_vgrid=False):
        # x1(1, 1+2500, 512), x2(1, 1+2500, 512)
        # x1 = x1 + self.attn(self.norm(x1))
        x, attn_path = self.attn2d(self.norm(x1).transpose(1, 2), self.norm(x2).transpose(1, 2), return_vgrid=False)
        x = x1 + x.transpose(1, 2) # x(B, 2500, 128)

        # #---->Clustering and Merging
        # # init token dict
        # B, N, _ = x.shape
        # device = x.device
        # idx_token = torch.arange(N)[None, :].repeat(B, 1).to(device)
        # agg_weight = x.new_ones(B, N, 1)
        # token_dict = {'x': x,
        #                 'token_num': N,
        #                 'idx_token': idx_token,
        #                 'agg_weight': agg_weight}
        # x_q = token_dict['x']

        # token_dict, _ = self.cluster_merge(token_dict) # down sample
        # x_kv = token_dict['x']
        
        # #---->Translayer x1
        # # print('testing:', x_q.shape, x_kv.shape)
        # x_out = self.cross_att(x_q, x_kv) #[B,L1,D] [B,L2,D]

        # # x = token_dict['x']
        # # print('k=5 x.shape:', x.shape) 
        # # print('k=5 attn_path.shape:', attn_path.shape)
        # return x_out, attn_path #[B,32,128],[B,8,2500,144]


        #---->Clustering and Merging
        # init token dict
        B, N, _ = x.shape
        device = x.device
        idx_token = torch.arange(N)[None, :].repeat(B, 1).to(device)
        agg_weight = x.new_ones(B, N, 1)
        token_dict = {'x': x,
                        'token_num': N,
                        'idx_token': idx_token,
                        'agg_weight': agg_weight}
        x = token_dict['x']
        token_dict, _ = self.cluster_merge(token_dict) # down sample
        x = token_dict['x'] ##[B, cluster_num, C]
        return x, attn_path


class UniTeacherNet(nn.Module):
    def __init__(self, args):
        super(UniTeacherNet, self).__init__()
        self._fc1 = nn.Sequential(nn.Linear(1024, args.path_dim), nn.ReLU())
        # self.cls_token = nn.Parameter(torch.randn(1, 1, args.path_dim))
        self.args = args
        self.encoder = UniTeacherEncoder(args=self.args, dim=args.path_dim)
        self.norm = nn.LayerNorm(args.path_dim)
        self.pooler1 = Pooler(args.path_dim)
        self.pooler2 = Pooler(args.path_dim)
        self.classifier = nn.Linear(args.path_dim * 2, args.label_dim)
    
    def forward(self, path, omic_list=None):

        path = path.float() #[B, n, 1024]
        path = self._fc1(path) #[B, n, 512]
        
        # elif omic_list is not None:
        omic1 = omic_list[0].float() 
        # print('omic1 original.shape:', omic1.shape) #[B,128]
        omic1 = omic1.unsqueeze(1).repeat(1, path.shape[1], 1) #[B,2500,128]
        omic2 = omic_list[1].float() 
        # print('omic2 original.shape:', omic2.shape) #[B,128]
        omic2 = omic2.unsqueeze(1).repeat(1, path.shape[1], 1) #[B,2500,128]

        #---->Fusion
        # omic1 = self.fusion_layer(path, omic1) #[B, N, 512] [B,2500,512], h(1, 2500, 512)
        # omic2 = self.fusion_layer(path, omic1) #[B, N, 512] [B,2500,512], h(1, 2500, 512)

        #---->Translayer x1
        feature1, feature2, att_omic1, att_omic2 = self.encoder([path,path], path, self.args.attn_dim) #[B, C2, 128], C2=2, #[B,2500,128],[B,8,2500,144]
        feature1 = self.pooler1(self.norm(feature1)) #[B, 1, 128]->[B,128]
        feature2 = self.pooler2(self.norm(feature2)) #[B, 1, 128]->[B,128]
        concat_features = torch.cat((feature1, feature2), dim=-1)
        #---->predict
        logits = self.classifier(concat_features)
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        risk = -torch.sum(S, dim=1)
        # y_hat = torch.argmax(logits, dim=1)
        return logits, hazards, S, risk, feature1, feature2, att_omic1, att_omic2 #[B, 128]


class TeacherNet(nn.Module):
    def __init__(self, args):
        super(TeacherNet, self).__init__()
        self._fc1 = nn.Sequential(nn.Linear(1024, args.path_dim), nn.ReLU())
        # self.cls_token = nn.Parameter(torch.randn(1, 1, args.path_dim))
        self.args = args
        self.encoder = TeacherEncoder(args=self.args, dim=args.path_dim)
        self.norm = nn.LayerNorm(args.path_dim)
        self.pooler1 = Pooler(args.path_dim)
        self.pooler2 = Pooler(args.path_dim)
        self.classifier = nn.Linear(args.path_dim * 2, args.label_dim)
    
    def forward(self, path, omic_list=None):

        path = path.float() #[B, n, 1024]
        path = self._fc1(path) #[B, n, 512]
        
        # elif omic_list is not None:
        omic1 = omic_list[0].float() 
        # print('omic1 original.shape:', omic1.shape) #[B,128]
        omic1 = omic1.unsqueeze(1).repeat(1, path.shape[1], 1) #[B,2500,128]
        omic2 = omic_list[1].float() 
        # print('omic2 original.shape:', omic2.shape) #[B,128]
        omic2 = omic2.unsqueeze(1).repeat(1, path.shape[1], 1) #[B,2500,128]

        #---->Fusion
        # omic1 = self.fusion_layer(path, omic1) #[B, N, 512] [B,2500,512], h(1, 2500, 512)
        # omic2 = self.fusion_layer(path, omic1) #[B, N, 512] [B,2500,512], h(1, 2500, 512)

        #---->Translayer x1
        feature1, feature2, att_omic1, att_omic2 = self.encoder([omic1,omic2], path, self.args.attn_dim) #[B, C2, 128], C2=2, #[B,2500,128],[B,8,2500,144]
        feature1 = self.pooler1(self.norm(feature1)) #[B, 1, 128]->[B,128]
        feature2 = self.pooler2(self.norm(feature2)) #[B, 1, 128]->[B,128]
        concat_features = torch.cat((feature1, feature2), dim=-1)
        #---->predict
        logits = self.classifier(concat_features)
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        risk = -torch.sum(S, dim=1)
        # y_hat = torch.argmax(logits, dim=1)
        return logits, hazards, S, risk, feature1, feature2, att_omic1, att_omic2 #[B, 128]


class StudentNet_old(nn.Module):
    def __init__(self, args):
        super(StudentNet, self).__init__()
        self._fc1 = nn.Sequential(nn.Linear(1024, args.path_dim), nn.ReLU())
        # self.cls_token = nn.Parameter(torch.randn(1, 1, args.path_dim))
        self.args = args
        # self.layer1 = TransLayer(dim=512)
        # self.layer2 = TransLayer(dim=512)
        self.encoder = StudentEncoder(args=self.args, dim=args.path_dim)
        self.norm = nn.LayerNorm(args.path_dim)
        self.pooler1 = Pooler(args.path_dim)
        self.classifier = nn.Linear(args.path_dim, args.label_dim)
    
    def forward(self, path, omic_list=None):

        path = path.float() #[B, n, 1024]
        path = self._fc1(path) #[B, n, 512]
        
        # if omic_list is None:
        #---->Translayer x1
        feature, att = self.encoder(path, path, self.args.attn_dim) #[B, C1, 128] #[B,2500,128],[B,8,2500,144]
        feature = self.pooler1(self.norm(feature)) #[B, C1, 128]->[B, 128]
        # print('feature.shape:', feature.shape) #[B, 2500, 128] for w/o clustermerge; [B, 32, 128] for w clustermerge
        #---->predict
        logits = self.classifier(feature)
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        risk = -torch.sum(S, dim=1)
        return logits, hazards, S, risk, feature, att #[B, 128]

class StudentNet(nn.Module):
    def __init__(self, args):
        super(StudentNet, self).__init__()
        self._fc1 = nn.Sequential(nn.Linear(1024, args.path_dim), nn.ReLU())
        # self.cls_token = nn.Parameter(torch.randn(1, 1, args.path_dim))
        self.args = args
        # self.layer1 = TransLayer(dim=512)
        # self.layer2 = TransLayer(dim=512)
        self.encoder = StudentEncoder(args=self.args, dim=args.path_dim)
        self.norm = nn.LayerNorm(args.path_dim)
        self.pooler1 = Pooler(args.path_dim)
        self.classifier = nn.Linear(args.path_dim*2, args.label_dim)
    
    def forward(self, path, omic_list=None):

        path = path.float() #[B, n, 1024]
        path = self._fc1(path) #[B, n, 512]
        
        # if omic_list is None:
        #---->Translayer x1
        feature, att = self.encoder(path, path, self.args.attn_dim) #[B, C1, 128] #[B,2500,128],[B,8,2500,144]
        # feature = self.pooler1(self.norm(feature)) #[B, Cluster_num, 128]->[B, 128]
        feature = torch.cat((feature[:,0,:], feature[:,1,:]), dim=-1) #[B, 128]->[B, 128*2]
        # print('feature.shape:', feature.shape) #[B, 2500, 128] for w/o clustermerge; [B, 32, 128] for w clustermerge
        #---->predict
        logits = self.classifier(feature)
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        risk = -torch.sum(S, dim=1)
        return logits, hazards, S, risk, feature, att #[B, 128]


class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # torch.Size([90, 64, 768]) language #[B, N, 512]
        # torch.Size([90, 82, 768]) vision
        # first_token: torch.Size([90, 768])
        # pooled_output1: torch.Size([90, 768])
        # pooled_output2: torch.Size([90, 768])
        
        # method 1:
        # first_token_tensor = hidden_states[:, 0]
        # pooled_output = self.dense(first_token_tensor)
        # pooled_output = self.activation(pooled_output)

        # mean_token_tensor: torch.Size([90, 768])
        # pooled_output1: torch.Size([90, 768])
        # pooled_output2: torch.Size([90, 768])
        
        # method 2:
        # 2.1 calculate average embeddings
        pooled_token_tensor = torch.mean(hidden_states, dim = 1) # avg_embeddings shape: (90, 768)
        # print('avg_token_tensor.shape:', avg_token_tensor.shape)

        # 2.2 calculate pooled embeddings
        # pooled_token_tensor = torch.max(hidden_states, dim=1)[0] # pooled_embeddings shape: (90, 768)

        pooled_output = self.dense(pooled_token_tensor)
        pooled_output = self.activation(pooled_output)

        return pooled_output