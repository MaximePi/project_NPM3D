#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM
"""


import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import manhattan_distances

def knn(x, k, distance='euclidean'):
    if distance=='euclidean':
        inner = -2*torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x**2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
        idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    elif distance=='manhattan':
        device = torch.device('cuda')
        pairwise_distance = -torch.zeros((x.size(0),x.size(2),x.size(2)),q)
        #x = x.cpu().detach().clone().numpy() 
        for i in range(x.size(1)):
            print(i)
            #out = manhattan_distances(cloud.T,cloud.T)
            pairwise_distance -= torch.abs(x[:,i:i+1,:]-x[:,i:i+1,:].transpose(2,1)).cpu()
            #pairwise_distance[i,:,:] = out
        #pairwise_distance = torch.tensor(pairwise_distance,device=device)
        idx = pairwise_distance.topk(k=k, dim=-1)[1]
        idx = idx.to(device)
    return idx


def knn_with_dist(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    dist,idx = pairwise_distance.topk(k=k, dim=-1)   # (batch_size, num_points, k)
    
    return dist, idx

def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2)
  
    return feature


def get_neighbors_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    gdist = None
    device = torch.device('cuda')
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        dist,idx = knn_with_dist(x, k=k)   # (batch_size, num_points, k)
        #dist = dist.to(device)
        dist = dist.view(batch_size, 1, num_points, k)
        gdist = torch.exp(0.5*dist) # dist was already neg
        
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    feature = feature.permute(0, 3, 1, 2) 
 
		
    return feature, gdist


class PointNet(nn.Module):
    def __init__(self, args, output_channels=40):
        super(PointNet, self).__init__()
        self.args = args
        if self.args.add_features:
            self.input_channels = 6
        else:
            self.input_channels = 3
        self.conv1 = nn.Conv1d(self.input_channels, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, output_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x


class PointNetPlus(nn.Module):
    def __init__(self, args, output_channels=40):
        super(PointNetPlus, self).__init__()
        self.args = args
        self.k = args.k
        if self.args.add_features:
            self.input_channels = 6
        else:
            self.input_channels = 3
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(self.input_channels, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)

        idx_neighbors = knn(x, self.k)

        x,_ = get_neighbors_feature(x, k=self.k, idx=idx_neighbors)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x,_ = get_neighbors_feature(x1, k=self.k, idx=idx_neighbors)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x,_ = get_neighbors_feature(x2, k=self.k, idx=idx_neighbors)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]
        
        x,_ = get_neighbors_feature(x3, k=self.k, idx=idx_neighbors)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x

class DGCNN(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DGCNN, self).__init__()
        self.args = args
        self.k = args.k
        if self.args.add_features:
            self.input_channels = 12
        else:
            self.input_channels = 6
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(self.input_channels, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        if self.args.aggregation=='max':
            x1 = x.max(dim=-1, keepdim=False)[0]
        if self.args.aggregation =='mean':
            x1 = x.mean(dim=-1, keepdim=False)
        if self.args.aggregation =='sum':
            x1 = x.sum(dim=-1, keepdim=False)
        

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        if self.args.aggregation=='max':
            x2 = x.max(dim=-1, keepdim=False)[0]
        if self.args.aggregation =='mean':
            x2 = x.mean(dim=-1, keepdim=False)
        if self.args.aggregation =='sum':
            x2 = x.sum(dim=-1, keepdim=False)
        
        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        if self.args.aggregation=='max':
            x3 = x.max(dim=-1, keepdim=False)[0]
        if self.args.aggregation =='mean':
            x3 = x.mean(dim=-1, keepdim=False)
        if self.args.aggregation =='sum':
            x3 = x.sum(dim=-1, keepdim=False)
        
        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        if self.args.aggregation=='max':
            x4 = x.max(dim=-1, keepdim=False)[0]
        if self.args.aggregation =='mean':
            x4 = x.mean(dim=-1, keepdim=False)
        if self.args.aggregation =='sum':
            x4 = x.sum(dim=-1, keepdim=False)
        
        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x


class DGCNN_seg(nn.Module):
    def __init__(self, args, n_classes=10):
        super(DGCNN_seg, self).__init__()
        self.args = args
        self.k = args.k

        self.input_channels = 6
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.bn6 = nn.BatchNorm2d(512)
        self.bn7 = nn.BatchNorm2d(256)

        self.conv1 = nn.Sequential(nn.Conv2d(self.input_channels, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(256, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        

        self.out1 = nn.Sequential(nn.Conv2d(258, 512, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.out2 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.out3 = nn.Conv2d(256, n_classes, kernel_size=1, bias=False)


    def forward(self, x):
        batch_size = x.size(0)

        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        
        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        
        x_m = torch.cat((x1, x2, x3), dim=1)
        x = self.conv5(x_m)
        x1 = F.adaptive_max_pool2d(x, [1024,1])
        x2 = F.adaptive_avg_pool2d(x, [1024,1])
        x = torch.cat((x1, x2), -1)

        x = torch.cat((x,x_m.transpose(2,1)),axis=-1)
        x = x.unsqueeze(-1).transpose(2,1)
        x = self.out1(x)
        x = self.out2(x)
        x = self.out3(x)
        x = x.transpose(2,1).squeeze()  
        
        # pred batch_size*N_points*N_classes
        
        return x
    

class MoNet(nn.Module):
    def __init__(self, args, output_channels=40):
        super(MoNet, self).__init__()
        self.args = args
        self.k = args.k
        if self.args.add_features:
            self.input_channels = 6
        else:
            self.input_channels = 3
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(self.input_channels, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        
        batch_size = x.size(0)
        
        x ,gdist1 = get_neighbors_feature(x, k=self.k)
        x = self.conv1(x)
        
        x = torch.mul(x,gdist1.repeat(1, x.size(1), 1, 1))
        x1 = x.sum(dim=-1, keepdim=False)
        
        x ,gdist2 = get_neighbors_feature(x1, k=self.k)
        x = self.conv2(x)

        x = torch.mul(x,gdist2.repeat(1, x.size(1), 1, 1))
        x2 = x.sum(dim=-1, keepdim=False)
        
        x ,gdist3 = get_neighbors_feature(x2, k=self.k)
        x = self.conv3(x)        
        x = torch.mul(x,gdist3.repeat(1, x.size(1), 1, 1))
        x3 = x.sum(dim=-1, keepdim=False)
        
        x ,gdist4 = get_neighbors_feature(x3, k=self.k)
        x = self.conv4(x)
        x = torch.mul(x,gdist4.repeat(1, x.size(1), 1, 1))
        x4 = x.sum(dim=-1, keepdim=False)

        x = torch.cat((x1, x2, x3, x4), dim=1)
        
        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x
