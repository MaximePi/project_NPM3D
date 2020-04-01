#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: data.py
@Time: 2018/10/13 6:21 PM
"""


import os
import sys
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
from sklearn.neighbors import KDTree
from prep_Paris_Lille import extract_from_ply

def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))

#def download_Paris_Lille():
#    https://cloud.mines-paristech.fr/index.php/s/JhIxgyt0ALgRZ1O/download
#    https://cloud.mines-paristech.fr/index.php/s/JhIxgyt0ALgRZ1O/download


def load_data(partition):
    download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    if clip is not None:
        pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    else:
        pointcloud += sigma * np.random.randn(N,C)
    return pointcloud

def rotate_pointcloud(pointcloud, angle):
    # pointcloud (3,N), rotation along Z axis
    
    rot_mat_Z = np.array([[np.cos(angle),-np.sin(angle),0],
                        [np.sin(angle),np.cos(angle),0],
                        [0,0,1]])
    rot_mat_Y = np.array([[np.cos(angle),0,-np.sin(angle)],
                        [0,1,0],
                        [np.sin(angle),0,np.cos(angle)]])
    pointcloud = (pointcloud@rot_mat_Z)@rot_mat_Y

def grid_subsampling(pointcloud, num_points, voxel_size=0.05):
    min_coords, max_coords = pointcloud.min(0), pointcloud.max(0)
    pts_indices = np.floor((pointcloud - min_coords)/voxel_size)
    _, pts_grids = np.unique(pts_indices, return_inverse = True, axis = 0)
    num_grids = pts_grids.max() + 1
    subsampled_points = []
    for i in range(num_grids):
        mask = (pts_grids == i)
        subsampled_points.append(pointcloud[mask].mean(0))
    subsampled_points = np.array(subsampled_points)
    N = subsampled_points.shape[0]
    
    if N>num_points:
        idx = np.random.randint(0,N,num_points)
        subsampled_points = subsampled_points[idx,:]
    else:
        subsampled_points = np.concatenate((subsampled_points,np.zeros((num_points-N,subsampled_points.shape[1]))))
    return subsampled_points.astype('float32')

def local_PCA(points):
    mean = np.mean(points,axis=0).reshape(1,3)
    cov = np.zeros((3,3))
    cov = (points-mean).T@(points-mean)/points.shape[0]
    eigenvalues,eigenvectors = np.linalg.eig(cov)
    ind_s = np.argsort(eigenvalues)
    return eigenvalues[ind_s], eigenvectors[ind_s]

def neighborhood_PCA(query_points, cloud_points, radius):
    all_eigenvalues = np.zeros((query_points.shape[0],3))
    all_eigenvectors = np.zeros((query_points.shape[0],3,3))

    tree = KDTree(cloud_points)
    for i in range(query_points.shape[0]):
        ind = tree.query_radius(query_points[i,:].reshape(-1,1).T, r=radius)[0]
        if len(ind)>0:
            all_eigenvalues[i,:], all_eigenvectors[i,:,:] = local_PCA(cloud_points[ind,:])

    return all_eigenvalues, all_eigenvectors

def compute_features(query_points, cloud_points, radius):
    eps=0.0001
    all_val, all_vect = neighborhood_PCA(query_points,cloud_points,radius)
    verticality = (2/np.pi)*np.arcsin(np.abs(all_vect[:,0,2]))
    planarity = np.divide(all_val[:,1]-all_val[:,0],all_val[:,2]+eps)
    sphericity = np.divide(all_val[:,0],all_val[:,2]+eps)
    
    return np.stack((verticality, planarity, sphericity)).astype('float32')
    
class ModelNet40(Dataset):
    def __init__(self, num_points, sigma, angle, grid_samp=False, add_features=False, partition='train'):
        self.data, self.label = load_data(partition)
        self.num_points = num_points
        self.partition = partition        
        self.sigma = sigma
        self.angle = angle
        self.grid_samp = grid_samp
        self.add_features=add_features
        
    def __getitem__(self, item):
        if self.grid_samp:
            pointcloud = grid_subsampling(self.data[item],self.num_points)
        else:
            pointcloud = self.data[item][:self.num_points]
        label = self.label[item]

        if self.partition == 'train':
            
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
            if self.add_features:
                new_feat = compute_features(pointcloud,self.data[item], radius=0.3).T
                pointcloud = np.concatenate((pointcloud,new_feat),axis=1)
                
        if self.partition == 'test':
            if self.add_features:
                new_feat = compute_features(pointcloud,self.data[item], radius=0.3).T
                pointcloud = np.concatenate((pointcloud,new_feat),axis=1)
            
            #pointcloud = translate_pointcloud(pointcloud)
            #pointcloud = jitter_pointcloud(pointcloud, sigma=self.sigma,clip=10)
            #point_cloud = rotate_pointcloud(pointcloud, self.angle)
            else:
                pass
            
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


def load_data_Paris(num_points,grid_size,partition):
    
    BASE_PATH = os.getcwd()
    DATA_PATH = os.path.join(os.path.join(BASE_PATH,'data'),'Paris_Lille')
    
    if partition == 'evaluate': ## no labels in evaluate
        cloudnames = os.path.join(DATA_PATH,'test')
        if not os.path.exists(os.path.join(cloudnames,'evaluate_ds'+'.hdf5')):
            extract_from_ply(num_points,grid_size,DATA_PATH,partition)
            f = h5py.File(os.path.join(cloudnames,'evaluate_ds'+'.hdf5'), 'r')    
            all_data = f['pointclouds/'+'evaluate_clouds'][:].astype('float32')
            return all_data, None
        
    cloudnames = os.path.join(DATA_PATH,'train')
    if not os.path.exists(os.path.join(cloudnames,partition+'ing_ds'+'.hdf5')):
        extract_from_ply(num_points,grid_size,DATA_PATH,partition)
    f = h5py.File(os.path.join(cloudnames,partition+'ing_ds'+'.hdf5'), 'r')    
    all_data = f['pointclouds/'+partition+'ing_clouds'][:].astype('float32')
    all_label = f['labels/'+partition+'ing_labels'][:].astype('int64')
    f.close()

    return all_data, all_label

class ParisLille(Dataset):
    def __init__(self, num_points, grid_size, partition='train'):
        self.data, self.label = load_data_Paris(num_points,grid_size,partition)
        self.num_points = num_points
        self.partition = partition        
        
    def __getitem__(self, item):

        pointcloud = self.data[item]
        label = self.label[item]

        if self.partition == 'train':            
            pointcloud = translate_pointcloud(pointcloud) # data augmentation
            
        if self.partition == 'evaluate':
            return pointcloud
                
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
#    train = ModelNet40(1024,1,1,True,True)
#    test = ModelNet40(1024,1,1,True,True,'test')
    train = ParisLille(1024,10,'train')
    for data, label in train:
        print(data.shape)
        print(label.shape)
