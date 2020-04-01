# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 11:02:29 2020

@author: Pietrantoni Maxime
"""

from utils import read_ply
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import os
import h5py



def extract_from_ply(num_points, grid_size, DATA_PATH, partition, verbose=True):
    
    ## Read ply files in /data/Paris_Lille/train or test
    ## Sample num_points per cube of volume 1m3 per ply file
    ## Save them h5py dataset
    
    if partition=='train':
        data_train = os.listdir(os.path.join(DATA_PATH,'train'))
        data_train = [c for c in data_train if c[-3:]=='ply']
        training_clouds = []
        training_labels = []
                
        for cloudpoint_name in data_train:
            print(cloudpoint_name)
            
            cloud_dir = os.path.join(os.path.join(DATA_PATH,'train'),cloudpoint_name)
            data = read_ply(cloud_dir)    
            points = np.vstack((data['x'], data['y'], data['z'])).T
            label = data['class']
            max_coord = np.max(points,axis=0)
            min_coord = np.min(points,axis=0)
            
        
            
            X,Y,Z = np.mgrid[int(min_coord[0]):int(max_coord[0]):grid_size,int(min_coord[1]):int(max_coord[1]):grid_size,int(min_coord[2]):int(max_coord[2]):grid_size]
            coordinates = np.stack([X.flatten(),Y.flatten(),Z.flatten()],axis=1)
            
            for i,coords in enumerate(coordinates):
                x,y,z = coords
                
                x_ind = np.logical_and(points[:,0]>x,points[:,0]<x+grid_size)
                y_ind = np.logical_and(points[:,1]>y,points[:,1]<y+grid_size)
                z_ind = np.logical_and(points[:,2]>z,points[:,2]<z+grid_size)
                
                ind_inside = np.logical_and(np.logical_and(x_ind, y_ind), z_ind)
                points_inside = points[ind_inside,:]
                labels_inside = label[ind_inside]
                
                if points_inside.shape[0]>=num_points:
                    cube_data = np.concatenate((points_inside,np.expand_dims(labels_inside,axis=0).T),axis=1)
                    np.random.shuffle(cube_data)
                    cube_data_sampled = cube_data[:num_points,:]
                    training_clouds.append(cube_data_sampled[:,:3])
                    training_labels.append(cube_data_sampled[:,-1])
        
                if verbose and i%100==0:
                    print(len(training_clouds))
        #            if i==200:
        #                break
        
        training_clouds = np.stack(training_clouds)
        training_labels = np.stack(training_labels)
    
        X_train, X_test, y_train, y_test = train_test_split(training_clouds, training_labels, test_size=0.2, random_state=1)
        
        f = h5py.File(os.path.join(DATA_PATH,os.path.join(partition,'training_ds'+'.hdf5')), 'w')
        pc = f.create_group("pointclouds")
        l = f.create_group("labels")
        pc.create_dataset("training_clouds", data=X_train)
        l.create_dataset("traisning_labels", data=y_train)
        f.close()
        
        f = h5py.File(os.path.join(DATA_PATH,os.path.join(partition,'testing_ds'+'.hdf5')), 'w')
        pc = f.create_group("pointclouds")
        l = f.create_group("labels")
        pc.create_dataset("training_clouds", data=X_test)
        l.create_dataset("traisning_labels", data=y_test)
        f.close()
        
        
    if partition=='evaluate':
        data_test = os.listdir(os.path.join(DATA_PATH,'test'))
        data_test = [c for c in data_test if c[-3:]=='ply']
        
        evluate_clouds = []
                
        for cloudpoint_name in data_test:
            print(cloudpoint_name)
            
            cloud_dir = os.path.join(os.path.join(DATA_PATH,'test'),cloudpoint_name)
            data = read_ply(cloud_dir)    
            points = np.vstack((data['x'], data['y'], data['z'])).T

            max_coord = np.max(points,axis=0)
            min_coord = np.min(points,axis=0)
            
            X,Y,Z = np.mgrid[int(min_coord[0]):int(max_coord[0]):grid_size,int(min_coord[1]):int(max_coord[1]):grid_size,int(min_coord[2]):int(max_coord[2]):grid_size]
            coordinates = np.stack([X.flatten(),Y.flatten(),Z.flatten()],axis=1)
            
            for i,coords in enumerate(coordinates):
                x,y,z = coords
                
                x_ind = np.logical_and(points[:,0]>x,points[:,0]<x+grid_size)
                y_ind = np.logical_and(points[:,1]>y,points[:,1]<y+grid_size)
                z_ind = np.logical_and(points[:,2]>z,points[:,2]<z+grid_size)
                
                ind_inside = np.logical_and(np.logical_and(x_ind, y_ind), z_ind)
                points_inside = points[ind_inside,:]
                
                if points_inside.shape[0]>=num_points:
                    cube_data = points_inside
                    np.random.shuffle(cube_data)
                    cube_data_sampled = cube_data[:num_points,:]
                    evluate_clouds.append(cube_data_sampled)
        
                if verbose and i%100==0:
                    print(len(evluate_clouds))

        
        evluate_clouds = np.stack(evluate_clouds)
    
        f = h5py.File(os.path.join(DATA_PATH,os.path.join('test','evaluate_ds'+'.hdf5')), 'w')
        pc = f.create_group("pointclouds")
        pc.create_dataset("evaluate_clouds", data=evluate_clouds)
        f.close()
        
        
if __name__=='__main__':
    BASE_PATH = os.getcwd()
    DATA_PATH = os.path.join(os.path.join(BASE_PATH,'data'),'Paris_Lille')
    num_points = 1024
    grid_size = 20
    partition = 'train'
    extract_from_ply(num_points, grid_size, DATA_PATH, partition, verbose=True)