# -*- coding: utf-8 -*-
"""
Create datasets and dataloaders for models
"""

## Python standard libraries
from __future__ import print_function
from __future__ import division
import numpy as np
import itertools
import pdb
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split

## PyTorch dependencies
import torch
from torchvision import transforms

## Local external libraries
from Datasets.Pytorch_Datasets import *
from Datasets.Get_transform import *

from Datasets import loader
from Datasets import preprocess


def Prepare_DataLoaders(Network_parameters, split,input_size=224):
    
    Dataset = Network_parameters['Dataset']
    data_dir = Network_parameters['data_dir']
    dataset_sampler = None

    # Data augmentation and normalization for training
    # Just normalization and resize for test
    # Data transformations as described in:
    # http://openaccess.thecvf.com/content_cvpr_2018/papers/Xue_Deep_Texture_Manifold_CVPR_2018_paper.pdf

    if Dataset == "MSTAR":
        global data_transforms
        data_transforms, mean, std = get_transform(Network_parameters, input_size=input_size)
        Network_parameters['mean'] = mean
        Network_parameters['std'] = std
    else:
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(Network_parameters['resize_size']),
                transforms.RandomResizedCrop(input_size,scale=(.8,1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(Network_parameters['center_size']),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                 ]),
        }
    
 
    if Dataset == 'FashionMNIST': #See people also use .5, .5 for normalization
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(Network_parameters['resize_size']),
                transforms.RandomResizedCrop(input_size,scale=(.8,1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(Network_parameters['center_size']),
                transforms.CenterCrop(input_size),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
  
        train_dataset = FashionMNIST_Index(data_dir,train=True,transform=data_transforms['train'],
                                       download=True)
        y = train_dataset.targets
        indices = np.arange(len(y))
        indices = torch.as_tensor(indices)
        _, _, _, _, train_indices, val_indices = train_test_split(y, y, indices, 
                                                          test_size=0.1, 
                                                          stratify=y, random_state=42)
        
        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        dataset_sampler = {'train': train_sampler, 'val': val_sampler, 'test': None}
        
        test_dataset = FashionMNIST_Index(data_dir,train=False,transform=data_transforms['test'],
                                       download=True)
        
    elif Dataset == 'SVHN': #See people also use .5, .5 for normalization
        train_dataset = SVHN_Index(data_dir,train=True,transform=data_transforms['train'],
                                       download=True)
        y = train_dataset.targets
        indices = np.arange(len(y))
        _, _, _, _, train_indices, val_indices = train_test_split(y, y, indices, 
                                                          test_size=0.1, 
                                                          stratify=y, random_state=42)
        
        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        dataset_sampler = {'train': train_sampler, 'val': val_sampler, 'test': None}
        
        test_dataset = SVHN_Index(data_dir,train=False,transform=data_transforms['test'],
                                       download=True)
    
    elif Dataset == 'MSTAR':
        # train_dataset = MSTAR_Index(data_dir,train=True,transform=data_transforms['train'],
        #                                download=True)
        # y = train_dataset.targets
        # indices = np.arange(len(y))
        # _, _, _, _, train_indices, val_indices = train_test_split(y, y, indices, 
        #                                                   test_size=0.1, 
        #                                                   stratify=y, random_state=42)
        
        # # Creating PT data samplers and loaders:
        # train_sampler = SubsetRandomSampler(train_indices)
        # val_sampler = SubsetRandomSampler(val_indices)
        # dataset_sampler = {'train': train_sampler, 'val': val_sampler, 'test': None}
        
        # test_dataset = MSTAR_Index(data_dir,train=False,transform=data_transforms['test'],
        #                                download=True)
        train_dataset = loader.MSTAR_Dataset(path=data_dir, name='soc', is_train=True,
        transform=data_transforms['test'])
        
        X = np.arange(0,len(train_dataset))  # array [0, 1, 2, ..., 134602] indices of images
        y = train_dataset.targets            # array [..., 0,6,9,8,5,0] targets for the images
        indices = np.arange(len(y))          # array same as X

        #Set random state to keep the data the same order for each model
        X_train, X_val, y_train, y_val, train_indices, val_indices = train_test_split(X,y,indices,test_size=.1,stratify=y, random_state=42)
        
        train_dataset = torch.utils.data.Subset(loader.MSTAR_Dataset(data_dir, name='soc', is_train=True,transform=data_transforms['train']),X_train)
        val_dataset =  torch.utils.data.Subset(loader.MSTAR_Dataset(data_dir, name='soc', is_train=True,transform=data_transforms['test']),X_val)
        test_dataset = loader.MSTAR_Dataset(data_dir, name='soc', is_train=False,transform=data_transforms['test'])
       
    elif Dataset == 'CIFAR10': #See people also use .5, .5 for normalization
        train_dataset = CIFAR10_Index(data_dir,train=True,transform=data_transforms['train'],
                                       download=True)
        y = train_dataset.targets
        indices = np.arange(len(y))
        _, _, _, _, train_indices, val_indices = train_test_split(y, y, indices, 
                                                          test_size=0.1, 
                                                          stratify=y, random_state=42)
        
        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        dataset_sampler = {'train': train_sampler, 'val': val_sampler, 'test': None}
        
        test_dataset = CIFAR10_Index(data_dir,train=False,transform=data_transforms['test'],
                                       download=True)
        
    elif Dataset == 'CIFAR100': #See people also use .5, .5 for normalization
        train_dataset = CIFAR100_Index(data_dir,train=True,transform=data_transforms['train'],
                                       download=True)
        y = train_dataset.targets
        indices = np.arange(len(y))
        _, _, _, _, train_indices, val_indices = train_test_split(y, y, indices, 
                                                          test_size=0.1, 
                                                          stratify=y, random_state=42)
        
        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        dataset_sampler = {'train': train_sampler, 'val': val_sampler, 'test': None}
        
        test_dataset = CIFAR100_Index(data_dir,train=False,transform=data_transforms['test'],
                                       download=True)
        
    elif Dataset == 'CIFAR100_Coarse': #See people also use .5, .5 for normalization
        train_dataset = CIFAR100_Index(data_dir,train=True,transform=data_transforms['train'],
                                       download=True,coarse=True)
        y = train_dataset.targets
        indices = np.arange(len(y))
        _, _, _, _, train_indices, val_indices = train_test_split(y, y, indices, 
                                                          test_size=0.1, 
                                                          stratify=y, random_state=42)
        
        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        dataset_sampler = {'train': train_sampler, 'val': val_sampler, 'test': None}
        
        test_dataset = CIFAR100_Index(data_dir,train=False,transform=data_transforms['test'],
                                       download=True,coarse=True)
        
    else:
        raise RuntimeError('Dataset not implemented') 

    #Create dataloaders
    if dataset_sampler is not None:
        image_datasets = {'train': train_dataset, 'val': train_dataset, 'test': test_dataset}
        # Create training and test dataloaders
        dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                           batch_size=Network_parameters['batch_size'][x],
                                                           sampler=dataset_sampler[x],
                                                           num_workers=Network_parameters['num_workers'],
                                                           pin_memory=Network_parameters['pin_memory'])
                                                           for x in ['train', 'val', 'test']}
        dataloaders_dict['train_full'] = torch.utils.data.DataLoader(train_dataset,
                                                           batch_size=Network_parameters['batch_size']['val'],
                                                           sampler=None,
                                                           num_workers=Network_parameters['num_workers'],
                                                           pin_memory=Network_parameters['pin_memory'])
    
    else:
        image_datasets = {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}
        # Create training and test dataloaders
        dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                           batch_size=Network_parameters['batch_size'][x],
                                                           shuffle=True,
                                                           num_workers=Network_parameters['num_workers'],
                                                           pin_memory=Network_parameters['pin_memory'])
                                                           for x in ['train', 'val','test']}
        # only when using MSTAR
        dataloaders_dict['train_full'] = torch.utils.data.DataLoader(loader.MSTAR_Dataset(path=data_dir, name='soc', is_train=True,transform=data_transforms['test']),
                                                           batch_size=Network_parameters['batch_size']['val'],
                                                           sampler=None,
                                                           num_workers=Network_parameters['num_workers'],
                                                           pin_memory=Network_parameters['pin_memory'])
        
    #If training dataset is larger than number of images for TSNE, subsampl
    TSNE_indices = {}
    dataset_indices = {'train': train_indices, 'val': val_indices, 
                       'test': np.arange(len(image_datasets['test']))}
    phase_count = 0
    
    for phase in ['train', 'val','test']:
        indices = np.arange(len(dataloaders_dict[phase].sampler))
        # mstar test is loaded differently (change this to if mstar)
        if phase != 'test' and Dataset == 'MSTAR':
            y = np.array(image_datasets[phase].dataset.targets)[dataset_indices[phase]]
        else:
            y = np.array(image_datasets[phase].targets)[dataset_indices[phase]]
        #Use stratified split to balance training validation splits, 
        #set random state to be same for each encoding method
        try:
            _,_,_,_,_,temp_indices = train_test_split(y,y,indices,
                                                      stratify=y,
                                                      test_size = Network_parameters['Num_TSNE_images'],
                                                      random_state=phase_count)
        except:
            #For CIFAR10, only 5,000 train images (use all)
            temp_indices = indices 
            
        TSNE_indices[phase] = dataset_indices[phase][temp_indices]
        phase_count += 1
    
    # Creating PT data samplers and loaders:
    TSNE_sampler = {'train': TSNE_indices['train'], 'val': TSNE_indices['val'],'test': TSNE_indices['test']}
    dataloaders_dict['TSNE'] = TSNE_sampler 

    return dataloaders_dict
