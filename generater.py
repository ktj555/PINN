import torch
from torch.utils.data import Dataset, DataLoader
from torch import zeros, empty
from numpy import linspace, array, vstack, hstack, ones, tile, float32
from numpy.random import randint,uniform
from copy import deepcopy
import os
import pickle

class Domain:
    def __init__(self,x_range,y_range):
        self.x_range = x_range
        self.y_range = y_range

class BC(Dataset):
    def __init__(self,domain,n_data,num_each_bc):
        self.x_min, self.x_max = domain.x_range
        self.y_min, self.y_max = domain.y_range
        self.labels = zeros(n_data,1)
        self.length = n_data
        self.num_bc = num_each_bc
        x = linspace(self.x_min,self.x_max,num_each_bc + 2)
        y = linspace(self.y_min,self.y_max,num_each_bc + 2)
        self.BCdomain = {'up':array([[x_i,self.y_max] for x_i in x[1:-1]], dtype = float32),
                       'down':array([[x_i,self.y_min] for x_i in x[1:-1]], dtype = float32),
                       'right':array([[self.x_max,y_i] for y_i in y[1:-1]], dtype = float32),
                       'left':array([[self.x_min,y_i] for y_i in y[1:-1]], dtype = float32)}
        self.BCvalues = {'up':0.0,
                         'down':0.0,
                         'right':0.0,
                         'left':0.0}
        self.BCtype = {'up':'value',
                       'down':'value',
                       'right':'value',
                       'left':'value'}
        self.BCinput = None
        self.Randominput = None
        self.BClabels = None
        self.build()

    def __len__(self):
        return self.length
        
    def __getitem__(self,index):
        return [self.BCinput[index], self.Randominput[index]], [self.labels[index],self.BClabels[index]]    

    def set_BC_value(self,**item):
        for key,value in item.items():
            if key not in ['up','down','right','left']:
                raise KeyError('Enter up, down, right, left only')
            self.BCvalues[key]=value[0]
            self.BCtype[key]=value[1]
        self.build()
        return self
    
    def build(self):
        typeindex = [['up','down','right','left'][i] for i in randint(low = 0,high = 4,size=self.length)]
        domainindex = randint(low = 0, high = self.num_bc, size = self.length)
        bc_sub_domain=array([self.BCdomain[t][d] for t,d in zip(typeindex,domainindex)], dtype = float32)
        bc_labels=array([[self.BCvalues[key]] for key in typeindex], dtype = float32)
            
        random_sub_domain=hstack([uniform(self.x_min,self.x_max,size=[self.length,1]).astype(float32),uniform(self.y_min,self.y_max,size=[self.length,1]).astype(float32)])
        
        bcfull_up = hstack((self.BCdomain['up'],ones([self.num_bc,1],dtype = float32) * self.BCvalues['up'])).flatten()
        bcfull_down = hstack((self.BCdomain['down'],ones([self.num_bc,1], dtype = float32) * self.BCvalues['down'])).flatten()
        bcfull_left = hstack((self.BCdomain['left'],ones([self.num_bc,1], dtype=float32) * self.BCvalues['left'])).flatten()
        bcfull_right = hstack((self.BCdomain['right'],ones([self.num_bc,1], dtype = float32) * self.BCvalues['right'])).flatten()
        bcfull = hstack((bcfull_up,bcfull_down,bcfull_right,bcfull_left))
        bcfully = tile(bcfull,(self.length,1))

        bc_sub_domain = torch.from_numpy(bc_sub_domain)
        bc_sub_domain.requires_grad_(True)
        random_sub_domain = torch.from_numpy(random_sub_domain)
        random_sub_domain.requires_grad_(True)

        self.BClabels = torch.from_numpy(bc_labels)
        self.BCinput = torch.hstack((bc_sub_domain,torch.from_numpy(bcfully)))
        self.Randominput = torch.hstack((random_sub_domain,torch.from_numpy(bcfully)))

class ChangableBC(BC):
    def __init__(self,domain,n_data,num_each_bc,max_noize):
        super().__init__(domain,n_data,num_each_bc)
        self.origin_value = deepcopy(self.BCvalues)
        self.max = max_noize

    def changeBC(self):
        for key,value in self.origin_value.items():
            self.BCvalues[key] = value + self.max * uniform(-1,1,size = 1).astype(float32)[0]
        self.build()
        return self