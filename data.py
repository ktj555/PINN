import torch
from torch.utils.data import Dataset
from math import floor, ceil

class Condition:
    def __init__(self,x_min, x_max, y_min, y_max, num_each_bc):
        self.domain = [[x_min,x_max],[y_min,y_max]]
        self.num_each_bc = num_each_bc
        self.BC_ = None
        self.BC_data = None
        self.IN = None
    
    def BC_domain(self):
        if(self.BC_ is None):
            x = torch.linspace(self.domain[0][0],self.domain[0][1],self.num_each_bc+2,requires_grad=True)[1:-1]
            y = torch.linspace(self.domain[1][0],self.domain[1][1],self.num_each_bc+2,requires_grad=True)[1:-1]
            up = torch.hstack((x.view(-1,1),torch.ones_like(x.view(-1,1)) * self.domain[1][1]))
            down = torch.hstack((x.view(-1,1),torch.ones_like(x.view(-1,1)) * self.domain[1][0]))
            right = torch.hstack((torch.ones_like(y.view(-1,1)) * self.domain[0][1],y.view(-1,1)))
            left = torch.hstack((torch.ones_like(y.view(-1,1)) * self.domain[0][0],y.view(-1,1)))
            self.BC_ = torch.vstack([up,down,right,left])
        return self.BC_
    
    def get_BC_feature(self,up,down,right,left):
        D = self.BC_domain()
        u = torch.ones(self.num_each_bc) * up
        d = torch.ones(self.num_each_bc) * down
        r = torch.ones(self.num_each_bc) * right
        l = torch.ones(self.num_each_bc) * left
        label = torch.vstack([u.reshape(-1,1),d.reshape(-1,1),r.reshape(-1,1),l.reshape(-1,1)])
        return torch.hstack([D,label])

    def IN_domain(self, n_data):
        if(self.IN is None):
            x_min, x_max = self.domain[0]
            y_min, y_max = self.domain[1]
            x, y = torch.meshgrid(torch.linspace(x_min,x_max,floor(n_data ** 0.5) + 3, requires_grad=True)[1:-1],torch.linspace(y_min,y_max,floor(n_data ** 0.5) + 3,requires_grad=True)[1:-1],indexing='ij')
            self.IN = torch.hstack([x.reshape(-1,1), y.reshape(-1,1)])[:n_data,:]
        return self.IN
    
    def random_IN(self,n_data):
        x_min, x_max = self.domain[0]
        y_min, y_max = self.domain[1]
        x_rand = x_min + (x_max - x_min) * torch.rand(n_data,1,requires_grad=True)
        y_rand = y_min + (y_max - y_min) * torch.rand(n_data,1,requires_grad=True)
        return torch.hstack([x_rand,y_rand])
    
    def get_IN_feature(self,n_data,mode=0):
        if(mode != 0):
            return self.random_IN(n_data)
        else:
            return self.IN_domain(n_data) 
    
    
class gen(Dataset):
    def __init__(self,condition,n_data,mode,BC_values):
        self.condition = condition
        self.n_data = n_data
        self.mode = mode
        self.BC_values = BC_values
        self.BC_data = self.condition.get_BC_feature(self.BC_values['up'],self.BC_values['down'],self.BC_values['right'],self.BC_values['left'])
        self.figure = self.condition.get_IN_feature(self.n_data,self.mode)

        return_BC_sub = self.BC_data.tile([ceil(self.n_data / self.condition.num_each_bc), 1])
        self.return_BC = return_BC_sub[torch.randperm(return_BC_sub.size(0))][:self.n_data,:]

    def __len__(self):
        return self.n_data

    def __getitem__(self,index):
        return self.figure[index], self.BC_data.flatten(), self.return_BC[index][:-1], self.return_BC[index][-1]
    
    def update(self,BC_values):
        self.BC_data = self.condition.get_BC_feature(BC_values['up'],BC_values['down'],BC_values['right'],BC_values['left'])
        return_BC_sub = self.BC_data.tile([ceil(self.n_data / self.condition.num_each_bc), 1])
        self.return_BC = return_BC_sub[torch.randperm(return_BC_sub.size(0))][:self.n_data,:]