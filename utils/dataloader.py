import torch
from typing import Union

EPS = 1e-8

class JetDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            data: torch.Tensor, 
            norm_constants: dict, 
            labels: Union[torch.Tensor, None] = None, 
        ):
        self.data = data
        self.labels = labels
        assert isinstance(norm_constants, dict)
        self.norm = norm_constants
    
    def __len__(self):
        return self.data.shape[0]
    
    def make_padding_mask(self, x):
        return x[...,0] == 0
    
    def normalize(self, x):

        pt_raw = x[:,0]
        eta_raw = x[:,1]
        phi_raw = x[:,2]
        mass_raw = x[:,3]
        nconst_raw = x[:,4]
        nsv_raw = x[:,5]
        area_raw = x[:,6]
    
        pt = torch.zeros_like(pt_raw)
        eta = torch.zeros_like(eta_raw)
        phi = torch.zeros_like(phi_raw)
        mass = torch.zeros_like(mass_raw)

        valid = pt_raw > 0

        if valid.any():
            pt_log = torch.log(pt_raw[valid])
            pt[valid] = (pt_log - self.norm["pt_min"]) / (self.norm["pt_max"] - self.norm["pt_min"] + EPS)
            eta[valid] = (eta_raw[valid] - self.norm["eta_min"]) / (self.norm["eta_max"] - self.norm["eta_min"] + EPS)
            phi[valid] = (phi_raw[valid] - self.norm["phi_min"]) / (self.norm["phi_max"] - self.norm["phi_min"] + EPS)
            mass[valid] = (mass_raw[valid] - self.norm["mass_min"]) / (self.norm["mass_max"] - self.norm["mass_min"] + EPS)
        
        return torch.stack([pt, eta, phi, mass, nconst_raw, nsv_raw, area_raw], dim=-1)

    def __getitem__(self, idx):
        x = self.normalize(self.data[idx])
        mask = self.make_padding_mask(x)
        y = None if self.labels is None else self.labels[idx]
        to_return = (x, mask) + (() if y is None else (y,))
        return to_return