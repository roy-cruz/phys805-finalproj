import torch
import uproot
import awkward as ak
import matplotlib.pyplot as plt
from typing import Union
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc

EPS = 1e-8

def load_data(ds_cfg, qcd_or_emj, filter_name= None, entry_stop=None):
    redir = ds_cfg[qcd_or_emj]['redir']
    path = redir + ds_cfg[qcd_or_emj]['datasets'][0][list(ds_cfg[qcd_or_emj]['datasets'][0].keys())[0]]
    data = uproot.open(path)["Events"].arrays(
        entry_stop=entry_stop,
        filter_name=filter_name,
    )
    if isinstance(filter_name, list):
        data = data[filter_name]
    return data

def ak_to_torch(ak_arr, ftrs, num_jets, label=None, extra_branches=None):
    # ftr_tensor shape: (B, num_jets, num_features)
    ftr_list = []
    extra_list = []
    for ftr in ftrs:
        ftr_data = ak.pad_none(ak_arr[ftr], target=num_jets, axis=1, clip=True).to_numpy() # (B, num_jets)
        ftr_tensor = torch.tensor(ftr_data, dtype=torch.float32).unsqueeze(-1)  # (B, num_jets, 1)
        ftr_list.append(ftr_tensor)
    ftr_tensor = torch.cat(ftr_list, dim=-1)  # (B, num_jets, num_features)
    if label is not None:
        labels = torch.full((ftr_tensor.shape[0], num_jets, 1), label, dtype=torch.float32) # (B, num_jets, 1)
        ftr_tensor = torch.cat([ftr_tensor, labels], dim=-1)  # (B, num_jets, num_features + 1)

    # Turn nan to zero
    ftr_tensor = torch.nan_to_num(ftr_tensor, nan=0.0)
    return ftr_tensor

def get_pu_weights(sig_pu, bkg_pu, bins=50, range=(0, 50)):
    sig_hist, bin_edges = ak.histogram(sig_pu, bins=bins, range=range)
    bkg_hist, _ = ak.histogram(bkg_pu, bins=bins, range=range)
    sig_hist = sig_hist / ak.sum(sig_hist)
    bkg_hist = bkg_hist / ak.sum(bkg_hist)
    weights = bkg_hist / sig_hist
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    return bin_centers, weights

def compute_norm_constants(data: torch.Tensor) -> dict:
    jet_pt = data[...,0]
    jet_eta = data[...,1]
    jet_mass = data[...,3]

    valid = jet_pt > 0

    pt_max = torch.log(jet_pt[valid]).max()
    pt_min = torch.log(jet_pt[valid]).min()
    eta_max = jet_eta[valid].max()
    eta_min = jet_eta[valid].min()
    phi_max = torch.pi
    phi_min = -torch.pi
    mass_max = jet_mass[valid].max()
    mass_min = jet_mass[valid].min()

    return {
        "pt_min": pt_min,
        "pt_max": pt_max,
        "eta_max": eta_max,
        "eta_min": eta_min,
        "phi_max": phi_max,
        "phi_min": phi_min,
        "mass_max": mass_max,
        "mass_min": mass_min,
    }