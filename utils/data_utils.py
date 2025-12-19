import torch
import uproot
import numpy as np
import awkward as ak

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

def ak_to_torch(ak_arr, ftrs, num_jets, label=None):
    # ftr_tensor shape: (B, num_jets, num_features)
    ftr_list = []
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
    jet_nconst = data[...,4]
    jet_nsv = data[...,5]
    jet_area = data[...,6]

    valid = jet_pt > 0

    pt_max = torch.log(jet_pt[valid]).max().item()
    pt_min = torch.log(jet_pt[valid]).min().item()
    eta_max = jet_eta[valid].max().item()
    eta_min = jet_eta[valid].min().item()
    phi_max = torch.pi
    phi_min = -torch.pi
    mass_max = jet_mass[valid].max().item()
    mass_min = jet_mass[valid].min().item()
    nconst_max = jet_nconst[valid].max().item()
    nconst_min = jet_nconst[valid].min().item()
    nsv_max = jet_nsv[valid].max().item()
    nsv_min = jet_nsv[valid].min().item()
    area_max = jet_area[valid].max().item()
    area_min = jet_area[valid].min().item()

    return {
        "pt_min": pt_min,
        "pt_max": pt_max,
        "eta_max": eta_max,
        "eta_min": eta_min,
        "phi_max": phi_max,
        "phi_min": phi_min,
        "mass_max": mass_max,
        "mass_min": mass_min,
        "nconst_max": nconst_max,
        "nconst_min": nconst_min,
        "nsv_max": nsv_max,
        "nsv_min": nsv_min,
        "area_max": area_max,
        "area_min": area_min,
    }

def match_pu(sig, bkg):
    sig_pu = sig["Pileup_nPU"]
    sig_pu_bins = np.arange(0, ak.max(sig_pu)+1, 1)
    sig_bin_counts, _ = np.histogram(sig_pu, bins=sig_pu_bins) # num_bins, num_bins+1
    pu_probs = sig_bin_counts / ak.sum(sig_bin_counts)

    bkg_events = []
    for bin_pu, pu_prob in enumerate(pu_probs):
        bkg_events_in_bin = bkg[bkg["Pileup_nPU"] == bin_pu]
        n_sig_in_bin = sig_bin_counts[bin_pu]
        bkg_events.append(
            bkg_events_in_bin[0:n_sig_in_bin]
        )

    bkg_undersampled = ak.concatenate(bkg_events)
    bkg_undersampled = bkg_undersampled[np.random.permutation(len(bkg_undersampled))]
    return bkg_undersampled