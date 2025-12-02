import uproot
import awkward as ak
import torch

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

def ak_to_torch(ak_arr, label=None):
    ftr_arr = None
    for ftr in ak_arr.fields:
        ftr_arr = ak_arr[ftr] if ftr_arr is None else ak.concatenate([ftr_arr, ak_arr[ftr]], axis=1)

    ftr_tensor = torch.tensor(ak.to_numpy(ftr_arr))
    if label is not None:
        labels = torch.full((ftr_tensor.shape[0], 1), label)
        ftr_tensor = torch.cat([ftr_tensor, labels], dim=1)
    
    return ftr_tensor

def get_pu_weights(sig_pu, bkg_pu, bins=50, range=(0, 50)):
    sig_hist, bin_edges = ak.histogram(sig_pu, bins=bins, range=range)
    bkg_hist, _ = ak.histogram(bkg_pu, bins=bins, range=range)
    sig_hist = sig_hist / ak.sum(sig_hist)
    bkg_hist = bkg_hist / ak.sum(bkg_hist)
    weights = bkg_hist / sig_hist
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    return bin_centers, weights

class JetDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels=None):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        if self.labels is not None:
            return self.data[idx], self.labels[idx]
        return self.data[idx]