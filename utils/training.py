import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07, class_similarity=None):
        super().__init__()
        self.temperature = temperature
        self.class_similarity = class_similarity

    def forward(self, features, labels):
        device = features.device
        batch_size = features.size(0)

        features = F.normalize(features, dim=1)
        sim_matrix = torch.matmul(features, features.T) / self.temperature

        labels = labels.view(-1, 1)

        positive_mask = torch.eq(labels, labels.T).float()
        positive_mask.fill_diagonal_(0)

        negatives_mask = (~torch.eye(batch_size, dtype=torch.bool, device=device)).float()
        neg_weights = negatives_mask

        sim_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        sim_matrix = sim_matrix - sim_max.detach()

        exp_sim = torch.exp(sim_matrix) * neg_weights
        log_sum_exp = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        log_prob = sim_matrix - log_sum_exp
        pos_log_prob = log_prob * positive_mask

        num_positives = positive_mask.sum(dim=1)
        valid_samples = num_positives > 0
        if valid_samples.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        sample_losses = pos_log_prob.sum(dim=1) / (num_positives + 1e-8)
        sample_losses = sample_losses[valid_samples]

        return -sample_losses.mean()