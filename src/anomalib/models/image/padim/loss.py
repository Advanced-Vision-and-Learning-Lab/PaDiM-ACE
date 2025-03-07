import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
import torch.nn as nn


class ACELoss(nn.Module):
    """LACE loss term.
       Learn target signatures and background stats
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
        device (str): device selected for memory allocation
    """

    def __init__(self, sig_mat: torch.Tensor, feat_dim=2, device='cuda',eps=1e-7):
        super(ACELoss, self).__init__()
        
        self.feat_dim = feat_dim
        self.device = device
        self.eps = eps
        self.signatures = sig_mat.to(self.device)
        
    def forward(self,X, b_mean, b_covs, cov_type):
        """
        Args:
            X (torch tensor): batch of image embeddings for every patch in the image
            b_mean (torch tensor): gaussian mean for every patch of normal image distribution
            b_covs (torch tensor): inverse gaussian covariance matrix for every patch of normal image distribution
        Returns:
            ACE_targets (torch tensor): 
        """
        cov_mat = torch.linalg.inv(b_covs)
        diag_mat = torch.diag_embed(torch.diagonal(cov_mat, dim1=1, dim2=2))
        inv_diag_mat = torch.linalg.inv(diag_mat)
        
        avg_variance = torch.mean(torch.diagonal(cov_mat, dim1=1,dim2=2),dim=1)  # Compute average variance
        inv_iso_mat = avg_variance[:, None, None] * torch.eye(cov_mat.shape[1]).to(self.device)

        if cov_type == "full":
            b_covs_inv = b_covs.to(self.device)
        elif cov_type == "diagonal":
            b_covs_inv = inv_diag_mat.to(self.device)
        elif cov_type == "isotropic":
            b_covs_inv = inv_iso_mat.to(self.device)
        
        X = X.to(self.device)
        b_mean = b_mean.to(self.device)             

        # Whiten dataspace
        X_centered = (X-b_mean)  
        X_whitened = torch.einsum("ijk,kmj->ikm",X_centered, b_covs_inv)

        # Normalize X and compute similarity
        x_norm = F.normalize(X_whitened, dim=-1)
        sig_norm = F.normalize(self.signatures, dim=0)
        ACE_targets = torch.einsum("ijk,kj->ij",x_norm, sig_norm)

        # Apply cross entropy
        # ACE_targets = torch.clamp(ACE_targets, min=1e-7, max=1-1e-7)
        # loss = -torch.log(ACE_targets)
        
        return ACE_targets
    