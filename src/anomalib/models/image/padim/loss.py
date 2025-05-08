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

    def __init__(self, sig_mat: torch.Tensor, num_patches: int,feat_dim=100, device='cuda',eps=1e-7):
        super(ACELoss, self).__init__()
        
        self.feat_dim = feat_dim
        self.device = device
        self.eps = eps
        self.num_patches = num_patches
        self.signatures = (sig_mat).to(self.device)
        
    def forward(self,X, b_mean, b_cov, cov_type):
        """
        Args:
            X (torch tensor): batch of image embeddings for every patch in the image
            b_mean (torch tensor): Mean for every patch of normal image distribution
            b_covs (torch tensor): inverse covariance matrix for every patch of the normal image distribution
        Returns:
            ACE_targets (torch tensor): 
        """
        if cov_type == "full":
            b_cov = b_cov.to(self.device)

        elif cov_type == "diagonal":
            b_cov = torch.diag_embed(torch.diagonal(b_cov, dim1=-2, dim2=-1)).to(self.device)

        elif cov_type == "isotropic":
            avg_variance = torch.mean(torch.diagonal(b_cov, dim1=-2, dim2=-1),dim=-1)  # Compute average variance (mean of diagonal)
            avg_full = torch.mean(b_cov, dim=(-2,-1))                                           # Compute average of full mat
            det = torch.logdet(b_cov)
            trace = torch.einsum("bii->b", b_cov)
            b_cov = det[:, None, None] * torch.eye(b_cov.shape[-1]).to(self.device)
        # Compute U (eigenvectors) and D (eigvenvalues) 
        try:
            U_mat, eigenvalues, _ = torch.svd(b_cov)
            
        except:
            U_mat, eigenvalues, _ = torch.svd(b_cov+
                                              (1e-5*torch.eye(b_cov.shape[1],
                                                              device=self.device)))
    
        #Compute D^-1/2 power
        D_mat = torch.diag_embed(torch.pow(eigenvalues, -1 / 2))

        #Compute matrix product DU^-1/2, should be
        #Perform transpose operation along DxD dimension (follow paper)
        DU = torch.bmm(D_mat, U_mat.transpose(1,2))

        # Center using mean
        X_centered = X-b_mean
        
        # Compute xHat
        X_centered_T = X_centered.permute(0, 2, 1)
        xHat = torch.einsum('pij, bpj -> bpi', DU, X_centered_T)

        # Compute sHat
        sHat = torch.einsum('pij, jp -> pi', DU, self.signatures)

        # Compute ACE score between features and signature, 
        # score between of -1 and 1 
        # L2 normalization done in function
        xHat = F.normalize(xHat, dim=2)
        sHat = F.normalize(sHat,dim=1)


        ACE_targets = torch.einsum('jk, bjk -> bj', sHat, xHat)        

        return ACE_targets
    