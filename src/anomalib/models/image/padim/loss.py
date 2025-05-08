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

        self.b_cov = nn.Parameter(torch.randn(self.num_patches,self.feat_dim,self.feat_dim).to('cuda'))
        self.b_mean = nn.Parameter(torch.randn(self.feat_dim, self.num_patches).to('cuda'))
        self.signatures = nn.Parameter(sig_mat.to(self.device))
        
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
            self.b_cov = self.b_cov.to(self.device)

        elif cov_type == "diagonal":
            self.b_cov = torch.diag_embed(torch.diagonal(self.b_cov, dim1=-2, dim2=-1)).to(self.device)

        elif cov_type == "isotropic":
            avg_variance = torch.mean(torch.diagonal(self.b_cov, dim1=-2, dim2=-1),dim=-1)  # Compute average variance (mean of diagonal)
            avg_full = torch.mean(self.b_cov, dim=(-2,-1))                                           # Compute average of full mat
            det = torch.logdet(self.b_cov)
            trace = torch.einsum("bii->b", self.b_cov)
            self.b_cov = det[:, None, None] * torch.eye(self.b_cov.shape[-1]).to(self.device)
        # Compute U (eigenvectors) and D (eigvenvalues) 
        try:
            U_mat, eigenvalues, _ = torch.svd(self.b_cov)
            
        except:
            U_mat, eigenvalues, _ = torch.svd(self.b_cov+
                                              (1e-5*torch.eye(self.b_cov.shape[1],
                                                              device=self.device)))
    
        #Compute D^-1/2 power
        D_mat = torch.diag_embed(torch.pow(eigenvalues, -1 / 2))

        #Compute matrix product DU^-1/2, should be
        #Perform transpose operation along DxD dimension (follow paper)
        DU = torch.bmm(D_mat, U_mat.transpose(1,2))

        # Center using mean
        X_centered = X-self.b_mean
        
        # Compute xHat
        # xHat = torch.empty(X_centered.shape[0],DU.shape[0],DU.shape[1])
        # for i in range(X_centered.shape[0]):
        #     for j in range(DU.shape[0]):
        #         xHat[i][j]=torch.matmul(DU[j],X_centered[i][:,j])
        X_centered_T = X_centered.permute(0, 2, 1)
        xHat = torch.einsum('pij, bpj -> bpi', DU, X_centered_T)

        # Compute sHat
        # sHat = torch.empty(DU.shape[0],DU.shape[1])
        # for i in range(DU.shape[0]):
        #     sHat[i]=torch.matmul(DU[i],self.signatures[:,i])
        sHat = torch.einsum('pij, jp -> pi', DU, self.signatures)

        # Compute ACE score between features and signature, 
        # score between of -1 and 1 
        # L2 normalization done in function
        xHat = F.normalize(xHat, dim=2)
        sHat = F.normalize(sHat,dim=1)
        
        # ACE_targets = torch.empty(X_centered.shape[0],DU.shape[0])
        # for i in range(ACE_targets.shape[0]):
        #     for j in range(ACE_targets.shape[1]):
        #         ACE_targets[i][j]=torch.matmul(sHat.transpose(0,1)[:,j],xHat[i][j])
        # ACE_targets=ACE_targets.to(self.device)

        ACE_targets = torch.einsum('jk, bjk -> bj', sHat, xHat)        

        return ACE_targets
    