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
        
    def forward(self,X, b_mean, b_cov, cov_type):
        """
        Args:
            X (torch tensor): batch of image embeddings for every patch in the image
            b_mean (torch tensor): Mean for every patch of normal image distribution
            b_covs (torch tensor): inverse covariance matrix for every patch of the normal image distribution
        Returns:
            ACE_targets (torch tensor): 
        """
        
        # inv_diag_mat = torch.linalg.inv(diag_mat)

        if cov_type == "full":
            b_cov = b_cov.to(self.device)
        elif cov_type == "diagonal":
            # b_covs_inv = inv_diag_mat.to(self.device)
            b_cov = torch.diag_embed(torch.diagonal(b_cov, dim1=1, dim2=2))

        elif cov_type == "isotropic":
            # b_covs_inv = inv_iso_mat.to(self.device)
            avg_variance = torch.mean(torch.diagonal(b_cov, dim1=1,dim2=2),dim=1)  # Compute average variance
            b_cov = avg_variance[:, None, None] * torch.eye(b_cov.shape[1]).to(self.device)
        
        # Full covariance matrix
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

        X_centered = X-b_mean
        
        #Compute x_hat
        # xHat = torch.matmul(DU, X_centered.T)
        # xHat = torch.einsum('pfd, bfp -> bpf', DU, X_centered)
        xHat = torch.empty(X_centered.shape[0],DU.shape[0],DU.shape[1])
        for i in range(X_centered.shape[0]):
            for j in range(DU.shape[0]):
                xHat[i][j]=torch.matmul(DU[j],X_centered[i][:,j])
        
        # sHat = torch.einsum('pfd, fp -> pf', DU, self.signatures)
        sHat = torch.empty(DU.shape[0],DU.shape[1])
        for i in range(DU.shape[0]):
            sHat[i]=torch.matmul(DU[i],self.signatures[:,i])
        # sHat = torch.bmm(DU, self.signatures.transpose(0,1).reshape(self.signatures.shape[1],self.signatures.shape[0],1))
        
        #Compute ACE score between features and signatures (NxC, one vs all), 
        #score between of -1 and 1 
        #L2 normalization done in function
        xHat = F.normalize(xHat, dim=2)
        sHat = F.normalize(sHat,dim=1)
        
        # ACE_targets = torch.mm(xHat,sHat) + torch.ones(batch_size,self.num_classes).to(self.device)
        # ACE_targets = torch.einsum('pf, bfp -> bp', sHat, xHat)
        ACE_targets = torch.empty(X_centered.shape[0],DU.shape[0])
        for i in range(ACE_targets.shape[0]):
            for j in range(ACE_targets.shape[1]):
                ACE_targets[i][j]=torch.matmul(sHat.transpose(0,1)[:,j],xHat[i][j])
        
        ACE_targets=ACE_targets.to(self.device)
        return ACE_targets
    