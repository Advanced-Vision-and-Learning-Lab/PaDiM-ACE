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
            b_mean (torch tensor): gaussian mean for every patch of normal image distribution
            b_covs (torch tensor): inverse gaussian covariance matrix for every patch of normal image distribution
        Returns:
            ACE_targets (torch tensor): 
        """
        b_cov = torch.linalg.inv(b_cov)
        # pdb.set_trace()

        
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
            U_mat, eigenvalues, _ = torch.svd(torch.einsum('pfd,pdf-> pfd',b_cov,b_cov.transpose(1,2)))
            
        except:
            U_mat, eigenvalues, _ = torch.svd(torch.mm(b_cov,b_cov.t())+
                                              (1e-5*torch.eye(b_cov.shape[0],
                                                              device=self.device)))
    
        #Compute D^-1/2 power
        D_mat = torch.diag_embed(torch.pow(eigenvalues, -1 / 2))

        #Compute matrix product DU^-1/2, should be
        #Perform transpose operation along DxD dimension (follow paper)
        DU = torch.matmul(D_mat, U_mat.transpose(1,2))

        X_centered = X-b_mean
        
        #Compute x_hat
        # xHat = torch.matmul(DU, X_centered.T)
        xHat = torch.einsum('pfd, bfp -> bpd', DU, X_centered)
        sHat = torch.einsum('pfd, pf -> pd',DU, self.signatures.T)
        
        #Compute ACE score between features and signatures (NxC, one vs all), 
        #score between of -1 and 1 
        #L2 normalization done in function
        xHat = F.normalize(xHat.T, dim=2)
        sHat = F.normalize(sHat.T,dim=1)
        
        # ACE_targets = torch.mm(xHat,sHat) + torch.ones(batch_size,self.num_classes).to(self.device)
        ACE_targets = torch.einsum('fpb, fp -> bp', xHat, sHat)
    
        #Uniqueness of signatures in whitened dataspace
        #ACE scores in angular softmax
        # labels = labels.long()
        # numerator = torch.diagonal(ACE_targets.transpose(0, 1)[labels])
        # excl = torch.cat([torch.cat((ACE_targets[i, :y], ACE_targets[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        # denominator = torch.exp(numerator) + torch.sum(torch.exp(excl), dim=1)
        # loss = (numerator - torch.log(denominator))
        # loss = -torch.mean(loss)
        
        return ACE_targets
    