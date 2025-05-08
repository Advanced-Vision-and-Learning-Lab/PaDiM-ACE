"""PyTorch model for the PaDiM model implementation.

This module implements the PaDiM model architecture using PyTorch. PaDiM models the
distribution of patch embeddings at each spatial location using multivariate
Gaussian distributions.

The model extracts features from multiple layers of pretrained CNN backbones to
capture both semantic and low-level visual information. During inference, it
computes Mahalanobis distances between test patch embeddings and their
corresponding reference distributions.

Example:
    >>> from anomalib.models.image.padim.torch_model import PadimModel
    >>> model = PadimModel(
    ...     backbone="resnet18",
    ...     layers=["layer1", "layer2", "layer3"],
    ...     pre_trained=True,
    ...     n_features=100
    ... )
    >>> input_tensor = torch.randn(32, 3, 224, 224)
    >>> output = model(input_tensor)

Paper: https://arxiv.org/abs/2011.08785

See Also:
    - :class:`anomalib.models.image.padim.lightning_model.Padim`:
        Lightning implementation of the PaDiM model
    - :class:`anomalib.models.image.padim.anomaly_map.AnomalyMapGenerator`:
        Anomaly map generation for PaDiM using Mahalanobis distance
    - :class:`anomalib.models.components.MultiVariateGaussian`:
        Multivariate Gaussian distribution modeling
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from random import sample
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812
import pdb
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from anomalib.data import InferenceBatch
from anomalib.models.components import MultiVariateGaussian, TimmFeatureExtractor
from anomalib.models.components.feature_extractors import dryrun_find_featuremap_dims

from .anomaly_map import AnomalyMapGenerator

if TYPE_CHECKING:
    from anomalib.data.utils.tiler import Tiler

# defaults from the paper
_N_FEATURES_DEFAULTS = {
    "resnet18": 100,
    "wide_resnet50_2": 550,
}



def _deduce_dims(
    feature_extractor: TimmFeatureExtractor,
    input_size: tuple[int, int],
    layers: list[str],
) -> tuple[int, int]:
    """Run a dry run to deduce the dimensions of the extracted features.

    This function performs a forward pass to determine the dimensions of features
    extracted from the specified layers of the backbone network.

    Args:
        feature_extractor (TimmFeatureExtractor): Feature extraction model
        input_size (tuple[int, int]): Input image dimensions (height, width)
        layers (list[str]): Names of layers to extract features from

    Important:
        ``layers`` is assumed to be ordered and the first (``layers[0]``)
        is assumed to be the layer with largest resolution.

    Returns:
        tuple[int, int]: Dimensions of extracted features:
            - n_dims_original: Total number of feature dimensions
            - n_patches: Number of spatial patches
    """
    dimensions_mapping = dryrun_find_featuremap_dims(feature_extractor, input_size, layers)

    # the first layer in `layers` has the largest resolution
    first_layer_resolution = dimensions_mapping[layers[0]]["resolution"]
    n_patches = torch.tensor(first_layer_resolution).prod().int().item()

    # the original embedding size is the sum of the channels of all layers
    n_features_original = sum(dimensions_mapping[layer]["num_features"] for layer in layers)  # type: ignore[misc]

    return n_features_original, n_patches


class PadimModel(nn.Module):
    """Padim Module.

    Args:
        layers (list[str]): Layers used for feature extraction
        backbone (str, optional): Pre-trained model backbone. Defaults to
            ``resnet18``.
        pre_trained (bool, optional): Boolean to check whether to use a
            pre_trained backbone. Defaults to ``True``.
        n_features (int, optional): Number of features to retain in the dimension
            reduction step. Default values from the paper are available for:
            resnet18 (100), wide_resnet50_2 (550). Defaults to ``None``.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        layers: list[str] = ["layer1", "layer2", "layer3"],  # noqa: B006
        pre_trained: bool = True,
        n_features: int | None = None,
        loss: str = "mahalanobis",
        cov_type = "full"
    ) -> None:
        super().__init__()
        self.tiler: Tiler | None = None

        self.backbone = backbone
        self.layers = layers
        self.feature_extractor = TimmFeatureExtractor(
            backbone=self.backbone,
            layers=layers,
            pre_trained=pre_trained,
        ).eval()
        self.n_features_original = sum(self.feature_extractor.out_dims)
        self.n_features = n_features or _N_FEATURES_DEFAULTS.get(self.backbone)
        if self.n_features is None:
            msg = (
                f"n_features must be specified for backbone {self.backbone}. "
                f"Default values are available for: {sorted(_N_FEATURES_DEFAULTS.keys())}"
            )
            raise ValueError(msg)

        if not (0 < self.n_features <= self.n_features_original):
            msg = f"For backbone {self.backbone}, 0 < n_features <= {self.n_features_original}, found {self.n_features}"
            raise ValueError(msg)

        # Since idx is randomly selected, save it with model to get same results
        self.register_buffer(
            "idx",
            torch.tensor(sample(range(self.n_features_original), self.n_features)),
        )
        self.idx: torch.Tensor
        self.loss = loss
        self.cov_type = cov_type
        self.sig_mat = None
        self.gaussian = MultiVariateGaussian()

    def forward(self, input_tensor: torch.Tensor, anom_path: str | None) -> torch.Tensor | InferenceBatch:
        """Forward-pass image-batch (N, C, H, W) into model to extract features.

        Args:
            input_tensor (torch.Tensor): Image batch with shape (N, C, H, W)
            anom_path (str): Path to the anomalous training images 

        Returns:
            torch.Tensor | InferenceBatch: If training, returns the embeddings.
                If inference, returns ``InferenceBatch`` containing prediction
                scores and anomaly maps.

        Example:
            >>> model = PadimModel()
            >>> x = torch.randn(32, 3, 224, 224)
            >>> features = model.extract_features(x)
            >>> features.keys()
            dict_keys(['layer1', 'layer2', 'layer3'])
            >>> [v.shape for v in features.values()]
            [torch.Size([32, 64, 56, 56]),
            torch.Size([32, 128, 28, 28]),
            torch.Size([32, 256, 14, 14])]
        """
        output_size = input_tensor.shape[-2:]
        if self.sig_mat is None and self.loss == "lace":
            H, W = output_size
            self.sig_mat = self.init_signitures(anom_path)
            self.anomaly_map_generator = AnomalyMapGenerator(sig_mat=self.sig_mat,loss=self.loss,cov_type=self.cov_type, n_features=self.n_features,num_patches=int((H/4)*(W/4)))
        
        if self.tiler:
            input_tensor = self.tiler.tile(input_tensor)

        with torch.no_grad():
            features = self.feature_extractor(input_tensor)
            embeddings = self.generate_embedding(features)

        if self.tiler:
            embeddings = self.tiler.untile(embeddings)

        if self.training:
            
            return embeddings

        anomaly_map = self.anomaly_map_generator(
            embedding=embeddings,
            mean=self.gaussian.mean,
            inv_covariance=self.gaussian.inv_covariance,
            image_size=output_size
        )

        pred_score = torch.amax(anomaly_map, dim=(-2, -1))
        return InferenceBatch(pred_score=pred_score, anomaly_map=anomaly_map)

    def generate_embedding(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        """Generate embedding from hierarchical feature map.

        This method combines features from multiple layers of the backbone network
        to create a rich embedding that captures both low-level and high-level
        image features.

        Args:
            features (dict[str, torch.Tensor]): Dictionary mapping layer names to
                their feature tensors extracted from the backbone CNN.

        Returns:
            torch.Tensor: Embedding tensor combining features from all specified
                layers, with dimensions reduced according to ``n_features``.
        """
        embeddings = features[self.layers[0]]
        
        if self.backbone == 'swinv2_tiny_window8_256':
            embeddings = embeddings.permute(0,3,1,2)

        for layer in self.layers[1:]:
            layer_embedding = features[layer]

            if self.backbone == 'swinv2_tiny_window8_256':
                layer_embedding = layer_embedding.permute(0,3,1,2)
            
            layer_embedding = F.interpolate(layer_embedding, size=embeddings.shape[-2:], mode="nearest")
            embeddings = torch.cat((embeddings, layer_embedding), 1)

        # subsample embeddings
        idx = self.idx.to(embeddings.device)
        return torch.index_select(embeddings, 1, idx)
    
    def init_signitures(self, anom_path: str):
        """Create target signiture matrix for LACE

        This method takes all the anomalous training images, extracts their features 
        from the first three layers, randomly reduces them to n_features,generates embeddings 
        from the features, and takes the mean of the feature embedding. 

        Returns:
            signiture_matrix: target signiture matrix to be used in LACE
        """
        transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
        ])
        norm_path = anom_path.rsplit("/anom", 1)[0]+"/norm"
        anom_dataset = datasets.ImageFolder(root= anom_path, transform=transform)
        # norm_dataset = datasets.ImageFolder(root= norm_path, transform=transform)

        anom_dataloader = DataLoader(anom_dataset, batch_size=16, shuffle=False)
        # norm_dataloader = DataLoader(norm_dataset, batch_size=16, shuffle=False)

        with torch.no_grad():
            norm_embeddings = []
            anom_embeddings = []
            
            for _, (images, labels) in enumerate(anom_dataloader):
                anom_features = self.feature_extractor(images.to('cuda'))           # Extract features
                anom_embeddings.append(self.generate_embedding(anom_features))      # Generate Embeddings
            anom_embeddings = torch.cat(anom_embeddings, dim=0)

            # for _, (images, labels) in enumerate(norm_dataloader):
            #     norm_features = self.feature_extractor(images.to('cuda'))               # Extract features
            #     norm_embeddings.append(self.generate_embedding(norm_features))          # Generate Embeddings
            # norm_embeddings = torch.cat(norm_embeddings, dim=0)
            

        # mean_norm_embeddings = torch.mean(norm_embeddings.permute(0,2,3,1), dim=0)  # compute the mean
        mean_anom_embeddings = torch.mean(anom_embeddings.permute(0,2,3,1), dim=0)

        # difference between mean anom and mean norm
        # mean_embeddings = mean_anom_embeddings - mean_norm_embeddings

        # mean anom 
        mean_embeddings = mean_anom_embeddings

        # median anom
        # mean_embeddings, _ = torch.median(anom_embeddings.permute(0,2,3,1), dim=0)

        # difference^2 -l2 norm
        # mean_embeddings = mean_embeddings**2

        h, w, feat_dim = mean_embeddings.shape
        signiture_matrix = mean_embeddings.reshape(feat_dim, h*w)
        return signiture_matrix
