# https://www.youtube.com/watch?v=ovB0ddFtzzA&ab_channel=mildlyoverfitted
import torch
import torch.nn as nn

class PatchEmbeding(nn.Module):
    """ Split image into patches and then embed them. """
    def __init__(self, img_size, patch_size, in_channels=3, embeding_dim=768) -> None:
        # Expcts square image !!!!
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.number_patches = (img_size // patch_size) ** 2

        # Convlution layer that does the splitting into patches and thair embedding
        self.proj = nn.Conv2d(
            in_channels,
            embeding_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
    def forward(self, x):
        """
        Parameters
        ----------
        x = torch.Tensor
            Shape( n_samples, in_chans, img_size, img_size)
        Returns
        -------
        torch.Tensor
            Shape(n_samples, n_patches, embed_dim)
        """
        self.proj(
            x
        ) # (n_samples, embed_dim, n_patches ** 0.5, n_atches ** 0.5)
        x = x.flatten(2) # (n_samples, embed_dim, n_patches)
        x = x.transpose(1, 2) # (n_samples, n_patches, embed_dim)
        return x

class Attention(nn.Module):
    