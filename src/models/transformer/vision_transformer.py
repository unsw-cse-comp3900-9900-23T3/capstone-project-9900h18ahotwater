import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, img_size=[224,224], patch_size=16 , in_channels=3, embed_dim= 16*16*3):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        

