import argparse
import os
import tqdm

import torch
from vit_model import VisionTransformer
from bert import Bert
from model import SFSC, DFDC
from torch.utils.tensorboard import SummaryWriter



# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print('Using {} device'.format(device))
# model = VisionTransformer().to(device)
# print(model)

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")
    tb_writer = SummaryWriter()
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)
