import torch
from vit_model import VisionTransformer
from bert import Bert


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))
model = VisionTransformer().to(device)
print(model)