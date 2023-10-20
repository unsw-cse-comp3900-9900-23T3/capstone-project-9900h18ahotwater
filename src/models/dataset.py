from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import re
import pandas as pd
from io import StringIO
import numpy as np
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class MyDataSet(Dataset):

    def __init__(self, 
                 path, 
                 img_path, 
                 range_label, 
                 transform=None):
        # e.g. img_path = "./dataset/dataset1/data/"
        # labels are from 1 to 19, range_label = (1,20)
        self.range_label = range_label
        self.img_path = img_path
        self.dataset = self.get_data(path)
        self.transform = transform

    def get_data(self, path):
        with open(path) as file:
            lines = [re.sub(r'([^,])"(\s*[^\n])', r'\1/"\2', line) for line in file]
        df = pd.read_csv(StringIO(''.join(lines)), escapechar="/")
        for id in range(len(df)):
            item = np.array(df["Labels"][id].split(" "), dtype=int)
            vector = [int(i in item) for i in range(self.range_label[0], self.range_label[1])]
            df["Labels"][id] = vector
        return df
    

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        img = Image.open(self.img_path+self.dataset["ImageID"][item])
        # RGB is colorful img，L is gray img
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.dataset["Labels"][item]

        if self.transform is not None:
            img1 = self.transform(img)
            img2 = self.transform(img)
        
        text = self.dataset["Caption"][item].lower()

        return img1, img2, text, label
    

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', cache_dir='./models')

def collate_fn(batch):
    # print("collate_fn:\n")
    # print(np.array(batch).shape)
    images1,images2, text, labels = tuple(zip(*batch))
    labels = torch.as_tensor(labels)
    labels = labels.to(torch.float)
    texts = tokenizer(
        text,
        padding="max_length", 
        truncation=True, 
        return_tensors="pt",
        max_length=224
    )
    texts = torch.stack(([texts["input_ids"], texts["token_type_ids"], texts["attention_mask"]]), dim=0)
    batch_size = len(labels)
    #[3,batchsize,length] -> [batchsize,3,length]
    texts = texts.permute(1,0,2)
    texts = texts.repeat(1,1,224)
    texts = texts.reshape(batch_size,3,224,224)
    images1 = torch.stack(images1, dim=0)
    images2 = torch.stack(images2, dim=0)
    # print("texts.shape: ",texts.shape)
    # print("images1.shape: ",images1.shape)
    # print("images2.shape: ",images2.shape)
    x = torch.stack((images1,images2,texts),dim=1)

    return x, labels