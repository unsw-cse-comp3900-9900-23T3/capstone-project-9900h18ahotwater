from vit_model import VisionTransformer
from bert import Bert
from torch import nn
import torch


class SFSC(nn.Module):
    #simple fusion with simple classifier
    #input: 2 images and 1 sentence
    #
    def __init__(self,
                 out_dim=768,
                 num_classes=2):
        super(SFSC, self).__init__()
        self.vit = VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, out_dim=768)
        self.bert = Bert(out_dim=768)
        self.linear = nn.Linear(768, out_dim)
        self.classfier = nn.Linear(out_dim, num_classes)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        #x -> [b,3,3,224,224] second 3 is for 3 views
        x1 = x[:,0,:,:,:]
        x2 = x[:,1,:,:,:]
        x3 = x[:,2,:,:,:]
        x3 = x3[:,:,0]
        # x1 -> [b, 3, 224, 224]
        # x2 -> [b, 3, 224, 224]
        # x3 -> [b, 3, sentence_len]
        vit_out = self.vit(x1)
        vit_out2 = self.vit(x2)
        bert_out = self.bert(x3)
        out = vit_out + bert_out + vit_out2
        out = self.linear(out)
        out = self.relu(out)
        # out,vit_out2,bert_out mean pooling
        out = (out+vit_out2+bert_out)/3
        out = self.classfier(out)
        out = self.sigmoid(out)
        #[b,768] -> [b,1,768]
        #[b,3,768]
        #out = torch.cat((out.unsqueeze(1), bert_out.unsqueeze(1), vit_out2.unsqueeze(1)), dim=1)
        return out


class DFDC(nn.Module):
    #paper structure
    def __init__(self,
                 out_dim=768,
                 num_classes=2):
        super(DFDC, self).__init__()
        self.vit = VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, out_dim=768)
        self.bert = Bert(out_dim=768)
        self.fc1 = nn.Linear(768*2, out_dim)
        self.fc2 = nn.Linear(768, out_dim)
        self.classfier = nn.Linear(out_dim, num_classes)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self,x):
        #x -> [b,3,3,224,224] second 3 is for 3 views
        x1 = x[:,0,:,:,:]
        x2 = x[:,1,:,:,:]
        x3 = x[:,2,:,:,:]
        x3 = x3[:,:,:,0]
        # x1 -> [b, 3, 224, 224]
        # x2 -> [b, 3, 224, 224]
        # x3 -> [b, 3, sentence_len]
        x1 = self.vit(x1)
        x2 = self.vit(x2)
        x3 = self.bert(x3)
        f1 = torch.cat((x1, x2), dim=1)
        f2 = torch.cat((x2, x3), dim=1)
        f1 = self.fc1(f1)
        f1 = self.relu(f1)
        f2 = self.fc1(f2)
        f2 = self.relu(f2)
        x = self.fc2(x1)
        x = self.relu(x)
        x = f1+f2+x
        x = (x+x2+x3)/3
        x = self.classfier(x)
        x = self.sigmoid(x)
        return x
