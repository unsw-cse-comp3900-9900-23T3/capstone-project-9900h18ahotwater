from src.models.vit_model import VisionTransformer
from src.models.bert import Bert
from torch import nn
import torch
import torchvision.models as models

class new_design1(nn.Module):
    def __init__(self,
                 out_dim=768,
                 num_classes=2):
        super(new_design1, self).__init__()
        self.vit = VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, out_dim=768)
        self.resnet = models.resnet50(pretrained=True)  # 使用ResNet50，也可以选择其他版本
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        self.bert = Bert(out_dim=768)
        self.resnet_dim = nn.Linear(2048, 768)  # 将BERT的输出维度从768转换为2048
        self.linear = nn.Linear(768, out_dim)
        self.classifier = nn.Linear(out_dim, num_classes)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = x[:, 0, :, :, :]
        x2 = x[:, 1, :, :, :]
        x3 = x[:, 2, :, :, :]
        x3 = x3[:, :, 0]

        vit_out = self.vit(x1)
        resnet_out = self.resnet(x2)  # 使用ResNet替代第二个ViT
        resnet_out = torch.flatten(resnet_out, 1)
        resnet_out = self.resnet_dim(resnet_out)

        bert_out = self.bert(x3)
         # 转换BERT的输出维度

        out = vit_out + resnet_out + bert_out
        out = self.linear(out)  # 使用self.linear处理合并后的输出
        out = self.relu(out)
        #print(out.shape, resnet_out.shape, bert_out.shape)
        out = (out + resnet_out + bert_out) / 3
        out = self.classifier(out)
        out = self.sigmoid(out)

        return out


class new_design2(nn.Module):
    def __init__(self,
                 out_dim=768,
                 num_classes=2):
        super(new_design2, self).__init__()
        self.vit = VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, out_dim=768)
        self.mobilenet = models.mobilenet_v2(pretrained=True).features
        #self.mobilenet = nn.Sequential(*list(self.mobilenet.children())[:-1])
        self.bert = Bert(out_dim=768)
        self.mobilenet_dim = nn.Linear(1280, 768)  # 将BERT的输出维度从768转换为2048
        self.linear = nn.Linear(768, out_dim)
        self.classifier = nn.Linear(out_dim, num_classes)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))


    def forward(self, x):
        x1 = x[:, 0, :, :, :]
        x2 = x[:, 1, :, :, :]
        x3 = x[:, 2, :, :, :]
        x3 = x3[:, :, 0]

        vit_out = self.vit(x1)
        mobilenet_out = self.mobilenet(x2)  # 使用ResNet替代第二个ViT
        mobilenet_out = self.avgpool(mobilenet_out)
        mobilenet_out = torch.flatten(mobilenet_out, 1)
        mobilenet_out = self.mobilenet_dim(mobilenet_out)

        bert_out = self.bert(x3)
         # 转换BERT的输出维度

        out = vit_out + mobilenet_out + bert_out
        out = self.linear(out)  # 使用self.linear处理合并后的输出
        out = self.relu(out)
        #print(out.shape, resnet_out.shape, bert_out.shape)
        out = (out + mobilenet_out + bert_out) / 3
        out = self.classifier(out)
        out = self.sigmoid(out)

        return out


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
