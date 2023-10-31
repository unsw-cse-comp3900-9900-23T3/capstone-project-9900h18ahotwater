import os
import json

import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer
import matplotlib.pyplot as plt

from model import SFSC, DFDC


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    threshold = 0.5
    lable_map = {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck',
    9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
    16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra',
    25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
    35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
    41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup',
    48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
    56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch',
    64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
    75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink',
    82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier',
    90: 'toothbrush'}

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', cache_dir='./models')
    # text = "There is a person in the picture. She is a woman."
    # text = "there are some people in the picture. they all have some cups in their hands."
    # text = "There are some motrocycles in the picture. bikes are also in the picture."
    text = "there is two cars in the picture. One is a car and the other is a truck. the car is a taxi."
    texts = tokenizer(
        text,
        padding="max_length", 
        truncation=True, 
        return_tensors="pt",
        max_length=224
    )
    texts = torch.stack(([texts["input_ids"], texts["token_type_ids"], texts["attention_mask"]]), dim=0)
    batch_size = 1
    #[3,batchsize,length] -> [batchsize,3,length]
    texts = texts.permute(1,0,2)
    texts = texts.repeat(1,1,224)
    texts = texts.reshape(batch_size,3,224,224)
    

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    # load image
    img_path = "test/3.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    if img.mode != 'RGB':
        # raise ValueError("image: {} isn't RGB mode.".format(self.dataset["ImagePath"][item]))
        img = img.convert('RGB')
    plt.imshow(img)
    # plt.show()
    # [N, C, H, W]
    images1 = data_transform(img)
    images2 = data_transform(img)
    # plt.imshow(images1.permute(1, 2, 0))
    # plt.show()
    # img = data_transform(img)
    # expand batch dimension
    images1 = torch.unsqueeze(images1, dim=0)
    images2 = torch.unsqueeze(images2, dim=0)
    # images1 = torch.stack(images1, dim=0)
    # images2 = torch.stack(images2, dim=0)
    # print("texts.shape: ",texts.shape)
    # print("images1.shape: ",images1.shape)
    # print("images2.shape: ",images2.shape)
    # print("img.shape: ",img.shape)
    x = torch.stack((images1,images2,texts),dim=1)
    # print("x.shape: ",x.shape)

    # read class_indict
    # json_path = './class_indices.json'
    # assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    # with open(json_path, "r") as f:
    #     class_indict = json.load(f)

    # create model
    model = SFSC(num_classes=90).to(device)
    # load model weights
    model_weight_path = "./weights/SFSC/model-best.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    with torch.no_grad():
        # predict class
        pred = model(x.to(device))
        # print(pred)
        pred_class = torch.where(pred > threshold, torch.ones_like(pred), torch.zeros_like(pred))[0]
        # print(pred_class)
        pred_class = pred_class.nonzero()
        # print(pred_class)
        # predict_cla = torch.where(predict_cla == 1)[0]
        # print(predict_cla)
    res = ""
    for i in range(len(pred_class)):
        res += "class: {:10}   prob: {:.3} \n".format(lable_map[pred_class[i].item()+1],
                                                  pred[0][pred_class[i]].item())
    plt.title(res)
    print(res)
    plt.show()


if __name__ == '__main__':
    main()