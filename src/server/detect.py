from flask import Blueprint, request, session
import os
import time
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer
from flask import jsonify

from src.models.model import SFSC, DFDC, new_design1, new_design2
from src.server.sql import dbsession, Data, History, User

detect = Blueprint('detect', __name__)

class Predict:
    def __init__(self, 
                 num_img, 
                 img_path, 
                 text, 
                 threshold=0.5, 
                #  weight_path="src/models/weights/SFSC/model-best.pth",
                 model="model1"
                #  model,
                 ):
        self.num_img = num_img
        self.img_path = img_path
        self.text = text
        self.threshold = threshold
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if model == 'model1':
            self.model = SFSC(num_classes=90).to(device)
            weight_path = "src/models/weights/SFSC/model-best.pth"
        elif model == 'model2':
            self.model = DFDC(num_classes=2).to(device)
            weight_path = "src/models/weights/DFDC/model-best.pth"
        elif model == 'model3':
            self.model = new_design1(num_classes=2).to(device)
            weight_path = "src/models/weights/new_design1/model-best.pth"
        else:
            self.model = new_design2(num_classes=2).to(device)
            weight_path = "src/models/weights/new_design2/model-best.pth"
        # self.model = SFSC(num_classes=90).to(device)
        # load model weights
        self.model_weight_path = weight_path
        self.transform = transforms.Compose(
            [transforms.Resize(256),
             transforms.CenterCrop(224),
             transforms.ToTensor(),
             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        self.label_map = {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck',
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



    def generateX(self):
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', cache_dir='src/models/models')
        texts = tokenizer(
            self.text,
            padding="max_length", 
            truncation=True, 
            return_tensors="pt",
            max_length=224
        )
        texts = torch.stack(([texts["input_ids"], texts["token_type_ids"], texts["attention_mask"]]), dim=0)
        #[3,batchsize,length] -> [batchsize,3,length]
        texts = texts.permute(1,0,2)
        texts = texts.repeat(1,1,224)
        texts = texts.reshape(1,3,224,224)
        if self.num_img == 1:
            # load image
            img_path = self.img_path[0]
            assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
            img = Image.open(img_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            # [N, C, H, W]
            images1 = self.transform(img)
            images2 = self.transform(img)
            # expand batch dimension
            images1 = torch.unsqueeze(images1, dim=0)
            images2 = torch.unsqueeze(images2, dim=0)
            x = torch.stack((images1,images2,texts),dim=1)
            return x
        elif self.num_img == 2:
            imgg_path1 = self.img_path[0]
            imgg_path2 = self.img_path[1]
            assert os.path.exists(imgg_path1), "file: '{}' dose not exist.".format(imgg_path1)
            assert os.path.exists(imgg_path2), "file: '{}' dose not exist.".format(imgg_path2)
            img1 = Image.open(imgg_path1)
            img2 = Image.open(imgg_path2)
            if img1.mode != 'RGB':
                img1 = img1.convert('RGB')
            if img2.mode != 'RGB':
                img2 = img2.convert('RGB')
            images1 = self.transform(img1)
            images2 = self.transform(img2)
            images1 = torch.unsqueeze(images1, dim=0)
            images2 = torch.unsqueeze(images2, dim=0)
            x = torch.stack((images1,images2,texts),dim=1)
            return x
        else:
            return None
        
    def predict(self):
        x = self.generateX()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = self.model
        model.load_state_dict(torch.load(self.model_weight_path, map_location=device))
        model.eval()
        with torch.no_grad():
            # predict class
            pred = model(x.to(device))
            # print(pred)
            pred_class = torch.where(pred > self.threshold, torch.ones_like(pred), torch.zeros_like(pred))[0]
            # print(pred_class)
            pred_class = pred_class.nonzero()
            # print(pred_class)
            # predict_cla = torch.where(predict_cla == 1)[0]
            # print(predict_cla)
        res_json = {}
        classes = []
        prob = []
        for i in range(len(pred_class)):
            classes.append(self.label_map[pred_class[i].item()+1])
            prob.append(round(pred[0][pred_class[i]].item(), 4))

        
        res_json['classes'] = classes
        res_json['prob'] = prob
        return res_json

###TODO: save img to local

@detect.route('/detect', methods=['POST'])
def getDetect():
    data = request.get_json()
    num_of_img = data['num_of_img']
    img_path = ["./src/resources/"+i for i in data['img_path']]
    text = data['text']
    model = data['model']
    predict = Predict(num_of_img, img_path, text, model=model)
    res = predict.predict()

    # save to database
    if session.get('isLogin'):
        user_eamil = session.get('email')
        user_id = User().find_by_email(user_eamil).user_id
        if num_of_img == 1:
            img1 = img_path[0]
            img2 = img_path[0]
        else:
            img1 = img_path[0]
            img2 = img_path[1]
        if Data().find_data(img1,img2,text) is None:
            data = Data(num_img=num_of_img, img1=img1, img2=img2, text=text)
            dbsession.add(data)
            dbsession.commit()
        data_id = Data().find_data(img1,img2,text).data_id
        history = History(user_id=user_id, data_id=data_id, history_model=model, history_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) , classes=",".join(res['classes']), probability=",".join([str(i) for i in res['prob']]))
        dbsession.add(history)
        dbsession.commit()
    return res

@detect.route('/uploadphoto', methods=['POST'])
def uploadphoto():
    data = request.files['file']
    files = os.listdir('./src/resources/data/')
    exist = [int(i.split(".")[0]) for i in files]
    if len(exist) == 0:
        data.save('./src/resources/data/1.'+data.filename.split(".")[-1])
    else:
        data.save('./src/resources/data/'+str(max(exist)+1)+'.'+data.filename.split(".")[-1])
    return jsonify({'status': 'success', 'path': 'data/'+str(max(exist)+1)+'.'+data.filename.split(".")[-1]})
    


