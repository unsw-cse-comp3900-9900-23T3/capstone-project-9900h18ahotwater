import os
import math
import argparse

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from src.models.model import SFSC, DFDC, new_design1, new_design2
from src.models.dataset import COCODataSet, collate_fn
from src.models.utils import train_one_epoch, evaluate
import pandas as pd



def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("src/models/dataset") is False:
        os.makedirs("src/models/dataset")

    if os.path.exists("src/models/models") is False:
        os.makedirs("src/models/models")

    if os.path.exists("src/models/weights") is False:
        os.makedirs("src/models/weights")
    
    if os.path.exists("src/models/weights/{}".format(args.model_name)) is False:
            os.makedirs("src/models/weights/{}".format(args.model_name))

    tb_writer = SummaryWriter()
    train_path = "src/models/dataset/" + args.data_path

    img_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}
    
    # train_dataset = MyDataSet(train_path+"/train.csv",train_path+"/data/",range_label=args.range_label,transform=img_transform["train"])
    # val_dataset = MyDataSet(train_path+"/val.csv",train_path+"/data/",range_label=args.range_label,transform=img_transform["val"])

    train_dataset = COCODataSet(path=train_path, range_label=args.range_label, train=True, transform=img_transform["train"])
    val_dataset = COCODataSet(path=train_path, range_label=args.range_label, train=False, transform=img_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=collate_fn)
    
    if args.model == "SFSC":
        model = SFSC(num_classes=args.num_classes).to(device)
    elif args.model == "DFDC":
        model = DFDC(num_classes=args.num_classes).to(device)
    elif args.model == "new_design1":
        model = new_design1(num_classes=args.num_classes).to(device)
    elif args.model == "new_design2":
        model = new_design2(num_classes=args.num_classes).to(device)
    else:
        raise ValueError("model name not found, you can choose SFSC or DFDC")
    # model = DFDC(num_classes=args.num_classes).to(device)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        # delete weights that doen't need
        # del_keys = ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
        # for k in del_keys:
        #     del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    # if args.freeze_layers:
    #     for name, para in model.named_parameters():
    #         # 除head, pre_logits外，其他权重全部冻结
    #         if "head" not in name and "pre_logits" not in name:
    #             para.requires_grad_(False)
    #         else:
    #             print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    best_f1 = 0.0
    best_epoch = 0
    tags = ["epoch","train_loss", "train_precision", "train_recall", "train_f1", "val_loss", "val_precision", "val_recall",'val_f1' , "learning_rate"]
    if os.path.exists("src/models/weights/{}/log.csv".format(args.model_name)) is False:
        pd.DataFrame(columns=tags).to_csv("src/models/weights/{}/log.csv".format(args.model_name), index=False)
    for epoch in range(args.epochs):
        # train
        train_loss, train_precision, train_recall, train_f1 = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)

        scheduler.step()

        # validate
        val_loss, val_precision, val_recall, val_f1 = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)

        
        tb_writer.add_scalar(tags[1], train_loss, epoch)
        tb_writer.add_scalar(tags[2], train_precision, epoch)
        tb_writer.add_scalar(tags[3], train_recall, epoch)
        tb_writer.add_scalar(tags[4], train_f1, epoch)
        tb_writer.add_scalar(tags[5], val_loss, epoch)
        tb_writer.add_scalar(tags[6], val_precision, epoch)
        tb_writer.add_scalar(tags[7], val_recall, epoch)
        tb_writer.add_scalar(tags[8], val_f1, epoch)
        tb_writer.add_scalar(tags[8], optimizer.param_groups[0]["lr"], epoch)
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            #save model

        torch.save(model.state_dict(), "src/models/weights/{}/model-{}.pth".format(args.model_name,epoch))
        pd.DataFrame([[epoch,train_loss, train_precision, train_recall, train_f1, val_loss, val_precision, val_recall, val_f1, optimizer.param_groups[0]["lr"]]], columns=tags).to_csv("src/models/weights/{}/log.csv".format(args.model_name), mode='a', header=False, index=False)

        #every 10 epoch save best model and delete from epoch-20 to epoch-10 models
        if epoch%10 == 9:
            if best_epoch>=epoch-19:
                try:
                    os.system("cp -f src/models/weights/{}/model-{}.pth src/models/weights/{}/model-best.pth".format(args.model_name,best_epoch,args.model_name))
                except:
                    pass
            if epoch > 10:
                for i in range(epoch-19,epoch-9):
                    try:
                        os.system("rm src/models/weights/{}/model-{}.pth".format(args.model_name,i))
                    except:
                        pass


    print("best f1: {}, best epoch: {}".format(best_f1, best_epoch))
    pd.DataFrame([[best_f1, best_epoch]], columns=["best_f1", "best_epoch"]).to_csv("src/models/weights/{}/best.csv".format(args.model_name), index=False)
    tb_writer.close()


if __name__ == '__main__':
    #python train.py --data_path coco --epochs 100 --model_name coco1 --batch_size 16
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=90)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)
    # model type: SFSC or DFDC
    parser.add_argument('--model', type=str, default="SFSC")

    # path of dataset,it should be under the path of the dataset, i.e. rather FILE than /dataset/FILES
    parser.add_argument('--data_path', type=str,
                        default="coco")
    #range of label from the least to the most+1, for example (1,91) means label from 1 to 90
    parser.add_argument('--range_label', type=tuple,default=(1,91),help='range of label')
    parser.add_argument('--model_name', default='model1', help='create model name')

    # path of pre-trained model，if not then make it to blank i.e. "",  default='./vit_base_patch16_224_in21k.pth'
    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    # frozen layers or not
    parser.add_argument('--freeze_layers', type=bool, default=True)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)