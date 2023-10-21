import os
import math
import argparse

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from model import SFSC, DFDC
from dataset import MyDataSet, collate_fn
from utils import train_one_epoch, evaluate



def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./dataset") is False:
        os.makedirs("./dataset")

    if os.path.exists("./models") is False:
        os.makedirs("./models")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")
    
    if os.path.exists("./weights/{}".format(args.model_name)) is False:
            os.makedirs("./weights/{}".format(args.model_name))

    tb_writer = SummaryWriter()
    train_path = "./dataset/" + args.data_path

    img_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}
    
    train_dataset = MyDataSet(train_path+"/train.csv",train_path+"/data/",range_label=args.range_label,transform=img_transform["train"])
    val_dataset = MyDataSet(train_path+"/val.csv",train_path+"/data/",range_label=args.range_label,transform=img_transform["val"])

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

    model = SFSC(num_classes=args.num_classes).to(device)
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

        tags = ["train_loss", "train_precision", "train_recall", "train_f1", "val_loss", "val_precision", "val_recall", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_precision, epoch)
        tb_writer.add_scalar(tags[2], train_recall, epoch)
        tb_writer.add_scalar(tags[3], train_f1, epoch)
        tb_writer.add_scalar(tags[4], val_loss, epoch)
        tb_writer.add_scalar(tags[5], val_precision, epoch)
        tb_writer.add_scalar(tags[6], val_recall, epoch)
        tb_writer.add_scalar(tags[7], optimizer.param_groups[0]["lr"], epoch)
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            #save model

        torch.save(model.state_dict(), "./weights/{}/model-{}.pth".format(args.model_name,epoch))

        #every 10 epoch save best model and delete from epoch-20 to epoch-10 models
        if epoch%10 == 0 and epoch != 0:
            if best_epoch>=epoch-20:
                try:
                    os.system("cp -f ./weights/{}/model-{}.pth ./weights/{}/model-best.pth".format(args.model_name,best_epoch,args.model_name))
                except:
                    pass
            if epoch > 10:
                for i in range(epoch-20,epoch-9):
                    try:
                        os.system("rm ./weights/{}/model-{}.pth".format(args.model_name,i))
                    except:
                        pass


    print("best f1: {}, best epoch: {}".format(best_f1, best_epoch))
    tb_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=19)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)

    # path of dataset,it should be under the path of the dataset, i.e. rather FILE than /dataset/FILES
    parser.add_argument('--data_path', type=str,
                        default="dataset1")
    #range of label from the least to the most+1, for example (1,20) means label from 1 to 19
    parser.add_argument('--range_label', type=tuple,default=(1,20),help='range of label')
    parser.add_argument('--model_name', default='model1', help='create model name')

    # path of pre-trained model，if not then make it to blank i.e. "",  default='./vit_base_patch16_224_in21k.pth'
    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    # frozen layers or not
    parser.add_argument('--freeze_layers', type=bool, default=True)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)