import os
import argparse
import random
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from matplotlib import pyplot as plt
from my_dataset import MyDataSet
from ronghe import getModel
from utils import read_split_data, create_lr_scheduler, get_params_groups, train_one_epoch, evaluate
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES=True


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    tb_writer = SummaryWriter()

    seed=1
    random.seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.daterministic=True
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    img_size = 224
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    #nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    #print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               #num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             #num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    # build models
    model = getModel().to(device)

    # if args.weights1 != "":
    #     assert os.path.exists(args.weights1), "weights file: '{}' not exist.".format(args.weights1)
    #     weights_dict = torch.load(args.weights1, map_location=device)
    #     # 删除有关分类类别的权重
    #     for k in list(weights_dict.keys()):
    #         if "classifier" in k:
    #             del weights_dict[k]
    #     print(model.load_state_dict(weights_dict, strict=False))

    if torch.cuda.is_available() :
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = model.cuda()
        model = torch.nn.DataParallel(model)


    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head外，其他权重全部冻结
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))



    # pg = [p for p in model.parameters() if p.requires_grad]
    pg1 = get_params_groups(model, weight_decay=args.wd1)
    optimizer1 = optim.AdamW(pg1, lr=args.lr1, weight_decay=args.wd1)
    lr_scheduler1 = create_lr_scheduler(optimizer1, len(train_loader), args.epochs,
                                       warmup=True, warmup_epochs=1)
    Epoch=[]
    Train_loss1=[]
    Train_acc1=[]
    Val_loss1=[]
    Val_acc1=[]
    best_acc1 = 0.
    for epoch in range(args.epochs):
        # train
        train_loss1, train_acc1 = train_one_epoch(model=model,
                                                optimizer=optimizer1,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch,
                                                lr_scheduler=lr_scheduler1)

        # validate
        val_loss1, val_acc1 = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)
        print("*********************************")
        tags = ["train_loss1", "train_acc1", "val_loss1", "val_acc1", "learning_rate1"]
        tb_writer.add_scalar(tags[0], train_loss1, epoch)
        tb_writer.add_scalar(tags[1], train_acc1, epoch)
        tb_writer.add_scalar(tags[2], val_loss1, epoch)
        tb_writer.add_scalar(tags[3], val_acc1, epoch)
        tb_writer.add_scalar(tags[4], optimizer1.param_groups[0]["lr"], epoch)

        Epoch.append(epoch)
        Train_loss1.append(train_loss1)
        Train_acc1.append(train_acc1)
        Val_loss1.append(val_loss1)
        Val_acc1.append(val_acc1)
        if best_acc1 < val_acc1:
            torch.save(model.state_dict(), "rongheweights/")
            best_acc1 = val_acc1

    results=pd.DataFrame({'epoch':Epoch,'train_loss':Train_loss1, 'val_loss':Val_loss1,
                                        'train_acc': Train_acc1,  'val_acc': Val_acc1})
    results.to_csv(r'D:/PythonProject/The-one/result_csv/dataset_covid-ct/ronghe.csv',index=None,encoding='utf8')

    print("The Best Result :")
    print("best acc1:{}".format(best_acc1))
    plt.figure(1)
    plt.title('ronghe-Loss Function Curve')  # 图片标题
    plt.xlabel('Epoch')  # x轴变量名称
    plt.ylabel('Loss/Acc')  # y轴变量名称
    plt.plot(Epoch, Train_loss1, label="$TLoss$")  # 逐点画出trian_loss_results值并连线，连线图标是Loss
    plt.plot(Epoch, Train_acc1, label="$TAcc$")
    plt.plot(Epoch, Val_loss1, label="$VLoss$")
    plt.plot(Epoch, Val_acc1, label="$VAcc$")
    plt.legend()  # 画出曲线图标
    plt.show()  # 画出图像'''
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr1', type=float, default=5e-4)#0.001
    parser.add_argument('--wd1', type=float, default=5e-2)#0.1


    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default="./Datasets/COVID-CT/")

    # 预训练权重路径，如果不想载入就设置为空字符
    # 链接: https://pan.baidu.com/s/1aNqQW4n_RrUlWUBNlaJRHA  密码: i83t
    parser.add_argument('--weights1', type=str, default='',
                        help='initial weights path')
    # 是否冻结head以外所有权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
