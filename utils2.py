import os
import sys
import json
import pickle
import random
import math
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from metrics import compute_confusion_matrix,compute_indexes,AUC

class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val=0
        self.avg=0
        self.sum=0
        self.count=0
    def update(self,val,n=1):
        self.val=val
        self.sum+=val*n
        self.count+=n
        self.avg=self.sum/self.count

def accuracy(output,target):
    batchsize=target.size(0)
    preds=output.argmax(dim=1)
    ##correct_k=torch.eq(preds, target.cuda()).sum()
    correct_k=torch.sum(preds.cpu().view(-1)==target.view(-1)).item()
    correct_k=correct_k/batchsize
    return correct_k

def f1(output, target):
    preds = output.argmax(dim=1)
    return f1_score(target.data.cpu().numpy(), preds.data.cpu().numpy(),average='macro')

def precision(output, target):
    preds = output.argmax(dim=1)
    return precision_score(target.data.cpu().numpy(), preds.data.cpu().numpy())

def recall(output, target):
    preds = output.argmax(dim=1)
    return recall_score(target.data.cpu().numpy(), preds.data.cpu().numpy())

def auc(output, target):
    preds = output.argmax(dim=1)
    return roc_auc_score(target.data.cpu().numpy(), preds.data.cpu().numpy())


def read_split_data(root: str, val_rate: float = 0.3):
    random.seed(10)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "not find data for train."
    assert len(val_images_path) > 0, "not find data for eval"

    plot_image = False
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(flower_class)), flower_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list

def kd_loss(logits_student, logits_teacher, temperature):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd

def feature_loss_function(fea,target_fea):
    loss = (fea -target_fea)**2*((fea>0)|(target_fea>0)).float()
    return torch.abs(loss).sum()

def train_one_epoch(model1,model2,optimizer1,optimizer2, data_loader, device, epoch,lr_scheduler1,lr_scheduler2):
    model1.train()
    model2.train()


    # batch_time=AverageMeter()
    # f1s1=AverageMeter()
    # acy1=AverageMeter()
    # precisions1=AverageMeter()
    # recalls1=AverageMeter()
    # aucs1=AverageMeter()
    # f1s2=AverageMeter()
    # precisions2=AverageMeter()
    # recalls2=AverageMeter()
    # aucs2=AverageMeter()


    loss_ce = torch.nn.CrossEntropyLoss()
    loss_kl = nn.KLDivLoss(reduction='batchmean')


    accu_loss1 = torch.zeros(1).to(device)  # 累计损失
    accu_loss2 = torch.zeros(1).to(device)  # 累计损失
    accu_num1 = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_num2 = torch.zeros(1).to(device)  # 累计预测正确的样本数

    accu_tp1 = torch.zeros(1).to(device)
    accu_fp1 = torch.zeros(1).to(device)
    accu_tn1 = torch.zeros(1).to(device)
    accu_fn1 = torch.zeros(1).to(device)
    accu_tp2 = torch.zeros(1).to(device)
    accu_fp2 = torch.zeros(1).to(device)
    accu_tn2 = torch.zeros(1).to(device)
    accu_fn2 = torch.zeros(1).to(device)
    #accu_aucs1 = torch.zeros(1).to(device)
    #accu_aucs2 = torch.zeros(1).to(device)

    optimizer1.zero_grad()
    optimizer2.zero_grad()
    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred1,fc = model1(images.to(device))
        pred2,ft = model2(images.to(device))
        pred_classes1 = torch.max(pred1, dim=1)[1]
        pred_classes2 = torch.max(pred2, dim=1)[1]
        accu_num1 += torch.eq(pred_classes1, labels.to(device)).sum()
        accu_num2 += torch.eq(pred_classes2, labels.to(device)).sum()

        ce_loss1 = loss_ce(pred1, labels.to(device))
        ce_loss2 = loss_ce(pred2, labels.to(device))
        kl_loss1 = loss_kl(F.log_softmax(pred1, dim=1), F.softmax(pred2, dim=1))
        kl_loss2 = loss_kl(F.log_softmax(pred2, dim=1), F.softmax(pred1, dim=1))
        fl_loss= feature_loss_function(fc[-1],ft[3])
        #kd_loss1 = kd_loss(pred1, pred2, 4)
        #kd_loss2 = kd_loss(pred2, pred1, 4)
        #loss = ce_loss + kl_loss / (self.model_num - 1)
        #loss1 = ce_loss1
        loss1 = ce_loss1+0.000001*fl_loss
        loss2 = ce_loss2
        #loss1=ce_loss1+0.5*kl_loss1
        #loss2=ce_loss2+0.5*kl_loss2
        #loss1=ce_loss1
        #loss2=ce_loss2
        loss1.backward(retain_graph=True)
        loss2.backward()


        accu_loss1 += loss1.detach()
        accu_loss2 += loss2.detach()

        tp1,fp1,tn1,fn1 = compute_confusion_matrix(pred_classes1,labels)
        tp2,fp2,tn2,fn2 = compute_confusion_matrix(pred_classes2,labels)
        accu_tp1 += tp1
        accu_fp1 += fp1
        accu_tn1 += tn1
        accu_fn1 += fn1
        accu_tp2 += tp2
        accu_fp2 += fp2
        accu_tn2 += tn2
        accu_fn2 += fn2
        acy1, precisions1, recalls1, f1s1 = compute_indexes(accu_tp1.item(), accu_fp1.item(), accu_tn1.item(),
                                                            accu_fn1.item())
        acy2, precisions2, recalls2, f1s2 = compute_indexes(accu_tp2.item(), accu_fp2.item(), accu_tn2.item(),
                                                            accu_fn2.item())

        #accu_aucs1 += AUC(pred_classes1,labels)
        #accu_aucs2 += AUC(pred_classes2, labels)
        # #jisuan SCORE
        # acy_score=accuracy(pred1,labels)
        # #f1_score1=f1(pred1,labels)
        # f1_score1 = f1_score(labels,pred_classes1.cpu(),average='binary')
        # precision_score1=precision(pred1,labels)
        # #recall_score1=recall(pred1,labels)
        # recall_score1 = recall_score(labels, pred_classes1.cpu())
        # auc_score1=0
        # try:
        #     auc_score1=auc(pred1,labels)
        # except ValueError:
        #     pass
        #
        # f1_score2 = f1(pred2, labels)
        # precision_score2 = precision(pred2, labels)
        # recall_score2 = recall(pred2, labels)
        # auc_score2 = 0
        # try:
        #     auc_score2 = auc(pred2, labels)
        # except ValueError:
        #     pass
        #
        # acy1.update(acy_score,images.size()[0])
        # f1s1.update(f1_score1,images.size()[0])
        # precisions1.update(precision_score1,images.size()[0])
        # recalls1.update(recall_score1,images.size()[0])
        # aucs1.update(auc_score1,images.size()[0])
        # f1s2.update(f1_score2, images.size()[0])
        # precisions2.update(precision_score2, images.size()[0])
        # recalls2.update(recall_score2, images.size()[0])
        # aucs2.update(auc_score2, images.size()[0])

        data_loader.desc = "[model_train epoch {}] loss1: {:.3f}, acc1: {:.3f}, " \
                           "lr: {:.5f},         loss2: {:.3f}, acc2: {:.3f}".format(
                    epoch+1,
                    accu_loss1.item() / (step + 1),
                    accu_num1.item() / sample_num,
                    optimizer1.param_groups[0]["lr"],
                    accu_loss2.item() / (step + 1),
                    accu_num2.item() / sample_num,

            )



        if not torch.isfinite(loss1):
            print('WARNING: non-finite loss, ending training ', loss1)
            sys.exit(1)

        if not torch.isfinite(loss2):
            print('WARNING: non-finite loss, ending training ', loss2)
            sys.exit(1)

        optimizer1.step()
        optimizer2.step()
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        lr_scheduler1.step()
        lr_scheduler2.step()


    return accu_loss1.item() / (step + 1), accu_num1.item() / sample_num, acy1,f1s1,precisions1,recalls1,\
           accu_loss2.item() / (step + 1), accu_num2.item() / sample_num, f1s2,precisions2,recalls2


@torch.no_grad()
def evaluate(model1,model2, data_loader, device, epoch):
    loss_ce = torch.nn.CrossEntropyLoss()
    loss_kl = nn.KLDivLoss(reduction='batchmean')

    model1.eval()
    model2.eval()

    # #batch_time = AverageMeter()
    # acy1=AverageMeter()
    # f1s1 = AverageMeter()
    # precisions1 = AverageMeter()
    # recalls1 = AverageMeter()
    # aucs1 = AverageMeter()
    # f1s2 = AverageMeter()
    # precisions2 = AverageMeter()
    # recalls2 = AverageMeter()
    # aucs2 = AverageMeter()


    accu_loss1 = torch.zeros(1).to(device)  # 累计损失
    accu_loss2 = torch.zeros(1).to(device)  # 累计损失
    accu_num1 = torch.zeros(1).to(device)  # 累计预测正确的样本数
    accu_num2 = torch.zeros(1).to(device)  # 累计预测正确的样本数

    accu_tp1 = torch.zeros(1).to(device)
    accu_fp1 = torch.zeros(1).to(device)
    accu_tn1 = torch.zeros(1).to(device)
    accu_fn1 = torch.zeros(1).to(device)
    accu_tp2 = torch.zeros(1).to(device)
    accu_fp2 = torch.zeros(1).to(device)
    accu_tn2 = torch.zeros(1).to(device)
    accu_fn2 = torch.zeros(1).to(device)

    #accu_aucs1 = torch.zeros(1).to(device)
    #accu_aucs2 = torch.zeros(1).to(device)

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred1,fc = model1(images.to(device))
        pred2,ft = model2(images.to(device))
        pred_classes1 = torch.max(pred1, dim=1)[1]
        pred_classes2 = torch.max(pred2, dim=1)[1]
        accu_num1 += torch.eq(pred_classes1, labels.to(device)).sum()
        accu_num2 += torch.eq(pred_classes2, labels.to(device)).sum()

        ce_loss1 = loss_ce(pred1, labels.to(device))
        ce_loss2 = loss_ce(pred2, labels.to(device))
        kl_loss1 = loss_kl(F.log_softmax(pred1, dim=1), F.softmax(pred2, dim=1))
        kl_loss2 = loss_kl(F.log_softmax(pred2, dim=1), F.softmax(pred1, dim=1))
        fl_loss = feature_loss_function(fc[-1],ft[3])
        #kd_loss1 = kd_loss(pred1, pred2, 4)
        #kd_loss2 = kd_loss(pred2, pred1, 4)
        # loss = ce_loss + kl_loss / (self.model_num - 1)
        #loss1 = ce_loss1
        loss1 = ce_loss1+0.000001*fl_loss
        loss2 = ce_loss2
        #loss1=ce_loss1+0.5*kl_loss1
        #loss2=ce_loss2+0.5*kl_loss2
        #loss1 = ce_loss1
        #loss2 = ce_loss2
        accu_loss1 += loss1.detach()
        accu_loss2 += loss2.detach()

        tp1, fp1, tn1, fn1 = compute_confusion_matrix(pred_classes1, labels)
        tp2, fp2, tn2, fn2 = compute_confusion_matrix(pred_classes2, labels)
        accu_tp1 += tp1
        accu_fp1 += fp1
        accu_tn1 += tn1
        accu_fn1 += fn1
        accu_tp2 += tp2
        accu_fp2 += fp2
        accu_tn2 += tn2
        accu_fn2 += fn2
        acy1, precisions1, recalls1, f1s1 = compute_indexes(accu_tp1.item(), accu_fp1.item(), accu_tn1.item(), accu_fn1.item())
        acy2, precisions2, recalls2, f1s2 = compute_indexes(accu_tp2.item(), accu_fp2.item(), accu_tn2.item(), accu_fn2.item())

        #accu_aucs1 += AUC(pred_classes1, labels)
        #accu_aucs2 += AUC(pred_classes2, labels)
        # # jisuan SCORE
        # acy_score=accuracy(pred1, labels)
        # #f1_score1 = f1(pred1, labels)
        # f1_score1 = f1_score(labels, pred_classes1.cpu(), average='binary')
        # precision_score1 = precision(pred1, labels)
        # #recall_score1 = recall(pred1, labels)
        # recall_score1 = recall_score(labels, pred_classes1.cpu())
        # auc_score1 = 0
        # try:
        #     auc_score1 = auc(pred1, labels)
        # except ValueError:
        #     pass
        #
        # f1_score2 = f1(pred2, labels)
        # precision_score2 = precision(pred2, labels)
        # recall_score2 = recall(pred2, labels)
        # auc_score2 = 0
        # try:
        #     auc_score2 = auc(pred2, labels)
        # except ValueError:
        #     pass
        #
        # acy1.update(acy_score,images.size()[0])
        # f1s1.update(f1_score1, images.size()[0])
        # precisions1.update(precision_score1, images.size()[0])
        # recalls1.update(recall_score1, images.size()[0])
        # aucs1.update(auc_score1, images.size()[0])
        # f1s2.update(f1_score2, images.size()[0])
        # precisions2.update(precision_score2, images.size()[0])
        # recalls2.update(recall_score2, images.size()[0])
        # aucs2.update(auc_score2, images.size()[0])

        data_loader.desc = "[model_valid epoch {}] loss1: {:.3f}, acc1: {:.3f}" \
                           "                       loss2: {:.3f}, acc2: {:.3f}".format(
                    epoch+1,
                    accu_loss1.item() / (step + 1),
                    accu_num1.item() / sample_num,
                    accu_loss2.item() / (step + 1),
                    accu_num2.item() / sample_num
                )



    # return accu_loss1.item() / (step + 1), accu_num1.item() / sample_num,\
    #        accu_loss2.item() / (step + 1), accu_num2.item() / sample_num

    return accu_loss1.item() / (step + 1), accu_num1.item() / sample_num, acy1,f1s1,precisions1,recalls1,\
           accu_loss2.item() / (step + 1), accu_num2.item() / sample_num, f1s2,precisions2,recalls2

def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-6):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            # warmup后lr倍率因子从1 -> end_factor
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def get_params_groups(model: torch.nn.Module, weight_decay: float = 1e-5):
    # 记录optimize要训练的权重参数
    parameter_group_vars = {"decay": {"params": [], "weight_decay": weight_decay},
                            "no_decay": {"params": [], "weight_decay": 0.}}

    # 记录对应的权重名称
    parameter_group_names = {"decay": {"params": [], "weight_decay": weight_decay},
                             "no_decay": {"params": [], "weight_decay": 0.}}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        if len(param.shape) == 1 or name.endswith(".bias"):
            group_name = "no_decay"
        else:
            group_name = "decay"

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)

    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())
