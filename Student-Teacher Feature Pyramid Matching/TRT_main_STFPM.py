import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.optim
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from evaluate import evaluate
from sklearn.model_selection import train_test_split
from glob import glob
from PIL import Image
# from torch.utils.data import ConcatDataset
import matplotlib.pyplot as plt
import copy
import tensorrt as trt
from torch2trt import torch2trt

device = torch.device("cuda:0")
print(f'Device used: {device}')

class MVTecDataset(object):
    def __init__(self, image_list, transform=None):
        self.image_list = image_list
        self.transform = transform
        if self.transform is None:
            self.transform = transforms.ToTensor()
        self.dataset = self.load_dataset()

    def load_dataset(self):
        return [Image.open(p).convert('RGB') for p in self.image_list]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.transform(self.dataset[idx])
        return self.image_list[idx], image


class ResNet18_MS3(nn.Module):

    def __init__(self, pretrained=False):
        super(ResNet18_MS3, self).__init__()     
        net = models.resnet18(pretrained=pretrained)
        # ignore the last block and fc
        self.model = torch.nn.Sequential(*(list(net.children())[:-2]))

    def forward(self, x):
        res = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in ['4', '5', '6']:
                res.append(x)
        return res



def load_gt(root, cls):
    gt = []
    gt_dir = os.path.join(root, cls, 'ground_truth')
    # sub_dirs = sorted(os.listdir(gt_dir))
    # for sb in sub_dirs:
    #     for fname in sorted(os.listdir(os.path.join(gt_dir, sb))):
    #         temp = cv2.imread(os.path.join(gt_dir, sb, fname), cv2.IMREAD_GRAYSCALE)
    #         temp = cv2.resize(temp, (256, 256)).astype(np.bool)[None, ...]
    #         gt.append(temp)
    # sorted(glob(os.path.join(args.mvtec_ad, category, 'img', 'test_good_*.png')))
    for fname in sorted(glob(os.path.join(gt_dir, 'ground_truth_*.png'))):
        temp = cv2.imread(os.path.join(fname), cv2.IMREAD_GRAYSCALE)
        # print('GT',temp)
        temp = cv2.resize(temp, (256, 256)).astype(np.bool_)[None, ...]
        gt.append(temp)
    gt = np.concatenate(gt, 0)
    return gt


def main():
    parser = argparse.ArgumentParser(description="Anomaly Detection")
    parser.add_argument("split", nargs="?", choices=["train", "test"])
    # required training super-parameters
    parser.add_argument("--checkpoint", type=str, default='snapshots_class', help="student checkpoint")
    parser.add_argument("--category", type=str, nargs = '+', default='', help="category name for MvTec AD dataset")
    parser.add_argument("--epochs", type=int, default=100, help='number of epochs')

    parser.add_argument("--checkpoint-epoch", type=int, default=100, help="checkpoint resumed for testing (1-based)")
    parser.add_argument("--batch-size", type=int, default=1, help='batch size')
    # trivial parameters
    parser.add_argument("--result-path", type=str, default='results', help="save results")
    parser.add_argument("--save-fig", action='store_true', help="save images with anomaly score")
    parser.add_argument("--mvtec-ad", type=str, default='../STAD/data', help="MvTec-AD dataset path")
    parser.add_argument('--model-save-path', type=str, default='snapshots', help='path where student models are saved')

    args = parser.parse_args()

    np.random.seed(0)
    torch.manual_seed(0)
    
    transform = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if args.split == 'train':
        image_list = sorted(glob(os.path.join(args.mvtec_ad, '**/img/train_good*.png')))
        # print('image_list', image_list)
        train_image_list, val_image_list = train_test_split(image_list, test_size=0.2, random_state=0)
        # tr_anomaly_datasets = []
        # for i in args.N:
        #     # print('dataset path', dataset_path)
        #     # datasets.append(dataset_path)
        tr_anomaly_datasets = MVTecDataset(train_image_list, transform=transform)
        # print()
        # tr_concat_dataset = ConcatDataset(tr_anomaly_datasets)
        # train_dataset = MVTecDataset(train_image_list, transform=transform)
        train_loader = DataLoader(tr_anomaly_datasets, batch_size=args.batch_size, shuffle=True, drop_last=False)
        val_dataset = MVTecDataset(val_image_list, transform=transform)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    # elif args.split == 'test':
    #     test_neg_image_list = sorted(glob(os.path.join(args.mvtec_ad, args.category, 'test', 'good', '*.png')))
    #     test_pos_image_list = set(glob(os.path.join(args.mvtec_ad, args.category, 'test', '*', '*.png'))) - set(test_neg_image_list)
    #     test_pos_image_list = sorted(list(test_pos_image_list))
    #     test_neg_dataset = MVTecDataset(test_neg_image_list, transform=transform)
    #     test_pos_dataset = MVTecDataset(test_pos_image_list, transform=transform)
    #     test_neg_loader = DataLoader(test_neg_dataset, batch_size=1, shuffle=False, drop_last=False)
    #     test_pos_loader = DataLoader(test_pos_dataset, batch_size=1, shuffle=False, drop_last=False)

    teacher = ResNet18_MS3(pretrained=True)
    
    teacher=teacher.eval().to('cuda')
    x=torch.ones([1,3,256,256]).to('cuda')
    teacher=torch2trt(teacher,[x],int8_mode=False,fp16_mode=True)
    torch.save(teacher.state_dict(), 'teacher_oneclass_16.pth')
    
    student = ResNet18_MS3(pretrained=False)
    
    # device = torch.device("cuda:1")
    # print(f'Device used: {device}')
    teacher.to(device)
    student.to(device)

    if args.split == 'train':
        train_val(teacher, student, train_loader, val_loader, args)
    elif args.split == 'test':
        print('entered in test')
        for i, category in enumerate(args.category):
            test_neg_image_list = sorted(glob(os.path.join(args.mvtec_ad, category, 'img', 'test_good_*.png')))
            #print('neg image_list', test_neg_image_list)
            test_pos_image_list = set(glob(os.path.join(args.mvtec_ad, category, 'img', 'test_*.png'))) - set(test_neg_image_list)
            test_pos_image_list = sorted(list(test_pos_image_list))
            #print('pos image_list', test_pos_image_list)
            test_neg_dataset = MVTecDataset(test_neg_image_list, transform=transform)
            test_pos_dataset = MVTecDataset(test_pos_image_list, transform=transform)
            test_neg_loader = DataLoader(test_neg_dataset, batch_size=1, shuffle=False, drop_last=False)
            test_pos_loader = DataLoader(test_pos_dataset, batch_size=1, shuffle=False, drop_last=False)
            #print(args.checkpoint)
            #saved_dict = torch.load(os.path.join(args.checkpoint, category, 'best_394_83.pt'))
            saved_dict = torch.load(os.path.join(args.checkpoint, category, 'best.pth.tar'))
            category = category
            gt = load_gt(args.mvtec_ad, category)

            print('load ' + args.checkpoint)
            student.load_state_dict(saved_dict['state_dict'])
            #student.load_state_dict(saved_dict)
            
            student=student.eval().to('cuda')
            x=torch.ones([1,3,256,256]).to('cuda')
            student=torch2trt(student,[x],int8_mode=False,fp16_mode=True)
            torch.save(student.state_dict(), 'student_oneclass_16.pth')

            pos, time1 = test(teacher, student, test_pos_loader)
            neg, time2 = test(teacher, student, test_neg_loader)
            print(f'Inf time for {category}', (time1+time2)/2.0)
            with open('oneclass_fp16size_Inf_time.txt', 'a') as f:
                f.write(f'oneclass_fp16size_Category: {category} Time: {(time1+time2)/2.0} \n')

            scores = []
            for i in range(len(pos)):
                temp = cv2.resize(pos[i], (256, 256))
                scores.append(temp)
            for i in range(len(neg)):
                temp = cv2.resize(neg[i], (256, 256))
                scores.append(temp)

            scores = np.stack(scores)
            neg_gt = np.zeros((len(neg), 256, 256), dtype=np.bool_)
            gt_pixel = np.concatenate((gt, neg_gt), 0)
            gt_image = np.concatenate((np.ones(pos.shape[0], dtype=np.bool_), np.zeros(neg.shape[0], dtype=np.bool_)), 0)        

            pro = evaluate(gt_pixel, scores, metric='pro')
            auc_pixel = evaluate(gt_pixel.flatten(), scores.flatten(), metric='roc')
            auc_image_max = evaluate(gt_image, scores.max(-1).max(-1), metric='roc')
            
            print('Catergory: {:s}\tPixel-AUC: {:.6f}\tImage-AUC: {:.6f}\tPRO: {:.6f}\n'.format(category, auc_pixel, auc_image_max, pro))
            with open('oneclass_fp16size_output_stfpm.txt', 'a') as f:
                f.write(f'oneclass_fp16size_Category: {category}\tPixel-AUC: {auc_pixel}\tImage-AUC: {auc_image_max}\tPRO: {pro}'.format(category, auc_pixel, auc_image_max, pro)+' '+'\n')
     


def test(teacher, student, loader):
    teacher.eval()
    student.eval()
    loss_map = np.zeros((len(loader.dataset), 64, 64))
    #GPU warmup
    loader1 = copy.deepcopy(loader)
    test = iter(loader1)
    _,test = next(test)
    test = test.to(device)
    _ = teacher(test) 

    i = 0
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    elapsed_time = 0.0
    print('length of loader', len(loader))
    print('length of dataset', len(loader.dataset))
    for batch_data in loader:
        _, batch_img = batch_data
        batch_img = batch_img.to(device)
        #print('batch', batch_img.size())
        starter.record()
        with torch.no_grad():
            t_feat = teacher(batch_img)
            s_feat = student(batch_img)
        score_map = 1.
        for j in range(len(t_feat)):
            t_feat[j] = F.normalize(t_feat[j], dim=1)
            s_feat[j] = F.normalize(s_feat[j], dim=1)
            sm = torch.sum((t_feat[j] - s_feat[j]) ** 2, 1, keepdim=True)
            sm = F.interpolate(sm, size=(64, 64), mode='bilinear', align_corners=False)
            # aggregate score map by element-wise product
            score_map = score_map * sm
        ender.record()
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        elapsed_time += curr_time

        # print("elapsed time", curr_time)
        loss_map[i: i + batch_img.size(0)] = score_map.squeeze().cpu().data.numpy()
        i += batch_img.size(0)
    elapsed_time_per = elapsed_time/(len(loader.dataset))
    print("Total elapsed time per image", elapsed_time_per)
    return loss_map, elapsed_time_per
    

def train_val(teacher, student, train_loader, val_loader, args):
    min_err = np.inf
    min_err_it = np.inf
    teacher.eval()
    student.train()
    tr_loss_ls = []
    val_loss_ls = []
    tr_loss_it_ls = []
    val_loss_it_ls = []
    
    optimizer = torch.optim.SGD(student.parameters(), 0.4, momentum=0.9, weight_decay=1e-4)
    print('length', len(train_loader))
    for epoch in range(args.epochs):
        student.train()
        running_loss = 0.0
        running_loss_it = 0.0
        for j, batch_data in tqdm(enumerate(train_loader)):
            _, batch_img = batch_data
            batch_img = batch_img.to(device)

            with torch.no_grad():
                t_feat = teacher(batch_img)
            s_feat = student(batch_img)

            loss =  0
            for i in range(len(t_feat)):
                t_feat[i] = F.normalize(t_feat[i], dim=1)
                s_feat[i] = F.normalize(s_feat[i], dim=1)
                loss += torch.sum((t_feat[i] - s_feat[i]) ** 2, 1).mean()

            print('[%d/%d] loss: %f' % (epoch, args.epochs, loss.item()))
            with open("output.txt", 'a') as f1:
                f1.write(f"[{epoch+1}/{args.epochs}] Loss: {loss.item()} \n")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_loss_it += loss.item()
            # print('i',j)
            if (j+1) % 7 == 0:
                err_it = test(teacher, student, val_loader).mean()
                tr_loss_it_ls.append((running_loss_it/(7*args.batch_size)))
                val_loss_it_ls.append((err_it.item()/(len(val_loader)*args.batch_size)))
                # print('exec')
                print('Valid Loss iteration: {:.7f}'.format(err_it.item()))
                with open("output_it.txt", 'a') as f2:
                    f2.write(f"Train Loss iteration: {running_loss_it/(7*args.batch_size)} Valid Loss iteration: {err_it.item()/(len(val_loader)*args.batch_size)} \n")
                if err_it < min_err_it:
                    min_err_it = err_it
                    save_name = os.path.join(args.model_save_path, '15_class', f'best_{epoch}_{j}.pt')
                    state_dict = student.state_dict()
                    torch.save(state_dict, save_name)
                running_loss_it = 0.0

        
        err = test(teacher, student, val_loader).mean()
        print('Valid Loss epoch: {:.7f}'.format(err.item()))
        # err.item():.{7}f
        with open("output.txt", 'a') as f1:
            f1.write(f"Train Loss epoch: {running_loss/(len(train_loader)*args.batch_size)} Valid Loss epoch: {err.item()/(len(val_loader)*args.batch_size)} \n")

        tr_loss_ls.append((running_loss/(len(train_loader)*args.batch_size)))
        val_loss_ls.append((err.item()/(len(val_loader)*args.batch_size)))
        # save_name = os.path.join(args.model_save_path, '15_class', f'epoch_{epoch}.pt')
        # # dir_name = os.path.dirname(save_name)
        # state_dict = student.state_dict()
        torch.save(state_dict, save_name)
        if err < min_err:
            min_err = err
            save_name = os.path.join(args.model_save_path, '15_class', f'best_{epoch}.pt')
            dir_name = os.path.dirname(save_name)
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name)
            # state_dict = {
            #     # 'category': args.category,
            #     'state_dict': student.state_dict()
            # }
            state_dict = student.state_dict()
            torch.save(state_dict, save_name)

    epochs = range(len(tr_loss_ls))
    epochs_it = range(len(tr_loss_it_ls))

    plt.figure(figsize=(30,6))
    plt.subplot(1,2,1)
    plt.plot(epochs_it, tr_loss_it_ls, color='blue')
    plt.plot(epochs_it, val_loss_it_ls, color='red')
    plt.xlabel('epochs')
    plt.ylabel('Loss')
    plt.title(f"Train vs Val iteration")

    plt.subplot(1,2,2)
    plt.plot(epochs, tr_loss_ls, color='blue')
    plt.plot(epochs, val_loss_ls, color='red')
    plt.xlabel('epochs')
    plt.ylabel('Loss')
    plt.title(f"Train vs Val Epoch")
    plt.savefig(f'Plots.png')


if __name__ == "__main__":
    main()
