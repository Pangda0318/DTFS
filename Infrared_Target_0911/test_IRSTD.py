import os
import time
import torch
import wandb
import argparse
from tqdm import tqdm
import numpy as np
import os.path as ops
from datetime import datetime
from utils.ISDTDDataset import ISDTDDataset
from torch.utils.data import DataLoader
# from model.DefineNet import define_trained_DCHFRnet, define_ISDTDnet
from utils.utils import set_device,seed_pytorch
from utils.BCEDiceloss import BCEDiceloss
from torch.optim import lr_scheduler
from utils.Imetrics import SigmoidMetric, SamplewiseSigmoidMetric
from utils.Pmetric import ROCMetric, PD_FA
from model.ISDTDNet import ISDTDNet
from PIL import Image
import torchvision.transforms as transforms
def main(args):
    device = torch.device('cuda')

    train_set = ISDTDDataset(args.root, 'train', if_transform=True, base_size=args.base_size, dataset_type=args.dataset_type)
    val_set = ISDTDDataset(args.root, 'val', if_transform=True, base_size=args.base_size, dataset_type=args.dataset_type)
    print("---------------finished loading dataset---------------")
    print('train data number:',len(train_set))
    print('val data number:', len(val_set))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_worker)
    val_loader = DataLoader(val_set, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_worker)
    print("---------------finished loading dataloader---------------")

    net = set_device(ISDTDNet(args), device)
    loss_func = BCEDiceloss().to(device)
    eval_iou = SigmoidMetric()
    eval_niou = SamplewiseSigmoidMetric(1, score_thresh=0.5)
    eval_ROC = ROCMetric(1, 10)
    eval_PD_FA = PD_FA(1, 10)
    
    net.train()
    optim_params = list(net.parameters())
    opt = torch.optim.Adam(optim_params, lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100, eta_min=0, last_epoch=-1)
    # scheduler = lr_scheduler.MultiStepLR(opt, milestones=[50,100,150], gamma=0.5)
    print("---------------finished loading ISDTD model---------------")

    current_epoch = 0
    n_epoch = args.n_epoch
    best_iou = 0
    best_niou = 0
    current_step = 0

    if args.phase == 'test':
        output_dir = ""
        os.makedirs(output_dir, exist_ok=True) 
        pkl_path = os.path.join(args.plk_path)
        val_tbar = tqdm(val_loader)
        eval_iou.reset()
        eval_niou.reset()
        eval_PD_FA.reset()
        eval_ROC.reset()
        eval_loss = 0.0
        net.load_state_dict(torch.load(pkl_path))
        net.eval()
        for i, val_data in enumerate(val_tbar):
            val_img = val_data['img'] 
            val_img = set_device(val_img, device)
            with torch.no_grad():
                val_pred = net(val_img)
                val_label = val_data['label']
                val_label = set_device(val_label, device)
                val_label = val_label[:, 0:1, :, :]/255
                if 'name' in val_data:
                    filename = os.path.basename(val_data['name'][0])  
                else:
                    filename = f"pred_{i:04d}.png"

                
                val_pred = val_pred
                val_loss_ISDTD, val_bce, val_dice = loss_func(val_pred, val_label)
                eval_loss += val_loss_ISDTD
                val_label = val_label.cpu()
                val_pred = val_pred.cpu()
                
                pred_np = val_pred.squeeze().numpy()  

                pred_np = (pred_np * 255).clip(0, 255).astype(np.uint8)
                output_path = os.path.join(output_dir, filename)
                Image.fromarray(pred_np).save(output_path)
                
                
                eval_iou.update(val_pred, val_label)
                eval_niou.update(val_pred, val_label)
                eval_ROC.update(val_pred, val_label)
                eval_PD_FA.update(val_pred, val_label)
        FA, PD = eval_PD_FA.get(len(val_loader))
        _, IoU = eval_iou.get()
        _, nIoU = eval_niou.get()
        _, _, _, _, F1_score = eval_ROC.get()
        FA = FA[5] * 1000000
        PD = PD[5] * 100
        F1_score = F1_score[5]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default=', help='choose datasets')
    parser.add_argument('--dataset_type', type=str, default='IRSTD', help='BigReal or BigSim or NUAA or NUDT or IRSTD')
    parser.add_argument('--base_size', type=int, default=512, help='img_size')
    parser.add_argument('--batch_size', type=int, default=1, help='train_batch_size')
    parser.add_argument('--val_batch_size', type=int, default=1, help='val_batch_size')
    parser.add_argument('--num_worker', type=int, default=0, help='num_workers')
    parser.add_argument('--n_epoch', type=int, default=1000, help='iter number')
    parser.add_argument('--phase', type=str, default='test', help='train or test')
    parser.add_argument('--val_freq', type=int, default=1, help='validation frequent')
    parser.add_argument('--results_mask', type=str, default='results', help='mask result fold')
    parser.add_argument('--save_path', type=str, default='./checkpoints/IRSTD', help='checkpoint fold')
    parser.add_argument('--name', type=str, default='')

    parser.add_argument("--seed", type=int, default=42, help="Threshold for test")
    parser.add_argument("--plk_path", type=str, default="")
    parser.add_argument('--resume', type=str, default=False)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    seed_pytorch(args.seed)
    if args.phase == 'train':
        os.environ["WANDB_API_KEY"] = "×××××"
        wandb.login() 
        nowtime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
       

    main(args)
