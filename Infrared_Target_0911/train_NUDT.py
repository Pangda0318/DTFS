import os
import torch
import wandb
import argparse
from tqdm import tqdm
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
    #net.load_state_dict(torch.load(args.resume_path))
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
    if args.phase == 'train':
        while current_epoch < n_epoch:
            train_loss = 0.0
            tbar = tqdm(train_loader)
            for _, train_data in enumerate(tbar):
                train_data = set_device(train_data, device)
                img = train_data['img']
                pred = net(img)
                label = train_data['label']
                label = label[:, 0:1, :, :]/255
                loss_ISDTD, bce, dice = loss_func(pred,label)
                current_step += 1
                opt.zero_grad()
                loss_ISDTD.backward()
                opt.step()
                scheduler.step()
                train_loss += loss_ISDTD
                tbar.set_description('Epoch %d, train loss %.4f, bce %.4f, dice %.4f' % (current_epoch, loss_ISDTD, bce, dice))
            wandb.log({'train_epoch': current_epoch, 'train_loss': train_loss/len(train_loader)})
            current_epoch += 1
            if current_epoch % args.val_freq == 0:
                print("---------------validation---------------")
                val_tbar = tqdm(val_loader)
                eval_iou.reset()
                eval_niou.reset()
                eval_PD_FA.reset()
                eval_ROC.reset()
                eval_loss = 0.0
                net.eval()
                for _, val_data in enumerate(val_tbar):
                    val_img = val_data['img'] 
                    val_img = set_device(val_img, device)
                    with torch.no_grad():
                        val_pred = net(val_img)
                        val_label = val_data['label']
                        val_label = set_device(val_label, device)
                        val_label = val_label[:, 0:1, :, :]/255
                        val_pred = val_pred
                        val_loss_ISDTD, val_bce, val_dice = loss_func(val_pred, val_label)
                        eval_loss += val_loss_ISDTD
                        val_label = val_label.cpu()
                        val_pred = val_pred.cpu()
                        eval_iou.update(val_pred, val_label)
                        eval_niou.update(val_pred, val_label)
                        eval_ROC.update(val_pred, val_label)
                        eval_PD_FA.update(val_pred, val_label)
                    val_tbar.set_description('Epoch %d, val loss %.4f, bce %.4f, dice %.4f' % (current_epoch, val_loss_ISDTD, val_bce, val_dice))
                FA, PD = eval_PD_FA.get(len(val_loader))
                _, IoU = eval_iou.get()
                _, nIoU = eval_niou.get()
                _, _, _, _, F1_score = eval_ROC.get()
                FA = FA[5] * 1000000
                PD = PD[5] * 100
                F1_score = F1_score[5]
                wandb.log({'epoch': current_epoch, "ioU": IoU, "nioU": nIoU, 'test_loss': eval_loss/len(val_loader), "PD": PD, "FA": FA, "F1_score": F1_score})
                if IoU > best_iou:
                    best_iou = IoU
                    if IoU > 0.60:
                        pkl_name = 'best-%s-Epoch-%3d_IoU-%.4f_nIoU-%.4f.pkl' % (args.name,current_epoch, best_iou, nIoU)
                        torch.save(net.state_dict(), ops.join(args.save_path, pkl_name))
                if nIoU > best_niou:
                    best_niou = nIoU
                    
                wandb.log({'epoch': current_epoch, "best_ioU": best_iou, "best_nioU": best_niou})
                net.train()
                
    if args.phase == 'test':
        pkl_path = os.path.join(args.plk_path)
        print("---------------validation---------------")
        val_tbar = tqdm(val_loader)
        eval_iou.reset()
        eval_niou.reset()
        eval_PD_FA.reset()
        eval_ROC.reset()
        eval_loss = 0.0
        #net.load_state_dict(torch.load(pkl_path))
        net.eval()
        for _, val_data in enumerate(val_tbar):
            val_img = val_data['img'] 
            val_img = set_device(val_img, device)
            with torch.no_grad():
                val_pred = net(val_img)
                val_label = val_data['label']
                val_label = set_device(val_label, device)
                val_label = val_label[:, 0:1, :, :]/255
                val_pred = val_pred
                val_loss_ISDTD, val_bce, val_dice = loss_func(val_pred, val_label)
                eval_loss += val_loss_ISDTD
                val_label = val_label.cpu()
                val_pred = val_pred.cpu()
                eval_iou.update(val_pred, val_label)
                eval_niou.update(val_pred, val_label)
                eval_ROC.update(val_pred, val_label)
                eval_PD_FA.update(val_pred, val_label)
            # val_tbar.set_description('Epoch %d, val loss %.4f, bce %.4f, dice %.4f' % (current_epoch, val_loss_ISDTD, val_bce, val_dice))
        FA, PD = eval_PD_FA.get(len(val_loader))
        _, IoU = eval_iou.get()
        _, nIoU = eval_niou.get()
        _, _, _, _, F1_score = eval_ROC.get()
        FA = FA[5] * 1000000
        PD = PD[5] * 100
        F1_score = F1_score[5]
        print("---------------Validation Results---------------")
        print(f"{'Loss':<10} {'FA':<10} {'PD':<10} {'IoU':<10} {'nIoU':<10} {'F1 Score':<10}")
        print(f"{eval_loss:.4f}   {FA:.2f}   {PD:.2f}   {IoU:.4f}   {nIoU:.4f}   {F1_score:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/home2/hugl/IR/datasets/IRSTD-1K', help='choose datasets')
    parser.add_argument('--dataset_type', type=str, default='IRSTD', help='BigReal or BigSim or NUAA or NUDT or IRSTD')
    parser.add_argument('--checkpoint_path', type=str, default='/xxxx_gen.pth', help='DCHFR checkpoint path')
    parser.add_argument('--ISDTD_checkpoint_path', type=str, default='.pth', help='trained ISDTD checkpoint path')
    # /home2/hugl/IR/Infrared_Target_1129/checkpoints/IRSTD/best-IRSTD_no7_19_WT+Mamba+FAS+CFAS+FAI_1024_22_tiqu_tiquzhulu_0.5_0.5-Epoch-365_IoU-0.7068_nIoU-0.6553.pkl
    parser.add_argument('--resume_path', type=str, default='/home2/hugl/IR/Infrared_Target_1129/checkpoints/NUAA/best-NUDT_no7_19_WT+Mamba+FAS+CFAS+FAI_1024_22_tiqu_tiquzhulu_0.55_0.45-Epoch-222_IoU-0.7882_nIoU-0.7879.pkl', help='trained ISDTD checkpoint path')
    parser.add_argument('--base_size', type=int, default=512, help='img_size')
    parser.add_argument('--batch_size', type=int, default=1, help='train_batch_size')
    parser.add_argument('--val_batch_size', type=int, default=1, help='val_batch_size')
    parser.add_argument('--num_worker', type=int, default=0, help='num_workers')
    parser.add_argument('--n_epoch', type=int, default=1000, help='iter number')
    parser.add_argument('--phase', type=str, default='train', help='train or test')
    parser.add_argument('--val_freq', type=int, default=1, help='validation frequent')
    parser.add_argument('--results_mask', type=str, default='results', help='mask result fold')
    parser.add_argument('--save_path', type=str, default='./checkpoints/IRSTD1', help='checkpoint fold')
    parser.add_argument('--name', type=str, default='IRSTD_no8_WT+Mamba+FAS+CFAS+FAI_1024_22_tiqu_tiquzhulu_0.5_0.5_0217')#NUAA_no7_WT+Mamba+FAS+CFAS+FAI_1024_22_tiqu_tiquzhulu_0.5_0.5_0217
    # parser.add_argument('--name', type=str, default='NUDT_no6_4_cnn+wt_swin_fusion_fas+cfas2')
    parser.add_argument("--seed", type=int, default=42, help="Threshold for test")
    parser.add_argument("--plk_path", type=str, default="/home2/hugl/IR/Infrared_Target_1129/checkpoints/NUAA/best-NUAA_no7_19_WT+Mamba+FAS+CFAS+FAI_1024_22_tiqu_tiquzhulu_0.5_0.4-Epoch-417_IoU-0.7793_nIoU-0.7626.pkl")
    parser.add_argument('--resume', type=str, default=False)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    seed_pytorch(args.seed)
    if args.phase == 'train':
        os.environ["WANDB_API_KEY"] = "f616ad77b967e58b1837619821297dfe3d274f01"
        wandb.login() 
        nowtime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    #wandb.init(project='no8_22_3_26', config=args.__dict__, name=args.name + nowtime, save_code=False) #NUAA_plus5 
       # wandb.init(project='IRSTD_8_07_4yue01', config=args.__dict__, name=args.name + nowtime, save_code=False)
        wandb.init(project='IRSTD_8_4.02_qu Net 1*1..', config=args.__dict__, name=args.name + nowtime, save_code=False, settings=wandb.Settings(init_timeout=120))
       

    main(args)