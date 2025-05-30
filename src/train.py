import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dataset import make_dataloader
from model import BBox3DModel
from utils import save_checkpoint
from losses import IoULoss3D
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils import clip_grad_norm_
from config import DATA_ROOT, OUTPUT_DIR, LOG_DIR
def train_one(model, loader, opt, l1_loss, iou_loss, alpha, device):
    model.train()
    running_loss = 0.0

    for b_idx, batch in enumerate(tqdm(loader, desc="  batches")):
        img   = batch['image'].to(device).float()
        mask  = batch['mask'].to(device).float()
        pc    = batch['pc'].to(device).float()
        gt    = batch['bbox3d'].to(device)

        opt.zero_grad()
        preds = model(img, mask, pc)

        loss_l1  = l1_loss(preds, gt)
        loss_iou = iou_loss(preds, gt)
        loss     = loss_l1 + alpha * loss_iou

        loss.backward()
        opt.step()

        running_loss += loss.item() * img.size(0)
        if (b_idx + 1) % 20 == 0:
            print(f"    batch {b_idx+1}/{len(loader)} — "
                  f"L1: {loss_l1.item():.4f}, IoU: {loss_iou.item():.4f}, "
                  f"Total: {loss.item():.4f}")

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train 3D BBox Model')
    parser.add_argument('--data',       default=DATA_ROOT, help=f'Path to root data folder (default: {DATA_ROOT})')
    parser.add_argument('--epochs',     type=int,   default=30, help='Number of epochs')
    parser.add_argument('--bs',         type=int,   default=16, help='Batch size')
    parser.add_argument('--lr',         type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--alpha',      type=float, default=1.0,
                        help='Weight for IoU loss term')
    parser.add_argument('--out',        default=OUTPUT_DIR, help=f'Checkpoint output dir (default: {OUTPUT_DIR})')

    parser.add_argument('--logdir',     default=LOG_DIR, help=f'TensorBoard log dir (default: {LOG_DIR})')
    parser.add_argument('--max-models', type=int, default=None,
                        help='Maximum number of best checkpoints to keep')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dl = make_dataloader(args.data, 'train', args.bs)
    val_dl   = make_dataloader(args.data, 'val',   args.bs)

    model    = BBox3DModel(pretrained=True, freeze_backbone=False).to(device)
    opt = AdamW([{'params': model.feature.parameters(), 'lr': args.lr * 0.1, 'weight_decay':1e-4}, {'params': model.head.parameters(), 'lr': args.lr, 'weight_decay':1e-6}])
    l1_loss  = nn.L1Loss()
    iou_loss = IoULoss3D()
    alpha    = args.alpha
    sched = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
    writer = SummaryWriter(log_dir='runs')
    best_val = float('inf')

    for epoch in range(1, args.epochs + 1):
        print(f"\n==> Epoch {epoch}/{args.epochs}")
        train_loss = train_one(model, train_dl, opt, l1_loss, iou_loss, alpha, device)

        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for batch in val_dl:
                img   = batch['image'].to(device).float()
                mask  = batch['mask'].to(device).float()
                pc    = batch['pc'].to(device).float()
                gt    = batch['bbox3d'].to(device)

                preds = model(img, mask, pc)
                loss_l1  = l1_loss(preds, gt)
                loss_iou = iou_loss(preds, gt)
                loss     = loss_l1 + alpha * loss_iou

                val_running += loss.item() * img.size(0)

        val_loss = val_running / len(val_dl.dataset)
        print(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}")

        writer.add_scalars('Loss', {
            'train_total': train_loss,
            'train_l1':    train_loss - alpha * loss_iou.item(),
            'train_iou':   loss_iou.item(),
            'val_total':   val_loss
        }, epoch)

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, opt, epoch, val_loss, args.out)

            if args.max_models is not None:
                files = [f for f in os.listdir(args.out) if f.endswith('.pth')]
                if len(files) > args.max_models:
                    losses = []
                    for f in files:
                        try:
                            loss_val = float(f.split('_vl')[1].split('.pth')[0])
                            losses.append((f, loss_val))
                        except:
                            continue
                    worst_file = max(losses, key=lambda x: x[1])[0]
                    os.remove(os.path.join(args.out, worst_file))
                    print(f"Deleted old checkpoint {worst_file} to keep top {args.max_models} models.")
        sched.step()
    writer.close()
    
