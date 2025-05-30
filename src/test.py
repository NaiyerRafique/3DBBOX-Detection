import numpy as np
import torch
import argparse
from dataset import make_dataloader
from model import BBox3DModel
from utils import load_checkpoint
from config import DATA_ROOT, OUTPUT_DIR, LOG_DIR

def test(data, ckpt, bs=16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dl = make_dataloader(data, 'test', bs)

    model = model = BBox3DModel(pretrained=True, freeze_backbone=False).to(device)
    load_checkpoint(model, ckpt)
    model.eval()

    errs = []
    with torch.no_grad():
        for b in dl:
            img  = b['image'].to(device)
            mask = b['mask'].to(device).float()
            pc   = b['pc'].to(device).float()
            gt   = b['bbox3d'].to(device)

            pr = model(img, mask, pc).float()
            errs.append(torch.mean(torch.abs(pr - gt), dim=[1,2]).cpu().numpy())

    errs = np.concatenate(errs, axis=0)
    print(f"Test set: mean L1 error = {errs.mean():.4f} ± {errs.std():.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test 3D BBox model')
    parser.add_argument('--data', default=DATA_ROOT,
                        help=f'Path to data folder (default from config: {DATA_ROOT})')
    parser.add_argument('--ckpt', required=True,
                        help='Path to checkpoint .pth file')
    parser.add_argument('--bs', type=int, default=16,
                        help='Batch size')
    args = parser.parse_args()
    test(args.data, args.ckpt, args.bs)
    

