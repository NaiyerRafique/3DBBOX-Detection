import os
import numpy as np
import torch
import argparse
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import BBox3DModel
from utils import load_checkpoint


def preprocess(img_path, mask, pc):
    img = np.array(Image.open(img_path).convert('RGB'))
    tr = A.Compose([
        A.Resize(224,224),
        A.Normalize(mean=(0.485,0.456,0.406),
                    std=(0.229,0.224,0.225)),
        ToTensorV2()
    ], additional_targets={'pc':'image'})
    aug = tr(image=img,
             mask=mask.astype('uint8'),
             pc=pc.astype('float32'))
    img_t  = aug['image']
    mask_t = aug['mask'][None].float()
    pc_t   = aug['pc'].float()
    return img_t, mask_t, pc_t


def main():
    parser = argparse.ArgumentParser(description='Torch inference for 3D BBox model')
    parser.add_argument('--ckpt', required=True, help='Path to .pth checkpoint')
    parser.add_argument('--img',  required=True, help='Path to rgb.jpg sample')
    parser.add_argument('--freeze', action='store_true', help='Whether to load model with frozen backbone')
    args = parser.parse_args()

    base = os.path.dirname(args.img)
    mask_arr = np.load(os.path.join(base, 'mask.npy'))
    pc_arr   = np.moveaxis(np.load(os.path.join(base, 'pc.npy')), 0, 2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BBox3DModel(pretrained=True, freeze_backbone=args.freeze).to(device)
    load_checkpoint(model, args.ckpt)
    model.eval()

    all_preds = []
    for inst in range(mask_arr.shape[0]):
        img_t, mask_t, pc_t = preprocess(args.img, mask_arr[inst], pc_arr)
        img_t, mask_t, pc_t = img_t.to(device), mask_t.to(device), pc_t.to(device)
        with torch.no_grad():
            pred = model(img_t.unsqueeze(0), mask_t.unsqueeze(0), pc_t.unsqueeze(0))
        all_preds.append(pred[0].cpu().numpy())

    all_preds = np.stack(all_preds, axis=0)
    print(f"Predicted {all_preds.shape[0]} boxes:")
    print(all_preds)

if __name__ == '__main__':
    main()
    
