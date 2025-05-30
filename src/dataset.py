import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import torch.nn.functional as F


class BBox3DDataset(Dataset):
    def __init__(self, root_dir, split='train', transforms=None):
        self.root = root_dir
        all_ids = sorted(os.listdir(self.root))
        n = len(all_ids)
        if split=='train':
            ids = all_ids[:int(0.8*n)]
        elif split=='val':
            ids = all_ids[int(0.8*n):int(0.9*n)]
        else:
            ids = all_ids[int(0.9*n):]
        self.index = []
        for idx in ids:
            mask = np.load(os.path.join(self.root, idx, 'mask.npy'))
            for inst in range(mask.shape[0]):
                self.index.append((idx, inst))
        self.transforms = transforms or A.Compose([
            A.Resize(224, 224),
            A.ColorJitter(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ], additional_targets={'pc': 'image'})
           

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        folder, inst = self.index[i]
        base = os.path.join(self.root, folder)
        img = np.array(Image.open(f"{base}/rgb.jpg").convert('RGB'))
        mask   = np.load(f"{base}/mask.npy")[inst]
        bbox3d = np.load(f"{base}/bbox3d.npy")[inst]
        pc = np.moveaxis(np.load(f"{base}/pc.npy"), 0, 2)
        ys, xs = np.where(mask>0)
        ys, xs = np.where(mask>0)
        if ys.size and xs.size:
            y0,y1 = ys.min(), ys.max()
            x0,x1 = xs.min(), xs.max()
            pad = 10
            y0, y1 = max(0, y0-pad), min(mask.shape[0], y1+pad)
            x0, x1 = max(0, x0-pad), min(mask.shape[1], x1+pad)
            img   = img  [y0:y1, x0:x1]
            mask  = mask [y0:y1, x0:x1]
            pc     = pc  [y0:y1, x0:x1]
        aug = self.transforms(
            image=img,
            mask=mask.astype('uint8'),
            pc=pc.astype('float32'),
        )

        img_t  = aug['image']                          
        mask_t = aug['mask'].unsqueeze(0).float()     

        pc_aug = aug['pc']
        if isinstance(pc_aug, np.ndarray):
            pc_t = torch.from_numpy(pc_aug.transpose(2,0,1)).float()
        else:
            pc_t = pc_aug
            if pc_t.ndim==3 and pc_t.shape[0] not in (1,3):
                pc_t = pc_t.permute(2,0,1)
            pc_t = pc_t.float()

        if pc_t.shape[1:] != (224,224):
            pc_t = F.interpolate(
                pc_t.unsqueeze(0), size=(224,224),
                mode='nearest', align_corners=False
            ).squeeze(0)

        return {
            'image' : img_t,
            'mask'  : mask_t,
            'pc'    : pc_t,
            'bbox3d': torch.from_numpy(bbox3d).float()
        }

def make_dataloader(root, split, bs, workers=2):
    ds = BBox3DDataset(root, split)
    return DataLoader(
        ds, batch_size=bs,
        shuffle=(split=='train'),
        num_workers=workers,
        pin_memory=True
    )
    
