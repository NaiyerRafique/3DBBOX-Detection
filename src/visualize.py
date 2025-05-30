
import os
import numpy as np
import cv2
import torch
import argparse
import matplotlib.pyplot as plt

from model import BBox3DModel
from utils import load_checkpoint
from inference import preprocess

def get_camera_intrinsics(width, height, fov_deg=60):

    fov = np.deg2rad(fov_deg)
    fx  = (width  / 2) / np.tan(fov/2)
    fy  = (height / 2) / np.tan(fov/2)
    cx  = width  / 2
    cy  = height / 2
    return np.array([[fx,  0, cx],
                     [ 0, fy, cy],
                     [ 0,  0,  1]], dtype=np.float32)

def project_points(pts3d, K):

    proj = (K @ pts3d.T).T      # (N×3)
    uv   = proj[:, :2] / proj[:, 2:3]
    return uv

def draw_wireframe(img, pts2d, color, thickness=2):

    edges = [
      (0,1),(1,2),(2,3),(3,0),
      (4,5),(5,6),(6,7),(7,4),
      (0,4),(1,5),(2,6),(3,7)
    ]
    h, w = img.shape[:2]
    for i,j in edges:
        u0, v0 = int(np.clip(pts2d[i,0], 0, w-1)), int(np.clip(pts2d[i,1], 0, h-1))
        u1, v1 = int(np.clip(pts2d[j,0], 0, w-1)), int(np.clip(pts2d[j,1], 0, h-1))
        cv2.line(img, (u0,v0), (u1,v1), color, thickness)
    return img

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Visualize ALL GT & Pred 3D boxes on an RGB image"
    )
    parser.add_argument(
        '--ckpt',   required=True,
        help='path to your .pth checkpoint'
    )
    parser.add_argument(
        '--sample', required=True,
        help='path to one sample folder (contains rgb.jpg, mask.npy, pc.npy, bbox3d.npy)'
    )
    parser.add_argument(
        '--fov', type=float, default=60,
        help='(optional) horizontal field of view in degrees for default intrinsics'
    )
    args = parser.parse_args()


    base = args.sample
    img_path = os.path.join(base, 'rgb.jpg')
    mask_arr = np.load(os.path.join(base, 'mask.npy'))        
    pc_arr   = np.moveaxis(np.load(os.path.join(base, 'pc.npy')), 0, 2) 
    gt_all   = np.load(os.path.join(base, 'bbox3d.npy'))    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = BBox3DModel(pretrained=True).to(device)
    load_checkpoint(model, args.ckpt)
    model.eval()


    img_bgr = cv2.imread(img_path)
    h, w    = img_bgr.shape[:2]
    K       = get_camera_intrinsics(w, h, args.fov)

    overlay = img_bgr.copy()


    for inst in range(mask_arr.shape[0]):

        gt_box = gt_all[inst]                        
        gt2d   = project_points(gt_box, K)          
        overlay = draw_wireframe(
            overlay, gt2d, color=(255,0,0), thickness=2
        )


        mask_i = mask_arr[inst]
        img_t, mask_t, pc_t = preprocess(img_path, mask_i, pc_arr)
        with torch.no_grad():
            pred_box = model(
                img_t.unsqueeze(0).to(device),
                mask_t.unsqueeze(0).to(device),
                pc_t.unsqueeze(0).to(device)
            )[0].cpu().numpy()  # (8,3)

        pred2d = project_points(pred_box, K)
        overlay = draw_wireframe(
            overlay, pred2d, color=(0,255,0), thickness=2
        )

    out_fn = 'overlay_all.jpg'
    cv2.imwrite(out_fn, overlay)
    print(f"Saved 2D overlay with all boxes to {out_fn}")

    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10,8))
    plt.imshow(overlay_rgb)
    plt.axis('off')
    plt.title('GT boxes in BLUE, Pred boxes in GREEN')
    plt.show()
