import os
import numpy as np
import torch
import onnxruntime
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

def export_onnx(model, img_t, mask_t, pc_t, onnx_path='bbox3d.onnx'):
    inp = (img_t.unsqueeze(0), mask_t.unsqueeze(0), pc_t.unsqueeze(0))
    torch.onnx.export(
        model, inp, onnx_path,
        input_names=['image','mask','pc'],
        output_names=['bbox3d'],
        dynamic_axes={
            'image':{0:'batch'},
            'mask': {0:'batch'},
            'pc':   {0:'batch'},
            'bbox3d': {0:'batch'}
        },
        opset_version=14
    )
    print(f"ONNX model saved to {onnx_path}")

def run_onnx(onnx_path, img_t, mask_t, pc_t):
    sess = onnxruntime.InferenceSession(onnx_path)
    inp = {
        sess.get_inputs()[0].name: img_t.unsqueeze(0).numpy(),
        sess.get_inputs()[1].name: mask_t.unsqueeze(0).numpy(),
        sess.get_inputs()[2].name: pc_t.unsqueeze(0).numpy(),
    }
    return sess.run(None, inp)[0]  # (1,8,3)

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True,
                        help='Path to .pth checkpoint')
    parser.add_argument('--img', required=True,
                        help='Path to a single rgb.jpg sample')
    parser.add_argument('--onnx', default='bbox3d.onnx',
                        help='Path to export/use ONNX model')
    args = parser.parse_args()

    base = os.path.dirname(args.img)
    mask_arr = np.load(os.path.join(base, 'mask.npy'))                    
    pc_arr   = np.moveaxis(np.load(os.path.join(base, 'pc.npy')), 0, 2)    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model = BBox3DModel(pretrained=True, freeze_backbone=False).to(device)
    load_checkpoint(model, args.ckpt)

    img_t, mask_t, pc_t = preprocess(args.img, mask_arr[0], pc_arr)
    img_t = img_t.to(device)
    mask_t = mask_t.to(device)
    pc_t   = pc_t.to(device)
    export_onnx(model, img_t, mask_t, pc_t, args.onnx)

    all_preds = []
    for inst in range(mask_arr.shape[0]):
        img_t, mask_t, pc_t = preprocess(args.img, mask_arr[inst], pc_arr)
        pred = run_onnx(args.onnx, img_t, mask_t, pc_t)
        all_preds.append(pred[0])  

    all_preds = np.stack(all_preds, 0) 
    print(f"Predicted {all_preds.shape[0]} boxes:")
    print(all_preds)
    
