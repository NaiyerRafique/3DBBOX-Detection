import os, torch

def save_checkpoint(model, optimizer, epoch, val_loss, out):
    os.makedirs(out, exist_ok=True)
    path = f"{out}/epoch{epoch:02d}_vl{val_loss:.4f}.pth"
    torch.save({'epoch':epoch,'vl':val_loss,'state':model.state_dict(),
                'opt':optimizer.state_dict()}, path)
    print("Saved:", path)

def load_checkpoint(model, ckpt):
    d = torch.load(ckpt, map_location='cpu')
    model.load_state_dict(d['state'])
    print("Loaded:", ckpt)
    
