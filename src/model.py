
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import vit_b_16, ViT_B_16_Weights

class BBox3DHead(nn.Module):
    def __init__(self, in_dim, hidden=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), 
            nn.ReLU(True),
            nn.Dropout(0.5),  
            nn.Linear(hidden, hidden), 
            nn.ReLU(True),
            nn.Dropout(0.3), 
            nn.Linear(hidden, 24)
        )
    def forward(self, x):
        return self.mlp(x).view(-1,8,3)

class BBox3DModel(nn.Module):
    def __init__(self, pretrained=True, freeze_backbone: bool = False):
        super().__init__()
        weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        vit     = vit_b_16(weights=weights)
        vit.heads = nn.Identity()
        self.feature = vit              

        if freeze_backbone:
            for p in self.feature.parameters():
                p.requires_grad = False

        self.mask_net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))   
        )

        self.pc_net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  
        )

        fusion_dim = vit.hidden_dim + 32 + 32

        self.head = BBox3DHead(fusion_dim)

    def forward(self, img, mask, pc):
        x_img = self.feature(img).flatten(1)  

        x_mask = self.mask_net(mask).flatten(1)

        x_pc   = self.pc_net(pc).flatten(1)

        x = torch.cat([x_img, x_mask, x_pc], dim=1)     
        return self.head(x)

        
