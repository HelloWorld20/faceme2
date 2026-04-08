import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import os
from arch.iresnet import Backbone

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        # Use VGG16 for perceptual loss
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).eval()
        # Extract features from relu1_2, relu2_2, relu3_3, relu4_3
        blocks = []
        blocks.append(vgg.features[:4])
        blocks.append(vgg.features[4:9])
        blocks.append(vgg.features[9:16])
        blocks.append(vgg.features[16:23])
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = nn.ModuleList(blocks)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target):
        # input and target should be in [0, 1]
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        loss = 0.0
        x = input
        y = target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += F.l1_loss(x, y)
        return loss

class ArcFaceLoss(nn.Module):
    def __init__(self, pretrained_path=None):
        super(ArcFaceLoss, self).__init__()
        # Initialize the IR-SE50 backbone which is standard for ArcFace
        # num_layers=50, drop_ratio=0.6, mode='ir_se'
        self.resnet = Backbone(50, 0.6, 'ir_se')
        
        if pretrained_path is None:
            # Default fallback path, you can replace it with your downloaded weights
            pretrained_path = "models/model_ir_se50.pth"
            pretrained_url = "https://github.com/TreB1eN/InsightFace_Pytorch/releases/download/1.0/model_ir_se50.pth"
            os.makedirs("models", exist_ok=True)
            if not os.path.exists(pretrained_path):
                print(f"Downloading pre-trained ArcFace weights... (Note: this URL might be broken, please download manually if it fails)")
                try:
                    torch.hub.download_url_to_file(pretrained_url, pretrained_path)
                except Exception as e:
                    print(f"Warning: Failed to download ArcFace weights: {e}. Please download `model_ir_se50.pth` manually to {pretrained_path}")
        
        if os.path.exists(pretrained_path):
            print(f"Loading ArcFace pre-trained weights from {pretrained_path}")
            state_dict = torch.load(pretrained_path, map_location='cpu')
            self.resnet.load_state_dict(state_dict, strict=False)
        else:
            print("Warning: No ArcFace pre-trained weights found. Using random weights.")
            
        self.resnet.eval()
        for param in self.resnet.parameters():
            param.requires_grad = False
            
    def extract_features(self, x):
        # The input x is assumed to be normalized in [0, 1] with shape (B, 3, H, W)
        # Interpolate to 112x112 which is standard for face recognition networks
        x = F.interpolate(x, size=(112, 112), mode='bilinear', align_corners=False)
        # Standardize using mean=0.5, std=0.5 as required by InsightFace PyTorch models
        x = (x - 0.5) / 0.5
        features = self.resnet(x)
        # Normalize the embedding to unit length
        features = F.normalize(features, p=2, dim=1)
        return features

    def forward(self, input_img, target_img):
        id_pred = self.extract_features(input_img)
        id_target = self.extract_features(target_img)
        # Cosine distance loss: 1 - cosine_similarity
        loss = 1.0 - (id_pred * id_target).sum(dim=1).mean()
        return loss
