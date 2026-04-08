import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

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
    def __init__(self, pretrained=True):
        super(ArcFaceLoss, self).__init__()
        # According to the proposal, ResNet-18 is used as the Identity Branch
        # In practice, this should be loaded with ArcFace pre-trained weights on a face dataset (e.g. WebFace or MS1MV2)
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        # Remove the classification head to get the 512-dim embedding
        self.resnet.fc = nn.Identity()
        self.resnet.eval()
        for param in self.resnet.parameters():
            param.requires_grad = False
            
    def extract_features(self, x):
        # Interpolate to 112x112 which is standard for face recognition networks
        x = F.interpolate(x, size=(112, 112), mode='bilinear', align_corners=False)
        # Assuming input is [0, 1], normalize with ImageNet stats (or face model stats if custom weights are loaded)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        x = (x - mean) / std
        features = self.resnet(x)
        return F.normalize(features, p=2, dim=1)

    def forward(self, input_img, target_img):
        # Compute cosine similarity between extracted ID features
        id_pred = self.extract_features(input_img)
        id_target = self.extract_features(target_img)
        # Cosine distance: 1 - cosine_similarity
        loss = 1.0 - (id_pred * id_target).sum(dim=1).mean()
        return loss
