import torch
import torch.nn as nn
from transformers.models.clip.modeling_clip import CLIPVisionModelWithProjection
from transformers.models.clip.configuration_clip import CLIPVisionConfig
import os
from safetensors.torch import save_file, load_file


# CLIP Vision 模型配置字典
# 基于 ViT-Large 结构: 24层, 1024维隐藏层, 16个注意力头
VISION_CONFIG_DICT = {
    "hidden_size": 1024,
    "intermediate_size": 4096,
    "num_attention_heads": 16,
    "num_hidden_layers": 24,
    "patch_size": 14,
    "projection_dim": 768
}


class PhotoMakerIDEncoder(CLIPVisionModelWithProjection):
    """PhotoMaker ID 编码器
    
    继承自 CLIPVisionModelWithProjection，用于提取人脸图像的 CLIP 视觉特征。
    输出维度为 1280维 (768 + 512)，包含两个投影分支的特征融合。
    """
    def __init__(self):
        super().__init__(CLIPVisionConfig(**VISION_CONFIG_DICT))
        # 第二个视觉投影层：将1024维映射到512维
        self.visual_projection_2 = nn.Linear(1024, 1280, bias=False)
        
    def forward(self, id_pixel_values):
        """前向传播
        
        Args:
            id_pixel_values: 人脸图像像素值
            
        Returns:
            id_embeds: 1280维的人脸嵌入向量
        """
        # 假设只输入一张参考图像
        # b, c, h, w = id_pixel_values.shape
        
        # 通过 CLIP 视觉编码器获取共享嵌入
        shared_id_embeds = self.vision_model(id_pixel_values)[1]
        # 第一个投影分支：768维
        id_embeds = self.visual_projection(shared_id_embeds)
        # 第二个投影分支：512维
        id_embeds_2 = self.visual_projection_2(shared_id_embeds)
        
        # 拼接两个分支的特征：768 + 512 = 1280维
        id_embeds = torch.cat((id_embeds, id_embeds_2), dim=-1)
        
        return id_embeds


class MLP(nn.Module):
    """带残差连接的多层感知机
    
    包含 LayerNorm、GELU激活函数和两层线性变换。
    支持可选的残差连接。
    """
    def __init__(self, in_dim, out_dim, hidden_dim, use_residual=True):
        super().__init__()
        if use_residual:
            assert in_dim == out_dim
        self.layernorm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.use_residual = use_residual
        self.act_fn = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.layernorm(x)
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        if self.use_residual:
            x = x + residual
        return x


class MLPs(nn.Module):
    """多层感知机堆叠
    
    由4个MLP组成，逐步扩大隐藏层维度：
    in_dim -> 2*in_dim -> 2*in_dim -> 4*in_dim -> 4*in_dim
    用于处理 ID 嵌入特征。
    """
    def __init__(self, in_dim):
        super().__init__()
        self.mlp1 = MLP(in_dim, in_dim * 2, in_dim * 2, False)
        self.mlp2 = MLP(in_dim * 2, in_dim * 2, in_dim * 2, True)
        self.mlp3 = MLP(in_dim * 2, in_dim * 4, in_dim * 4, False)
        self.mlp4 = MLP(in_dim * 4, in_dim * 4, in_dim * 4, True)
        
    def forward(self, x):
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)
        x = self.mlp4(x)
        return x 


class Mix(nn.Module):
    """混合模块
    
    用于融合 CLIP 视觉嵌入和人脸 ID 嵌入的模块。
    流程：
    1. 将 ID 嵌入通过 MLPs 处理
    2. 将 CLIP 嵌入和处理后的 ID 嵌入拼接
    3. 通过 MLP 进行特征融合
    4. 输出最终的混合嵌入（用于文本提示）
    
    输入:
        clip_emb: CLIP 视觉嵌入 (batch, 1280)
        id_emb: 人脸 ID 嵌入 (batch, 512)
    输出:
        emb: 混合嵌入 (batch, 2048)
    """
    def __init__(self, embed_dim=2048):
        super().__init__()
        # ID 嵌入处理器：512 -> 2048
        self.pro_id = MLPs(512)
        # 融合 MLP：拼接后 (1280 + 2048) -> 2048
        self.mlp1 = MLP(embed_dim * 2, embed_dim, embed_dim, use_residual=False)
        self.mlp2 = MLP(embed_dim, embed_dim, embed_dim, use_residual=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        
    def forward(self, clip_emb, id_emb):
        """前向传播
        
        Args:
            clip_emb: CLIP 视觉嵌入 [batch, 1280]
            id_emb: 人脸 ID 嵌入 [batch, 512]
            
        Returns:
            emb: 混合嵌入 [batch, 2048]
        """
        # 处理 ID 嵌入
        id_emb = self.pro_id(id_emb)
        # 拼接 CLIP 和处理后的 ID 嵌入
        emb = self.mlp1(torch.cat([clip_emb, id_emb], dim=-1))
        # LayerNorm
        emb = self.norm1(emb)
        # 最后的 MLP 处理
        emb = self.mlp2(emb)
        return emb



    def from_pretrained(self, save_dir):
        """从预训练权重加载模型"""
        mix_path = os.path.join(save_dir, "mix.safetensors")
        state_dict = load_file(mix_path)
        self.load_state_dict(state_dict)
        
    def save_pretrained(self, save_dir):
        """保存模型到指定目录"""
        os.makedirs(save_dir, exist_ok=True)
        mix_path = os.path.join(save_dir, "mix.safetensors")
        save_file(self.state_dict(), mix_path)
        print(f"Mix saved to {mix_path}")
