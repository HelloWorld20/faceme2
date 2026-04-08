import argparse
import os
import torch
import torch.nn.functional as F
import json 
import math
import random
import itertools
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from diffusers import UNet2DConditionModel as OriginalUNet2DConditionModel
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
)
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module
from huggingface_hub import hf_hub_download
from utils.load_photomaker import load_photomaker
from arch.idencoder import Mix
from arch.swinir import SwinIRQualityBranch
from arch.losses import PerceptualLoss, ArcFaceLoss

if is_wandb_available():
    pass

logger = get_logger(__name__, log_level="INFO")

def load_config(config_path):
    """
    加载配置文件并返回配置字典。
    """
    with open(config_path, "r") as file:
        config = json.load(file)
    return config


def make_train_dataset(args, tokenizer, text_encoder):
    # 编码文本提示词
    def encode_prompt(text_encoders, text_input_ids_list=None):
        prompt_embeds_list = []
        for i, text_encoder in enumerate(text_encoders):
            prompt_embeds = text_encoder(
                text_input_ids_list[i].to(text_encoder.device), output_hidden_states=True, return_dict=False
            )
            # 我们总是只关注最终文本编码器的池化输出 (pooled output)
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds[-1][-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)
        # 拼接多个文本编码器的输出
        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
        return prompt_embeds, pooled_prompt_embeds

    
    tokens_one = tokenizer[0]("A photo of face.",padding="max_length",
                              max_length=tokenizer[0].model_max_length,
                              truncation=True,
                              return_tensors="pt",).input_ids
    tokens_two = tokenizer[1]("A photo of face.",padding="max_length",
                              max_length=tokenizer[1].model_max_length,
                              truncation=True,
                              return_tensors="pt",).input_ids
    prompt_embeds , pooled_prompt_embeds = encode_prompt(text_encoders=text_encoder, text_input_ids_list=[tokens_one, tokens_two])
    prompt_embeds = prompt_embeds.detach().cpu()
    pooled_prompt_embeds = pooled_prompt_embeds.detach().cpu()
    crops_coords_top_left = (0, 0)
    original_size = (args.resolution, args.resolution)
    target_size = (args.resolution, args.resolution)
    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    add_time_ids = torch.tensor([add_time_ids])
    
    from dataset import FaceMeDataset
    train_dataset = FaceMeDataset(file_json=args.train_data_dir, 
                                  resolution=args.resolution,
                                  prompt_embeds=prompt_embeds.squeeze(dim=0), 
                                  pooled_prompt_embeds=pooled_prompt_embeds.squeeze(dim=0), 
                                  tokens_one=tokens_one.squeeze(dim=0), 
                                  add_time_ids=add_time_ids.squeeze(dim=0), )    
   
      
    if args.max_train_samples is not None:
        train_dataset = torch.utils.data.Subset(train_dataset, list(range(min(args.max_train_samples, len(train_dataset)))))
    

    return train_dataset

import gc
import sys

def log_vram(msg):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        print(f"[{msg}] VRAM Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

def main(args):

    logging_dir = os.path.join(args.output_dir, "log")
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            
    # 添加初始显存日志
    import torch as th  # 使用别名以避免 UnboundLocalError
    if th.cuda.is_available():
        allocated = th.cuda.memory_allocated() / (1024 ** 3)
        reserved = th.cuda.memory_reserved() / (1024 ** 3)
        logger.info(f"Initial GPU Memory: {allocated:.2f} GB Allocated, {reserved:.2f} GB Reserved")

    if accelerator.is_main_process:
        log_vram("Before loading models")

    # 初始化分词器
    tokenizer_one = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        use_fast=False,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        use_fast=False,
    )
    
    token_id_one = tokenizer_one.encode("face")[1]
    token_id_two = tokenizer_two.encode("face")[1]
    
    # 初始化文本编码器
    text_encoder_one = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", )
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_2", )

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    
    # 初始化 VAE 和 UNet
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae",)
    unet = OriginalUNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", )

    if accelerator.is_main_process:
        log_vram("After loading base models (VAE, UNet, TextEncoders)")

    # 合并 photomaker 权重
    photomaker_path = "./models/photomaker-v1.bin"
    if not os.path.exists(photomaker_path):
        logger.info(f"Downloading PhotoMaker checkpoint to {photomaker_path}...")
        photomaker_path = hf_hub_download(repo_id="TencentARC/PhotoMaker", filename="photomaker-v1.bin", repo_type="model", local_dir="./models")
    _, unet = load_photomaker(photomaker_path, clip_id_encoder=None, unet=unet)
     ###

    ## 从 unet 初始化 controlnet 权重 
    controlnet = ControlNetModel.from_unet(unet)

    # 添加 id mix 模块
    mix = Mix()
    if args.mix_pretrained_path is not None and args.mix_pretrained_path.lower() != "none":
        mix.from_pretrained(args.mix_pretrained_path)
    ###

    # 初始化 SwinIR 质量分支，以及 Perceptual Loss 和 ArcFace Loss
    swinir = SwinIRQualityBranch(use_checkpoint=args.gradient_checkpointing)
    perceptual_loss = PerceptualLoss()
    arcface_loss = ArcFaceLoss()

    if accelerator.is_main_process:
        log_vram("After loading ControlNet, Mix, SwinIR, and Losses")

    vae.requires_grad_(False)
    vae.enable_slicing()
    vae.enable_tiling()
    
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    unet.requires_grad_(False)
    if args.mix_pretrained_path is not None:
        mix.requires_grad_(False)
    else:
        mix.requires_grad_(True)    
        
    # === 完全冻结 ControlNet 以极致节省显存 ===
    controlnet.requires_grad_(False) # 先把全部参数冻结
    
    # 由于 OOM 严重，目前完全冻结 ControlNet，仅训练 SwinIR（和可能的 Mix 模块）
    # for name, param in controlnet.named_parameters():
    #     if "mid_block" in name or "controlnet_down_blocks" in name:
    #         param.requires_grad = True
            
    logger.info("Fully frozen ControlNet: Only training SwinIR (and Mix) to aggressively save VRAM.")
    # ======================================

    # 如果 xformers 可用，则使用内存高效的注意力机制 (memory efficient attention)
    import diffusers
    if diffusers.utils.is_xformers_available():
        import xformers
        from packaging import version
        xformers_version = version.parse(xformers.__version__)
        if xformers_version == version.parse("0.0.16"):
            logger.warning(
                "xFormers 0.0.16 无法在某些 GPU 上用于训练。如果在训练期间观察到问题，请将 xFormers 更新至至少 0.0.17。详情请见 https://huggingface.co/docs/diffusers/main/en/optimization/xformers。"
            )
        unet.enable_xformers_memory_efficient_attention()
        controlnet.enable_xformers_memory_efficient_attention()
    else:
        logger.warning("xformers 不可用。请确保其已正确安装")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        controlnet.enable_gradient_checkpointing()
        logger.info("已对 unet 和 controlnet 启用梯度检查点 (以计算 SwinIR 的反向传播)。")

    # 如果 PyTorch >= 2.0 可用，尝试使用 torch.compile 进行加速
    try:
        import torch._dynamo
        if hasattr(torch, "compile"):
            logger.info("由于 DDP 冲突（DDPOptimizer 后端：发现高阶操作），暂时禁用了 torch.compile。")
            # unet = torch.compile(unet, mode="reduce-overhead", fullgraph=True)
            # controlnet = torch.compile(controlnet, mode="reduce-overhead", fullgraph=True)
            
            # 如果必须使用它，我们可以抑制错误：
            # torch._dynamo.config.suppress_errors = True
            # torch._dynamo.config.optimize_ddp = False
    except ImportError:
        logger.warning("torch.compile 不可用。请升级到 PyTorch 2.0+ 以获得更好的性能和显存优化。")

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    controlnet.to(accelerator.device, dtype=weight_dtype)
    if args.mix_pretrained_path is not None:
        mix.to(accelerator.device, dtype=weight_dtype)

    swinir.to(accelerator.device, dtype=weight_dtype)
    perceptual_loss.to(accelerator.device, dtype=weight_dtype)
    arcface_loss.to(accelerator.device, dtype=weight_dtype)

    if accelerator.is_main_process:
        log_vram("Before optimizer creation")
        
    if args.mix_pretrained_path is not None:
        params_to_optimize = filter(lambda p: p.requires_grad, itertools.chain(controlnet.parameters(), swinir.parameters()))
    else :
        params_to_optimize = filter(lambda p: p.requires_grad, itertools.chain(controlnet.parameters(), mix.parameters(), swinir.parameters()))
        
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_class = bnb.optim.AdamW8bit
            logger.info("使用 8-bit AdamW 优化器以节省显存。")
        except ImportError:
            optimizer_class = torch.optim.AdamW
            logger.warning("bitsandbytes 不可用，回退到标准的 AdamW。")
    else:
        optimizer_class = torch.optim.AdamW
        logger.info("使用标准的 AdamW 优化器。")

    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08,
    )
    
    if accelerator.is_main_process:
        log_vram("After optimizer creation")

    train_dataset = make_train_dataset(args, [tokenizer_one, tokenizer_two], [text_encoder_one, text_encoder_two])
    # 自定义批处理合并函数 (collate_fn)
    def custom_collate_fn(batch):
        gt = torch.stack([torch.tensor(item['target']) for item in batch])
        control = torch.stack([torch.tensor(item['control']) for item in batch])
        prompt_embeds = torch.stack([item['prompt_embeds'] for item in batch])
        pooled_prompt_embeds = torch.stack([item['pooled_prompt_embeds'] for item in batch])
        tokens_one = torch.stack([item['tokens_one'] for item in batch])
        add_time_ids = torch.stack([item['add_time_ids'] for item in batch])

        # 处理是否使用空提示词进行无条件生成
        if args.mix_pretrained_path is not None and random.random() > float(args.null_prompt_p):
            return dict(target=gt,control=control, prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds ,add_time_ids=add_time_ids)
        else :  
            ref_id_emb = torch.stack([item['ref_id_emb'] for item in batch])
            ref_clip_emb = torch.stack([item['ref_clip_emb'] for item in batch])
            
            random_num = random.randint(1, 4)        
            ref_id_emb = ref_id_emb[:, :random_num, :]
            ref_clip_emb = ref_clip_emb[:, :random_num, :]
            # 找到特定 token 进行特征替换/混合
            index = torch.where(tokens_one == token_id_one)[1][0]
            pref = prompt_embeds[:, :index, :]
            sufx = prompt_embeds[:, index + 1:, :]
            
            return dict(target=gt,control=control, ref_id_emb=ref_id_emb, ref_clip_emb=ref_clip_emb, pref=pref, sufx=sufx, pooled_prompt_embeds=pooled_prompt_embeds ,add_time_ids=add_time_ids)
            
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=custom_collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
    )

    # 计算训练步数及学习率调度器相关的数学运算。
    # 详细解释请查看 PR https://github.com/huggingface/diffusers/pull/8312。
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = args.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,

    )
    if accelerator.is_main_process:
        log_vram("Before accelerator.prepare")
        
    if args.mix_pretrained_path is not None:
        swinir, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            swinir, optimizer, train_dataloader, lr_scheduler
        )   
    else :   
        mix, swinir, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            mix, swinir, optimizer, train_dataloader, lr_scheduler
        )

    if accelerator.is_main_process:
        log_vram("After accelerator.prepare")

  
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    ## 注册保存与加载模型的钩子函数
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for model in models:    
                if isinstance(unwrap_model(model), ControlNetModel):
                    model.save_pretrained(os.path.join(output_dir, 'controlnet'))
                    accelerator.print("成功保存 controlnet！")
                elif isinstance(unwrap_model(model), Mix):
                    model.save_pretrained(os.path.join(output_dir, 'mix'))
                    accelerator.print("成功保存 id mix！")
                elif isinstance(unwrap_model(model), SwinIRQualityBranch):
                    torch.save(unwrap_model(model).state_dict(), os.path.join(output_dir, 'swinir.pth'))
                    accelerator.print("成功保存 swinir！")
                else:
                    raise ValueError(f"遇到未预期的保存模型类型: {model.__class__}")
                if weights:
                    weights.pop()
    def load_model_hook(models, input_dir):
        while len(models) > 0:
            model = models.pop()
            if isinstance(unwrap_model(model), ControlNetModel):
                model.from_pretrained(os.path.join(input_dir, 'controlnet'))
                accelerator.print("成功加载 controlnet！")
            elif isinstance(unwrap_model(model), Mix):
                model.from_pretrained(os.path.join(input_dir, 'mix'))
                accelerator.print("成功加载 id mix！")
            elif isinstance(unwrap_model(model), SwinIRQualityBranch):
                unwrap_model(model).load_state_dict(torch.load(os.path.join(input_dir, 'swinir.pth')))
                accelerator.print("成功加载 swinir！")
            else:
                raise ValueError(f"遇到未预期的加载模型类型: {model.__class__}")
    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # 由于训练数据加载器的大小可能已发生更改，我们需要重新计算总训练步数
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps * accelerator.num_processes:
            logger.warning(
                f"经过 'accelerator.prepare' 处理后，'train_dataloader' 的长度 ({len(train_dataloader)}) "
                f"与创建学习率调度器时的预期长度 ({len_train_dataloader_after_sharding}) 不符。"
                f"这种不一致可能会导致学习率调度器无法正常运行。"
            )
    # 随后我们重新计算训练的 epoch 数量
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.exp_name, config=tracker_config)


    # 开始训练！
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** 开始训练 *****")
    logger.info(f"  样本数量 = {len(train_dataset)}")
    logger.info(f"  Epoch 数量 = {args.num_train_epochs}")
    logger.info(f"  每台设备的瞬时批次大小 = {args.train_batch_size}")
    logger.info(f"  总训练批次大小 (包含并行、分布式和梯度累积) = {total_batch_size}")
    logger.info(f"  梯度累积步数 = {args.gradient_accumulation_steps}")
    logger.info(f"  总优化步数 = {args.max_train_steps}")
    
    if th.cuda.is_available():
        allocated = th.cuda.memory_allocated() / (1024 ** 3)
        reserved = th.cuda.memory_reserved() / (1024 ** 3)
        logger.info(f"训练循环前 GPU 显存: 已分配 {allocated:.2f} GB, 已保留 {reserved:.2f} GB")
    global_step = 0
    first_epoch = 0
    initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="训练步数",
        # 仅在每台机器的主进程上显示一次进度条。
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        epoch_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            if step == 0 and accelerator.is_main_process:
                log_vram(f"Start of Epoch {epoch}, Step 0")
                
            with accelerator.accumulate(controlnet):
                # 将图像转换为潜在空间表示（latents）
                latents = vae.encode(batch["target"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # 采样要添加到 latents 中的噪声
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
        
                # 为每张图像采样一个随机的时间步
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # 根据每个时间步的噪声幅度，将噪声添加到 latents 中（前向扩散过程）
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps).to(dtype=weight_dtype)

                # SwinIR 质量分支：从模糊图像中提取纹理并修复
                degraded_image = batch["control"].to(dtype=weight_dtype)
                controlnet_image = swinir(degraded_image).to(dtype=weight_dtype)
                
                if 'prompt_embeds' in batch.keys():
                    id_prompt_embeds = batch['prompt_embeds'].to(dtype=weight_dtype)
                else :
                    ref_clip_emb = batch['ref_clip_emb'].to(dtype=weight_dtype)
                    ref_id_emb = batch['ref_id_emb'].to(dtype=weight_dtype) 
                    pref = batch['pref'].to(dtype=weight_dtype)
                    sufx = batch['sufx'].to(dtype=weight_dtype)
    
                    mix_emb = mix(clip_emb=ref_clip_emb , id_emb=ref_id_emb)
                    id_prompt_embeds = torch.cat([pref, mix_emb, sufx] , dim=1)[:,:77,:]
                    id_prompt_embeds = id_prompt_embeds.to(dtype=weight_dtype)
                        
                unet_added_conditions = {'text_embeds':batch['pooled_prompt_embeds'].to(dtype=weight_dtype), 'time_ids': batch['add_time_ids'].to(dtype=weight_dtype)}

                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states= id_prompt_embeds ,
                    added_cond_kwargs= unet_added_conditions,
                    controlnet_cond=controlnet_image,
                    return_dict=False,
                )
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states= id_prompt_embeds,
                    added_cond_kwargs= unet_added_conditions,
                    down_block_additional_residuals=[
                        sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                    ],
                    mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                    return_dict=False,
                )[0]

                # 根据预测类型获取计算损失的目标值
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"未知的预测类型 {noise_scheduler.config.prediction_type}")
                
                # 1. 计算原始的扩散模型噪声 MSE Loss
                loss_mse = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # ----------------- 新增：反推 x0 并计算 Proposal 的 Fusion Loss -----------------
                # 2. 从预测的噪声 (model_pred) 反推当前步预测的干净潜变量 (pred_x0)
                alpha_prod_t = noise_scheduler.alphas_cumprod[timesteps].view(-1, 1, 1, 1).to(latents.device)
                beta_prod_t = 1 - alpha_prod_t

                # 公式: pred_x0 = (noisy_latents - sqrt(1 - alpha) * noise_pred) / sqrt(alpha)
                if noise_scheduler.config.prediction_type == "epsilon":
                    pred_original_sample = (noisy_latents - beta_prod_t ** (0.5) * model_pred) / alpha_prod_t ** (0.5)
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    pred_original_sample = (alpha_prod_t**0.5) * noisy_latents - (beta_prod_t**0.5) * model_pred
                
                # 3. 通过 VAE 解码得到预测的 RGB 图像 (注意显存消耗)
                pred_images = vae.decode(pred_original_sample.to(weight_dtype) / vae.config.scaling_factor).sample
                gt_images = batch["target"].to(dtype=weight_dtype)

                # 4. 计算 L1 和 Perceptual Loss (质量分支的 Loss)
                # 将 [-1, 1] 的图像转换到 [0, 1]
                pred_images_norm = (pred_images / 2 + 0.5).clamp(0, 1)
                gt_images_norm = (gt_images / 2 + 0.5).clamp(0, 1)

                loss_l1 = F.l1_loss(pred_images_norm, gt_images_norm)
                loss_perceptual = perceptual_loss(pred_images_norm, gt_images_norm)

                # 5. 计算 ArcFace Loss (身份分支的 Loss)
                loss_id = arcface_loss(pred_images_norm, gt_images_norm)

                # 6. Loss 融合 (对应 Proposal 里的 λ 权重)
                lambda_l1 = 1.0
                lambda_p = 1.0
                lambda_id = 0.5 # 对应 RQ2 中的 λ Weight
                
                # 动态权重：在 timestep 很大（噪声大）时，反推的图像很模糊，计算 ID Loss 会造成干扰
                time_weight = (1000 - timesteps.float().mean()) / 1000.0
                
                total_loss = loss_mse + lambda_l1 * loss_l1 + lambda_p * loss_perceptual + (lambda_id * time_weight) * loss_id

                accelerator.backward(total_loss)
                
                if step == 0 and accelerator.is_main_process:
                    log_vram(f"After Backward Pass Epoch {epoch}, Step 0")
                    
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(controlnet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                
                # 显式清空 CUDA 缓存，防止显存碎片堆积
                if step % 100 == 0:
                    torch.cuda.empty_cache()

            # 检查加速器是否在后台执行了优化步骤（梯度同步完成）
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                # 添加步级别的显存日志
                if global_step % 10 == 0 and th.cuda.is_available():
                    allocated = th.cuda.memory_allocated() / (1024 ** 3)
                    reserved = th.cuda.memory_reserved() / (1024 ** 3)
                    logger.info(f"[Step {global_step}] GPU 显存: 已分配 {allocated:.2f} GB, 已保留 {reserved:.2f} GB")
                
                # 修复卡死问题：所有进程必须同步等待保存操作
                # 在 DeepSpeed 环境下，save_state() 是一个集体通信操作（Collective Operation），
                # 所有的 GPU 进程都必须同时调用 accelerator.save_state() 才能把分散的显存权重合并收集起来。
                # 绝对不能把它包裹在 `if accelerator.is_main_process:` 里面，否则其他卡就会在旁边死等。
                if global_step % args.checkpoint_steps == 0:
                    accelerator.wait_for_everyone() # 确保所有卡都跑到这一步
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    
                    # ⚠️ DeepSpeed 下所有卡都必须参与 save_state 才能拼出完整的权重
                    accelerator.save_state(save_path)
                    
                    if accelerator.is_main_process:
                        logger.info(f"Saved state to {save_path}")
                    accelerator.wait_for_everyone() # 确保保存完大家再一起往下跑

            logs = {
                "loss": total_loss.detach().item(), 
                "mse": loss_mse.detach().item(),
                "l1": loss_l1.detach().item(),
                "perceptual": loss_perceptual.detach().item(),
                "id": loss_id.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0]
            }
            epoch_loss += total_loss.detach().item()
            
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
            
        epoch_loss /= len(train_dataloader)
        accelerator.log({"epoch_loss": epoch_loss}, step=global_step)
    accelerator.wait_for_everyone()
    accelerator.end_training()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, )
    parser.add_argument("--mix_pretrained_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./result")
    parser.add_argument("--resolution", type=int, default=512,)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=200)
    parser.add_argument("--max_train_steps", type=int, default=None) 
    parser.add_argument("--checkpoint_steps", type=int, default=50000) 
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--lr_scheduler", type=str, default="constant",)    
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--mixed_precision", type=str, default="fp16")
    parser.add_argument("--seed", type=str, default=233)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--train_data_dir", type=str, )
    parser.add_argument("--null_prompt_p", type=float, default=0.5)
    parser.add_argument("--exp_name", type=str, default="faceme")
    parser.add_argument("--max_train_samples", type=int, default=None, help="限制训练样本数量，用于快速测试")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="是否启用梯度检查点以节省显存")
    parser.add_argument("--use_8bit_adam", action="store_true", help="是否使用 8-bit Adam 优化器以节省显存")

    
    args = parser.parse_args()
    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."
        )
    
    main(args)