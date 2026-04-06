CUDA_VISIBLE_DEVICES=1,2,3,4,5 accelerate launch --num_processes=5 train.py \
 --pretrained_model_name_or_path "/data/weijianghong/workspace/faceme2/models/RealVisXL_V3.0" \
 --mix_pretrained_path "None" \
 --output_dir "./output/train_results" \
 --train_data_dir "output/train_json/train.json" \
 --resolution 512 \
 --report_to "wandb" \
 --learning_rate 5e-5 \
 --train_batch_size 1 \
 --mixed_precision fp16 \
 --num_workers 4 \
 --gradient_accumulation_steps 2 \
 --num_train_epochs 100 \
 --checkpoint_steps 1000 \
 --max_train_samples 1000
The following values were not passed to `accelerate launch` and had defaults used instead:
More than one GPU was found, enabling multi-GPU training.
If this was unintended please pass in `--num_processes=1`.
`--num_machines` was set to a value of `1`
`--mixed_precision` was set to a value of `'no'`
`--dynamo_backend` was set to a value of `'no'`
To avoid this warning pass in values for each of the problematic parameters or run `accelerate config`.
/data/weijianghong/workspace/faceme2/utils/load_photomaker.py:56: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
state_dict = torch.load(model_file, map_location="cpu")
/data/weijianghong/workspace/faceme2/utils/load_photomaker.py:56: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
state_dict = torch.load(model_file, map_location="cpu")
/data/weijianghong/workspace/faceme2/utils/load_photomaker.py:56: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
state_dict = torch.load(model_file, map_location="cpu")
/data/weijianghong/workspace/faceme2/utils/load_photomaker.py:56: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
state_dict = torch.load(model_file, map_location="cpu")
/data/weijianghong/workspace/faceme2/utils/load_photomaker.py:56: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
state_dict = torch.load(model_file, map_location="cpu")
Loading PhotoMaker components [1] id_encoder from [./models]...
Loading PhotoMaker components [1] id_encoder from [./models]...
Loading PhotoMaker components [1] id_encoder from [./models]...
Loading PhotoMaker components [1] id_encoder from [./models]...
Loading PhotoMaker components [1] id_encoder from [./models]...
wandb: [wandb.login()] Loaded credentials for https://api.wandb.ai from /home/weijianghong/.netrc.
wandb: Currently logged in as: weijianghong007 (weijianghong007-) to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.25.1
wandb: Run data is saved locally in /data/weijianghong/workspace/faceme2/wandb/run-20260406_103527-ylinaht1
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run dax-tribble-15
wandb: ⭐️ View project at https://wandb.ai/weijianghong007-/faceme
wandb: 🚀 View run at https://wandb.ai/weijianghong007-/faceme/runs/ylinaht1
wandb: Detected [huggingface_hub.inference] in use.
wandb: Use W&B Weave for improved LLM call tracing. Install Weave with `pip install weave` then add `import weave` to the top of your script.
wandb: For more information, check out the docs at: https://weave-docs.wandb.ai/
Steps: 0%| | 0/10000 [00:00<?, ?it/s][ WARN:0@9.503] global loadsave.cpp:1671 imencodeWithMetadata Unsupported depth image for selected encoder is fallbacked to CV_8U.
[ WARN:0@9.503] global loadsave.cpp:1671 imencodeWithMetadata Unsupported depth image for selected encoder is fallbacked to CV_8U.
[ WARN:0@9.510] global loadsave.cpp:1671 imencodeWithMetadata Unsupported depth image for selected encoder is fallbacked to CV_8U.
[ WARN:0@8.816] global loadsave.cpp:1671 imencodeWithMetadata Unsupported depth image for selected encoder is fallbacked to CV_8U.
[ WARN:0@9.511] global loadsave.cpp:1671 imencodeWithMetadata Unsupported depth image for selected encoder is fallbacked to CV_8U.
[ WARN:0@8.816] global loadsave.cpp:1671 imencodeWithMetadata Unsupported depth image for selected encoder is fallbacked to CV_8U.
[ WARN:0@8.820] global loadsave.cpp:1671 imencodeWithMetadata Unsupported depth image for selected encoder is fallbacked to CV_8U.
[ WARN:0@9.598] global loadsave.cpp:1671 imencodeWithMetadata Unsupported depth image for selected encoder is fallbacked to CV_8U.
[ WARN:0@9.504] global loadsave.cpp:1671 imencodeWithMetadata Unsupported depth image for selected encoder is fallbacked to CV_8U.
[ WARN:0@9.512] global loadsave.cpp:1671 imencodeWithMetadata Unsupported depth image for selected encoder is fallbacked to CV_8U.
[ WARN:0@9.506] global loadsave.cpp:1671 imencodeWithMetadata Unsupported depth image for selected encoder is fallbacked to CV_8U.
[ WARN:0@8.819] global loadsave.cpp:1671 imencodeWithMetadata Unsupported depth image for selected encoder is fallbacked to CV_8U.
[ WARN:0@9.602] global loadsave.cpp:1671 imencodeWithMetadata Unsupported depth image for selected encoder is fallbacked to CV_8U.
[ WARN:0@9.602] global loadsave.cpp:1671 imencodeWithMetadata Unsupported depth image for selected encoder is fallbacked to CV_8U.
[ WARN:0@8.824] global loadsave.cpp:1671 imencodeWithMetadata Unsupported depth image for selected encoder is fallbacked to CV_8U.
[ WARN:0@8.822] global loadsave.cpp:1671 imencodeWithMetadata Unsupported depth image for selected encoder is fallbacked to CV_8U.
[ WARN:0@9.517] global loadsave.cpp:1671 imencodeWithMetadata Unsupported depth image for selected encoder is fallbacked to CV_8U.
[ WARN:0@8.828] global loadsave.cpp:1671 imencodeWithMetadata Unsupported depth image for selected encoder is fallbacked to CV_8U.
[ WARN:0@9.607] global loadsave.cpp:1671 imencodeWithMetadata Unsupported depth image for selected encoder is fallbacked to CV_8U.
[ WARN:0@8.859] global loadsave.cpp:1671 imencodeWithMetadata Unsupported depth image for selected encoder is fallbacked to CV_8U.
Steps: 0%| | 0/10000 [00:02<?, ?it/s, loss=0.285, lr=5e-5][rank4]: Traceback (most recent call last):
[rank4]: File "/data/weijianghong/workspace/faceme2/train.py", line 439, in <module>
[rank4]: main(args)
[rank4]: File "/data/weijianghong/workspace/faceme2/train.py", line 348, in main
[rank4]: down_block_res_samples, mid_block_res_sample = controlnet(
[rank4]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1736, in \_wrapped_call_impl
[rank4]: return self.\_call_impl(*args, \*\*kwargs)
[rank4]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1747, in \_call_impl
[rank4]: return forward_call(*args, **kwargs)
[rank4]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/parallel/distributed.py", line 1643, in forward
[rank4]: else self.\_run_ddp_forward(\*inputs, **kwargs)
[rank4]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/parallel/distributed.py", line 1459, in \_run_ddp_forward
[rank4]: return self.module(*inputs, \*\*kwargs) # type: ignore[index]
[rank4]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1736, in \_wrapped_call_impl
[rank4]: return self.\_call_impl(*args, **kwargs)
[rank4]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1747, in \_call_impl
[rank4]: return forward_call(\*args, **kwargs)
[rank4]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/accelerate/utils/operations.py", line 823, in forward
[rank4]: return model_forward(*args, \*\*kwargs)
[rank4]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/accelerate/utils/operations.py", line 811, in **call**
[rank4]: return convert_to_fp32(self.model_forward(*args, **kwargs))
[rank4]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/amp/autocast_mode.py", line 44, in decorate_autocast
[rank4]: return func(\*args, **kwargs)
[rank4]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/diffusers/models/controlnet.py", line 807, in forward
[rank4]: sample, res_samples = downsample_block(
[rank4]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1736, in \_wrapped_call_impl
[rank4]: return self.\_call_impl(*args, \*\*kwargs)
[rank4]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1747, in \_call_impl
[rank4]: return forward_call(*args, **kwargs)
[rank4]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/diffusers/models/unets/unet_2d_blocks.py", line 1288, in forward
[rank4]: hidden_states = attn(
[rank4]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1736, in \_wrapped_call_impl
[rank4]: return self.\_call_impl(\*args, **kwargs)
[rank4]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1747, in \_call_impl
[rank4]: return forward_call(*args, \*\*kwargs)
[rank4]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/diffusers/models/transformers/transformer_2d.py", line 442, in forward
[rank4]: hidden_states = block(
[rank4]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1736, in \_wrapped_call_impl
[rank4]: return self.\_call_impl(*args, **kwargs)
[rank4]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1747, in \_call_impl
[rank4]: return forward_call(\*args, **kwargs)
[rank4]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/diffusers/models/attention.py", line 545, in forward
[rank4]: attn_output = self.attn2(
[rank4]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1736, in \_wrapped_call_impl
[rank4]: return self.\_call_impl(*args, \*\*kwargs)
[rank4]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1747, in \_call_impl
[rank4]: return forward_call(*args, **kwargs)
[rank4]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/diffusers/models/attention_processor.py", line 495, in forward
[rank4]: return self.processor(
[rank4]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/diffusers/models/attention_processor.py", line 2365, in **call**
[rank4]: key = attn.to_k(encoder_hidden_states)
[rank4]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1736, in \_wrapped_call_impl
[rank4]: return self.\_call_impl(\*args, **kwargs)
[rank4]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1747, in \_call_impl
[rank4]: return forward_call(*args, \*\*kwargs)
[rank4]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/linear.py", line 125, in forward
[rank4]: return F.linear(input, self.weight, self.bias)
[rank4]: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB. GPU 4 has a total capacity of 23.56 GiB of which 15.88 MiB is free. Including non-PyTorch memory, this process has 23.54 GiB memory in use. Of the allocated memory 22.95 GiB is allocated by PyTorch, and 162.12 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation. See documentation for Memory Management (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
[rank3]: Traceback (most recent call last):
[rank3]: File "/data/weijianghong/workspace/faceme2/train.py", line 439, in <module>
[rank3]: main(args)
[rank3]: File "/data/weijianghong/workspace/faceme2/train.py", line 348, in main
[rank3]: down_block_res_samples, mid_block_res_sample = controlnet(
[rank3]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1736, in \_wrapped_call_impl
[rank3]: return self.\_call_impl(*args, **kwargs)
[rank3]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1747, in \_call_impl
[rank3]: return forward_call(\*args, **kwargs)
[rank3]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/parallel/distributed.py", line 1643, in forward
[rank3]: else self.\_run_ddp_forward(*inputs, \*\*kwargs)
[rank3]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/parallel/distributed.py", line 1459, in \_run_ddp_forward
[rank3]: return self.module(*inputs, **kwargs) # type: ignore[index]
[rank3]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1736, in \_wrapped_call_impl
[rank3]: return self.\_call_impl(\*args, **kwargs)
[rank3]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1747, in \_call_impl
[rank3]: return forward_call(*args, \*\*kwargs)
[rank3]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/accelerate/utils/operations.py", line 823, in forward
[rank3]: return model_forward(*args, **kwargs)
[rank3]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/accelerate/utils/operations.py", line 811, in **call**
[rank3]: return convert_to_fp32(self.model_forward(\*args, **kwargs))
[rank3]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/amp/autocast_mode.py", line 44, in decorate_autocast
[rank3]: return func(*args, \*\*kwargs)
[rank3]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/diffusers/models/controlnet.py", line 807, in forward
[rank3]: sample, res_samples = downsample_block(
[rank3]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1736, in \_wrapped_call_impl
[rank3]: return self.\_call_impl(*args, **kwargs)
[rank3]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1747, in \_call_impl
[rank3]: return forward_call(\*args, **kwargs)
[rank3]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/diffusers/models/unets/unet_2d_blocks.py", line 1288, in forward
[rank3]: hidden_states = attn(
[rank3]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1736, in \_wrapped_call_impl
[rank3]: return self.\_call_impl(*args, \*\*kwargs)
[rank3]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1747, in \_call_impl
[rank3]: return forward_call(*args, **kwargs)
[rank3]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/diffusers/models/transformers/transformer_2d.py", line 442, in forward
[rank3]: hidden_states = block(
[rank3]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1736, in \_wrapped_call_impl
[rank3]: return self.\_call_impl(\*args, **kwargs)
[rank3]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1747, in \_call_impl
[rank3]: return forward_call(*args, \*\*kwargs)
[rank3]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/diffusers/models/attention.py", line 545, in forward
[rank3]: attn_output = self.attn2(
[rank3]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1736, in \_wrapped_call_impl
[rank3]: return self.\_call_impl(*args, **kwargs)
[rank3]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1747, in \_call_impl
[rank3]: return forward_call(\*args, **kwargs)
[rank3]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/diffusers/models/attention_processor.py", line 495, in forward
[rank3]: return self.processor(
[rank3]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/diffusers/models/attention_processor.py", line 2365, in **call**
[rank3]: key = attn.to_k(encoder_hidden_states)
[rank3]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1736, in \_wrapped_call_impl
[rank3]: return self.\_call_impl(*args, \*\*kwargs)
[rank3]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1747, in \_call_impl
[rank3]: return forward_call(*args, **kwargs)
[rank3]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/linear.py", line 125, in forward
[rank3]: return F.linear(input, self.weight, self.bias)
[rank3]: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB. GPU 3 has a total capacity of 23.56 GiB of which 15.88 MiB is free. Including non-PyTorch memory, this process has 23.54 GiB memory in use. Of the allocated memory 22.95 GiB is allocated by PyTorch, and 162.12 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation. See documentation for Memory Management (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
[rank2]: Traceback (most recent call last):
[rank2]: File "/data/weijianghong/workspace/faceme2/train.py", line 439, in <module>
[rank2]: main(args)
[rank2]: File "/data/weijianghong/workspace/faceme2/train.py", line 348, in main
[rank2]: down_block_res_samples, mid_block_res_sample = controlnet(
[rank2]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1736, in \_wrapped_call_impl
[rank2]: return self.\_call_impl(\*args, **kwargs)
[rank2]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1747, in \_call_impl
[rank2]: return forward_call(*args, \*\*kwargs)
[rank2]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/parallel/distributed.py", line 1643, in forward
[rank2]: else self.\_run_ddp_forward(*inputs, **kwargs)
[rank2]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/parallel/distributed.py", line 1459, in \_run_ddp_forward
[rank2]: return self.module(\*inputs, **kwargs) # type: ignore[index]
[rank2]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1736, in \_wrapped_call_impl
[rank2]: return self.\_call_impl(*args, \*\*kwargs)
[rank2]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1747, in \_call_impl
[rank2]: return forward_call(*args, **kwargs)
[rank2]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/accelerate/utils/operations.py", line 823, in forward
[rank2]: return model_forward(\*args, **kwargs)
[rank2]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/accelerate/utils/operations.py", line 811, in **call**
[rank2]: return convert_to_fp32(self.model_forward(*args, \*\*kwargs))
[rank2]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/amp/autocast_mode.py", line 44, in decorate_autocast
[rank2]: return func(*args, **kwargs)
[rank2]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/diffusers/models/controlnet.py", line 807, in forward
[rank2]: sample, res_samples = downsample_block(
[rank2]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1736, in \_wrapped_call_impl
[rank2]: return self.\_call_impl(\*args, **kwargs)
[rank2]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1747, in \_call_impl
[rank2]: return forward_call(*args, \*\*kwargs)
[rank2]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/diffusers/models/unets/unet_2d_blocks.py", line 1288, in forward
[rank2]: hidden_states = attn(
[rank2]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1736, in \_wrapped_call_impl
[rank2]: return self.\_call_impl(*args, **kwargs)
[rank2]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1747, in \_call_impl
[rank2]: return forward_call(\*args, **kwargs)
[rank2]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/diffusers/models/transformers/transformer_2d.py", line 442, in forward
[rank2]: hidden_states = block(
[rank2]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1736, in \_wrapped_call_impl
[rank2]: return self.\_call_impl(*args, \*\*kwargs)
[rank2]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1747, in \_call_impl
[rank2]: return forward_call(*args, **kwargs)
[rank2]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/diffusers/models/attention.py", line 545, in forward
[rank2]: attn_output = self.attn2(
[rank2]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1736, in \_wrapped_call_impl
[rank2]: return self.\_call_impl(\*args, **kwargs)
[rank2]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1747, in \_call_impl
[rank2]: return forward_call(*args, \*\*kwargs)
[rank2]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/diffusers/models/attention_processor.py", line 495, in forward
[rank2]: return self.processor(
[rank2]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/diffusers/models/attention_processor.py", line 2365, in **call**
[rank2]: key = attn.to_k(encoder_hidden_states)
[rank2]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1736, in \_wrapped_call_impl
[rank2]: return self.\_call_impl(*args, **kwargs)
[rank2]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1747, in \_call_impl
[rank2]: return forward_call(\*args, **kwargs)
[rank2]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/linear.py", line 125, in forward
[rank2]: return F.linear(input, self.weight, self.bias)
[rank2]: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB. GPU 2 has a total capacity of 23.56 GiB of which 15.88 MiB is free. Including non-PyTorch memory, this process has 23.54 GiB memory in use. Of the allocated memory 22.95 GiB is allocated by PyTorch, and 162.12 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation. See documentation for Memory Management (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
[rank1]: Traceback (most recent call last):
[rank1]: File "/data/weijianghong/workspace/faceme2/train.py", line 439, in <module>
[rank1]: main(args)
[rank1]: File "/data/weijianghong/workspace/faceme2/train.py", line 348, in main
[rank1]: down_block_res_samples, mid_block_res_sample = controlnet(
[rank1]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1736, in \_wrapped_call_impl
[rank1]: return self.\_call_impl(*args, \*\*kwargs)
[rank1]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1747, in \_call_impl
[rank1]: return forward_call(*args, **kwargs)
[rank1]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/parallel/distributed.py", line 1643, in forward
[rank1]: else self.\_run_ddp_forward(\*inputs, **kwargs)
[rank1]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/parallel/distributed.py", line 1459, in \_run_ddp_forward
[rank1]: return self.module(*inputs, \*\*kwargs) # type: ignore[index]
[rank1]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1736, in \_wrapped_call_impl
[rank1]: return self.\_call_impl(*args, **kwargs)
[rank1]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1747, in \_call_impl
[rank1]: return forward_call(\*args, **kwargs)
[rank1]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/accelerate/utils/operations.py", line 823, in forward
[rank1]: return model_forward(*args, \*\*kwargs)
[rank1]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/accelerate/utils/operations.py", line 811, in **call**
[rank1]: return convert_to_fp32(self.model_forward(*args, **kwargs))
[rank1]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/amp/autocast_mode.py", line 44, in decorate_autocast
[rank1]: return func(\*args, **kwargs)
[rank1]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/diffusers/models/controlnet.py", line 807, in forward
[rank1]: sample, res_samples = downsample_block(
[rank1]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1736, in \_wrapped_call_impl
[rank1]: return self.\_call_impl(*args, \*\*kwargs)
[rank1]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1747, in \_call_impl
[rank1]: return forward_call(*args, **kwargs)
[rank1]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/diffusers/models/unets/unet_2d_blocks.py", line 1288, in forward
[rank1]: hidden_states = attn(
[rank1]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1736, in \_wrapped_call_impl
[rank1]: return self.\_call_impl(\*args, **kwargs)
[rank1]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1747, in \_call_impl
[rank1]: return forward_call(*args, \*\*kwargs)
[rank1]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/diffusers/models/transformers/transformer_2d.py", line 442, in forward
[rank1]: hidden_states = block(
[rank1]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1736, in \_wrapped_call_impl
[rank1]: return self.\_call_impl(*args, **kwargs)
[rank1]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1747, in \_call_impl
[rank1]: return forward_call(\*args, **kwargs)
[rank1]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/diffusers/models/attention.py", line 545, in forward
[rank1]: attn_output = self.attn2(
[rank1]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1736, in \_wrapped_call_impl
[rank1]: return self.\_call_impl(*args, \*\*kwargs)
[rank1]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1747, in \_call_impl
[rank1]: return forward_call(*args, **kwargs)
[rank1]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/diffusers/models/attention_processor.py", line 495, in forward
[rank1]: return self.processor(
[rank1]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/diffusers/models/attention_processor.py", line 2365, in **call**
[rank1]: key = attn.to_k(encoder_hidden_states)
[rank1]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1736, in \_wrapped_call_impl
[rank1]: return self.\_call_impl(\*args, **kwargs)
[rank1]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1747, in \_call_impl
[rank1]: return forward_call(*args, \*\*kwargs)
[rank1]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/linear.py", line 125, in forward
[rank1]: return F.linear(input, self.weight, self.bias)
[rank1]: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB. GPU 1 has a total capacity of 23.56 GiB of which 15.88 MiB is free. Including non-PyTorch memory, this process has 23.54 GiB memory in use. Of the allocated memory 22.95 GiB is allocated by PyTorch, and 162.12 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation. See documentation for Memory Management (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
Traceback (most recent call last):
File "/data/weijianghong/workspace/faceme2/train.py", line 439, in <module>
main(args)
File "/data/weijianghong/workspace/faceme2/train.py", line 348, in main
down_block_res_samples, mid_block_res_sample = controlnet(
File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1736, in \_wrapped_call_impl
return self.\_call_impl(*args, **kwargs)
File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1747, in \_call_impl
return forward_call(\*args, **kwargs)
File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/parallel/distributed.py", line 1643, in forward
else self.\_run_ddp_forward(*inputs, \*\*kwargs)
File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/parallel/distributed.py", line 1459, in \_run_ddp_forward
return self.module(*inputs, **kwargs) # type: ignore[index]
File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1736, in \_wrapped_call_impl
return self.\_call_impl(\*args, **kwargs)
File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1747, in \_call_impl
return forward_call(*args, \*\*kwargs)
File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/accelerate/utils/operations.py", line 823, in forward
return model_forward(*args, **kwargs)
File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/accelerate/utils/operations.py", line 811, in **call**
return convert_to_fp32(self.model_forward(\*args, **kwargs))
File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/amp/autocast_mode.py", line 44, in decorate_autocast
return func(*args, \*\*kwargs)
File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/diffusers/models/controlnet.py", line 807, in forward
sample, res_samples = downsample_block(
File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1736, in \_wrapped_call_impl
return self.\_call_impl(*args, **kwargs)
File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1747, in \_call_impl
return forward_call(\*args, **kwargs)
File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/diffusers/models/unets/unet_2d_blocks.py", line 1288, in forward
hidden_states = attn(
File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1736, in \_wrapped_call_impl
return self.\_call_impl(*args, \*\*kwargs)
File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1747, in \_call_impl
return forward_call(*args, **kwargs)
File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/diffusers/models/transformers/transformer_2d.py", line 442, in forward
hidden_states = block(
File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1736, in \_wrapped_call_impl
return self.\_call_impl(\*args, **kwargs)
File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1747, in \_call_impl
return forward_call(*args, \*\*kwargs)
File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/diffusers/models/attention.py", line 545, in forward
attn_output = self.attn2(
File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1736, in \_wrapped_call_impl
return self.\_call_impl(*args, **kwargs)
File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1747, in \_call_impl
return forward_call(\*args, **kwargs)
File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/diffusers/models/attention_processor.py", line 495, in forward
return self.processor(
File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/diffusers/models/attention_processor.py", line 2365, in **call**
key = attn.to_k(encoder_hidden_states)
File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1736, in \_wrapped_call_impl
return self.\_call_impl(*args, \*\*kwargs)
File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1747, in \_call_impl
return forward_call(*args, **kwargs)
File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/linear.py", line 125, in forward
return F.linear(input, self.weight, self.bias)
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 has a total capacity of 23.56 GiB of which 15.88 MiB is free. Including non-PyTorch memory, this process has 23.54 GiB memory in use. Of the allocated memory 22.95 GiB is allocated by PyTorch, and 162.15 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation. See documentation for Memory Management (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
[rank0]: Traceback (most recent call last):
[rank0]: File "/data/weijianghong/workspace/faceme2/train.py", line 439, in <module>
[rank0]: main(args)
[rank0]: File "/data/weijianghong/workspace/faceme2/train.py", line 348, in main
[rank0]: down_block_res_samples, mid_block_res_sample = controlnet(
[rank0]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1736, in \_wrapped_call_impl
[rank0]: return self.\_call_impl(\*args, **kwargs)
[rank0]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1747, in \_call_impl
[rank0]: return forward_call(*args, \*\*kwargs)
[rank0]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/parallel/distributed.py", line 1643, in forward
[rank0]: else self.\_run_ddp_forward(*inputs, **kwargs)
[rank0]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/parallel/distributed.py", line 1459, in \_run_ddp_forward
[rank0]: return self.module(\*inputs, **kwargs) # type: ignore[index]
[rank0]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1736, in \_wrapped_call_impl
[rank0]: return self.\_call_impl(*args, \*\*kwargs)
[rank0]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1747, in \_call_impl
[rank0]: return forward_call(*args, **kwargs)
[rank0]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/accelerate/utils/operations.py", line 823, in forward
[rank0]: return model_forward(\*args, **kwargs)
[rank0]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/accelerate/utils/operations.py", line 811, in **call**
[rank0]: return convert_to_fp32(self.model_forward(*args, \*\*kwargs))
[rank0]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/amp/autocast_mode.py", line 44, in decorate_autocast
[rank0]: return func(*args, **kwargs)
[rank0]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/diffusers/models/controlnet.py", line 807, in forward
[rank0]: sample, res_samples = downsample_block(
[rank0]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1736, in \_wrapped_call_impl
[rank0]: return self.\_call_impl(\*args, **kwargs)
[rank0]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1747, in \_call_impl
[rank0]: return forward_call(*args, \*\*kwargs)
[rank0]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/diffusers/models/unets/unet_2d_blocks.py", line 1288, in forward
[rank0]: hidden_states = attn(
[rank0]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1736, in \_wrapped_call_impl
[rank0]: return self.\_call_impl(*args, **kwargs)
[rank0]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1747, in \_call_impl
[rank0]: return forward_call(\*args, **kwargs)
[rank0]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/diffusers/models/transformers/transformer_2d.py", line 442, in forward
[rank0]: hidden_states = block(
[rank0]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1736, in \_wrapped_call_impl
[rank0]: return self.\_call_impl(*args, \*\*kwargs)
[rank0]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1747, in \_call_impl
[rank0]: return forward_call(*args, **kwargs)
[rank0]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/diffusers/models/attention.py", line 545, in forward
[rank0]: attn_output = self.attn2(
[rank0]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1736, in \_wrapped_call_impl
[rank0]: return self.\_call_impl(\*args, **kwargs)
[rank0]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1747, in \_call_impl
[rank0]: return forward_call(*args, \*\*kwargs)
[rank0]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/diffusers/models/attention_processor.py", line 495, in forward
[rank0]: return self.processor(
[rank0]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/diffusers/models/attention_processor.py", line 2365, in **call**
[rank0]: key = attn.to_k(encoder_hidden_states)
[rank0]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1736, in \_wrapped_call_impl
[rank0]: return self.\_call_impl(*args, **kwargs)
[rank0]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1747, in \_call_impl
[rank0]: return forward_call(\*args, **kwargs)
[rank0]: File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/linear.py", line 125, in forward
[rank0]: return F.linear(input, self.weight, self.bias)
[rank0]: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 has a total capacity of 23.56 GiB of which 15.88 MiB is free. Including non-PyTorch memory, this process has 23.54 GiB memory in use. Of the allocated memory 22.95 GiB is allocated by PyTorch, and 162.15 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation. See documentation for Memory Management (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
wandb:
wandb: 🚀 View run dax-tribble-15 at: https://wandb.ai/weijianghong007-/faceme/runs/ylinaht1
wandb: Find logs at: wandb/run-20260406_103527-ylinaht1/logs
W0406 10:35:35.569000 1879814 site-packages/torch/distributed/elastic/multiprocessing/api.py:897] Sending process 1879932 closing signal SIGTERM
W0406 10:35:35.570000 1879814 site-packages/torch/distributed/elastic/multiprocessing/api.py:897] Sending process 1879934 closing signal SIGTERM
W0406 10:35:35.571000 1879814 site-packages/torch/distributed/elastic/multiprocessing/api.py:897] Sending process 1879935 closing signal SIGTERM
W0406 10:35:35.571000 1879814 site-packages/torch/distributed/elastic/multiprocessing/api.py:897] Sending process 1879936 closing signal SIGTERM
E0406 10:35:36.087000 1879814 site-packages/torch/distributed/elastic/multiprocessing/api.py:869] failed (exitcode: 1) local_rank: 1 (pid: 1879933) of binary: /data/weijianghong/envs/wei310/bin/python3.10
Traceback (most recent call last):
File "/data/weijianghong/envs/wei310/bin/accelerate", line 6, in <module>
sys.exit(main())
File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/accelerate/commands/accelerate_cli.py", line 48, in main
args.func(args)
File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/accelerate/commands/launch.py", line 1159, in launch_command
multi_gpu_launcher(args)
File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/accelerate/commands/launch.py", line 793, in multi_gpu_launcher
distrib_run.run(args)
File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/distributed/run.py", line 910, in run
elastic_launch(
File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/distributed/launcher/api.py", line 138, in **call**
return launch_agent(self.\_config, self.\_entrypoint, list(args))
File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/distributed/launcher/api.py", line 269, in launch_agent
raise ChildFailedError(
torch.distributed.elastic.multiprocessing.errors.ChildFailedError:
============================================================
train.py FAILED

---

Failures:
<NO_OTHER_FAILURES>

---

Root Cause (first observed failure):
[0]:
time : 2026-04-06_10:35:35
host : gpu05
rank : 1 (local_rank: 1)
exitcode : 1 (pid: 1879933)
error_file: <N/A>
traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
============================================================
