./run_train.sh 
Starting FaceMe2 Optimized Training...
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
Downloading: "https://download.pytorch.org/models/vgg16-397923af.pth" to /data/weijianghong/.cache/torch/hub/checkpoints/vgg16-397923af.pth
Downloading: "https://download.pytorch.org/models/vgg16-397923af.pth" to /data/weijianghong/.cache/torch/hub/checkpoints/vgg16-397923af.pth
  1%|▋                                                                                    | 4.62M/528M [00:00<00:43, 12.5MB/s]Downloading: "https://download.pytorch.org/models/vgg16-397923af.pth" to /data/weijianghong/.cache/torch/hub/checkpoints/vgg16-397923af.pth
  1%|█▎                                                                                   | 7.88M/528M [00:00<00:28, 19.2MB/s]Downloading: "https://download.pytorch.org/models/vgg16-397923af.pth" to /data/weijianghong/.cache/torch/hub/checkpoints/vgg16-397923af.pth
  0%|                                                                                       | 128k/528M [00:00<13:01, 708kB/s]Downloading: "https://download.pytorch.org/models/vgg16-397923af.pth" to /data/weijianghong/.cache/torch/hub/checkpoints/vgg16-397923af.pth
100%|██████████████████████████████████████████████████████████████████████████████████████| 528M/528M [00:38<00:00, 14.4MB/s]
 94%|████████████████████████████████████████████████████████████████████████████████▌     | 494M/528M [00:37<00:02, 17.2MB/s]Downloading: "https://download.pytorch.org/models/resnet18-f37072fd.pth" to /data/weijianghong/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth
100%|██████████████████████████████████████████████████████████████████████████████████████| 528M/528M [00:38<00:00, 14.2MB/s]
 97%|███████████████████████████████████████████████████████████████████████████████████▋  | 513M/528M [00:38<00:00, 28.5MB/s]Downloading: "https://download.pytorch.org/models/resnet18-f37072fd.pth" to /data/weijianghong/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth
100%|██████████████████████████████████████████████████████████████████████████████████████| 528M/528M [00:39<00:00, 14.0MB/s]
100%|██████████████████████████████████████████████████████████████████████████████████████| 528M/528M [00:39<00:00, 13.9MB/s]
 96%|██████████████████████████████████████████████████████████████████████████████████▊   | 508M/528M [00:39<00:00, 20.9MB/s]Downloading: "https://download.pytorch.org/models/resnet18-f37072fd.pth" to /data/weijianghong/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth
100%|████████████████████████████████████████████████████████████████████████████████████| 44.7M/44.7M [00:02<00:00, 20.5MB/s]
100%|██████████████████████████████████████████████████████████████████████████████████████| 528M/528M [00:40<00:00, 13.7MB/s]
100%|████████████████████████████████████████████████████████████████████████████████████| 44.7M/44.7M [00:02<00:00, 20.8MB/s]
100%|████████████████████████████████████████████████████████████████████████████████████| 44.7M/44.7M [00:01<00:00, 29.0MB/s]
wandb: [wandb.login()] Loaded credentials for https://api.wandb.ai from /home/weijianghong/.netrc.
wandb: Currently logged in as: weijianghong007 (weijianghong007-) to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.25.1
wandb: Run data is saved locally in /data/weijianghong/workspace/faceme2/wandb/run-20260408_103615-3sg5y54v
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run robust-totem-1
wandb: ⭐️ View project at https://wandb.ai/weijianghong007-/faceme2_dual_branch
wandb: 🚀 View run at https://wandb.ai/weijianghong007-/faceme2_dual_branch/runs/3sg5y54v
wandb: Detected [huggingface_hub.inference] in use.
wandb: Use W&B Weave for improved LLM call tracing. Install Weave with `pip install weave` then add `import weave` to the top of your script.
wandb: For more information, check out the docs at: https://weave-docs.wandb.ai/
训练步数:   0%|                                                                                     | 0/10000 [00:00<?, ?it/s][ WARN:0@14.967] global loadsave.cpp:1671 imencodeWithMetadata Unsupported depth image for selected encoder is fallbacked to CV_8U.
[ WARN:0@14.968] global loadsave.cpp:1671 imencodeWithMetadata Unsupported depth image for selected encoder is fallbacked to CV_8U.
[ WARN:0@15.010] global loadsave.cpp:1671 imencodeWithMetadata Unsupported depth image for selected encoder is fallbacked to CV_8U.
[ WARN:0@15.012] global loadsave.cpp:1671 imencodeWithMetadata Unsupported depth image for selected encoder is fallbacked to CV_8U.
[ WARN:0@14.977] global loadsave.cpp:1671 imencodeWithMetadata Unsupported depth image for selected encoder is fallbacked to CV_8U.
[ WARN:0@15.018] global loadsave.cpp:1671 imencodeWithMetadata Unsupported depth image for selected encoder is fallbacked to CV_8U.
[ WARN:0@15.019] global loadsave.cpp:1671 imencodeWithMetadata Unsupported depth image for selected encoder is fallbacked to CV_8U.
[ WARN:0@15.022] global loadsave.cpp:1671 imencodeWithMetadata Unsupported depth image for selected encoder is fallbacked to CV_8U.
[ WARN:0@15.022] global loadsave.cpp:1671 imencodeWithMetadata Unsupported depth image for selected encoder is fallbacked to CV_8U.
[ WARN:0@15.028] global loadsave.cpp:1671 imencodeWithMetadata Unsupported depth image for selected encoder is fallbacked to CV_8U.
[ WARN:0@15.039] global loadsave.cpp:1671 imencodeWithMetadata Unsupported depth image for selected encoder is fallbacked to CV_8U.
[ WARN:0@15.040] global loadsave.cpp:1671 imencodeWithMetadata Unsupported depth image for selected encoder is fallbacked to CV_8U.
[ WARN:0@15.041] global loadsave.cpp:1671 imencodeWithMetadata Unsupported depth image for selected encoder is fallbacked to CV_8U.
[ WARN:0@15.003] global loadsave.cpp:1671 imencodeWithMetadata Unsupported depth image for selected encoder is fallbacked to CV_8U.
[ WARN:0@15.043] global loadsave.cpp:1671 imencodeWithMetadata Unsupported depth image for selected encoder is fallbacked to CV_8U.
[ WARN:0@15.045] global loadsave.cpp:1671 imencodeWithMetadata Unsupported depth image for selected encoder is fallbacked to CV_8U.
[ WARN:0@15.051] global loadsave.cpp:1671 imencodeWithMetadata Unsupported depth image for selected encoder is fallbacked to CV_8U.
[ WARN:0@15.060] global loadsave.cpp:1671 imencodeWithMetadata Unsupported depth image for selected encoder is fallbacked to CV_8U.
[ WARN:0@15.066] global loadsave.cpp:1671 imencodeWithMetadata Unsupported depth image for selected encoder is fallbacked to CV_8U.
[ WARN:0@15.083] global loadsave.cpp:1671 imencodeWithMetadata Unsupported depth image for selected encoder is fallbacked to CV_8U.
[rank1]: Traceback (most recent call last):
[rank1]:   File "/data/weijianghong/workspace/faceme2/train.py", line 596, in <module>
[rank1]:     main(args)
[rank1]:   File "/data/weijianghong/workspace/faceme2/train.py", line 486, in main
[rank1]:     pred_images = vae.decode(pred_original_sample / vae.config.scaling_factor).sample
[rank1]:   File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/diffusers/utils/accelerate_utils.py", line 46, in wrapper
[rank1]:     return method(self, *args, **kwargs)
[rank1]:   File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/diffusers/models/autoencoders/autoencoder_kl.py", line 326, in decode
[rank1]:     decoded = self._decode(z).sample
[rank1]:   File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/diffusers/models/autoencoders/autoencoder_kl.py", line 295, in _decode
[rank1]:     z = self.post_quant_conv(z)
[rank1]:   File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
[rank1]:     return self._call_impl(*args, **kwargs)
[rank1]:   File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
[rank1]:     return forward_call(*args, **kwargs)
[rank1]:   File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/conv.py", line 554, in forward
[rank1]:     return self._conv_forward(input, self.weight, self.bias)
[rank1]:   File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/conv.py", line 549, in _conv_forward
[rank1]:     return F.conv2d(
[rank1]: RuntimeError: Input type (float) and bias type (c10::Half) should be the same
[rank4]: Traceback (most recent call last):
[rank4]:   File "/data/weijianghong/workspace/faceme2/train.py", line 596, in <module>
[rank4]:     main(args)
[rank4]:   File "/data/weijianghong/workspace/faceme2/train.py", line 486, in main
[rank4]:     pred_images = vae.decode(pred_original_sample / vae.config.scaling_factor).sample
[rank4]:   File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/diffusers/utils/accelerate_utils.py", line 46, in wrapper
[rank4]:     return method(self, *args, **kwargs)
[rank4]:   File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/diffusers/models/autoencoders/autoencoder_kl.py", line 326, in decode
[rank4]:     decoded = self._decode(z).sample
[rank4]:   File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/diffusers/models/autoencoders/autoencoder_kl.py", line 295, in _decode
[rank4]:     z = self.post_quant_conv(z)
[rank4]:   File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
[rank4]:     return self._call_impl(*args, **kwargs)
[rank4]:   File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
[rank4]:     return forward_call(*args, **kwargs)
[rank4]:   File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/conv.py", line 554, in forward
[rank4]:     return self._conv_forward(input, self.weight, self.bias)
[rank4]:   File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/conv.py", line 549, in _conv_forward
[rank4]:     return F.conv2d(
[rank4]: RuntimeError: Input type (float) and bias type (c10::Half) should be the same
[rank2]: Traceback (most recent call last):
[rank2]:   File "/data/weijianghong/workspace/faceme2/train.py", line 596, in <module>
[rank2]:     main(args)
[rank2]:   File "/data/weijianghong/workspace/faceme2/train.py", line 486, in main
[rank2]:     pred_images = vae.decode(pred_original_sample / vae.config.scaling_factor).sample
[rank2]:   File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/diffusers/utils/accelerate_utils.py", line 46, in wrapper
[rank2]:     return method(self, *args, **kwargs)
[rank2]:   File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/diffusers/models/autoencoders/autoencoder_kl.py", line 326, in decode
[rank2]:     decoded = self._decode(z).sample
[rank2]:   File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/diffusers/models/autoencoders/autoencoder_kl.py", line 295, in _decode
[rank2]:     z = self.post_quant_conv(z)
[rank2]:   File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
[rank2]:     return self._call_impl(*args, **kwargs)
[rank2]:   File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
[rank2]:     return forward_call(*args, **kwargs)
[rank2]:   File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/conv.py", line 554, in forward
[rank2]:     return self._conv_forward(input, self.weight, self.bias)
[rank2]:   File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/conv.py", line 549, in _conv_forward
[rank2]:     return F.conv2d(
[rank2]: RuntimeError: Input type (float) and bias type (c10::Half) should be the same
[rank3]: Traceback (most recent call last):
[rank3]:   File "/data/weijianghong/workspace/faceme2/train.py", line 596, in <module>
[rank3]:     main(args)
[rank3]:   File "/data/weijianghong/workspace/faceme2/train.py", line 486, in main
[rank3]:     pred_images = vae.decode(pred_original_sample / vae.config.scaling_factor).sample
[rank3]:   File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/diffusers/utils/accelerate_utils.py", line 46, in wrapper
[rank3]:     return method(self, *args, **kwargs)
[rank3]:   File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/diffusers/models/autoencoders/autoencoder_kl.py", line 326, in decode
[rank3]:     decoded = self._decode(z).sample
[rank3]:   File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/diffusers/models/autoencoders/autoencoder_kl.py", line 295, in _decode
[rank3]:     z = self.post_quant_conv(z)
[rank3]:   File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
[rank3]:     return self._call_impl(*args, **kwargs)
[rank3]:   File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
[rank3]:     return forward_call(*args, **kwargs)
[rank3]:   File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/conv.py", line 554, in forward
[rank3]:     return self._conv_forward(input, self.weight, self.bias)
[rank3]:   File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/conv.py", line 549, in _conv_forward
[rank3]:     return F.conv2d(
[rank3]: RuntimeError: Input type (float) and bias type (c10::Half) should be the same
Traceback (most recent call last):
  File "/data/weijianghong/workspace/faceme2/train.py", line 596, in <module>
    main(args)
  File "/data/weijianghong/workspace/faceme2/train.py", line 486, in main
    pred_images = vae.decode(pred_original_sample / vae.config.scaling_factor).sample
  File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/diffusers/utils/accelerate_utils.py", line 46, in wrapper
    return method(self, *args, **kwargs)
  File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/diffusers/models/autoencoders/autoencoder_kl.py", line 326, in decode
    decoded = self._decode(z).sample
  File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/diffusers/models/autoencoders/autoencoder_kl.py", line 295, in _decode
    z = self.post_quant_conv(z)
  File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
    return forward_call(*args, **kwargs)
  File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/conv.py", line 554, in forward
    return self._conv_forward(input, self.weight, self.bias)
  File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/conv.py", line 549, in _conv_forward
    return F.conv2d(
RuntimeError: Input type (float) and bias type (c10::Half) should be the same
[rank0]: Traceback (most recent call last):
[rank0]:   File "/data/weijianghong/workspace/faceme2/train.py", line 596, in <module>
[rank0]:     main(args)
[rank0]:   File "/data/weijianghong/workspace/faceme2/train.py", line 486, in main
[rank0]:     pred_images = vae.decode(pred_original_sample / vae.config.scaling_factor).sample
[rank0]:   File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/diffusers/utils/accelerate_utils.py", line 46, in wrapper
[rank0]:     return method(self, *args, **kwargs)
[rank0]:   File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/diffusers/models/autoencoders/autoencoder_kl.py", line 326, in decode
[rank0]:     decoded = self._decode(z).sample
[rank0]:   File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/diffusers/models/autoencoders/autoencoder_kl.py", line 295, in _decode
[rank0]:     z = self.post_quant_conv(z)
[rank0]:   File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
[rank0]:     return self._call_impl(*args, **kwargs)
[rank0]:   File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
[rank0]:     return forward_call(*args, **kwargs)
[rank0]:   File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/conv.py", line 554, in forward
[rank0]:     return self._conv_forward(input, self.weight, self.bias)
[rank0]:   File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/nn/modules/conv.py", line 549, in _conv_forward
[rank0]:     return F.conv2d(
[rank0]: RuntimeError: Input type (float) and bias type (c10::Half) should be the same
wandb: 
wandb: 🚀 View run robust-totem-1 at: https://wandb.ai/weijianghong007-/faceme2_dual_branch/runs/3sg5y54v
wandb: Find logs at: wandb/run-20260408_103615-3sg5y54v/logs
W0408 10:36:26.015000 3327702 site-packages/torch/distributed/elastic/multiprocessing/api.py:897] Sending process 3327820 closing signal SIGTERM
W0408 10:36:26.017000 3327702 site-packages/torch/distributed/elastic/multiprocessing/api.py:897] Sending process 3327822 closing signal SIGTERM
W0408 10:36:26.018000 3327702 site-packages/torch/distributed/elastic/multiprocessing/api.py:897] Sending process 3327823 closing signal SIGTERM
W0408 10:36:26.018000 3327702 site-packages/torch/distributed/elastic/multiprocessing/api.py:897] Sending process 3327824 closing signal SIGTERM
E0408 10:36:26.638000 3327702 site-packages/torch/distributed/elastic/multiprocessing/api.py:869] failed (exitcode: 1) local_rank: 1 (pid: 3327821) of binary: /data/weijianghong/envs/wei310/bin/python3.10
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
  File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/distributed/launcher/api.py", line 138, in __call__
    return launch_agent(self._config, self._entrypoint, list(args))
  File "/data/weijianghong/envs/wei310/lib/python3.10/site-packages/torch/distributed/launcher/api.py", line 269, in launch_agent
    raise ChildFailedError(
torch.distributed.elastic.multiprocessing.errors.ChildFailedError: 
============================================================
train.py FAILED
------------------------------------------------------------
Failures:
  <NO_OTHER_FAILURES>
------------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2026-04-08_10:36:26
  host      : gpu05
  rank      : 1 (local_rank: 1)
  exitcode  : 1 (pid: 3327821)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
============================================================
Training command executed.