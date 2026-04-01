python infer.py \
 --pretrained_model_name_or_path "SG161222/RealVisXL_V3.0" \ # Specify the pretrained model path
--mix_path "weights/mix" \ # Provide the path to the Mix‘s checkpoint
--controlnet_model_name_or_path "weights/controlnet" \ # Provide the path to the ControlNet's checkpoint
--input_dir "data/lq" \ # Directory containing low-quality input images
--ref_dir "data/ref" \ # Directory containing high-quality reference images
--result_dir "results" \ # Directory to save the resulting outputs
--color_correction \ # Apply color correction to the outputs
--seed=233 # Set a seed for reproducibility

python infer.py --pretrained_model_name_or_path "SG161222/RealVisXL_V3.0" --mix_path "weights/mix" --controlnet_model_name_or_path "weights/controlnet" --input_dir "data/lq" --ref_dir "data/ref" --result_dir "results" --color_correction --seed=233

python utils/preprocess.py \
 --input_dir "data/train" \
 --id_emb_save_dir "output/id_emb/" \
 --clip_emb_save_dir "output/clip_emb/" \
 --dataset_name ['FFHQ'/'FFHQRef']

python utils/preprocess.py --input_dir "data/train/FFHQRef/" --id_emb_save_dir "output/id_emb/" --clip_emb_save_dir "output/clip_emb/" --dataset_name 'FFHQRef'

python utils/preprocess.py --input_dir "data/train/FFHQ/" --id_emb_save_dir "output/id_emb/" --clip_emb_save_dir "output/clip_emb/" --dataset_name 'FFHQ'

python utils/create_train_json.py
--ffhq_dir 'data/train/FFHQ/'
--ffhq_emb_dir 'output/id_emb/'
--ffhqref_emb_dir 'output/clip_emb/'
--save_dir 'output/train_json/'

python utils/create_train_json.py --ffhq_dir 'data/train/FFHQ/' --ffhq_emb_dir 'output/id_emb/' --ffhqref_emb_dir 'output/clip_emb/' --save_dir 'output/train_json/'

用第八张卡、训练1000个数据集(单卡6个小时)、5w个steps，stage1
CUDA_VISIBLE_DEVICES=7 python train.py --pretrained_model_name_or_path "SG161222/RealVisXL_V3.0" --mix_pretrained_path None --output_dir "./output/train_results" --train_data_dir "output/train_json/train.json" --resolution 512 --report_to "wandb" --learning_rate 5e-5 --train_batch_size 2 --mixed_precision fp16 --num_workers 0 --gradient_accumulation_steps 1 --num_train_epochs 100 --checkpoint_steps 10000 --max_train_samples 1000

用第八张卡、训练1000个数据集(单卡6个小时)、5w个steps，stage2，用的是下载的mix.safetensors。
CUDA_VISIBLE_DEVICES=7 python train.py --pretrained_model_name_or_path "SG161222/RealVisXL_V3.0" --mix_pretrained_path "./weights/mix" --output_dir "./output/train_results" --train_data_dir "output/train_json/train.json" --resolution 512 --report_to "wandb" --learning_rate 5e-5 --train_batch_size 2 --mixed_precision fp16 --num_workers 0 --gradient_accumulation_steps 1 --num_train_epochs 100 --checkpoint_steps 10000 --max_train_samples 1000

用所有卡，训练5000个数据集，resolution改为1024
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes=8 train.py --pretrained_model_name_or_path "SG161222/RealVisXL_V3.0" --mix_pretrained_path "./weights/mix" --output_dir "./output/train_results" --train_data_dir "output/train_json/train.json" --resolution 1024 --report_to "wandb" --learning_rate 1e-5 --train_batch_size 2 --gradient_accumulation_steps 1 --lr_scheduler cosine --lr_warmup_steps 500 --num_workers 4 --mixed_precision fp16 --num_train_epochs 20 --max_train_samples 5000
