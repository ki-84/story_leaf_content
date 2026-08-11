#!/bin/bash
# Skyport キャスト LoRA 学習 (musubi-tuner / Qwen-Image)
# 前提: dataset/skyport/ に選別済み画像+キャプション、単一ファイル重みダウンロード済み
# 手順: cache_latents -> cache_te -> train の3段。個別に再実行可能。
set -eo pipefail
source /home/kuwabara/venvs/musubi/bin/activate
cd /home/kuwabara/devwork/musubi-tuner

HUB=~/.cache/huggingface/hub
DIT=$(ls $HUB/models--Comfy-Org--Qwen-Image_ComfyUI/snapshots/*/split_files/diffusion_models/qwen_image_bf16.safetensors | head -1)
TE=$(ls $HUB/models--Comfy-Org--Qwen-Image_ComfyUI/snapshots/*/split_files/text_encoders/qwen_2.5_vl_7b.safetensors | head -1)
VAE=$(ls $HUB/models--Comfy-Org--Qwen-Image_ComfyUI/snapshots/*/split_files/vae/qwen_image_vae.safetensors | head -1)
CFG=/home/kuwabara/devwork/story_leaf_content/tools/lora/skyport_dataset.toml
OUT=/home/kuwabara/devwork/story_leaf_content/tools/lora/output

step=${1:-all}

if [ "$step" = "cache" ] || [ "$step" = "all" ]; then
  python src/musubi_tuner/qwen_image_cache_latents.py \
    --dataset_config "$CFG" --vae "$VAE" --model_version original
  python src/musubi_tuner/qwen_image_cache_text_encoder_outputs.py \
    --dataset_config "$CFG" --text_encoder "$TE" --batch_size 1 --model_version original
fi

if [ "$step" = "train" ] || [ "$step" = "all" ]; then
  accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 \
    src/musubi_tuner/qwen_image_train_network.py \
    --dit "$DIT" --vae "$VAE" --text_encoder "$TE" \
    --model_version original \
    --dataset_config "$CFG" \
    --sdpa --mixed_precision bf16 \
    --fp8_base --fp8_scaled \
    --timestep_sampling qwen_shift --weighting_scheme none \
    --optimizer_type adamw8bit --learning_rate 5e-5 \
    --gradient_checkpointing \
    --max_data_loader_n_workers 2 --persistent_data_loader_workers \
    --network_module networks.lora_qwen_image --network_dim 16 \
    --max_train_epochs 10 --save_every_n_epochs 2 --seed 42 \
    --output_dir "$OUT" --output_name skyport-cast-v1
fi
