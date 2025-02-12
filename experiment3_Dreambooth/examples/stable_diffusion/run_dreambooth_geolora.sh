export MODEL_NAME="CompVis/stable-diffusion-v1-4" 
export INSTANCE_DIR="/dog-example"  ### update to your paths
export CLASS_DIR="/classes"
export OUTPUT_DIR="/tdlrt"

python3 -u examples/stable_diffusion/train_dreambooth_geolora.py \
 --pretrained_model_name_or_path=$MODEL_NAME  \
 --instance_data_dir=$INSTANCE_DIR \
 --class_data_dir=$CLASS_DIR \
 --output_dir=$OUTPUT_DIR \
 --train_text_encoder \
 --num_train_epochs 5 \
 --with_prior_preservation --prior_loss_weight=1.0 \
 --instance_prompt="A photo of a golden retriever. The dog catches a tennis ball, while it is jumping" \
 --class_prompt="A photo of a golden retriever. The dog catches a tennis ball, while it is jumping" \
 --resolution=512 \
 --train_batch_size=1 \
 --lr_scheduler="constant" \
 --lr_warmup_steps=0 \
 --num_class_images=200 \
  lora \
 --unet_r 8 \
