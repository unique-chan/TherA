# Text-guided image translation mode (llava on-the-fly)
python infer_custom.py \
    --rgb-dir example \
    --output-dir preds_new \
    --llava-base-path weights/llava-v1.5-7b \
    --llava-lora-path weights/TherA_VLM \
    --llava-prompt "how would this RGB scene appear in long-wave thermal infrared spectrum." \
    --checkpoint weights \
    --llava-device cuda:1

# LLava cache extraction mode
python extract_cache.py \
    --image example/fig1.jpg \
    --output weights/reference_caches/SUNNY.pt \
    --llava-base-path weights/llava-v1.5-7b \
    --llava-lora-path weights/TherA_VLM \
    --llava-device cuda:1 \
    --prompt "a bright sunny daytime scene in long-wave thermal infrared spectrum."


# Text-guided image translation mode (llava off-the-fly)
python infer_custom.py \
    --rgb-dir example \
    --output-dir preds_sunny \
    --reference-cache weights/reference_caches/SUNNY.pt \
    --checkpoint weights


# Reference-guided image translation mode (llava on-the-fly)
python infer_example_guided.py \
  --mode two-image \
  --reference-image example/fig1.jpg \
  --input-image example/fig2.jpg \
  --output preds/scene_tir.png \
  --llava-base-path weights/llava-v1.5-7b \
  --llava-lora-path weights/TherA_VLM \
  --checkpoint weights  \
  --llava-device cuda:1

# 

