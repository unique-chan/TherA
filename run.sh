python infer_example_guided.py \
  --mode two-image \
  --reference-image example/fig1.jpg \
  --input-image example/fig2.jpg \
  --output preds/scene_tir.png \
  --llava-base-path weights/llava-v1.5-7b \
  --llava-lora-path weights/TherA_VLM \
  --checkpoint weights  \
  --llava-device cuda:1

