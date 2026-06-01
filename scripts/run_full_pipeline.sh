#!/bin/bash
# run_full_pipeline.sh — End-to-end pipeline runner

set -e
echo "🛒 FMCG CV Pipeline — Full Run"
echo "================================"

# Step 1: Download images
echo -e "\n[1/4] Downloading product images..."
python src/data_collection.py \
    --images_per_class 10 \
    --output_dir data/raw \
    --delay 0.5

# Step 2: Semi-automatic annotation
echo -e "\n[2/4] Running semi-automatic annotation..."
python src/annotation_tool.py \
    --image_dir data/raw \
    --output_csv data/annotations.csv \
    --confidence_threshold 0.85 \
    --active_learning_budget 20 \
    --clip_model ViT-L-14 \
    --no_ui  # Remove this flag to use the Gradio UI

# Step 3: Train
echo -e "\n[3/4] Training model..."
python src/train.py --config configs/config.yaml

# Step 4: Evaluate
echo -e "\n[4/4] Evaluating model..."
python src/evaluate.py \
    --checkpoint checkpoints/final_model.pt \
    --output_dir results/

echo -e "\n✅ Pipeline complete! Check results/ for metrics and plots."
