# Run Instructions

## 1. Clone Repository and Install Dependencies

```bash
git clone https://github.com/sonali123321/fmcg-cv-solution.git
cd fmcg-cv-solution
pip install -r requirements.txt
```

## 2. Collect Images

```bash
python src/data_collection.py \
    --categories "cola_can,chips_lays,tide_detergent,maggi_noodles,amul_butter" \
    --images_per_class 20 \
    --output_dir data/raw
```

## 3. Run Semi-Automatic Annotation

```bash
python src/annotation_tool.py \
    --image_dir data/raw \
    --output_csv data/annotations.csv \
    --confidence_threshold 0.85
```

## 4. Train the Model

```bash
python src/train.py \
    --config configs/config.yaml \
    --data_dir data/raw \
    --annotations data/annotations.csv
```

## 5. Evaluate the Model

```bash
python src/evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --test_dir data/test
```

## Optional: Run Shell Scripts

```bash
bash scripts/run_annotation.sh
bash scripts/run_training.sh
bash scripts/run_evaluation.sh
```
