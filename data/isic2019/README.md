# ISIC 2019 training data

## 1. Download

1. Go to https://challenge.isic-archive.com/data/#2019  
2. Download **Training**:
   - Training Input (images)
   - Training Ground Truth (CSV)
   - Training Metadata (CSV)

## 2. Unzip into `raw/`

```
data/isic2019/raw/
  ISIC_2019_Training_Input/
  ISIC_2019_Training_GroundTruth.csv
  ISIC_2019_Training_Metadata.csv
```

## 3. Prepare folders (symlinks — saves disk)

```bash
source venv/bin/activate
python scripts/prepare_isic2019.py
```

Creates `splits/train/MEL/`, `splits/val/AK/`, etc. (patient-level split by `lesion_id`).

## 4. Train v3

```bash
python scripts/train_skin_classifier.py
```

Saves `models/skin_classifier_v3.pth`. The app uses v3 automatically when that file exists.

## 5. Test

```bash
python scripts/eval_classifier.py
```

Add your own AK photos under `data/eval/AK/` to track the fix.
