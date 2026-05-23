# ISIC 2019 training data

## 1. Download

Official links (ISIC Challenge 2019): https://challenge.isic-archive.com/data/#2019

| File | Size | URL |
|------|------|-----|
| Training images (zip) | ~9.1 GB | https://isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Training_Input.zip |
| Ground truth (csv) | ~1 MB | https://isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Training_GroundTruth.csv |
| Metadata (csv) | ~1 MB | https://isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Training_Metadata.csv |

Or download the three files from the website into `raw/` (do **not** use old `isic-challenge-2019` S3 URLs — they return Access Denied).

## 2. Unzip into `raw/`

```
data/isic2019/raw/
  ISIC_2019_Training_Input/
  ISIC_2019_Training_GroundTruth.csv
  ISIC_2019_Training_Metadata.csv
```

## 3. After the zip finishes (~9.1 GB)

**Easy way (one command after download):**

```bash
cd /Users/mac/Desktop/htmlcss/derma-rag
source venv/bin/activate
./scripts/run_after_isic_download.sh
```

**Or step by step — prepare folders (symlinks — saves disk):**

```bash
source venv/bin/activate
unzip -q -o data/isic2019/raw/ISIC_2019_Training_Input.zip -d data/isic2019/raw
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

---

## Train on Google Colab (GPU — recommended)

Training on a Mac CPU takes many hours. Use the notebook:

**`notebooks/Colab_Train_Skin_Classifier_v3.ipynb`**

1. Upload the notebook to [Google Colab](https://colab.research.google.com/) (File → Upload notebook).
2. **Runtime → Change runtime type → GPU**.
3. Put `models/skin_classifier_v2.pth` on Google Drive (or upload in the notebook).
4. Run all cells → download `skin_classifier_v3.pth` to your Mac → place in `models/`.
5. Restart the backend and test AK photos.
