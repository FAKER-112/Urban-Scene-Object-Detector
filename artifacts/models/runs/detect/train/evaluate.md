# 🧠 Model Evaluation Report

**Generated:** 2025-10-23 11:14:45

**Evaluation Source:** `artifacts/models/runs/detect/train`

## 📊 Final Epoch Metrics
- **epoch**: 100.0
- **time**: 2114.61
- **train/box_loss**: 0.8926
- **train/cls_loss**: 0.6516
- **train/dfl_loss**: 1.0268
- **metrics/precision(B)**: 0.5433
- **metrics/recall(B)**: 0.4171
- **metrics/mAP50(B)**: 0.4411
- **metrics/mAP50-95(B)**: 0.2628
- **val/box_loss**: 1.3411
- **val/cls_loss**: 1.1447
- **val/dfl_loss**: 1.3248
- **lr/pg0**: 0.0
- **lr/pg1**: 0.0
- **lr/pg2**: 0.0

## 🖼️ Visual Results
**Confusion Matrix:**

![Confusion Matrix](\confusion_matrix.png)

**Validation Sample Predictions:**

![val_batch0_labels.jpg](\val_batch0_labels.jpg)

![val_batch0_pred.jpg](\val_batch0_pred.jpg)

![val_batch1_labels.jpg](\val_batch1_labels.jpg)

![val_batch1_pred.jpg](\val_batch1_pred.jpg)

![val_batch2_labels.jpg](\val_batch2_labels.jpg)

![val_batch2_pred.jpg](\val_batch2_pred.jpg)


## 🔍 Insights
- Model performance measured using mAP@[.5:.95] and mAP@.5.
- Use confusion matrix to identify misclassifications.
- Check PR curve for class-wise precision/recall balance.

## 💡 Suggested Improvements
- Increase dataset diversity or use class-balanced sampling.
- Experiment with longer training or image augmentations.
- Fine-tune on small-object subsets for better pedestrian detection.
