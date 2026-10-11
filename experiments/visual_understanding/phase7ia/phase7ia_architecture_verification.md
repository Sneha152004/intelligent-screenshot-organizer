# Phase 7I-A: Architecture Verification Report (YOLO & Candidate Models)

## 1. Executive Summary

This report provides an independent architectural audit of the vision model candidates evaluated in Phase 7I, with special focus on resolving the architectural naming and parameter count ambiguity surrounding the `YOLO26n-cls` candidate.

---

## 2. YOLO Architecture Resolution & Ambiguity Fix

### Naming & Implementation Audit:
In Phase 7I artifacts, the candidate key `yolo26n_cls` was assigned the display label `"YOLO26n-cls (YOLOv8n-cls)"`.

An audit of `train_models.py` (lines 41-48) reveals:

```python
class YOLOClassificationWrapper(nn.Module):

  def __init__(self, num_classes: int = 5):
    super().__init__()
    yolo_wrapper = YOLO("yolov8n-cls.pt")
    self.backbone = copy.deepcopy(yolo_wrapper.model)
    in_features = (
        self.backbone.model[9].linear.in_features
        if hasattr(self.backbone.model[9], "linear")
        else 1280
    )
    self.backbone.model[9].linear = nn.Linear(in_features, num_classes)
```

### Key Architectural Findings:
1. **Exact Architecture Name & Version**: **`YOLOv8n-cls`** (Ultralytics YOLOv8 nano classification model).
2. **Pretrained Checkpoint Source**: `yolov8n-cls.pt` downloaded via official `ultralytics` PyTorch package.
3. **Candidate Designation**: `"YOLO26n-cls"` was a project experimental placeholder key specified in the experiment protocol. The wrapper code loaded `yolov8n-cls.pt` as the underlying backbone.
4. **Substitution Status**: No runtime failure or silent fallback occurred; `YOLOClassificationWrapper` explicitly instantiated `yolov8n-cls.pt`.
5. **Head Adaptation**: The original 1000-class ImageNet classification linear head (`self.backbone.model[9].linear`) was replaced with a 5-class linear output head `nn.Linear(in_features=1280, out_features=5)`.

---

## 3. Parameter Count Discrepancy Resolution

The reported parameter counts in `model_configs.json` and `phase7i_completion_report.md` reflected default 1000-class ImageNet estimates. Empirical PyTorch parameter counting on the actual 5-class models revealed:

| Model Candidate Key | Display Name | Config Reported Parameters | Actual 5-Class Model Parameters | Discrepancy |
| :--- | :--- | :---: | :---: | :---: |
| `mobilenet_v3_small` | MobileNetV3-Small | 2,542,856 | **1,522,981** | -1,019,875 (Classifier head adapted to 5 classes) |
| `efficientnet_lite0` | EfficientNet-Lite0 | 4,652,008 | **3,377,413** | -1,274,595 (Classifier head adapted to 5 classes) |
| `mobilevit_xs` | MobileViT-XS | 2,317,848 | **1,934,773** | -383,075 (Classifier head adapted to 5 classes) |
| `yolo26n_cls` | YOLOv8n-cls | 2,719,288 | **1,444,693** | -1,274,595 (Classifier head adapted to 5 classes) |

> [!NOTE]
> **Summary**: **YOLOv8n-cls** is actually the most parameter-efficient model candidate in the benchmark (**1,444,693 parameters**), rather than MobileNetV3-Small (1,522,981 parameters).

---

## 4. Architectural Summary Table

| Model Candidate | Backbone Base | PyTorch Module Class | Actual Parameters | Conv2d Layers | Linear Layers | Classifier Head |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **MobileNetV3-Small** | `mobilenetv3_small_100` | `timm.models.mobilenetv3` | 1,522,981 | 98.9% (1.50M) | 0.3% (5.1K) | `nn.Linear(1024, 5)` |
| **EfficientNet-Lite0** | `efficientnet_lite0` | `timm.models.efficientnet` | 3,377,413 | 98.6% (3.33M) | 0.2% (6.4K) | `nn.Linear(1280, 5)` |
| **MobileViT-XS** | `mobilevit_xs` | `timm.models.mobilevit` | 1,934,773 | 41.6% (805K) | 57.7% (1.11M) | `nn.Linear(384, 5)` |
| **YOLOv8n-cls** | `yolov8n-cls.pt` | `YOLOClassificationWrapper` | 1,444,693 | 99.1% (1.43M) | 0.4% (6.4K) | `nn.Linear(1280, 5)` |
