import os
import sys
import json
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

# ---------------------------------------------------------
# PATHS & SETUP
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent.parent.parent
DATASET_MANIFEST_PATH = BASE_DIR / "dataset" / "final_13class_manifest.csv"
MODEL_PATH = BASE_DIR / "models" / "yolov8n_cls_13class.pth"
INT8_MODEL_PATH = BASE_DIR / "models" / "yolov8n_cls_13class_int8.pth"
BENCHMARKS_DIR = BASE_DIR / "benchmarks"
FUSION_DIR = BASE_DIR / "fusion"
REPORTS_DIR = BASE_DIR / "reports"
DATA_SCREENSHOTS_DIR = Path("data/screenshots")

BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
FUSION_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "intelligence" / "final_refinement"))
from final_refinement_experiment import FinalRefinedClassifier

# ---------------------------------------------------------
# TAXONOMY & MAPS
# ---------------------------------------------------------
VIS_13_TO_SEMANTIC = {
    "code_editor_terminal": "Coding",
    "travel_booking": "Travel",
    "media_player_library": "Entertainment",
    "document_article_view": "Education",
    "dashboard_settings": "Connectivity",
    "product_listing": "Shopping",
    "product_detail": "Shopping",
    "cart_checkout_payment": "Payment",
    "login_authentication": "Connectivity",
    "messaging_chat": "Communication",
    "email_view": "Communication",
    "error_dialog_alert": "General",
    "other_screen": "General"
}

CANONICAL_SEMANTIC_CATEGORIES = [
    "Education", "Payment", "Shopping", "Travel", "Coding",
    "Entertainment", "Connectivity", "Work & Career", "Communication", "General"
]

# ---------------------------------------------------------
# LETTERBOX RESIZE & INFERENCE HELPERS
# ---------------------------------------------------------
def letterbox_resize(img: Image.Image, target_size: int = 224) -> Image.Image:
    if img.mode != 'RGB':
        img = img.convert('RGB')
    w, h = img.size
    scale = min(target_size / w, target_size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    img_resized = img.resize((nw, nh), Image.Resampling.BILINEAR)
    new_img = Image.new('RGB', (target_size, target_size), (114, 114, 114))
    new_img.paste(img_resized, ((target_size - nw) // 2, (target_size - nh) // 2))
    return new_img

transform_pipeline = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class YOLO13ClassModelWrapper(nn.Module):
    def __init__(self, num_classes: int = 13):
        super().__init__()
        yolo_base = YOLO("yolov8n-cls.pt")
        self.backbone = yolo_base.model
        in_features = self.backbone.model[9].linear.in_features if hasattr(self.backbone.model[9], "linear") else 1280
        self.backbone.model[9].linear = nn.Linear(in_features, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)
        if isinstance(out, (tuple, list)):
            return out[0]
        return out

def load_13class_model(model_path: Path, device: torch.device):
    ckpt = torch.load(model_path, map_location=device)
    class_to_idx = ckpt['class_to_idx']
    idx_to_class = ckpt['idx_to_class']
    model = YOLO13ClassModelWrapper(num_classes=len(class_to_idx)).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, class_to_idx, idx_to_class

# ---------------------------------------------------------
# MULTIMODAL FUSION EVALUATOR
# ---------------------------------------------------------
def run_multimodal_evaluation():
    print("Starting Part F & G: Multimodal Fusion Re-evaluation and Transition Analysis...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load 13-Class Model
    model, class_to_idx, idx_to_class = load_13class_model(MODEL_PATH, device)
    
    # 2. Load Semantic Classifier
    semantic_classifier = FinalRefinedClassifier()
    
    # 3. Load Validation Set
    df_manifest = pd.read_csv(DATASET_MANIFEST_PATH)
    val_df = df_manifest[df_manifest['partition'] == 'validation'].copy()
    print(f"Validation Partition Size: {len(val_df)} screenshots")
    
    # Load metadata notes/text lookup
    ocr_notes_cache = {}
    meta_path = PROJECT_ROOT / "data" / "metadata.csv"
    if meta_path.exists():
        df_meta = pd.read_csv(meta_path)
        for _, row in df_meta.iterrows():
            ocr_notes_cache[str(row['file'])] = str(row.get('notes', ''))
            
    # Evaluation Containers
    results = []
    
    # Timing benchmarks
    lat_semantic = []
    lat_visual = []
    lat_fusion = []
    
    for _, row in val_df.iterrows():
        fn = str(row['filename'])
        ss_id = str(row['screenshot_id'])
        gt_sem = str(row.get('original_semantic_category', 'General'))
        gt_vis = str(row.get('screen_type', 'other_screen'))
        
        notes_text = ocr_notes_cache.get(fn, fn)
        img_path = DATA_SCREENSHOTS_DIR / fn
        
        # A. Semantic Branch
        t0_sem = time.perf_counter()
        sem_res = semantic_classifier.predict_final(raw_text=notes_text, notes=notes_text, variant="C")
        t1_sem = time.perf_counter()
        lat_semantic.append((t1_sem - t0_sem) * 1000.0)
        
        sem_pred = sem_res.get('prediction', 'General')
        sem_conf = sem_res.get('confidence', 'Low')
        is_ambig = bool(sem_res.get('is_ambiguous', False))
        
        # B. 13-Class Visual Branch
        t0_vis = time.perf_counter()
        if img_path.exists():
            with Image.open(img_path) as img:
                img_lb = letterbox_resize(img, 224)
                t_img = transform_pipeline(img_lb).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(t_img)
                probs = torch.softmax(logits / 1.5, dim=1).cpu().numpy()[0]
                vis_idx = int(np.argmax(probs))
                vis_pred_st = idx_to_class[vis_idx]
                vis_conf = float(probs[vis_idx])
        else:
            vis_pred_st = 'other_screen'
            vis_conf = 0.5
        t1_vis = time.perf_counter()
        lat_visual.append((t1_vis - t0_vis) * 1000.0)
        
        vis_mapped_sem = VIS_13_TO_SEMANTIC.get(vis_pred_st, 'General')
        
        # C. 13-Class Selective Multimodal Fusion (Strategy E)
        t0_fus = time.perf_counter()
        fusion_applied = False
        fusion_reason = "Semantic classifier prediction authoritative"
        fused_pred = sem_pred
        
        # Selective Override Trigger Rule
        if (sem_conf == "Low" or is_ambig or sem_pred == "General") and vis_conf >= 0.40:
            if vis_pred_st in VIS_13_TO_SEMANTIC:
                override_target = VIS_13_TO_SEMANTIC[vis_pred_st]
                if override_target != sem_pred:
                    fused_pred = override_target
                    fusion_applied = True
                    fusion_reason = f"Visual layout '{vis_pred_st}' overrode low-confidence/ambiguous semantic prediction '{sem_pred}' -> '{fused_pred}'"
                    
        t1_fus = time.perf_counter()
        lat_fusion.append((t1_fus - t0_sem) * 1000.0) # Complete end-to-end pipeline latency
        
        # D. Baseline 5-Class Multimodal Result (Phase 7K emulation)
        baseline_5class_pred = sem_pred
        if (sem_conf == "Low" or is_ambig or sem_pred == "General"):
            if vis_pred_st == "code_editor_terminal": baseline_5class_pred = "Coding"
            elif vis_pred_st == "travel_booking": baseline_5class_pred = "Travel"
            
        results.append({
            "screenshot_id": ss_id,
            "filename": fn,
            "ground_truth_semantic": gt_sem,
            "ground_truth_visual": gt_vis,
            "opt_a_semantic_only": sem_pred,
            "opt_b_visual_only_mapped": vis_mapped_sem,
            "opt_c_selective_fusion_13class": fused_pred,
            "opt_d_baseline_fusion_5class": baseline_5class_pred,
            "fusion_applied": fusion_applied,
            "fusion_reason": fusion_reason
        })
        
    df_res = pd.DataFrame(results)
    
    # ---------------------------------------------------------
    # CALCULATE ACCURACY & METRICS FOR OPTIONS A, B, C, D
    # ---------------------------------------------------------
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, balanced_accuracy_score
    
    y_true = df_res['ground_truth_semantic'].tolist()
    
    acc_a = accuracy_score(y_true, df_res['opt_a_semantic_only'])
    acc_b = accuracy_score(y_true, df_res['opt_b_visual_only_mapped'])
    acc_c = accuracy_score(y_true, df_res['opt_c_selective_fusion_13class'])
    acc_d = accuracy_score(y_true, df_res['opt_d_baseline_fusion_5class'])
    
    _, _, f1_a, _ = precision_recall_fscore_support(y_true, df_res['opt_a_semantic_only'], average='macro', zero_division=0)
    _, _, f1_b, _ = precision_recall_fscore_support(y_true, df_res['opt_b_visual_only_mapped'], average='macro', zero_division=0)
    _, _, f1_c, _ = precision_recall_fscore_support(y_true, df_res['opt_c_selective_fusion_13class'], average='macro', zero_division=0)
    _, _, f1_d, _ = precision_recall_fscore_support(y_true, df_res['opt_d_baseline_fusion_5class'], average='macro', zero_division=0)
    
    bal_a = balanced_accuracy_score(y_true, df_res['opt_a_semantic_only'])
    bal_b = balanced_accuracy_score(y_true, df_res['opt_b_visual_only_mapped'])
    bal_c = balanced_accuracy_score(y_true, df_res['opt_c_selective_fusion_13class'])
    bal_d = balanced_accuracy_score(y_true, df_res['opt_d_baseline_fusion_5class'])
    
    # ---------------------------------------------------------
    # PART G: ERROR TRANSITION ANALYSIS
    # ---------------------------------------------------------
    sem_correct = df_res['opt_a_semantic_only'] == df_res['ground_truth_semantic']
    fus_correct = df_res['opt_c_selective_fusion_13class'] == df_res['ground_truth_semantic']
    
    t_cor_to_cor = int(np.sum(sem_correct & fus_correct))
    t_incor_to_cor = int(np.sum((~sem_correct) & fus_correct)) # Errors corrected
    t_cor_to_incor = int(np.sum(sem_correct & (~fus_correct))) # Correct predictions harmed
    t_incor_to_incor = int(np.sum((~sem_correct) & (~fus_correct))) # Both incorrect
    
    net_gain = t_incor_to_cor - t_cor_to_incor
    
    print("\n======================================================================")
    print("           MULTIMODAL FUSION COMPARISON (VALIDATION SET)")
    print("======================================================================")
    print(f"Option A: FinalRefinedClassifier Alone (Semantic Baseline):")
    print(f"  Top-1 Accuracy: {acc_a*100:.2f}% | Macro F1: {f1_a:.4f} | Balanced Acc: {bal_a*100:.2f}%")
    print(f"\nOption B: 13-Class Visual Classifier Alone (Mapped):")
    print(f"  Top-1 Accuracy: {acc_b*100:.2f}% | Macro F1: {f1_b:.4f} | Balanced Acc: {bal_b*100:.2f}%")
    print(f"\nOption C: 13-Class Selective Multimodal Fusion (PROPOSED):")
    print(f"  Top-1 Accuracy: {acc_c*100:.2f}% | Macro F1: {f1_c:.4f} | Balanced Acc: {bal_c*100:.2f}%")
    print(f"\nOption D: Existing 5-Class Multimodal Baseline (Phase 7K):")
    print(f"  Top-1 Accuracy: {acc_d*100:.2f}% | Macro F1: {f1_d:.4f} | Balanced Acc: {bal_d*100:.2f}%")
    
    print("\n--- PART G: ERROR TRANSITION ANALYSIS (Option A -> Option C) ---")
    print(f"  1. Semantic Correct -> Fusion Correct:   {t_cor_to_cor}")
    print(f"  2. Semantic Incorrect -> Fusion Correct: {t_incor_to_cor}  <-- ERRORS CORRECTED")
    print(f"  3. Semantic Correct -> Fusion Incorrect: {t_cor_to_incor}  <-- HARMED PREDICTIONS")
    print(f"  4. Both Incorrect:                      {t_incor_to_incor}")
    print(f"  NET GAIN (Corrected - Harmed):           +{net_gain}")
    
    # Timing metrics
    warm_med_sem = float(np.median(lat_semantic[1:]))
    warm_med_vis = float(np.median(lat_visual[1:]))
    warm_med_pipe = float(np.median(lat_fusion[1:]))
    warm_p95_pipe = float(np.percentile(lat_fusion[1:], 95))
    cold_pipe = lat_fusion[0]
    
    print(f"\n--- PART J: PIPELINE LATENCY PERFORMANCE ---")
    print(f"  Semantic-Only Warm Median Latency: {warm_med_sem:.2f} ms")
    print(f"  Visual-Only Warm Median Latency:   {warm_med_vis:.2f} ms")
    print(f"  Combined Pipeline Warm Median:      {warm_med_pipe:.2f} ms")
    print(f"  Combined Pipeline Warm p95:         {warm_p95_pipe:.2f} ms")
    print(f"  Cold Start Pipeline Latency:       {cold_pipe:.2f} ms")

    summary_fusion = {
        "validation_samples": len(val_df),
        "option_a_semantic_only": {"accuracy": round(acc_a, 4), "macro_f1": round(f1_a, 4), "balanced_acc": round(bal_a, 4)},
        "option_b_visual_only_mapped": {"accuracy": round(acc_b, 4), "macro_f1": round(f1_b, 4), "balanced_acc": round(bal_b, 4)},
        "option_c_selective_fusion_13class": {"accuracy": round(acc_c, 4), "macro_f1": round(f1_c, 4), "balanced_acc": round(bal_c, 4)},
        "option_d_baseline_fusion_5class": {"accuracy": round(acc_d, 4), "macro_f1": round(f1_d, 4), "balanced_acc": round(bal_d, 4)},
        "error_transition": {
            "semantic_correct_to_fusion_correct": t_cor_to_cor,
            "errors_corrected": t_incor_to_cor,
            "correct_predictions_harmed": t_cor_to_incor,
            "both_incorrect": t_incor_to_incor,
            "net_gain": net_gain
        },
        "latency_ms": {
            "semantic_only_median": round(warm_med_sem, 2),
            "visual_only_median": round(warm_med_vis, 2),
            "pipeline_warm_median": round(warm_med_pipe, 2),
            "pipeline_warm_p95": round(warm_p95_pipe, 2),
            "cold_start": round(cold_pipe, 2)
        }
    }
    
    with open(FUSION_DIR / "multimodal_fusion_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(summary_fusion, f, indent=2)

if __name__ == '__main__':
    run_multimodal_evaluation()
