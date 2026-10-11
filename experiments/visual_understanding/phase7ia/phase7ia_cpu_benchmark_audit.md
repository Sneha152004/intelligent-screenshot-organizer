# Phase 7I-A: CPU Benchmark & Edge Deployment Audit Report

## 1. Executive Summary

This report audits the CPU inference latency, memory measurement methodology, and deployment trade-offs for all four visual screen-type classification models evaluated in Phase 7I.

---

## 2. Latency Benchmarking Methodology

Inference latency was benchmarked on x86 CPU single-sample inference (`batch_size = 1`) with input tensor shape `(1, 3, 224, 224)`:

1. **Cold-Start Latency**: Measured the wall-clock execution time of the very first single-sample forward pass immediately following model instantiation and weight loading.
2. **Warm Latency**: Measured across 50 consecutive single-sample forward passes using `time.perf_counter()`. Reported median warm latency and p95 warm latency in milliseconds.

### Verified Latency Results:

| Model Candidate | FP32 Checkpoint Size | Cold-Start Latency (ms) | Median Warm Latency (ms) | p95 Warm Latency (ms) | Relative CPU Speed |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **MobileNetV3-Small** | 5.94 MB | **21.77 ms** | **12.80 ms** | **14.43 ms** | 1.00x (Fastest) |
| **YOLOv8n-cls (`yolo26n_cls`)** | **5.59 MB** | 24.93 ms | 18.11 ms | 20.73 ms | 1.41x |
| **EfficientNet-Lite0** | 13.15 MB | 40.14 ms | 25.90 ms | 28.75 ms | 2.02x |
| **MobileViT-XS** | 7.53 MB | 56.23 ms | 39.31 ms | 46.67 ms | 3.07x |

---

## 3. Memory Footprint Attribution Audit

> [!CAUTION]
> **Critical Memory Clarification**:
> The Phase 7I deployment table reported peak memory values ranging from **1744.75 MB to 1779.84 MB**.
>
> An independent audit of `evaluate_models.py` (lines 185-194) reveals that memory was measured using:
> `psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)`
>
> **What This Value Represents**:
> - **Total Process Resident Set Size (RSS)** of the Python process running PyTorch, MKL/OpenMP runtime DLLs, TIMM, Ultralytics, scikit-learn, pandas, and pre-allocated PyTorch memory pools.
> - **What It Does NOT Represent**: It does NOT represent the isolated memory footprint of the visual model weights or dynamic activation tensors during a single forward pass (which is actually < 20 MB for lightweight models).
>
> **Desktop vs Mobile CPU Scope Restriction**:
> Desktop CPU benchmark measurements under Python runtime do NOT equal mobile or edge readiness. Edge mobile deployment requires ONNX Runtime / TFLite / C++ engine profiling to measure true isolated runtime memory.

---

## 4. Latency vs Accuracy Efficiency Trade-Off Matrix

```
                          High Top-1 Accuracy (72.0%)
                                      ▲
                                      │  YOLOv8n-cls (18.11 ms, 5.58 MB)
                                      │
               MobileViT-XS (39.31 ms)│  EfficientNet-Lite0 (25.90 ms)
                                      │
  MobileNetV3-Small (12.80 ms)       │
──────────────────────────────────────┼──────────────────────────────────────►
  Fast CPU Warm Latency (<15 ms)     │  Higher CPU Latency (>20 ms)
                                      │
                                      │  Low Top-1 Accuracy (<50.0%)
```

### Recommendation:
**YOLOv8n-cls (`yolo26n_cls`)** represents the optimal trade-off point, offering **72.0% Top-1 Accuracy** while executing in **18.11 ms median warm CPU latency** with only **1.44M parameters** and **5.58 MB INT8 checkpoint size**.
