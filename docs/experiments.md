# Disaster UAV Assessment - Evaluation Methodology & Experiment Documentation

This document describes the reproducible evaluation methodology, experimental setup, evaluation metrics, benchmark datasets, and comparative ablations for the AI-Enabled UAV Disaster Assessment decision-support prototype.

---

## 1. Experimental Setup and Hardware Environment

All evaluations are structured to run reproducibly under documented CPU or GPU runtime environments.

| Parameter | Specification |
| :--- | :--- |
| **Operating System** | Windows 11 / x86_64 |
| **Python Runtime** | Python 3.11+ / 3.13 |
| **Inference Hardware** | Intel/AMD Host CPU (GPU acceleration supported when CUDA PyTorch is available) |
| **Evaluator Framework** | Custom zero-dependency metric calculation with exact geometric polygon IoU |
| **Strict Anti-Fabrication Rule** | If model weights or ground-truth annotations are missing, the evaluator outputs an explicit diagnostic status: `metric_cannot_yet_be_calculated` rather than guessing or fabricating numbers. |

---

## 2. Person Detection Evaluation Protocol

### A. Benchmark Dataset
* **VisDrone2019-DET Validation Split (`VisDrone2019-DET-val`)**:
  * High-resolution aerial UAV imagery captured across varying altitudes, angles, and weather conditions.
  * Person targets are extracted from VisDrone category IDs:
    * `Category 1`: Pedestrian
    * `Category 2`: People (crowds, seated, lying down)
  * Evaluated across bounding boxes with IoU threshold $\text{IoU} \ge 0.50$.

### B. Evaluated Metrics
1. **Precision ($P$)**: $\frac{TP}{TP + FP}$ — Measures purity of person detection alerts (avoiding false alarms).
2. **Recall ($R$)**: $\frac{TP}{TP + FN}$ — Measures coverage of actual persons on the ground (avoiding missed detections).
3. **F1-Score**: $\frac{2 \cdot P \cdot R}{P + R}$ — Harmonic mean of precision and recall.
4. **Average Precision (AP@50)**: Area under the Precision-Recall curve using 11-point interpolation across recall thresholds $[0.0, 0.1, \dots, 1.0]$.
5. **Inference & Processing Latency**: Mean execution time per image measured in milliseconds using high-precision timers (`time.perf_counter()`).

---

## 3. Multi-Object Tracking Evaluation Protocol

### A. Objective & The Duplicate Alert Rule
In emergency response operations, a naive detector running on 30 FPS video generates massive alert proliferation:
* **Naive Rule (Per-Frame Alerting)**: If an individual remains visible for 50 frames, the detector fires **50 distinct detection alerts**, misleading incident commanders into assuming 50 distinct survivors need rescue.
* **Proposed Rule (Track Deduplication)**: Spatial-temporal tracking associates bounding boxes across successive frames into a single persistent `track_id`. A notification is raised **once** per unique track.

### B. Evaluated Tracking Metrics
1. **Track Consistency**: Fraction of active frames where ground-truth targets maintain uninterrupted track ID consistency ($[0.0, 1.0]$).
2. **Duplicate Alert Reduction Percentage**:
   $$\text{Reduction \%} = \left(1.0 - \frac{\text{Unique Person Tracks}}{\text{Raw Per-Frame Detections}}\right) \times 100\%$$
3. **Tracking Failures**:
   * **ID Switches**: Number of times an active track ID assigned to an uninterrupted target shifts.
   * **False Terminations**: Number of times an active target track drops prematurely while the target is still visible.
   * **Phantom Tracks**: False positive tracks spawned from visual noise or false alarms.

---

## 4. Decision Workflow Evaluation Protocol

### A. Labeled Disaster Scenario Suite
The priority engine is validated against a standardized reference suite of labeled disaster scenarios:

| Scenario ID | Scenario Title | Key Factors | Expected Priority | Expected Rank |
| :--- | :--- | :--- | :--- | :--- |
| `SCENARIO-1` | Rooftop Flood Inundation | 4 persons, Critical flood hazard, isolated terrain, fresh observation | **CRITICAL** | **Rank 1** |
| `SCENARIO-2` | Debris Flow / Mudslide | 2 persons, High debris hazard, difficult terrain | **HIGH** | **Rank 2** |
| `SCENARIO-3` | Stale Perimeter Sighting | 1 person, Low hazard, accessible terrain, 35-minute-old observation | **MEDIUM** | **Rank 3** |
| `SCENARIO-4` | Open Roadside Individual | 1 person, Zero hazard, normal accessible road, fresh observation | **LOW** | **Rank 4** |

### B. Constraint & Safety Compliance Verification
The resource recommendation engine is evaluated for 100% compliance across all generated proposals:
1. **Availability Compliance (100% target)**: Zero recommendations proposing offline, busy, or 0-capacity resources.
2. **Capability Compliance (100% target)**: Zero recommendations proposing resources that lack the specific required capability (e.g. sending a transport truck to a deep water flood).
3. **Human Review Gate Compliance (100% target)**: 100% of recommendations must enforce `requires_human_authorization = True`.

---

## 5. Baseline vs. Proposed Workflow: Comparative Ablation

| Dimension | Naive Baseline (YOLO Only) | Proposed System Workflow | Operational Advantage |
| :--- | :--- | :--- | :--- |
| **Survivor Counting** | Per-frame detection count (e.g. 50 alerts for 1 person across 50 frames) | Temporal IoU tracking deduplication yields 1 persistent track ID | **95–98% alert spam reduction**, eliminating commander fatigue |
| **Priority Ranking** | Equates raw visual confidence directly to urgency | Decouples Urgency (threat to life) from Uncertainty (need for verification) | **Guarantees critical flood victims out-rank non-emergency individuals**, regardless of confidence |
| **Resource Allocation** | Greedy/unfiltered dispatch; risks proposing offline units or trucks to flood waters | Constraint-checked capability matching, availability filtering, and distance ranking | **100% constraint compliance**; prevents fatal dispatch errors |
| **Operational Boundary** | Autonomous actuation risks; no human gate | Strict human authorization gate (`requires_human_authorization = True`) | **Zero autonomous dispatch risk**; decision-support only |

---

## 6. How to Reproduce Results

Run the full evaluation harness directly from the command line:

```powershell
# Run full evaluation suite
python evaluate.py --all

# Run with JSON report output
python evaluate.py --all --json evaluation_report.json

# Run individual components
python evaluate.py --detection
python evaluate.py --tracking
python evaluate.py --workflow
```

---

## 7. Threats to Validity and Honest Limitations

1. **Synthetic vs. Real Disaster Video**: While VisDrone provides real aerial footage, tracking evaluations on synthetic trajectories establish upper-bound algorithmic baselines. Real-world UAV jitter, occlusions from tree canopies, and thermal shifts require active camera gimbal stabilization.
2. **Advisory Decision Support Only**: Priority scores and resource proposals are mathematical heuristics intended solely to assist incident commanders in organizing search queues. They must never be treated as ground truth survival confirmations.
