# Trinetra: AI-Enabled UAV Disaster Command Center

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0%2Bcu124-EE4C2C.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-RTX%202050%20Enabled-76B900.svg)](https://developer.nvidia.com/cuda-zone)
[![Tests Passing](https://img.shields.io/badge/Tests-115%2F115%20Passed-brightgreen.svg)]()

> **Trinetra** is an AI-driven aerial surveillance and emergency decision-support command center that processes real-time UAV imagery and video streams for **potential-person detection**, **12-class environmental hazard segmentation**, **transparent risk prioritization**, and **advisory resource recommendation**.

---

## Table of Contents
- [1. System Architecture](#1-system-architecture)
- [2. Key Capabilities](#2-key-capabilities)
- [3. Dual-Model Machine Learning Pipeline](#3-dual-model-machine-learning-pipeline)
- [4. Project Safety Constitution](#4-project-safety-constitution)
- [5. Master UI/UX Command Center Dashboard](#5-master-uiux-command-center-dashboard)
- [6. Project Layout](#6-project-layout)
- [7. Installation & Quick Start](#7-installation--quick-start)
- [8. Running the Application](#8-running-the-application)
- [9. REST API Reference](#9-rest-api-reference)
- [10. Automated Testing & Quality Assurance](#10-automated-testing--quality-assurance)
- [11. Supported Datasets & Research Notebooks](#11-supported-datasets--research-notebooks)
- [12. Contributing & License](#12-contributing--license)

---

## 1. System Architecture

```
                            ┌────────────────────────────────────────┐
                            │    UAV Aerial Video / Image Stream     │
                            └───────────────────┬────────────────────┘
                                                │
                                                ▼
                     ┌─────────────────────────────────────────────────────┐
                     │    FastAPI Gateway: POST /api/v1/analysis/multimodal │
                     └──────────┬───────────────────────────────┬──────────┘
                                │                               │
                                ▼                               ▼
       ┌───────────────────────────────────┐ ┌───────────────────────────────────┐
       │   YOLOv8n Person Detector (CUDA)  │ │ DeepLabV3-MobileNetV3 RescueNet   │
       │   VisDrone-Trained Aerial Weights │ │ 12-Class Hazard Semantic Segmentor│
       └─────────────────┬─────────────────┘ └─────────────────┬─────────────────┘
                         │                                     │
                         │ [Potential Persons + BBoxes]        │ [Water, Debris, Roads, Trees,
                         ▼                                     │  Building Damage Percentages]
       ┌───────────────────────────────────┐                   │
       │    Spatial Multi-Object Tracker   │                   │
       │    Persistent Track ID & IoU      │                   │
       └─────────────────┬─────────────────┘                   │
                         │                                     │
                         │ [Unique Tracks + Camera Pose]       │
                         ▼                                     │
       ┌───────────────────────────────────┐                   │
       │    Simulated Geolocation Engine   │                   │
       │    Raycasting + Uncertainty Radii │                   │
       └─────────────────┬─────────────────┘                   │
                         │                                     │
                         └───────────────┬─────────────────────┘
                                         │
                                         ▼
                     ┌───────────────────────────────────────┐
                     │     Transparent Priority Engine       │
                     │  - Urgency Score (0-100)              │
                     │  - Uncertainty Score (0-100)          │
                     │  - Composite Priority Level           │
                     └───────────────────┬───────────────────┘
                                         │
                                         ▼
                     ┌───────────────────────────────────────┐
                     │    Advisory Resource Recommender      │
                     │  - Capability & Terrain Matching      │
                     │  - Distance & Suitability Scoring     │
                     │  - Status: PENDING_REVIEW             │
                     │  - Human-in-the-Loop Authorization    │
                     └───────────────────┬───────────────────┘
                                         │
                                         ▼
                     ┌───────────────────────────────────────┐
                     │    Command Center Frontend Dashboard  │
                     │  - 70% Dominant Viewport (Sensor/Geo) │
                     │  - Interactive Bounding Box Inspector │
                     │  - 4-Section Contextual Drawer        │
                     │  - Bottom Analytics & Timeline        │
                     └───────────────────────────────────────┘
```

---

## 2. Key Capabilities

* **Real-Time Multimodal Assessment**: Processes aerial frames through both person detection and disaster semantic segmentation simultaneously on an NVIDIA RTX 2050 (or CPU fallback).
* **Person-Only Detection & Tracking**: Filters YOLO classes to person detections, tracking individuals across video frames using spatial IoU matching to prevent double-counting.
* **12-Class Disaster Segmentation**: Quantifies pixel-level scene coverage for water, structural debris, road infrastructure, vegetation, and damaged buildings via RescueNet.
* **Transparent Multi-Factor Risk Scoring**: Evaluates situation severity without opaque black-box scoring; exposes urgency, uncertainty, and contributing factors.
* **Advisory Emergency Resource Matching**: Ranks simulated rescue assets (e.g., Water Rescue, Canine Search, Heavy Debris Clearance, Urban SAR) based on required capabilities and estimated distances.
* **Strict Human-in-the-Loop Governance**: Autonomous resource dispatch is disabled by design. Every recommendation is marked as advisory (`PENDING_REVIEW`) requiring explicit human coordinator sign-off.
* **Kaggle-Inspired Operational UI**: Modern, clean, spatial monitoring interface featuring a 56px left navigation rail, dominant 70% central workspace, dual-view mode (Sensor View vs. Simulated Geo Map View), and a contextual inspection panel.

---

## 3. Dual-Model Machine Learning Pipeline

### Model 1: VisDrone Person Detector (YOLOv8n)
* **Architecture**: Ultralytics YOLOv8n fine-tuned for high-altitude UAV oblique viewpoints.
* **Dataset**: VisDrone 2019 Detection Benchmark.
* **Target Classes**: `person` (filtered strictly; vehicle classes suppressed for survivor triage).
* **Inference Pipeline**: Normalized coordinates $[0.0, 1.0]$, configurable confidence threshold ($0.10$–$0.90$), and collision-aware label pills (`TRK-01 • 91%`).

### Model 2: RescueNet Disaster Semantic Segmentor
* **Architecture**: DeepLabV3 with MobileNetV3-Large backbone (`num_classes=12`).
* **Input Resolution**: $512 \times 512$ RGB, bilinear image interpolation, nearest-neighbor mask scaling.
* **Validation Performance**: $45.53\%$ mIoU and $75.81\%$ pixel accuracy across 449 real disaster validation scenes.
* **12 RescueNet Semantic Classes**:
  1. Background (`0`)
  2. Water (`1`)
  3. Building-No-Damage (`2`)
  4. Building-Minor-Damage (`3`)
  5. Building-Major-Damage (`4`)
  6. Building-Destroyed (`5`)
  7. Vehicle (`6`)
  8. Road (`7`)
  9. Tree / Vegetation (`8`)
  10. Debris (`9`)
  11. Pool (`10`)
  12. Sand (`11`)

---

## 4. Project Safety Constitution

Trinetra enforces strict ethical and operational boundaries codified in `app/core/safety_constitution.py`:

1. **Non-Confirmatory Survivor Terminology**: Visual detections represent *potential persons* or detected human figures. The system never claims "confirmed survivors" without on-the-ground biological/medical verification.
2. **Simulated Geolocation Boundaries**: Coordinates and transit distances are projected using simulated UAV camera raycasting models and are explicitly marked as `SIMULATED LOCATION`.
3. **Advisory Decision Support**: Resource recommendations do not trigger automated field dispatches. All recommendations require verbal or signed human coordinator authorization.
4. **Dataset Independence**: The VisDrone person detector and RescueNet segmentor originate from separate datasets. Their multimodal combination represents an operational prototype rather than an aligned single ground-truth sensor.

---

## 5. Master UI/UX Command Center Dashboard

The frontend is built using zero-overhead **Vanilla HTML5, modern CSS3, and ES6 JavaScript**, avoiding heavy build pipelines (no Node/npm/Vite needed).

### Key Spatial Design Characteristics:
* **Left Navigation Rail (56px)**: Minimalist vertical rail with line-style SVG icons (`Monitor`, `Incidents`, `Detections`, `Resources`, `History`, `Safety`, `Settings`).
* **Minimal Top Header**: Active sortie chip (`Grid Delta-4`), compact system status (`● System ready`), and hardware telemetry pill (`RTX 2050 • CUDA`).
* **Dominant Monitoring Workspace (~70% width)**:
  * Floating toolbar: detection toggle, segmentation overlay toggle, confidence slider, fit-to-view, and fullscreen.
  * Direct Sensor Canvas with interactive bounding box selection and hover tooltips.
  * Dual-View Switcher: Toggle between **Sensor View** and **Simulated Geo Map View** (visualizing simulated drone paths, detection coordinates, and uncertainty radiuses).
  * In-scene calm loading state with stepper indicators (`Media Ingestion` $\rightarrow$ `YOLO` $\rightarrow$ `RescueNet` $\rightarrow$ `Priority`).
* **Right Contextual Inspection Panel (4 Concise Sections)**:
  1. *Situation & Risk*: Urgency (0–100) & Uncertainty (0–100) progress gauges, executive summary, and expandable contributing factors.
  2. *Environmental Context*: RescueNet breakdown bars (Water, Debris, Road, Tree, Building Damage) and accessibility status.
  3. *Detections & Selected Track*: Selected track details (confidence, simulated GPS, uncertainty $\pm 15$m) and entity list.
  4. *Recommended Response*: Advisory resource card, suitability percentage, transit distance, and a "Review Recommendation" button triggering coordinator sign-off.
* **Bottom Analytics & Timeline**: Displays inference latency, frame metadata, and interactive track chips.

---

## 6. Project Layout

```
disaster-uav/
├── app/
│   ├── api/
│   │   ├── deps.py               # Dependency injection (DB, detector, segmentor)
│   │   └── v1/
│   │       ├── api.py            # API router aggregator
│   │       └── endpoints/        # REST endpoints (analysis, media, incidents, resources, health)
│   ├── core/
│   │   ├── config.py             # App settings (Pydantic BaseSettings)
│   │   ├── database.py           # SQLAlchemy session management
│   │   └── safety_constitution.py# Safety rules, disclaimers, terminology enforcement
│   ├── models/                   # SQLAlchemy ORM models (Incident, MediaAsset, Detection, Track, etc.)
│   ├── schemas/                  # Pydantic domain, request, and response schemas
│   └── services/                 # Core business services:
│       ├── detector.py           # Ultralytics YOLOv8 detector adapter
│       ├── rescuenet.py          # RescueNet mask analysis & taxonomy
│       ├── rescuenet_segmentor.py# DeepLabV3-MobileNetV3 semantic segmentation inference
│       ├── tracker.py            # Video spatial IoU multi-object tracker
│       ├── location.py           # Geolocation estimation & raycasting
│       ├── priority.py           # Multi-factor urgency and uncertainty engine
│       ├── recommender.py        # Constraint-based emergency resource matching
│       └── workflow.py           # Unified AssessmentWorkflowService orchestrator
├── docs/                         # In-depth architectural & safety documentation
├── static/                       # Command Center frontend application
│   ├── index.html                # Single-page spatial monitoring application
│   ├── css/
│   │   └── style.css             # Tokenized Kaggle-inspired stylesheet
│   ├── js/
│   │   └── app.js                # Asynchronous client controller & canvas renderer
│   └── samples/                  # Sample VisDrone validation frames for instant testing
├── tests/
│   └── unit/                     # Complete pytest suite (115 tests)
├── requirements.txt              # Production and development dependencies
└── verify_frontend_e2e.py        # End-to-end multimodal API & UI verification script
```

---

## 7. Installation & Quick Start

### Prerequisites
* **Operating System**: Windows 10/11, Linux, or macOS.
* **Python**: Python 3.11 or 3.12 (64-bit).
* **GPU (Optional)**: NVIDIA GPU with CUDA 12.4+ (RTX 2050 or higher recommended). The system will automatically fall back to CPU if no CUDA device is present.

### Step 1: Clone the Repository
```bash
git clone https://github.com/Ashish-800/Trinetra.git
cd Trinetra
```

### Step 2: Create and Activate Virtual Environment
On Windows PowerShell:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
On Linux/macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

*For GPU acceleration on Windows with PyTorch + CUDA 12.4:*
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### Step 4: Verify Environment
```bash
python verify_env.py
```

---

## 8. Running the Application

### Start the FastAPI Server & Command Center
```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### Open the Dashboard
Navigate to:
```
http://127.0.0.1:8000/
```
*(Or interactive API Swagger UI at `http://127.0.0.1:8000/docs`)*

### Quick Test Workflow:
1. Open `http://127.0.0.1:8000/`.
2. Click **"Load Sample Recon"** (loads a real VisDrone validation reconnaissance frame).
3. Click **"Run Assessment"**.
4. Observe bounding boxes with track tags, environmental disaster distribution, urgency scores, and the advisory resource recommendation.

---

## 9. REST API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| **POST** | `/api/v1/analysis/multimodal` | Single-request multimodal assessment: uploads an aerial image, runs YOLOv8 person detector + RescueNet segmentor, spatial tracking, priority evaluation, and resource recommendation. |
| **POST** | `/api/v1/analysis/start` | Initiates analysis workflow on previously ingested media assets. |
| **POST** | `/api/v1/media/upload` | Ingests static imagery or video streams and extracts frame metadata. |
| **GET** | `/api/v1/incidents/` | Lists recorded disaster incidents. |
| **POST** | `/api/v1/incidents/` | Registers a new disaster incident area. |
| **GET** | `/api/v1/resources/` | Queries simulated emergency resource inventory. |
| **PATCH** | `/api/v1/resources/{id}` | Updates resource capacity and dispatch status. |
| **GET** | `/health` | Root health check and safety constitution status. |
| **GET** | `/` | Serves the Command Center frontend dashboard. |

---

## 10. Automated Testing & Quality Assurance

The codebase includes an exhaustive test suite covering data persistence, media ingestion, YOLO adapters, spatial tracking, geolocation raycasting, priority calculation, resource matching, and API endpoints.

```bash
# Run the complete test suite
pytest -q
```

### Verified Test Results:
* **CUDA Environment (`.venv-cuda` with RTX 2050)**: `115 passed` (100% PASS)
* **Standard CPU Environment (`.venv`)**: `114 passed, 1 skipped` (100% CPU Parity)

---

## 11. Supported Datasets & Research Notebooks

The repository includes research exploration notebooks across complementary disaster modalities:
* **`archive/VisDrone/`**: High-resolution UAV benchmark for person detection in aerial imagery.
* **`RescueNet/`**: High-resolution semantic segmentation dataset for disaster scenes (water, debris, collapsed buildings).
* **`person-detection (1).ipynb`**: Aerial survivor candidate localization baselines.
* **`flood-area-segmentation-unet-structure.ipynb`**: Binary flood and water segmentation structure.
* **`deep-learning-for-satellite-image-processing.ipynb`**: xBD building damage classification.
* **`predicting-building-damage-from-earthquakes.ipynb`**: Structural damage analysis from the 2015 Gorkha earthquake.

---

## 12. Contributing & License

Contributions to improve detection robustness, segmentation accuracy, and operational ergonomics are welcome! Please ensure all pull requests pass `pytest` and adhere to the [Project Safety Constitution](docs/safety_and_verification_rules.md).

Distributed under the **MIT License**.
