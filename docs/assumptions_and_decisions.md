# Assumptions and Unresolved Decisions

**Project:** AI-Enabled UAV System for Real-Time Disaster Assessment  
**Document Version:** 1.0.0 (Phase 0 Planning)  

---

## 1. Explicit Engineering Assumptions

### 1.1 Ingestion & Sensor Data
* **Assumption A1 (Media Formats):** The prototype will accept standard static RGB images (`.jpg`, `.jpeg`, `.png`) and standard container video files (`.mp4`, `.mov`, `.avi`) via HTTP multipart uploads. Live streaming protocols (e.g., RTSP, WebRTC, RTMP) are deferred to future phases.
* **Assumption A2 (Telemetry Availability):** Flight telemetry (UAV GPS latitude, longitude, altitude above ground level, gimbal pitch, and camera focal length) will either be provided as accompanying JSON metadata in the upload payload or extracted from EXIF metadata tags. When absent, the system will apply configurable default simulated flight parameters and explicitly flag coordinates as `is_simulated: true`.
* **Assumption A3 (Camera Orientation):** For the initial coordinate estimation mathematics, we assume nadir (downward-facing $90^\circ$) or oblique downward-facing gimbal angles ($30^\circ - 60^\circ$) over reasonably flat ground terrain (planar projection model). High-fidelity Digital Elevation Models (DEM) are not required for prototype demonstration.

### 1.2 Detection & Computer Vision
* **Assumption A4 (Model Classes):** Standard pretrained YOLO weights (`yolov8n.pt` / `yolov8s.pt` or mock equivalent) detect generic classes such as `person` (COCO class 0), `car`, `truck`, `boat`, etc. For disaster-specific hazards (e.g., collapsed structures, fire, flood water), the prototype will either leverage class-mapped proxies or synthetic hazard bounding boxes until specialized disaster aerial models are trained.
* **Assumption A5 (Zero-GPU & Offline Testability):** All automated unit and API integration tests must execute deterministically on CPU within 10 seconds without attempting to download weights from the internet or requiring GPU acceleration. A `MockDetector` component will simulate standardized detections.

### 1.3 Tracking & Temporal Association
* **Assumption A6 (Frame-Rate & Continuity):** Video processing extracts frames at a controlled sampling rate (e.g., 2 to 5 frames per second). Tracking will be based on 2D bounding box spatial overlap (Intersection over Union, IoU) and centroid proximity across consecutive extracted frames.

### 1.4 Human Review & Advisory Dispatch
* **Assumption A7 (No Autonomous External Side-Effects):** The system generates advisory resource allocation plans. It assumes a human operator is logged in, reviews the detections, and either confirms, modifies, or dismisses each recommendation. No automated external webhooks or dispatch SMS/APIs are fired.

---

## 2. Unresolved Decisions & Options Matrix

The following design decisions are documented for explicit alignment:

| Decision ID | Topic | Option A | Option B | Selected / Recommended for Prototype | Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **DEC-01** | **Telemetry Input Format** | Multi-part JSON alongside media file | Embedded EXIF / XMP metadata inside image files | **Hybrid Approach:** Check EXIF first; allow JSON payload override. | Ensures maximum flexibility for both standalone images and batch flight uploads. |
| **DEC-02** | **Default Model Weights** | Auto-download `yolov8n.pt` at runtime on first real inference | Require explicit manual weight placement in `weights/` directory; default to `MockDetector` if absent | **Option B (Safe Fallback):** Never download automatically in headless environments. Default to `MockDetector` for development/testing unless weights exist. | Complies with the safety rule: do not download weights during automated test execution. |
| **DEC-03** | **Video Processing Execution Mode** | Synchronous frame-by-frame processing in request cycle | Asynchronous background task (FastAPI `BackgroundTasks`) returning a task ID | **Option B (Async Background Task):** Return a `task_id` with progress endpoint (`GET /media/tasks/{id}`). | Prevents HTTP request timeouts on multi-megabyte video uploads. |
| **DEC-04** | **Database Storage for Media** | Store raw media bytes as BLOBs in SQLite | Store media on local disk (`uploads/`) and save relative file paths in SQLite | **Option B (Local File System):** Store paths in DB, files in storage directory. | Prevents SQLite database bloating and maintains high query performance. |
| **DEC-05** | **Scoring Function Formulations** | Dynamic linear weighted sum ($w_1 \cdot \text{conf} + w_2 \cdot \text{hazard\_dist} + \dots$) | Rule-based decision tree (e.g., If hazard < 20m $\rightarrow$ HIGH) | **Option A with Configurable Weights:** Linear normalized weighting with threshold buckets. | Transparent, tuneable, easy to test, and explainable to operators. |
