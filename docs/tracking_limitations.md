# Tracking Limitations & Operational Boundary Document

**Project:** AI-Enabled UAV System for Real-Time Disaster Assessment  
**Document Version:** 1.0.0 (Phase 5 Planning & Implementation)  

---

## 1. Fundamental Ethical and Safety Boundary

> [!IMPORTANT]
> **Safety Rule: Tracking is Not Identity or Vital Status Confirmation**
> 1. A tracking ID (e.g. `track_id: 1`) is an **ephemeral mathematical association** connecting bounding boxes across consecutive 2D camera frames within a single video stream.
> 2. A track **does not** prove biometric identity, personal recognition, life status, medical condition, entrapment, or survivor status.
> 3. Tracks are strictly labeled as `"person"` or `"potential survivor"`.
> 4. The system does not implement facial recognition, biometric identity retrieval, multi-camera re-identification, or thermal fusion.

---

## 2. Technical and Operational Limitations of Aerial UAV Tracking

### 2.1 Drone Ego-Motion & Parallax
* **Limitation:** In low-altitude UAV flights, drone panning, tilting, yaw rotation, and sudden wind gusts introduce large image displacements (camera ego-motion).
* **Impact:** Pure IoU-based spatial tracking assumes high overlap between consecutive frames. Rapid camera movement can cause an active track to lose overlap, triggering an unintended **ID switch** or premature track termination.
* **Mitigation:** Higher frame sampling rates (e.g. 2–5 frames per second) and linear motion prediction (Kalman filter / centroid velocity).

### 2.2 Occlusion by Debris, Rubble, Dust, and Vegetation
* **Limitation:** Disaster environments are visually cluttered with collapsed masonry, dust clouds, smoke plumes, and tree canopy cover.
* **Impact:** A detected person may walk or be carried behind an obstacle for several seconds. When the person re-emerges, the tracker may assign a new track ID if the elapsed duration exceeds `max_lost_frames`.
* **Mitigation:** Configurable `max_lost_frames` parameter allowing tracks to remain dormant while unobserved for short intervals before final termination.

### 2.3 Variable Altitude and Ground Sampling Distance (GSD)
* **Limitation:** When a UAV climbs to higher altitudes (e.g., $>60\text{m}$), people appear as tiny clusters (10–25 pixels high).
* **Impact:** Small pixel changes can drastically shift bounding box IoU values, increasing tracking volatility.

### 2.4 Crossing Paths & High Density Crowds
* **Limitation:** In crowded evacuation zones or search-and-rescue team operations, individuals frequently cross paths or huddle closely.
* **Impact:** 2D bounding boxes overlap substantially, which can lead to track swapping between two adjacent individuals.

### 2.5 Single-Camera Boundary (No Multi-Camera Re-ID)
* **Limitation:** If a UAV leaves an area and returns 5 minutes later, or if a second UAV flies over the same sector, the system treats detections as new candidate tracks.
* **Impact:** Re-identification across disparate flight legs or distinct drones is explicitly out of scope for this safety-critical prototype to prevent false identity assurances.

---

## 3. Preventing Duplicate Count Inflation

Without tracking, a 10-second video recorded at 30 fps showing 1 stationary person would produce **300 separate detection records**, creating severe operator alert fatigue and misrepresenting disaster casualty figures.

By incorporating `AerialIoUTracker`:
* The system maintains persistent `track_id` assignments across video frames.
* The pipeline aggregates detections and outputs `unique_person_tracks`.
* Operator dashboards and review queues display **unique track instances** rather than raw frame counts.
