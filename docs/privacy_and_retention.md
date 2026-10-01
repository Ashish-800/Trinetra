# UAV Aerial Imagery: Privacy, Ethics, and Data Retention Guidelines

This document outlines the privacy safeguards, ethical boundaries, and data-retention schedules for the **AI-Enabled UAV Disaster Assessment** decision-support system.

---

## 1. Privacy Impact and Ethical Principles

Aerial UAV reconnaissance during disaster recovery operations must balance urgent search-and-rescue needs against the fundamental privacy rights of affected citizens.

### A. Non-Emergency Bystander Privacy
1. **Minimization of Exposure**: UAV flight trajectories and camera fields of view must be focused strictly on designated disaster sectors, avoiding residential properties outside the immediate disaster perimeter.
2. **De-Identification & Facial Blurring**:
   * Aerial visual models operating at standard UAV survey altitudes ($30\text{m} - 60\text{m}$) inherently lack facial resolution.
   * If lower-altitude or high-zoom imagery resolves facial features or residential interiors, automated de-identification (Gaussian blurring or pixelation) must be applied prior to long-term archiving.
3. **Non-Survivor Exclusions**: The system operates strictly as a survivor detection aid. It must **never** be integrated with facial recognition, biographical databases, or law enforcement identity tracking.

---

## 2. EXIF and Metadata Sanitization

UAV media files often contain sensitive embedded camera metadata (operator identifiers, drone serial numbers, camera model fingerprints).
1. **EXIF Scrubbing**: Prior to persistence or API transport, non-essential EXIF tags must be scrubbed to prevent device fingerprinting.
2. **Spatial Privacy**: High-precision ground coordinates generated for human dwellings must be restricted to authorized incident command personnel to prevent scavenging, unauthorized trespass, or looting in evacuated sectors.

---

## 3. Data Retention Schedule

To avoid unbounded data growth and minimize liability, the following retention schedules are enforced:

| Asset Type | Storage Location | Retention Window | Purge Procedure |
| :--- | :--- | :--- | :--- |
| **Raw UAV Video Files** | `uploads/` | **24 to 48 hours** maximum | Automatically purged after frame extraction and mission confirmation. |
| **Extracted Negative Frames** (0 detections) | `uploads/extracted_frames/` | **7 days** | Deleted during routine cleanup cycles. |
| **Annotated Positive Detection Frames** | `uploads/extracted_frames/` | **30 days** | Retained for human incident post-mortem analysis, then permanently wiped. |
| **Mission & Assessment Audit Records** | SQLite / Postgres DB | **90 days** | Anonymized and archived for operational review. |

---

## 4. Access Control and Transport Security

1. **Storage Isolation**: Uploaded media must reside in a dedicated managed vault (`settings.UPLOAD_DIR`), disallowing arbitrary client path traversal or symlink following.
2. **Transport Encryption**: All API communications must enforce TLS 1.3 (HTTPS) in production to prevent interception of telemetry or location coordinates.
3. **Role-Based Authorization**: Advisory resource recommendations and incident logs are accessible only to verified incident commanders.
