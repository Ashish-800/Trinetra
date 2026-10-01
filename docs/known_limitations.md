# Known Technical Limitations, Environmental Constraints, and Non-Certification Disclaimers

> [!CAUTION]
> ### STRICT NON-CERTIFICATION & OPERATIONAL DISCLAIMER
> This software is an **experimental academic and engineering prototype** developed for algorithmic research into decision support for disaster response.
> 
> * **NOT CERTIFIED**: The software is **NOT certified** by the FAA, EASA, NFPA, ISO, or any civil defense or aviation safety authority for life-critical or autonomous emergency dispatch.
> * **NOT PRODUCTION READY**: The prototype must **NOT** be deployed as a sole or authoritative life-safety system.
> * **HUMAN-IN-THE-LOOP MANDATORY**: All detection outputs, priority tier rankings, and emergency resource recommendations are **strictly advisory proposals**. Every recommendation enforces `requires_human_authorization = True` and demands verified human sign-off before physical dispatch.

---

## 1. Computer Vision & Biological Status Limitations

1. **No Biological or Vital Sign Inference**:
   * Visual bounding boxes classify pixel patterns matching the shape of a person.
   * Visual detection **never** confirms biological status (whether an individual is conscious, breathing, injured, or deceased).
   * Terminology is strictly restricted to `"person"` or `"potential survivor"`.
2. **False Positives & False Negatives**:
   * Debris, overturned furniture, piles of clothing, mannequins, or shadow patterns can trigger false positive bounding boxes.
   * Individuals covered in mud, partially submerged, wearing camouflage, or lying motionless may be missed (false negatives).
3. **Scale & Resolution Limits**:
   * Below 15–20 pixels target height (e.g. UAV flight altitude $> 80\text{m}$ on standard 1080p sensors), model confidence drops significantly.

---

## 2. Multi-Object Tracking Limitations

1. **Camera Motion & Jitter**:
   * The prototype tracker relies on 2D spatial Intersection-over-Union (IoU) across sequential frames.
   * High UAV yaw rates, sudden wind gusts, or aggressive gimbal maneuvers disrupt bounding box overlap, leading to potential **identity switches (`id_switches`)**.
2. **Prolonged Occlusions**:
   * If a target moves under a bridge, thick tree canopy, or collapsed roof for longer than the tracker's termination threshold (30 frames / 1–2s), the track terminates. When the target reappears, a new track ID will be assigned.
3. **Crowd Merging**:
   * Groups of closely packed individuals in evacuation shelters or stranded boats may merge into a single bounding box and track ID.

---

## 3. Location Estimation & Telemetry Limitations

1. **Planar Flat-Earth Assumption**:
   * The prototype ray-casting estimator projects pixel offsets onto a flat horizontal ground plane ($z = 0$).
   * In steep mountainous terrain, deep canyons, or high-rise urban ravines, ground elevation variations introduce coordinate estimation errors ($\pm 10\text{m} - 50\text{m}$).
2. **Absence of Real Hardware Gimbal Feedback**:
   * When UAV gimbal pitch, yaw, or altitude telemetry is missing or simulated, ground coordinates are approximate and must be treated with wide uncertainty bounds (`accuracy_meters = 25.0+`).
   * The drone GPS position is **never** equated directly to the survivor's exact ground location.

---

## 4. Atmospheric & Environmental Constraints

1. **Adverse Weather & Obscurants**:
   * Optical RGB sensors fail in dense wildfire smoke, heavy fog, blizzards, or torrential rainfall.
   * Thermal fusion and multi-spectral sensors (outside the prototype scope) are required for penetration of heavy obscurants.
2. **Nighttime & Low-Light Operations**:
   * The baseline RGB detector requires ambient daylight illumination or high-power searchlights.

---

## 5. Decision & Resource Recommendation Boundaries

1. **Advisory Scoring Heuristics**:
   * Urgency and uncertainty formulas are transparent, rule-based initial assumptions designed to demonstrate explainability. They must be calibrated by operational incident commanders against regional Standard Operating Procedures (SOPs).
2. **Simulated Resource Inventory**:
   * The prototype operates against simulated inventory databases.
   * Response times, route speeds, and road navigability are approximations and do not account for live traffic, active debris blockages, or collapsed bridges unless manually flagged.
