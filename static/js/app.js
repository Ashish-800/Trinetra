/**
 * TRINETRA — Disaster UAV Command Center Client Application
 * Layout: Left Primary Analysis (~58%) + Right UAV Evidence Preview (~42%)
 * Connects directly to FastAPI POST /api/v1/analysis/multimodal
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements — Ingestion & View Management
  const imageUploadInput = document.getElementById("image-upload-input");
  const btnHeaderUpload = document.getElementById("btn-header-upload");
  const btnLoadSample = document.getElementById("btn-load-sample");
  const btnEmptySample = document.getElementById("btn-empty-sample");
  const emptyState = document.getElementById("empty-state");
  const activeDashboard = document.getElementById("active-dashboard");
  const labelMediaFilename = document.getElementById("label-media-filename");
  const previewMetaTag = document.getElementById("preview-meta-tag");

  // Navigation Items
  const navMonitor = document.getElementById("nav-monitor");
  const navHistory = document.getElementById("nav-history");
  const navSettings = document.getElementById("nav-settings");
  const btnOpenDiagnostics = document.getElementById("btn-open-diagnostics");

  // UAV Preview & Canvas Elements
  const uavCanvas = document.getElementById("uav-canvas");
  const ctx = uavCanvas.getContext("2d");
  const detectionTooltip = document.getElementById("detection-tooltip");
  const toggleYolo = document.getElementById("toggle-yolo");
  const toggleSeg = document.getElementById("toggle-seg");
  const confSlider = document.getElementById("conf-slider");
  const confVal = document.getElementById("conf-val");
  const btnZoomFit = document.getElementById("btn-zoom-fit");
  const tabSensorView = document.getElementById("tab-sensor-view");
  const tabGeoView = document.getElementById("tab-geo-view");
  const canvasHolder = document.getElementById("canvas-holder");
  const geoMapHolder = document.getElementById("geo-map-holder");
  const geoMarkersLayer = document.getElementById("geo-markers-layer");
  const loadingOverlay = document.getElementById("loading-overlay");

  // Primary Analysis: Section 1 Assessment
  const valPersonsCount = document.getElementById("val-persons-count");
  const badgePriority = document.getElementById("badge-priority");
  const valUrgencyScore = document.getElementById("val-urgency-score");
  const valUrgencyLevel = document.getElementById("val-urgency-level");
  const barUrgency = document.getElementById("bar-urgency");
  const valUncertaintyScore = document.getElementById("val-uncertainty-score");
  const valUncertaintyLevel = document.getElementById("val-uncertainty-level");
  const barUncertainty = document.getElementById("bar-uncertainty");
  const badgeHazard = document.getElementById("badge-hazard");
  const badgeAccess = document.getElementById("badge-access");
  const textSituationSummary = document.getElementById("text-situation-summary");
  const btnToggleFactors = document.getElementById("btn-toggle-factors");
  const factorsDrawer = document.getElementById("factors-drawer");
  const factorsList = document.getElementById("factors-list");

  // Primary Analysis: Section 2 Environmental Context
  const pctWater = document.getElementById("pct-water");
  const barWater = document.getElementById("bar-water");
  const pctDebris = document.getElementById("pct-debris");
  const barDebris = document.getElementById("bar-debris");
  const pctRoad = document.getElementById("pct-road");
  const barRoad = document.getElementById("bar-road");
  const pctTree = document.getElementById("pct-tree");
  const barTree = document.getElementById("bar-tree");
  const valDamagePct = document.getElementById("val-damage-pct");
  const btnToggleDamageDetails = document.getElementById("btn-toggle-damage-details");
  const damageBreakdownPopover = document.getElementById("damage-breakdown-popover");
  const damageSubclassesList = document.getElementById("damage-subclasses-list");
  const capabilitiesTagsList = document.getElementById("capabilities-tags-list");

  // Primary Analysis: Section 3 Recommendation
  const recResourceName = document.getElementById("rec-resource-name");
  const recResourceType = document.getElementById("rec-resource-type");
  const recSuitabilityVal = document.getElementById("rec-suitability-val");
  const recMatchedCapability = document.getElementById("rec-matched-capability");
  const recDistanceVal = document.getElementById("rec-distance-val");
  const badgeRecStatus = document.getElementById("badge-rec-status");
  const recRationaleText = document.getElementById("rec-rationale-text");
  const btnOpenReviewModal = document.getElementById("btn-open-review-modal");

  // Selected Detection Inspector
  const valSelectedTrackId = document.getElementById("val-selected-track-id");
  const valSelectedConf = document.getElementById("val-selected-conf");
  const valSelectedStatus = document.getElementById("val-selected-status");
  const valSelectedCoords = document.getElementById("val-selected-coords");
  const valSelectedUncertainty = document.getElementById("val-selected-uncertainty");
  const trackChipsList = document.getElementById("track-chips-list");

  // Bottom Timeline
  const bottomTimelineBar = document.getElementById("bottom-timeline-bar");
  const timelineLatencyText = document.getElementById("timeline-latency-text");

  // Modals
  const modalDiagnostics = document.getElementById("modal-diagnostics");
  const btnCloseDiagnostics = document.getElementById("btn-close-diagnostics");
  const btnDismissDiagnostics = document.getElementById("btn-dismiss-diagnostics");
  const diagDevice = document.getElementById("diag-device");
  const diagGpu = document.getElementById("diag-gpu");
  const diagCuda = document.getElementById("diag-cuda");
  const diagStatus = document.getElementById("diag-status");

  const modalHistory = document.getElementById("modal-history");
  const btnCloseHistory = document.getElementById("btn-close-history");
  const btnDismissHistory = document.getElementById("btn-dismiss-history");
  const historyList = document.getElementById("history-list");

  const modalLimitations = document.getElementById("modal-limitations");
  const btnOpenLimitations = document.getElementById("btn-open-limitations");
  const btnCloseLimitations = document.getElementById("btn-close-limitations");
  const btnDismissLimitations = document.getElementById("btn-dismiss-limitations");

  const modalReview = document.getElementById("modal-review");
  const btnCloseReview = document.getElementById("btn-close-review");
  const btnCancelReview = document.getElementById("btn-cancel-review");
  const btnConfirmReview = document.getElementById("btn-confirm-review");
  const reviewModalBody = document.getElementById("review-modal-body");

  const toastCenter = document.getElementById("toast-center");

  // Application State
  let currentFile = null;
  let currentImage = null;
  let currentResult = null;
  let selectedDetectionIndex = null;
  let renderedBoundingBoxes = [];
  const sessionHistory = [];

  // Initialize Diagnostics from backend
  fetchSystemHealth();

  // --------------------------------------------------------------------------
  // EVENT LISTENERS
  // --------------------------------------------------------------------------

  // Ingestion File Picker
  imageUploadInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelected(e.target.files[0]);
    }
  });

  // Sample Load Action
  const loadSampleAction = async () => {
    try {
      showToast("Loading VisDrone sample validation frame...", "info");
      const res = await fetch("/static/samples/visdrone_sample_1.jpg");
      if (res.ok) {
        const blob = await res.blob();
        const file = new File([blob], "visdrone_val_0000001.jpg", { type: "image/jpeg" });
        handleFileSelected(file);
      } else {
        createSyntheticSample();
      }
    } catch (err) {
      createSyntheticSample();
    }
  };

  btnLoadSample.addEventListener("click", loadSampleAction);
  btnEmptySample.addEventListener("click", loadSampleAction);

  // URL query parameter support for testing / QA
  const urlParams = new URLSearchParams(window.location.search);
  if (urlParams.get("sample") === "1") {
    setTimeout(loadSampleAction, 300);
  }

  // Drag & drop onto empty state
  emptyState.addEventListener("dragover", (e) => e.preventDefault());
  emptyState.addEventListener("drop", (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  // Controls: Confidence threshold & layer toggles
  confSlider.addEventListener("input", (e) => {
    confVal.textContent = parseFloat(e.target.value).toFixed(2);
    renderCanvas();
  });
  toggleYolo.addEventListener("change", () => renderCanvas());
  toggleSeg.addEventListener("change", () => renderCanvas());
  btnZoomFit.addEventListener("click", () => {
    uavCanvas.style.transform = "scale(1)";
  });

  // View Switcher (Sensor vs Geo View)
  tabSensorView.addEventListener("click", () => {
    tabSensorView.classList.add("active");
    tabGeoView.classList.remove("active");
    canvasHolder.style.display = "flex";
    geoMapHolder.style.display = "none";
  });

  tabGeoView.addEventListener("click", () => {
    tabGeoView.classList.add("active");
    tabSensorView.classList.remove("active");
    canvasHolder.style.display = "none";
    geoMapHolder.style.display = "block";
  });

  // Contributing Factors Accordion Toggle
  btnToggleFactors.addEventListener("click", () => {
    const isExpanded = btnToggleFactors.getAttribute("aria-expanded") === "true";
    btnToggleFactors.setAttribute("aria-expanded", !isExpanded);
    factorsDrawer.style.display = isExpanded ? "none" : "block";
  });

  // Priority Click Expands Contributing Factors (Requirement 15)
  const btnPriorityExpander = document.getElementById("btn-priority-expander");
  if (btnPriorityExpander) {
    btnPriorityExpander.addEventListener("click", () => {
      const isExpanded = btnToggleFactors.getAttribute("aria-expanded") === "true";
      btnToggleFactors.setAttribute("aria-expanded", !isExpanded);
      factorsDrawer.style.display = isExpanded ? "none" : "block";
    });
  }

  // Recommendation Click Expands Rationale / Opens Review (Requirement 15)
  const recHeadlineGroup = document.getElementById("rec-headline-group");
  if (recHeadlineGroup) {
    recHeadlineGroup.addEventListener("click", () => {
      if (!btnOpenReviewModal.disabled) {
        openReview();
      }
    });
  }

  // Environmental Category Click Highlights Layer (Requirement 15)
  document.querySelectorAll(".env-row").forEach(row => {
    row.addEventListener("click", () => {
      const envType = row.getAttribute("data-env");
      const wasHighlighted = row.classList.contains("highlighted");
      document.querySelectorAll(".env-row").forEach(r => r.classList.remove("highlighted"));
      if (!wasHighlighted) {
        row.classList.add("highlighted");
        const name = row.querySelector(".env-class-name")?.textContent || envType;
        showToast(`Environmental focus: ${name} layer active`, "info");
      } else {
        showToast("Environmental focus reset", "info");
      }
    });
  });

  // Building Damage Breakdown Popover Toggle
  btnToggleDamageDetails.addEventListener("click", (e) => {
    e.stopPropagation();
    const isVisible = damageBreakdownPopover.style.display === "block";
    damageBreakdownPopover.style.display = isVisible ? "none" : "block";
  });

  document.addEventListener("click", () => {
    damageBreakdownPopover.style.display = "none";
  });

  // Navigation Items
  navMonitor.addEventListener("click", () => {
    // Already in monitor view
  });

  const openHistory = () => { renderHistoryList(); modalHistory.style.display = "flex"; };
  const closeHistory = () => { modalHistory.style.display = "none"; };
  navHistory.addEventListener("click", openHistory);
  btnCloseHistory.addEventListener("click", closeHistory);
  btnDismissHistory.addEventListener("click", closeHistory);

  const openDiag = () => { modalDiagnostics.style.display = "flex"; };
  const closeDiag = () => { modalDiagnostics.style.display = "none"; };
  navSettings.addEventListener("click", openDiag);
  btnOpenDiagnostics.addEventListener("click", openDiag);
  btnCloseDiagnostics.addEventListener("click", closeDiag);
  btnDismissDiagnostics.addEventListener("click", closeDiag);

  const openLimits = () => { modalLimitations.style.display = "flex"; };
  const closeLimits = () => { modalLimitations.style.display = "none"; };
  btnOpenLimitations.addEventListener("click", openLimits);
  btnCloseLimitations.addEventListener("click", closeLimits);
  btnDismissLimitations.addEventListener("click", closeLimits);

  const openReview = () => { populateReviewModal(); modalReview.style.display = "flex"; };
  const closeReview = () => { modalReview.style.display = "none"; };
  btnOpenReviewModal.addEventListener("click", openReview);
  btnCloseReview.addEventListener("click", closeReview);
  btnCancelReview.addEventListener("click", closeReview);
  btnConfirmReview.addEventListener("click", () => {
    closeReview();
    badgeRecStatus.textContent = "AUTHORIZED (HUMAN SIGN-OFF)";
    badgeRecStatus.style.color = "#16A34A";
    badgeRecStatus.style.backgroundColor = "#F0FDF4";
    badgeRecStatus.style.borderColor = "#BBF7D0";
    showToast("Human Authorization Recorded: Advisory recommendation verified.", "success");
  });

  // Canvas Hit Testing (Hover & Click Bounding Box)
  uavCanvas.addEventListener("mousemove", (e) => {
    const rect = uavCanvas.getBoundingClientRect();
    const mouseX = (e.clientX - rect.left) * (uavCanvas.width / rect.width);
    const mouseY = (e.clientY - rect.top) * (uavCanvas.height / rect.height);

    let hit = null;
    for (let i = renderedBoundingBoxes.length - 1; i >= 0; i--) {
      const box = renderedBoundingBoxes[i];
      if (mouseX >= box.x1 && mouseX <= box.x2 && mouseY >= box.y1 && mouseY <= box.y2) {
        hit = box;
        break;
      }
    }

    if (hit) {
      uavCanvas.style.cursor = "pointer";
      detectionTooltip.style.display = "block";
      detectionTooltip.style.left = `${e.clientX + 10}px`;
      detectionTooltip.style.top = `${e.clientY + 10}px`;
      detectionTooltip.innerHTML = `<strong>${hit.trackId}</strong> &bull; ${(hit.confidence * 100).toFixed(0)}%`;
    } else {
      uavCanvas.style.cursor = "default";
      detectionTooltip.style.display = "none";
    }
  });

  uavCanvas.addEventListener("mouseleave", () => {
    detectionTooltip.style.display = "none";
  });

  uavCanvas.addEventListener("click", (e) => {
    const rect = uavCanvas.getBoundingClientRect();
    const mouseX = (e.clientX - rect.left) * (uavCanvas.width / rect.width);
    const mouseY = (e.clientY - rect.top) * (uavCanvas.height / rect.height);

    let hitIdx = null;
    for (let i = renderedBoundingBoxes.length - 1; i >= 0; i--) {
      const box = renderedBoundingBoxes[i];
      if (mouseX >= box.x1 && mouseX <= box.x2 && mouseY >= box.y1 && mouseY <= box.y2) {
        hitIdx = box.origIndex;
        break;
      }
    }

    selectDetection(hitIdx);
  });

  // --------------------------------------------------------------------------
  // FILE HANDLING & ANALYSIS WORKFLOW
  // --------------------------------------------------------------------------

  function handleFileSelected(file) {
    currentFile = file;
    labelMediaFilename.textContent = file.name;

    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => {
        currentImage = img;
        currentResult = null;
        selectedDetectionIndex = null;

        // Switch from Empty State to Active Dashboard
        emptyState.style.display = "none";
        activeDashboard.style.display = "grid";
        if (bottomTimelineBar) bottomTimelineBar.style.display = "none";

        if (previewMetaTag) previewMetaTag.textContent = `${img.naturalWidth} \u00d7 ${img.naturalHeight} px`;

        // Switch to video controls if video media
        const isVideo = file.type && file.type.startsWith("video");
        const imgControls = document.getElementById("image-controls");
        const vidControls = document.getElementById("video-controls");
        if (imgControls && vidControls) {
          imgControls.style.display = isVideo ? "none" : "flex";
          vidControls.style.display = isVideo ? "flex" : "none";
        }

        renderCanvas();
        // Immediately run multimodal assessment
        executeMultimodalAnalysis(file);
      };
      img.src = e.target.result;
    };
    reader.readAsDataURL(file);
  }

  async function executeMultimodalAnalysis(file) {
    showLoading(true);
    setLoadingStep("step-yolo", true);

    const startTime = performance.now();
    const formData = new FormData();
    formData.append("file", file);
    formData.append("confidence_threshold", confSlider.value);
    formData.append("enable_segmentation", "true");

    try {
      setTimeout(() => {
        setLoadingStep("step-yolo", false);
        setLoadingStep("step-seg", true);
      }, 400);

      setTimeout(() => {
        setLoadingStep("step-seg", false);
        setLoadingStep("step-risk", true);
      }, 800);

      const response = await fetch("/api/v1/analysis/multimodal", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server error (HTTP ${response.status})`);
      }

      const result = await response.json();
      currentResult = result;
      showLoading(false);

      const latencyMs = Math.round(performance.now() - startTime);
      timelineLatencyText.textContent = `Inference: ${latencyMs}ms`;

      // Record to session history
      sessionHistory.unshift({
        filename: file.name,
        timestamp: new Date().toLocaleTimeString(),
        personsCount: result.total_detections_found || 0,
        priority: result.priority_assessment ? result.priority_assessment.composite_priority : "NORMAL",
        result: result
      });

      // Update all dashboard sections
      updateAssessmentPanel(result);
      updateEnvironmentalPanel(result.scene_context);
      updateRecommendationPanel(result.recommendation_outcome);
      updateTelemetry(result.device_info);
      updateGeoMap(result.detections || []);
      renderCanvas();

      const count = result.total_detections_found || 0;
      showToast(`Assessment complete: ${count} potential person${count === 1 ? "" : "s"} identified.`, "success");

    } catch (err) {
      showLoading(false);
      showToast(`Analysis failed: ${err.message}`, "error");
    }
  }

  // --------------------------------------------------------------------------
  // UAV PREVIEW & CANVAS DRAWING
  // --------------------------------------------------------------------------

  function renderCanvas() {
    if (!currentImage) return;

    uavCanvas.width = currentImage.naturalWidth;
    uavCanvas.height = currentImage.naturalHeight;

    // Draw base aerial image
    ctx.clearRect(0, 0, uavCanvas.width, uavCanvas.height);
    ctx.drawImage(currentImage, 0, 0);

    renderedBoundingBoxes = [];

    if (!currentResult) return;

    // 1. Environmental segmentation overlay if enabled
    if (toggleSeg.checked && currentResult.scene_context) {
      drawSegmentationOverlay(currentResult.scene_context);
    }

    // 2. Person bounding boxes if enabled
    if (toggleYolo.checked && currentResult.detections) {
      drawBoundingBoxes(currentResult.detections);
    }
  }

  function drawBoundingBoxes(detections) {
    const thresh = parseFloat(confSlider.value);
    const filtered = detections.filter(d => d.confidence >= thresh);

    filtered.forEach((det, idx) => {
      const bx = det.bbox;
      const x1 = bx.x1 <= 1.0 ? bx.x1 * uavCanvas.width : bx.x1;
      const y1 = bx.y1 <= 1.0 ? bx.y1 * uavCanvas.height : bx.y1;
      const x2 = bx.x2 <= 1.0 ? bx.x2 * uavCanvas.width : bx.x2;
      const y2 = bx.y2 <= 1.0 ? bx.y2 * uavCanvas.height : bx.y2;
      const w = Math.max(2, x2 - x1);
      const h = Math.max(2, y2 - y1);

      const isSelected = selectedDetectionIndex === idx;
      const trackId = det.track_id ? `TRK-${String(det.track_id).padStart(2, "0")}` : `ID-${idx + 1}`;

      renderedBoundingBoxes.push({
        origIndex: idx,
        trackId: trackId,
        confidence: det.confidence,
        x1: x1,
        y1: y1,
        x2: x2,
        y2: y2
      });

      ctx.save();
      if (selectedDetectionIndex !== null && !isSelected) {
        ctx.globalAlpha = 0.45;
      }

      const boxColor = isSelected ? "#2563EB" : "#16A34A";
      const lineWidth = isSelected ? Math.max(3, Math.round(uavCanvas.width / 500)) : Math.max(2, Math.round(uavCanvas.width / 650));

      // Crisp border
      ctx.strokeStyle = boxColor;
      ctx.lineWidth = lineWidth;
      ctx.strokeRect(x1, y1, w, h);

      // Subtle translucent box tint
      ctx.fillStyle = isSelected ? "rgba(37, 99, 235, 0.12)" : "rgba(22, 163, 74, 0.08)";
      ctx.fillRect(x1, y1, w, h);

      // Compact collision-aware label pill: TRK-01 · 85%
      const confPct = Math.round(det.confidence * 100);
      const labelText = `${trackId} \u00b7 ${confPct}%`;
      const fontSize = Math.max(10, Math.min(13, Math.round(uavCanvas.width / 120)));
      ctx.font = `600 ${fontSize}px 'Inter', sans-serif`;

      const textWidth = ctx.measureText(labelText).width;
      const pillHeight = fontSize + 6;
      const pillWidth = textWidth + 8;

      let pillY = y1 - pillHeight - 2;
      if (pillY < 0) pillY = y1 + 2;

      ctx.fillStyle = boxColor;
      roundRect(ctx, x1, pillY, pillWidth, pillHeight, 2);
      ctx.fill();

      ctx.fillStyle = "#FFFFFF";
      ctx.fillText(labelText, x1 + 4, pillY + fontSize - 1);

      ctx.restore();
    });

    // Populate track chips in inspector
    updateTrackChips(filtered);
  }

  function drawSegmentationOverlay(scene) {
    if (!scene) return;
    ctx.save();
    // Subtle hazard status watermark in canvas corner
    ctx.fillStyle = "rgba(15, 23, 42, 0.75)";
    roundRect(ctx, 12, 12, 130, 22, 3);
    ctx.fill();

    const sev = (scene.inferred_hazard_severity || "NORMAL").toUpperCase();
    ctx.font = "600 10px 'Inter', sans-serif";
    ctx.fillStyle = sev === "SEVERE" ? "#F87171" : (sev === "MODERATE" ? "#FBBF24" : "#34D399");
    ctx.fillText(`Hazard: ${sev}`, 20, 26);
    ctx.restore();
  }

  // --------------------------------------------------------------------------
  // PRIMARY ANALYSIS WORKSPACE POPULATION
  // --------------------------------------------------------------------------

  function updateAssessmentPanel(result) {
    const prio = result.priority_assessment || {};
    const scene = result.scene_context || {};

    // 1. Potential Persons Count
    const personCount = result.total_detections_found || 0;
    valPersonsCount.textContent = personCount;
    const labelPersonsUnit = document.getElementById("label-persons-unit");
    if (labelPersonsUnit) {
      labelPersonsUnit.textContent = personCount === 1 ? "potential person" : "potential persons";
    }

    // 2. Priority Badge
    const prioLevel = (prio.composite_priority || "STANDBY").toUpperCase();
    badgePriority.textContent = prioLevel;
    badgePriority.className = "hero-priority-val " + (
      prioLevel.includes("CRITICAL") ? "critical" :
      prioLevel.includes("HIGH") ? "high" : ""
    );

    // 3. Urgency Score & Level
    const urgScore = prio.urgency_score !== undefined ? Math.round(prio.urgency_score * 100) : 0;
    valUrgencyScore.textContent = urgScore;
    valUrgencyLevel.textContent = prio.urgency_level || "Low";
    barUrgency.style.width = `${Math.min(100, urgScore)}%`;

    // 4. Uncertainty Score & Level
    const uncScore = prio.uncertainty_score !== undefined ? Math.round(prio.uncertainty_score * 100) : 0;
    valUncertaintyScore.textContent = uncScore;
    valUncertaintyLevel.textContent = prio.uncertainty_level || "Low";
    barUncertainty.style.width = `${Math.min(100, uncScore)}%`;

    // 5. Hazard & Accessibility Badges
    const hazard = (scene.inferred_hazard_severity || "NORMAL").toUpperCase();
    badgeHazard.textContent = hazard;

    const access = (scene.inferred_accessibility || "ACCESSIBLE").toUpperCase();
    badgeAccess.textContent = access;
    badgeAccess.className = "status-badge badge-access " + (
      access.includes("BLOCKED") ? "blocked" :
      access.includes("LIMITED") ? "limited" : ""
    );

    // 6. Unified Executive Summary
    if (personCount > 0) {
      textSituationSummary.textContent = `${personCount} potential person${personCount > 1 ? "s" : ""} detected in surveyed sector. Regional accessibility evaluated as ${access}.`;
    } else {
      textSituationSummary.textContent = `No potential persons localized in current reconnaissance frame. Baseline hazard evaluated as ${hazard}.`;
    }

    // 7. Contributing Factors Accordion
    factorsList.innerHTML = "";
    const factors = prio.contributing_factors || [];
    if (factors.length > 0) {
      factors.forEach(f => {
        const li = document.createElement("li");
        li.textContent = f;
        factorsList.appendChild(li);
      });
    } else {
      factorsList.innerHTML = "<li>Baseline aerial reconnaissance parameters.</li>";
    }
  }

  function updateEnvironmentalPanel(scene) {
    if (!scene) return;

    // Slim Progress Bars (Formatted to 2 decimal places per specification)
    const water = scene.water_coverage_pct || 0;
    pctWater.textContent = `${water.toFixed(2)}%`;
    barWater.style.width = `${Math.min(100, water)}%`;

    const debris = scene.debris_coverage_pct || 0;
    pctDebris.textContent = `${debris.toFixed(2)}%`;
    barDebris.style.width = `${Math.min(100, debris)}%`;

    const road = scene.road_coverage_pct || 0;
    pctRoad.textContent = `${road.toFixed(2)}%`;
    barRoad.style.width = `${Math.min(100, road)}%`;

    const tree = scene.tree_coverage_pct || 0;
    pctTree.textContent = `${tree.toFixed(2)}%`;
    barTree.style.width = `${Math.min(100, tree)}%`;

    // Building Damage
    const dist = scene.class_distribution_pct || {};
    const minor = dist["Building-Minor-Damage"] || 0;
    const major = dist["Building-Major-Damage"] || 0;
    const dest = dist["Building-Destroyed"] || 0;
    const totalDmg = minor + major + dest;
    valDamagePct.textContent = `${totalDmg.toFixed(1)}%`;
    damageSubclassesList.innerHTML = `Minor: ${minor.toFixed(1)}% &bull; Major: ${major.toFixed(1)}% &bull; Destroyed: ${dest.toFixed(1)}%`;

    // Required Capabilities
    capabilitiesTagsList.innerHTML = "";
    const caps = scene.inferred_required_capabilities || [];
    if (caps.length > 0) {
      caps.forEach(c => {
        const span = document.createElement("span");
        span.className = "cap-tag";
        span.textContent = c.replace(/_/g, " ");
        capabilitiesTagsList.appendChild(span);
      });
    } else {
      capabilitiesTagsList.innerHTML = '<span class="cap-tag-empty">Standard Response</span>';
    }
  }

  function updateRecommendationPanel(recOutcome) {
    if (!recOutcome) return;
    const recs = recOutcome.recommendations || [];

    if (recs.length === 0) {
      recResourceName.textContent = "No Immediate Suitable Asset";
      recResourceType.textContent = "Advisory Guidance";
      recSuitabilityVal.textContent = "\u2014";
      recMatchedCapability.textContent = "None matched";
      recDistanceVal.innerHTML = "\u2014";
      recRationaleText.textContent = recOutcome.rationale || "All recorded emergency resources unsuitable for sector constraints.";
      btnOpenReviewModal.disabled = true;
      return;
    }

    const r = recs[0];
    recResourceName.textContent = r.resource_name || "Emergency Response Team";
    recResourceType.textContent = (r.resource_type || "SAR Unit").replace(/_/g, " ");

    const suit = Math.round(r.suitability_score * 100);
    recSuitabilityVal.textContent = `${suit}%`;

    const caps = r.matched_capabilities || ["Rapid Response"];
    recMatchedCapability.textContent = caps.join(", ").replace(/_/g, " ");

    const dist = r.estimated_distance_meters ? Math.round(r.estimated_distance_meters) : 720;
    recDistanceVal.innerHTML = `${dist} m <small class="sim-tag">SIMULATED</small>`;

    recRationaleText.textContent = r.rationale || "Highest capability match score with shortest simulated approach distance.";
    btnOpenReviewModal.disabled = false;
  }

  // --------------------------------------------------------------------------
  // DETECTION INSPECTION & TRACK CHIPS
  // --------------------------------------------------------------------------

  function selectDetection(idx) {
    if (idx === null || !currentResult || !currentResult.detections || !currentResult.detections[idx]) {
      selectedDetectionIndex = null;
      renderCanvas();
      return;
    }

    selectedDetectionIndex = idx;
    const det = currentResult.detections[idx];
    const trackId = det.track_id ? `TRK-${String(det.track_id).padStart(2, "0")}` : `ID-${idx + 1}`;

    valSelectedTrackId.textContent = trackId;
    valSelectedConf.textContent = `${(det.confidence * 100).toFixed(0)}% confidence`;
    valSelectedStatus.textContent = det.review_status || "Pending Review";
    valSelectedUncertainty.textContent = `\u00b1 ${(det.uncertainty_radius_meters || 15.0).toFixed(1)} m`;

    if (det.simulated_lat && det.simulated_lon) {
      valSelectedCoords.innerHTML = `${det.simulated_lat.toFixed(4)}\u00b0 N, ${det.simulated_lon.toFixed(4)}\u00b0 W <small class="sim-tag">SIMULATED</small>`;
    } else {
      valSelectedCoords.innerHTML = `Simulated projection <small class="sim-tag">ACTIVE</small>`;
    }

    // Update chip styling
    document.querySelectorAll(".chip-track").forEach((chip, i) => {
      chip.classList.toggle("selected", i === idx);
    });

    renderCanvas();
  }

  function updateTrackChips(detections) {
    trackChipsList.innerHTML = "";
    if (detections.length === 0) {
      valSelectedTrackId.textContent = "None";
      valSelectedConf.textContent = "";
      valSelectedCoords.innerHTML = "&mdash;";
      valSelectedUncertainty.textContent = "";
      return;
    }

    detections.forEach((det, idx) => {
      const trackId = det.track_id ? `TRK-${String(det.track_id).padStart(2, "0")}` : `ID-${idx + 1}`;
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "chip-track" + (selectedDetectionIndex === idx ? " selected" : "");
      chip.textContent = trackId;
      chip.addEventListener("click", () => selectDetection(idx));
      trackChipsList.appendChild(chip);
    });

    if (selectedDetectionIndex === null || selectedDetectionIndex >= detections.length) {
      selectDetection(0);
    }
  }

  function updateGeoMap(detections) {
    geoMarkersLayer.innerHTML = "";
    detections.forEach((det, idx) => {
      const cx = 250 + (idx * 35);
      const cy = 160 + (idx * 20);

      const g = document.createElementNS("http://www.w3.org/2000/svg", "g");

      const circ = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      circ.setAttribute("cx", cx);
      circ.setAttribute("cy", cy);
      circ.setAttribute("r", "20");
      circ.setAttribute("fill", "#EFF6FF");
      circ.setAttribute("stroke", "#3B82F6");
      circ.setAttribute("stroke-width", "1");
      circ.setAttribute("stroke-dasharray", "2 2");
      g.appendChild(circ);

      const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      dot.setAttribute("cx", cx);
      dot.setAttribute("cy", cy);
      dot.setAttribute("r", "4");
      dot.setAttribute("fill", "#2563EB");
      g.appendChild(dot);

      const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
      text.setAttribute("x", cx + 7);
      text.setAttribute("y", cy + 3);
      text.setAttribute("font-size", "9");
      text.setAttribute("font-weight", "600");
      text.setAttribute("fill", "#1E40AF");
      text.textContent = det.track_id ? `TRK-${det.track_id}` : `ID-${idx + 1}`;
      g.appendChild(text);

      geoMarkersLayer.appendChild(g);
    });
  }

  // --------------------------------------------------------------------------
  // MODALS & DIAGNOSTICS
  // --------------------------------------------------------------------------

  function populateReviewModal() {
    if (!currentResult || !currentResult.recommendation_outcome) return;
    const recs = currentResult.recommendation_outcome.recommendations || [];
    if (recs.length === 0) return;

    const r = recs[0];
    reviewModalBody.innerHTML = `
      <div style="font-size:12px;line-height:1.6;color:#334155;">
        <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:6px;padding:12px;margin-bottom:12px;">
          <div style="font-size:10px;color:#64748B;text-transform:uppercase;font-weight:700;">Proposed Resource</div>
          <div style="font-size:15px;font-weight:600;color:#0F172A;">${r.resource_name}</div>
          <div style="font-size:11px;color:#475569;">Type: ${r.resource_type} &bull; Distance: ${Math.round(r.estimated_distance_meters || 720)}m (Simulated)</div>
        </div>
        <p style="margin-bottom:8px;"><strong>Rationale:</strong> ${r.rationale || 'Optimal match based on current environmental capabilities.'}</p>
        <div style="background:#FFFBEB;border:1px solid #FDE68A;border-radius:4px;padding:8px;font-size:11px;color:#92400E;">
          <strong>Human Authorization Gate:</strong> Autonomous dispatch is disabled. Confirming this action logs human coordinator verification.
        </div>
      </div>
    `;
  }

  function renderHistoryList() {
    historyList.innerHTML = "";
    if (sessionHistory.length === 0) {
      historyList.innerHTML = '<div class="history-empty-text">No prior sorties recorded in this session.</div>';
      return;
    }

    sessionHistory.forEach(item => {
      const row = document.createElement("div");
      row.className = "history-item";
      row.innerHTML = `
        <div>
          <strong>${item.filename}</strong>
          <div style="font-size:11px;color:#64748B;">${item.timestamp} &bull; ${item.personsCount} potential persons &bull; Priority: ${item.priority}</div>
        </div>
      `;
      historyList.appendChild(row);
    });
  }

  async function fetchSystemHealth() {
    try {
      const res = await fetch("/health");
      if (res.ok) {
        const data = await res.json();
        diagStatus.textContent = data.status || "Operational";
      }
    } catch (err) {
      diagStatus.textContent = "Offline";
    }
  }

  function updateTelemetry(deviceInfo) {
    if (!deviceInfo) return;
    diagDevice.textContent = deviceInfo.inference_device || "CUDA:0";
    diagGpu.textContent = deviceInfo.gpu_name || "NVIDIA GeForce RTX 2050";
    diagCuda.textContent = String(deviceInfo.cuda_available);
  }

  function setLoadingStep(stepId, isActive) {
    const el = document.getElementById(stepId);
    if (!el) return;
    el.className = isActive ? "step-row active" : "step-row done";
  }

  function showLoading(show) {
    loadingOverlay.style.display = show ? "flex" : "none";
  }

  function showToast(message, type = "info") {
    const toast = document.createElement("div");
    toast.className = `toast-item ${type}`;
    toast.textContent = message;
    toastCenter.appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = "0";
      setTimeout(() => toast.remove(), 200);
    }, 3200);
  }

  function roundRect(ctx, x, y, width, height, radius) {
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.lineTo(x + width - radius, y);
    ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
    ctx.lineTo(x + width, y + height - radius);
    ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
    ctx.lineTo(x + radius, y + height);
    ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
    ctx.lineTo(x, y + radius);
    ctx.quadraticCurveTo(x, y, x + radius, y);
    ctx.closePath();
  }

  function createSyntheticSample() {
    const offCanvas = document.createElement("canvas");
    offCanvas.width = 1024;
    offCanvas.height = 768;
    const offCtx = offCanvas.getContext("2d");

    const grad = offCtx.createLinearGradient(0, 0, 1024, 768);
    grad.addColorStop(0, "#2B3A4A");
    grad.addColorStop(0.5, "#4B5563");
    grad.addColorStop(1, "#1E293B");
    offCtx.fillStyle = grad;
    offCtx.fillRect(0, 0, 1024, 768);

    offCtx.fillStyle = "#64748B";
    offCtx.fillRect(100, 320, 824, 90);
    offCtx.fillRect(450, 60, 140, 648);

    offCanvas.toBlob((blob) => {
      const file = new File([blob], "visdrone_val_sample.jpg", { type: "image/jpeg" });
      handleFileSelected(file);
      showToast("Generated synthetic reconnaissance frame for demonstration", "info");
    }, "image/jpeg");
  }

  // Window exposures for test scripts
  window.drawBoundingBoxes = drawBoundingBoxes;
  window.executeMultimodalAnalysis = executeMultimodalAnalysis;
});
