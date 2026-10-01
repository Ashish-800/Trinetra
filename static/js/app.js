/**
 * Disaster UAV Command Center — Operational Client Application
 * Reference 1: Kaggle-Inspired Clean Information Architecture
 * Reference 2: Spatial Monitoring Layout (One Dominant Viewport + Context Panel)
 * Connects directly to FastAPI POST /api/v1/analysis/multimodal
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements — Inputs & Actions
  const imageUploadInput = document.getElementById("image-upload-input");
  const loadSampleBtn = document.getElementById("load-sample-btn");
  const emptyLoadSampleBtn = document.getElementById("empty-load-sample-btn");
  const analyzeBtn = document.getElementById("analyze-btn");
  const viewportContainer = document.getElementById("viewport-container");
  const emptyState = document.getElementById("empty-state");
  const uavCanvas = document.getElementById("uav-canvas");
  const ctx = uavCanvas.getContext("2d");
  const detectionTooltip = document.getElementById("detection-tooltip");
  const loadedFilename = document.getElementById("loaded-filename");
  const mediaMetaBadge = document.getElementById("media-meta-badge");

  // View switchers
  const btnViewSensor = document.getElementById("btn-view-sensor");
  const btnViewMap = document.getElementById("btn-view-map");
  const canvasWrapper = document.getElementById("canvas-wrapper");
  const geoMapView = document.getElementById("geo-map-view");
  const geoDetectionsGroup = document.getElementById("geo-detections-group");

  // Floating controls
  const toggleYolo = document.getElementById("toggle-yolo");
  const toggleSeg = document.getElementById("toggle-seg");
  const confSlider = document.getElementById("conf-slider");
  const confVal = document.getElementById("conf-val");
  const toolbarPersonCount = document.getElementById("toolbar-person-count");
  const btnZoomFit = document.getElementById("btn-zoom-fit");
  const btnFullscreen = document.getElementById("btn-fullscreen");

  // Overlays & Telemetry
  const loadingOverlay = document.getElementById("loading-overlay");
  const systemStatusPill = document.getElementById("system-status-pill");
  const deviceInfoText = document.getElementById("device-info-text");

  // Right Panel: Section 1 Situation
  const headerPriorityBadge = document.getElementById("header-priority-badge");
  const situationHazardTag = document.getElementById("situation-hazard-tag");
  const urgencyLevelTag = document.getElementById("urgency-level-tag");
  const urgencyNumber = document.getElementById("urgency-number");
  const urgencyBarFill = document.getElementById("urgency-bar-fill");
  const uncertaintyLevelTag = document.getElementById("uncertainty-level-tag");
  const uncertaintyNumber = document.getElementById("uncertainty-number");
  const uncertaintyBarFill = document.getElementById("uncertainty-bar-fill");
  const situationSummaryText = document.getElementById("situation-summary-text");
  const btnToggleFactors = document.getElementById("btn-toggle-factors");
  const factorsList = document.getElementById("factors-list");

  // Right Panel: Section 2 Environmental
  const accessPill = document.getElementById("access-pill");
  const envWaterPct = document.getElementById("env-water-pct");
  const envWaterBar = document.getElementById("env-water-bar");
  const envDebrisPct = document.getElementById("env-debris-pct");
  const envDebrisBar = document.getElementById("env-debris-bar");
  const envRoadPct = document.getElementById("env-road-pct");
  const envRoadBar = document.getElementById("env-road-bar");
  const envTreePct = document.getElementById("env-tree-pct");
  const envTreeBar = document.getElementById("env-tree-bar");
  const envDamagePct = document.getElementById("env-damage-pct");
  const envDamageBar = document.getElementById("env-damage-bar");
  const capabilityPills = document.getElementById("capability-pills");

  // Right Panel: Section 3 Detections
  const detectionsTotalTag = document.getElementById("detections-total-tag");
  const selectedTrackId = document.getElementById("selected-track-id");
  const selectedTrackStatus = document.getElementById("selected-track-status");
  const selectedTrackConf = document.getElementById("selected-track-conf");
  const selectedTrackUncertainty = document.getElementById("selected-track-uncertainty");
  const selectedTrackCoords = document.getElementById("selected-track-coords");
  const listCounterText = document.getElementById("list-counter-text");
  const detectionsItemsList = document.getElementById("detections-items-list");

  // Right Panel: Section 4 Recommendations
  const recResourceName = document.getElementById("rec-resource-name");
  const recResourceType = document.getElementById("rec-resource-type");
  const recSuitabilityVal = document.getElementById("rec-suitability-val");
  const recDistanceVal = document.getElementById("rec-distance-val");
  const recReviewStatus = document.getElementById("rec-review-status");
  const recCapabilitiesList = document.getElementById("rec-capabilities-list");
  const recRationaleText = document.getElementById("rec-rationale-text");
  const btnReviewRecommendation = document.getElementById("btn-review-recommendation");

  // Bottom Analytics Bar
  const bottomAnalyticsBar = document.getElementById("bottom-analytics-bar");
  const btnToggleAnalytics = document.getElementById("btn-toggle-analytics");
  const metricFrameStatus = document.getElementById("metric-frame-status");
  const metricLatency = document.getElementById("metric-latency");
  const metricTracks = document.getElementById("metric-tracks");
  const trackChipsScroll = document.getElementById("track-chips-scroll");
  const timelineEmptyHint = document.getElementById("timeline-empty-hint");

  // Modals
  const safetyModal = document.getElementById("safety-modal");
  const btnOpenSafetyDetails = document.getElementById("btn-open-safety-details");
  const btnCloseSafetyModal = document.getElementById("btn-close-safety-modal");
  const btnDismissSafetyModal = document.getElementById("btn-dismiss-safety-modal");
  const navSafety = document.getElementById("nav-safety");

  const reviewModal = document.getElementById("review-modal");
  const reviewDialogContent = document.getElementById("review-dialog-content");
  const btnCloseReviewModal = document.getElementById("btn-close-review-modal");
  const btnCancelReview = document.getElementById("btn-cancel-review");
  const btnAuthorizeReview = document.getElementById("btn-authorize-review");

  const toastCenter = document.getElementById("toast-center");

  // Application State
  let currentFile = null;
  let currentImage = null;
  let currentResult = null;
  let selectedDetectionIndex = null;
  let renderedBoundingBoxes = []; // Cached screen coordinates for hit testing

  // Initialize Diagnostics & Health Check
  fetchSystemHealth();

  // --------------------------------------------------------------------------
  // EVENT LISTENERS
  // --------------------------------------------------------------------------

  // Threshold slider
  confSlider.addEventListener("input", (e) => {
    confVal.textContent = parseFloat(e.target.value).toFixed(2);
    renderCanvas();
  });

  // Toggles
  toggleYolo.addEventListener("change", () => renderCanvas());
  toggleSeg.addEventListener("change", () => renderCanvas());

  // View Switcher (Sensor vs Geo View)
  btnViewSensor.addEventListener("click", () => switchView("sensor"));
  btnViewMap.addEventListener("click", () => switchView("map"));

  // File Upload
  imageUploadInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelected(e.target.files[0]);
    }
  });

  // Sample Load
  const loadSampleAction = async () => {
    try {
      showToast("Loading VisDrone sample validation frame...", "info");
      const res = await fetch("/static/samples/visdrone_sample_1.jpg");
      if (res.ok) {
        const blob = await res.blob();
        const file = new File([blob], "visdrone_val_0000001.jpg", { type: "image/jpeg" });
        handleFileSelected(file);
      } else {
        createSyntheticSampleRecon();
      }
    } catch (err) {
      createSyntheticSampleRecon();
    }
  };

  loadSampleBtn.addEventListener("click", loadSampleAction);
  if (emptyLoadSampleBtn) {
    emptyLoadSampleBtn.addEventListener("click", loadSampleAction);
  }

  // Assessment Trigger
  analyzeBtn.addEventListener("click", () => {
    if (currentFile) {
      executeMultimodalAnalysis(currentFile);
    }
  });

  // Drag and drop onto viewport
  viewportContainer.addEventListener("dragover", (e) => {
    e.preventDefault();
  });
  viewportContainer.addEventListener("drop", (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  // Fit & Fullscreen
  btnZoomFit.addEventListener("click", () => {
    uavCanvas.style.transform = "scale(1)";
    showToast("Viewport reset to fit", "info");
  });

  btnFullscreen.addEventListener("click", () => {
    if (!document.fullscreenElement) {
      viewportContainer.requestFullscreen().catch(() => {});
    } else {
      document.exitFullscreen().catch(() => {});
    }
  });

  // Bottom Analytics Toggle
  btnToggleAnalytics.addEventListener("click", () => {
    bottomAnalyticsBar.classList.toggle("collapsed");
    btnToggleAnalytics.classList.toggle("collapsed");
  });

  // Contributing Factors Toggle
  btnToggleFactors.addEventListener("click", () => {
    const isExpanded = btnToggleFactors.getAttribute("aria-expanded") === "true";
    btnToggleFactors.setAttribute("aria-expanded", !isExpanded);
    factorsList.style.display = isExpanded ? "none" : "block";
  });

  // Section Header Collapsibles in Right Panel
  document.querySelectorAll(".section-header-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const isExpanded = btn.getAttribute("aria-expanded") === "true";
      btn.setAttribute("aria-expanded", !isExpanded);
      const targetId = btn.getAttribute("data-target");
      const targetContent = document.getElementById(targetId);
      if (targetContent) {
        targetContent.style.display = isExpanded ? "none" : "block";
      }
    });
  });

  // Collapse/Expand all sections button
  const btnCollapseAll = document.getElementById("btn-collapse-all-sections");
  let allCollapsed = false;
  if (btnCollapseAll) {
    btnCollapseAll.addEventListener("click", () => {
      allCollapsed = !allCollapsed;
      document.querySelectorAll(".section-header-btn").forEach((btn) => {
        btn.setAttribute("aria-expanded", !allCollapsed);
        const targetId = btn.getAttribute("data-target");
        const targetContent = document.getElementById(targetId);
        if (targetContent) {
          targetContent.style.display = allCollapsed ? "none" : "block";
        }
      });
    });
  }

  // Modals Listeners
  const openSafety = () => { safetyModal.style.display = "flex"; };
  const closeSafety = () => { safetyModal.style.display = "none"; };
  btnOpenSafetyDetails.addEventListener("click", openSafety);
  if (navSafety) navSafety.addEventListener("click", openSafety);
  btnCloseSafetyModal.addEventListener("click", closeSafety);
  btnDismissSafetyModal.addEventListener("click", closeSafety);

  const openReview = () => { populateReviewModal(); reviewModal.style.display = "flex"; };
  const closeReview = () => { reviewModal.style.display = "none"; };
  btnReviewRecommendation.addEventListener("click", openReview);
  btnCloseReviewModal.addEventListener("click", closeReview);
  btnCancelReview.addEventListener("click", closeReview);
  btnAuthorizeReview.addEventListener("click", () => {
    closeReview();
    recReviewStatus.textContent = "Authorized (Coordinator Sign-Off)";
    recReviewStatus.className = "spec-val tag-success";
    showToast("Human Authorization Recorded: Advisory recommendation marked as reviewed.", "success");
  });

  // Canvas Mouse Interactions (Hit testing detections for clicks & tooltips)
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
      detectionTooltip.style.left = `${e.clientX + 12}px`;
      detectionTooltip.style.top = `${e.clientY + 12}px`;
      detectionTooltip.innerHTML = `
        <div style="font-weight:600;color:#60A5FA;">${hit.trackId}</div>
        <div>Confidence: ${(hit.confidence * 100).toFixed(1)}%</div>
        <div style="color:#94A3B8;font-size:10px;">${hit.coords}</div>
      `;
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
  // CORE FUNCTIONS: FILE SELECTION & INGESTION
  // --------------------------------------------------------------------------

  function handleFileSelected(file) {
    if (!file.type.startsWith("image/")) {
      showToast("Invalid file: Please provide an aerial reconnaissance image (JPEG, PNG, WebP).", "error");
      return;
    }

    currentFile = file;
    loadedFilename.textContent = file.name;
    analyzeBtn.disabled = false;

    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => {
        currentImage = img;
        currentResult = null;
        selectedDetectionIndex = null;

        emptyState.style.display = "none";
        uavCanvas.style.display = "block";
        mediaMetaBadge.textContent = `${img.naturalWidth} \u00d7 ${img.naturalHeight} px \u2022 ${(file.size / 1024).toFixed(0)} KB`;

        renderCanvas();
        resetPanelsToAwaitingState();
      };
      img.src = e.target.result;
    };
    reader.readAsDataURL(file);
  }

  // --------------------------------------------------------------------------
  // CORE FUNCTIONS: MULTIMODAL API INTEGRATION
  // --------------------------------------------------------------------------

  async function executeMultimodalAnalysis(file) {
    showLoading(true);
    setStepState("step-upload", "done");
    setStepState("step-yolo", "active");

    const startTime = performance.now();
    const formData = new FormData();
    formData.append("file", file);
    formData.append("confidence_threshold", confSlider.value);
    formData.append("enable_segmentation", "true");

    try {
      setTimeout(() => {
        setStepState("step-yolo", "done");
        setStepState("step-seg", "active");
      }, 350);

      setTimeout(() => {
        setStepState("step-seg", "done");
        setStepState("step-prio", "active");
      }, 700);

      const response = await fetch("/api/v1/analysis/multimodal", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server returned HTTP ${response.status}`);
      }

      const result = await response.json();
      currentResult = result;
      showLoading(false);

      const durationMs = Math.round(performance.now() - startTime);
      metricLatency.textContent = `Inference: ${durationMs}ms`;

      // Update All Redesigned Dashboard Panels
      updateTelemetry(result.device_info);
      renderCanvas();
      populateSituationPanel(result);
      populateEnvironmentalPanel(result.scene_context);
      populateDetectionsPanel(result.detections || []);
      populateRecommendationPanel(result.recommendation_outcome);
      populateBottomAnalytics(result, durationMs);
      updateGeoMap(result.detections || []);

      const count = result.total_detections_found || 0;
      showToast(`Assessment complete: ${count} potential person${count === 1 ? "" : "s"} identified.`, "success");

    } catch (err) {
      showLoading(false);
      showToast(`Multimodal analysis failed: ${err.message}`, "error");
    }
  }

  // --------------------------------------------------------------------------
  // VISUAL WORKSPACE: CANVAS RENDERING
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

    // 1. Environmental Segmentation Context Layer
    if (toggleSeg.checked && currentResult.scene_context) {
      drawSegmentationContextOverlay(currentResult.scene_context);
    }

    // 2. Person Detection Bounding Boxes
    if (toggleYolo.checked && currentResult.detections) {
      drawBoundingBoxes(currentResult.detections);
    }
  }

  function drawBoundingBoxes(detections) {
    const thresh = parseFloat(confSlider.value);
    const filtered = detections.filter(d => d.confidence >= thresh);
    toolbarPersonCount.textContent = filtered.length;

    filtered.forEach((det, idx) => {
      const bx = det.bbox;
      // Coordinate normalization check
      const x1 = bx.x1 <= 1.0 ? bx.x1 * uavCanvas.width : bx.x1;
      const y1 = bx.y1 <= 1.0 ? bx.y1 * uavCanvas.height : bx.y1;
      const x2 = bx.x2 <= 1.0 ? bx.x2 * uavCanvas.width : bx.x2;
      const y2 = bx.y2 <= 1.0 ? bx.y2 * uavCanvas.height : bx.y2;
      const w = Math.max(2, x2 - x1);
      const h = Math.max(2, y2 - y1);

      const isSelected = selectedDetectionIndex === idx;

      // Cache box for hit testing
      const trackId = det.track_id ? `TRK-${String(det.track_id).padStart(2, "0")}` : `ID-${idx + 1}`;
      const coordsText = det.simulated_lat && det.simulated_lon 
        ? `${det.simulated_lat.toFixed(4)}\u00b0 N, ${det.simulated_lon.toFixed(4)}\u00b0 W` 
        : "Simulated projection";

      renderedBoundingBoxes.push({
        origIndex: idx,
        trackId: trackId,
        confidence: det.confidence,
        coords: coordsText,
        x1: x1,
        y1: y1,
        x2: x2,
        y2: y2
      });

      // Box Styling — High legibility, restrained colors
      ctx.save();
      if (selectedDetectionIndex !== null && !isSelected) {
        // Dim unselected detections slightly
        ctx.globalAlpha = 0.45;
      }

      const boxColor = isSelected ? "#2563EB" : "#16A34A";
      const lineWidth = isSelected ? Math.max(3, Math.round(uavCanvas.width / 450)) : Math.max(2, Math.round(uavCanvas.width / 600));

      // Subtle translucent box fill
      ctx.fillStyle = isSelected ? "rgba(37, 99, 235, 0.15)" : "rgba(22, 163, 74, 0.08)";
      ctx.fillRect(x1, y1, w, h);

      // Clean border
      ctx.strokeStyle = boxColor;
      ctx.lineWidth = lineWidth;
      ctx.strokeRect(x1, y1, w, h);

      // Compact, collision-aware label pill
      const confPct = Math.round(det.confidence * 100);
      const labelText = `${trackId} \u2022 ${confPct}%`;

      const fontSize = Math.max(11, Math.min(14, Math.round(uavCanvas.width / 110)));
      ctx.font = `600 ${fontSize}px 'Inter', sans-serif`;
      const textWidth = ctx.measureText(labelText).width;
      const pillHeight = fontSize + 8;
      const pillWidth = textWidth + 12;

      // Position pill above box if room, else inside top
      let pillY = y1 - pillHeight - 2;
      if (pillY < 0) {
        pillY = y1 + 2;
      }

      ctx.fillStyle = boxColor;
      roundRect(ctx, x1, pillY, pillWidth, pillHeight, 3);
      ctx.fill();

      // Label text
      ctx.fillStyle = "#FFFFFF";
      ctx.fillText(labelText, x1 + 6, pillY + fontSize);

      ctx.restore();
    });
  }

  function drawSegmentationContextOverlay(scene) {
    if (!scene) return;
    ctx.save();

    // Subtle hazard sector indicator at canvas corner
    const pad = 16;
    const badgeW = Math.max(140, Math.round(uavCanvas.width / 6));
    const badgeH = 26;

    ctx.fillStyle = "rgba(15, 23, 42, 0.75)";
    roundRect(ctx, pad, pad, badgeW, badgeH, 4);
    ctx.fill();

    const sev = (scene.inferred_hazard_severity || "NONE").toUpperCase();
    ctx.font = "600 11px 'Inter', sans-serif";
    ctx.fillStyle = sev === "SEVERE" ? "#F87171" : (sev === "MODERATE" ? "#FBBF24" : "#34D399");
    ctx.fillText(`Hazard: ${sev}`, pad + 10, pad + 17);

    ctx.restore();
  }

  // --------------------------------------------------------------------------
  // RIGHT CONTEXT PANEL POPULATION
  // --------------------------------------------------------------------------

  function populateSituationPanel(result) {
    const prio = result.priority_assessment || {};
    const scene = result.scene_context || {};

    // Header priority badge
    const prioLevel = (prio.composite_priority || "STANDBY").toUpperCase();
    headerPriorityBadge.textContent = prioLevel;
    headerPriorityBadge.className = "priority-badge-compact " + (
      prioLevel.includes("CRITICAL") ? "critical" : 
      prioLevel.includes("HIGH") ? "high" : 
      prioLevel.includes("ELEVATED") ? "elevated" : ""
    );

    // Hazard Severity Tag
    const hazard = (scene.inferred_hazard_severity || "NORMAL").toUpperCase();
    situationHazardTag.textContent = hazard;

    // Urgency
    const urgScore = prio.urgency_score !== undefined ? Math.round(prio.urgency_score * 100) : 0;
    urgencyNumber.textContent = urgScore;
    urgencyLevelTag.textContent = prio.urgency_level || "Low";
    urgencyBarFill.style.width = `${Math.min(100, urgScore)}%`;

    // Uncertainty
    const uncScore = prio.uncertainty_score !== undefined ? Math.round(prio.uncertainty_score * 100) : 0;
    uncertaintyNumber.textContent = uncScore;
    uncertaintyLevelTag.textContent = prio.uncertainty_level || "Low";
    uncertaintyBarFill.style.width = `${Math.min(100, uncScore)}%`;

    // Concise Executive Summary
    const personCount = result.total_detections_found || 0;
    if (personCount > 0) {
      situationSummaryText.textContent = `${personCount} potential person${personCount > 1 ? "s" : ""} detected in sector. Terrain accessibility is evaluated as ${scene.inferred_accessibility || "Accessible"}.`;
    } else {
      situationSummaryText.textContent = "No potential persons detected. Environmental baseline clear.";
    }

    // Contributing Factors List
    factorsList.innerHTML = "";
    const factors = prio.contributing_factors || [];
    if (factors.length > 0) {
      factors.forEach(f => {
        const div = document.createElement("div");
        div.className = "factor-bullet";
        div.textContent = f;
        factorsList.appendChild(div);
      });
    } else {
      factorsList.innerHTML = '<div class="factor-bullet">Standard aerial surveillance parameters.</div>';
    }
  }

  function populateEnvironmentalPanel(scene) {
    if (!scene) return;

    // Accessibility Pill
    const access = (scene.inferred_accessibility || "ACCESSIBLE").toUpperCase();
    accessPill.textContent = access;
    accessPill.className = "access-pill " + (
      access.includes("BLOCKED") ? "blocked" :
      access.includes("LIMITED") ? "limited" : ""
    );

    // Coverage Bars
    const water = scene.water_coverage_pct || 0;
    envWaterPct.textContent = `${water.toFixed(1)}%`;
    envWaterBar.style.width = `${Math.min(100, water)}%`;

    const debris = scene.debris_coverage_pct || 0;
    envDebrisPct.textContent = `${debris.toFixed(1)}%`;
    envDebrisBar.style.width = `${Math.min(100, debris)}%`;

    const road = scene.road_coverage_pct || 0;
    envRoadPct.textContent = `${road.toFixed(1)}%`;
    envRoadBar.style.width = `${Math.min(100, road)}%`;

    const tree = scene.tree_coverage_pct || 0;
    envTreePct.textContent = `${tree.toFixed(1)}%`;
    envTreeBar.style.width = `${Math.min(100, tree)}%`;

    const dist = scene.class_distribution_pct || {};
    const damage = (dist["Building-Damaged"] || 0) + (dist["Building-Destroyed"] || 0);
    envDamagePct.textContent = `${damage.toFixed(1)}%`;
    envDamageBar.style.width = `${Math.min(100, damage)}%`;

    // Inferred Capabilities
    capabilityPills.innerHTML = "";
    const caps = scene.inferred_required_capabilities || [];
    if (caps.length > 0) {
      caps.forEach(c => {
        const pill = document.createElement("span");
        pill.className = "cap-pill";
        pill.textContent = c.replace(/_/g, " ");
        capabilityPills.appendChild(pill);
      });
    } else {
      capabilityPills.innerHTML = '<span class="cap-pill-empty">No hazardous capability dependencies</span>';
    }
  }

  function populateDetectionsPanel(detections) {
    detectionsTotalTag.textContent = `${detections.length} Person${detections.length === 1 ? "" : "s"}`;
    listCounterText.textContent = `${detections.length} record${detections.length === 1 ? "" : "s"}`;

    detectionsItemsList.innerHTML = "";

    if (detections.length === 0) {
      detectionsItemsList.innerHTML = '<div class="detection-empty-item">No potential persons detected in current frame.</div>';
      selectedTrackId.textContent = "No active tracks";
      selectedTrackStatus.textContent = "Standby";
      selectedTrackConf.textContent = "\u2014";
      selectedTrackUncertainty.textContent = "\u2014";
      selectedTrackCoords.textContent = "\u2014";
      return;
    }

    detections.forEach((det, idx) => {
      const trackId = det.track_id ? `TRK-${String(det.track_id).padStart(2, "0")}` : `ID-${idx + 1}`;
      const confPct = `${(det.confidence * 100).toFixed(1)}%`;

      const row = document.createElement("div");
      row.className = "detection-row-item" + (selectedDetectionIndex === idx ? " selected" : "");
      row.innerHTML = `
        <span class="row-track-id">${trackId}</span>
        <span class="row-conf">${confPct}</span>
      `;

      row.addEventListener("click", () => {
        selectDetection(idx);
      });

      detectionsItemsList.appendChild(row);
    });

    // Auto-select first detection if none selected
    if (selectedDetectionIndex === null || selectedDetectionIndex >= detections.length) {
      selectDetection(0);
    } else {
      selectDetection(selectedDetectionIndex);
    }
  }

  function selectDetection(idx) {
    if (idx === null || !currentResult || !currentResult.detections || !currentResult.detections[idx]) {
      selectedDetectionIndex = null;
      renderCanvas();
      return;
    }

    selectedDetectionIndex = idx;
    const det = currentResult.detections[idx];
    const trackId = det.track_id ? `TRK-${String(det.track_id).padStart(2, "0")}` : `ID-${idx + 1}`;

    selectedTrackId.textContent = trackId;
    selectedTrackStatus.textContent = det.review_status || "Pending Review";
    selectedTrackConf.textContent = `${(det.confidence * 100).toFixed(1)}%`;
    selectedTrackUncertainty.textContent = `\u00b1 ${(det.uncertainty_radius_meters || 15.0).toFixed(1)} m`;

    if (det.simulated_lat && det.simulated_lon) {
      selectedTrackCoords.textContent = `${det.simulated_lat.toFixed(5)}\u00b0 N, ${det.simulated_lon.toFixed(5)}\u00b0 W (Simulated)`;
    } else {
      selectedTrackCoords.textContent = "Simulated projection active";
    }

    // Update active row classes
    document.querySelectorAll(".detection-row-item").forEach((r, i) => {
      r.classList.toggle("selected", i === idx);
    });

    // Update bottom chips
    document.querySelectorAll(".track-chip").forEach((chip, i) => {
      chip.classList.toggle("selected", i === idx);
    });

    renderCanvas();
  }

  function populateRecommendationPanel(recOutcome) {
    if (!recOutcome) return;

    const recs = recOutcome.recommendations || [];
    if (recs.length === 0) {
      recResourceName.textContent = "No Suitable Resource";
      recResourceType.textContent = "Advisory";
      recSuitabilityVal.textContent = "\u2014";
      recDistanceVal.textContent = "\u2014";
      recRationaleText.textContent = recOutcome.rationale || "All simulated resource units currently committed or unsuitable for sector constraints.";
      btnReviewRecommendation.disabled = true;
      return;
    }

    const primaryRec = recs[0];
    recResourceName.textContent = primaryRec.resource_name || "Emergency Resource Team";
    recResourceType.textContent = (primaryRec.resource_type || "SAR Team").replace(/_/g, " ");

    const suit = Math.round(primaryRec.suitability_score * 100);
    recSuitabilityVal.textContent = `${suit}%`;
    recDistanceVal.textContent = primaryRec.estimated_distance_meters 
      ? `${Math.round(primaryRec.estimated_distance_meters)} m` 
      : "720 m (Est.)";

    recReviewStatus.textContent = primaryRec.status || "Pending Review";
    recRationaleText.textContent = primaryRec.rationale || "Optimal match based on capability coverage and shortest simulated approach corridor.";

    recCapabilitiesList.innerHTML = "";
    const caps = primaryRec.matched_capabilities || ["Rapid Response"];
    caps.forEach(c => {
      const tag = document.createElement("span");
      tag.className = "cap-tag-pill";
      tag.textContent = c.replace(/_/g, " ");
      recCapabilitiesList.appendChild(tag);
    });

    btnReviewRecommendation.disabled = false;
  }

  // --------------------------------------------------------------------------
  // BOTTOM ANALYTICS & TIMELINE REGION
  // --------------------------------------------------------------------------

  function populateBottomAnalytics(result, durationMs) {
    const tracks = result.detections || [];
    metricTracks.textContent = `Active Tracks: ${tracks.length}`;
    metricFrameStatus.textContent = `Sortie 001 \u2022 ${tracks.length} entities`;

    trackChipsScroll.innerHTML = "";
    if (tracks.length === 0) {
      timelineEmptyHint.style.display = "block";
    } else {
      timelineEmptyHint.style.display = "none";
      tracks.forEach((det, idx) => {
        const trackId = det.track_id ? `TRK-${String(det.track_id).padStart(2, "0")}` : `ID-${idx + 1}`;
        const confPct = `${(det.confidence * 100).toFixed(0)}%`;

        const chip = document.createElement("div");
        chip.className = "track-chip" + (selectedDetectionIndex === idx ? " selected" : "");
        chip.innerHTML = `
          <span class="chip-track-id">${trackId}</span>
          <span class="chip-conf">${confPct}</span>
          <span class="chip-coords">\u00b1${(det.uncertainty_radius_meters || 15).toFixed(0)}m</span>
        `;
        chip.addEventListener("click", () => selectDetection(idx));
        trackChipsScroll.appendChild(chip);
      });
    }
  }

  // --------------------------------------------------------------------------
  // SIMULATED GEO MAP VIEW
  // --------------------------------------------------------------------------

  function switchView(mode) {
    if (mode === "sensor") {
      btnViewSensor.classList.add("active");
      btnViewMap.classList.remove("active");
      canvasWrapper.style.display = "flex";
      geoMapView.style.display = "none";
    } else {
      btnViewSensor.classList.remove("active");
      btnViewMap.classList.add("active");
      canvasWrapper.style.display = "none";
      geoMapView.style.display = "flex";
    }
  }

  function updateGeoMap(detections) {
    geoDetectionsGroup.innerHTML = "";
    detections.forEach((det, idx) => {
      // Map pixel coordinates inside 800x500 SVG
      const cx = 320 + (idx * 45);
      const cy = 200 + (idx * 25);
      const rUncertainty = 30;

      // Group
      const g = document.createElementNS("http://www.w3.org/2000/svg", "g");

      // Uncertainty circle
      const circ = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      circ.setAttribute("cx", cx);
      circ.setAttribute("cy", cy);
      circ.setAttribute("r", rUncertainty);
      circ.setAttribute("fill", "#EFF6FF");
      circ.setAttribute("stroke", "#3B82F6");
      circ.setAttribute("stroke-width", "1");
      circ.setAttribute("stroke-dasharray", "3 3");
      circ.setAttribute("opacity", "0.7");
      g.appendChild(circ);

      // Person marker dot
      const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      dot.setAttribute("cx", cx);
      dot.setAttribute("cy", cy);
      dot.setAttribute("r", "5");
      dot.setAttribute("fill", "#2563EB");
      g.appendChild(dot);

      // Label
      const txt = document.createElementNS("http://www.w3.org/2000/svg", "text");
      txt.setAttribute("x", cx + 8);
      txt.setAttribute("y", cy + 4);
      txt.setAttribute("font-size", "10");
      txt.setAttribute("font-weight", "600");
      txt.setAttribute("fill", "#1E40AF");
      txt.textContent = det.track_id ? `TRK-${det.track_id}` : `ID-${idx + 1}`;
      g.appendChild(txt);

      geoDetectionsGroup.appendChild(g);
    });
  }

  // --------------------------------------------------------------------------
  // MODAL DIALOGS
  // --------------------------------------------------------------------------

  function populateReviewModal() {
    if (!currentResult || !currentResult.recommendation_outcome) return;
    const recs = currentResult.recommendation_outcome.recommendations || [];
    if (recs.length === 0) return;

    const r = recs[0];
    reviewDialogContent.innerHTML = `
      <div style="font-size:13px;line-height:1.6;color:#334155;">
        <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:6px;padding:12px;margin-bottom:12px;">
          <div style="font-size:11px;color:#64748B;text-transform:uppercase;font-weight:600;">Resource Candidate</div>
          <div style="font-size:15px;font-weight:600;color:#0F172A;">${r.resource_name}</div>
          <div style="font-size:12px;color:#475569;">Type: ${r.resource_type} &bull; Estimated Transit: ${Math.round(r.estimated_distance_meters || 720)}m</div>
        </div>

        <p style="margin-bottom:8px;"><strong>Rationale:</strong> ${r.rationale || 'Matched against sector constraints.'}</p>
        
        <div style="background:#FFFBEB;border:1px solid #FDE68A;border-radius:6px;padding:10px;margin-top:12px;font-size:11px;color:#92400E;">
          <strong>Human-in-the-Loop Gate:</strong> Autonomous resource dispatch is strictly blocked. Signing off records confirmation that an operational coordinator has reviewed the advisory recommendation.
        </div>
      </div>
    `;
  }

  // --------------------------------------------------------------------------
  // HEALTH & TELEMETRY
  // --------------------------------------------------------------------------

  async function fetchSystemHealth() {
    try {
      const res = await fetch("/health");
      if (res.ok) {
        const data = await res.json();
        systemStatusPill.querySelector(".status-text").textContent = "System ready";
      }
    } catch (err) {
      systemStatusPill.querySelector(".status-dot-mini").className = "status-dot-mini amber";
      systemStatusPill.querySelector(".status-text").textContent = "Connecting...";
    }
  }

  function updateTelemetry(deviceInfo) {
    if (!deviceInfo) return;
    if (deviceInfo.cuda_available) {
      deviceInfoText.textContent = `${deviceInfo.gpu_name || "RTX 2050"} \u2022 CUDA`;
    } else {
      deviceInfoText.textContent = "CPU Fallback";
    }
  }

  function resetPanelsToAwaitingState() {
    toolbarPersonCount.textContent = "0";
    headerPriorityBadge.textContent = "Standby";
    headerPriorityBadge.className = "priority-badge-compact";
    situationHazardTag.textContent = "Normal";
    urgencyNumber.textContent = "0";
    urgencyBarFill.style.width = "0%";
    uncertaintyNumber.textContent = "0";
    uncertaintyBarFill.style.width = "0%";
    situationSummaryText.textContent = "Media loaded. Click 'Run Assessment' to perform dual-model analysis.";
    factorsList.innerHTML = '<div class="factor-bullet">Awaiting model execution.</div>';
    accessPill.textContent = "Accessible";
    accessPill.className = "access-pill";
    capabilityPills.innerHTML = '<span class="cap-pill-empty">Pending assessment</span>';
    btnReviewRecommendation.disabled = true;
  }

  function setStepState(stepId, state) {
    const el = document.getElementById(stepId);
    if (!el) return;
    const ind = el.querySelector(".step-indicator");
    ind.className = `step-indicator ${state}`;
    if (state === "done") {
      ind.innerHTML = "&check;";
    } else if (state === "active") {
      ind.innerHTML = "&bull;";
    } else {
      ind.innerHTML = "&bull;";
    }
  }

  function showLoading(show) {
    loadingOverlay.style.display = show ? "flex" : "none";
  }

  function showToast(message, type = "info") {
    const toast = document.createElement("div");
    toast.className = `toast-msg ${type}`;
    toast.textContent = message;
    toastCenter.appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = "0";
      setTimeout(() => toast.remove(), 250);
    }, 3800);
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

  function createSyntheticSampleRecon() {
    const offCanvas = document.createElement("canvas");
    offCanvas.width = 1024;
    offCanvas.height = 768;
    const offCtx = offCanvas.getContext("2d");

    // Tactical aerial scenery
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
      showToast("Generated synthetic aerial frame for demonstration", "info");
    }, "image/jpeg");
  }

  // Expose key functions for testing & integration
  window.drawBoundingBoxes = drawBoundingBoxes;
  window.executeMultimodalAnalysis = executeMultimodalAnalysis;
});
