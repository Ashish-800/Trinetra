# Proposed Folder Structure

**Project:** AI-Enabled UAV System for Real-Time Disaster Assessment  
**Document Version:** 1.0.0 (Phase 0 Planning)  

---

```text
disaster-uav/
├── .gitignore                   # Git ignore patterns for venv, cache, weights, and datasets
├── README.md                    # Setup guide, execution instructions, and API reference
├── requirements.txt             # Locked Python dependencies
├── verify_env.py                # Environment and dependency verification script
├── app/                         # Core Application Source Code
│   ├── __init__.py
│   ├── main.py                  # FastAPI application entry point, lifecycle events & middleware
│   ├── api/                     # REST API Layer
│   │   ├── __init__.py
│   │   ├── deps.py              # Dependency injection (database session, detector singleton)
│   │   └── v1/
│   │       ├── __init__.py
│   │       ├── api.py           # v1 API router aggregation
│   │       └── endpoints/
│   │           ├── __init__.py
│   │           ├── health.py    # Health check & system diagnostics
│   │           ├── missions.py  # Mission management endpoints
│   │           ├── media.py     # Image and video upload / ingestion endpoints
│   │           ├── detections.py# Detection queries, spatial filters & priority queues
│   │           ├── resources.py # Emergency resource inventory & advisory allocations
│   │           └── reviews.py   # Human review, confirmation, and override actions
│   ├── core/                    # Core Infrastructure & Cross-Cutting Concerns
│   │   ├── __init__.py
│   │   ├── config.py            # Pydantic Settings (env variables, paths, thresholds)
│   │   ├── logging.py           # Structured JSON / formatted logging configuration
│   │   └── security.py          # Input sanitation, path traversal guards & API keys (if needed)
│   ├── db/                      # Database Infrastructure
│   │   ├── __init__.py
│   │   ├── base.py              # Base declarative class and ORM model imports
│   │   └── session.py           # Engine creation, sessionmaker, and table initialization
│   ├── models/                  # SQLAlchemy ORM Models (Persistence Domain)
│   │   ├── __init__.py
│   │   ├── mission.py           # Mission metadata & operational parameters
│   │   ├── frame.py             # AerialFrame record, telemetry attachment, file storage path
│   │   ├── detection.py         # Detected bounding boxes, person/hazard labels, coordinates
│   │   ├── resource.py          # Available emergency vehicles, medical kits, rescue teams
│   │   ├── recommendation.py    # Advisory resource assignment proposals
│   │   └── review_log.py        # Operator verification audit log
│   ├── schemas/                 # Pydantic Schemas (API Data Contracts & Validation)
│   │   ├── __init__.py
│   │   ├── common.py            # Shared types (GPS coordinates, BoundingBox, SeverityEnum)
│   │   ├── mission.py           # MissionCreate, MissionRead, MissionUpdate
│   │   ├── telemetry.py         # UAVTelemetryInput (altitude, lat, lon, gimbal pitch/yaw)
│   │   ├── frame.py             # FrameRead, FrameProcessingResult
│   │   ├── detection.py         # DetectionRead, DetectionFilter, PrioritySummary
│   │   ├── resource.py          # ResourceCreate, ResourceRead, ResourceAllocationPlan
│   │   └── review.py            # HumanReviewAction, HumanReviewRecord
│   └── services/                # Business Logic & Algorithms
│       ├── __init__.py
│       ├── preprocessor.py      # OpenCV video frame extraction and image normalization
│       ├── detector.py          # YOLO detector wrapper & MockDetector for tests
│       ├── tracker.py           # Aerial multi-object tracker (IoU / Centroid tracking)
│       ├── geo_estimator.py     # Camera ray-projection to estimated ground coordinates
│       ├── scorer.py            # Urgency & Uncertainty computation engine
│       └── recommender.py       # Decision-support resource matching engine
├── docs/                        # Project Documentation & Architecture
│   ├── architecture.md          # Comprehensive system architecture & data flow
│   ├── phase_plan.md            # Roadmap, phase dependencies & verification steps
│   ├── folder_structure.md      # Repository directory structure (this file)
│   ├── assumptions_and_decisions.md # Explicit engineering assumptions & open decisions
│   └── risk_assessment.md       # Operational, data quality, privacy & safety risk analysis
└── tests/                       # Automated Test Suite (pytest)
    ├── __init__.py
    ├── conftest.py              # Shared fixtures (in-memory SQLite session, test media, mock detector)
    ├── unit/                    # Unit Tests
    │   ├── test_schemas.py      # Pydantic validation & constraint tests
    │   ├── test_preprocessor.py # Image/video frame reader tests
    │   ├── test_detector.py     # Mock detector & YOLO wrapper tests
    │   ├── test_tracker.py      # Tracking ID consistency across frames
    │   ├── test_geo.py          # Coordinate estimation & projection mathematics
    │   ├── test_scorer.py       # Urgency & uncertainty score calculations
    │   └── test_recommender.py  # Resource recommendation logic & constraint tests
    └── api/                     # API Integration Tests
        ├── test_health.py       # System health & ping
        ├── test_missions.py     # Mission lifecycle endpoints
        ├── test_media.py        # File upload & frame processing
        ├── test_detections.py   # Detection retrieval & filtering
        └── test_reviews.py      # Human operator review & status transitions
```
