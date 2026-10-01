"""
Disaster UAV Assessment System - FastAPI Application Entry Point.
Provides RESTful APIs for aerial disaster assessment, detection inspection,
priority scoring, and advisory emergency resource recommendation.
Includes hardened error sanitization and defensive exception handling.
"""
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.v1.endpoints import health
from app.api.v1.router import api_router
from app.core.config import settings
from app.core.logging import setup_logging
from app.db.session import init_db

logger = logging.getLogger("app.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler initializing DB tables, directories, and sanitized logging."""
    setup_logging()
    init_db()
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=(
        "Decision-support system for aerial disaster imagery and video assessment. "
        "Adheres strictly to the Project Safety Constitution: all detection labels are 'person' "
        "or 'potential survivor', coordinates are explicitly labeled 'simulated', and resource "
        "recommendations are strictly advisory, requiring human authorization."
    ),
    lifespan=lifespan,
)

# Enable CORS for standard web and desktop API consumers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root-level health check
app.include_router(health.router)

# Mount API v1 router
app.include_router(api_router, prefix=settings.API_V1_STR)

# Mount static and uploads directories for frontend dashboard
static_dir = Path("static")
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

uploads_dir = settings.UPLOAD_DIR
uploads_dir.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(uploads_dir)), name="uploads")


@app.get("/", response_class=FileResponse, include_in_schema=False)
@app.get("/dashboard", response_class=FileResponse, include_in_schema=False)
async def serve_dashboard():
    """Serves the Tactical Disaster UAV Command Center Dashboard."""
    index_file = Path("static/index.html")
    if index_file.exists():
        return FileResponse(str(index_file))
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"status": "online", "message": "Disaster UAV Command Center API Online"}
    )



@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Custom handler for domain and safety validation errors."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": str(exc), "error_code": "VALIDATION_ERROR"},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Sanitized handler for known HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "error_code": f"HTTP_{exc.status_code}"},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Defensive global handler preventing raw stack traces or filesystem leakage."""
    logger.error(f"Unhandled server exception on {request.method} {request.url.path}: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An internal server error occurred. Sensitive execution traces are masked for security.",
            "error_code": "INTERNAL_SERVER_ERROR",
        },
    )
