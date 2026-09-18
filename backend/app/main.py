import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import connections as connections_router
from app.api import datasets as datasets_router
from app.api import discovery as discovery_router
from app.api import jobs as jobs_router
from app.config import get_settings
from app.schemas import ErrorEnvelope

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("automl")

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description="Dataset Discovery + Automated Machine Learning",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Never expose raw stack traces to the client."""
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content=ErrorEnvelope(
            code="INTERNAL_ERROR",
            message="An unexpected error occurred.",
            what_happened="The server hit an unexpected error while processing the request.",
            why="This is usually a bug in the application or an unavailable dependency.",
            what_to_do="Retry the operation. If it persists, check the backend logs.",
        ).model_dump(),
    )


@app.get("/api/health")
def health():
    return {"status": "ok"}


app.include_router(discovery_router.router)
app.include_router(datasets_router.router)
app.include_router(jobs_router.router)
app.include_router(connections_router.router)


@app.on_event("startup")
def create_tables():
    """Dev convenience: create tables when they don't exist. Alembic handles migrations."""
    from app.db import Base, engine

    import app.models  # noqa: F401  (registers ORM models on Base metadata)

    Base.metadata.create_all(bind=engine)
