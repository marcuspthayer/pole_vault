"""
VaultSense FastAPI application.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.jobs import router as jobs_router
from api.services.job_runner import cleanup_old_jobs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vaultsense")

ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,https://vaultsense.app,https://www.vaultsense.app"
).split(",")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start background job cleanup loop
    cleanup_task = asyncio.create_task(cleanup_old_jobs())
    logger.info("VaultSense API started")
    yield
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    logger.info("VaultSense API shut down")


app = FastAPI(
    title="VaultSense API",
    description="Pole vault video analysis — AI-powered biomechanics",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(jobs_router)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "vaultsense-api"}
