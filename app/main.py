"""AgentCare application entrypoint.

Run locally:  uvicorn app.main:app --reload
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from app.config import settings
from app.db.base import init_db

logging.basicConfig(
    level=logging.INFO if not settings.debug else logging.DEBUG,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
logger = logging.getLogger("agentcare")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    # Importing the tool modules is what registers them in the tool registry.
    from app.tools import documents, hospital, oversight  # noqa: F401
    from app.tools.registry import REGISTRY

    logger.info("Registered %d tools: %s", len(REGISTRY), ", ".join(sorted(REGISTRY)))
    if not settings.llm_enabled:
        logger.warning(
            "GROQ_API_KEY is not set — agents will run on deterministic fallbacks "
            "and workflows will be marked degraded."
        )
    yield


app = FastAPI(
    title="AgentCare",
    description="Agentic AI for patient administration and care coordination.",
    version="1.0.0",
    lifespan=lifespan,
)

from app.web.routes import router  # noqa: E402

app.include_router(router)


@app.exception_handler(401)
async def unauthorised(request: Request, exc):
    if request.url.path.startswith("/api"):
        return JSONResponse({"detail": "Not authenticated"}, status_code=401)
    # Clear the dead cookie on the way out. Without this, "/" would see a
    # cookie, bounce to /dashboard, fail again, and loop forever.
    response = RedirectResponse("/", status_code=302)
    response.delete_cookie("access_token")
    return response


@app.exception_handler(403)
async def forbidden(request: Request, exc):
    detail = getattr(exc, "detail", "Forbidden")
    if request.url.path.startswith("/api"):
        return JSONResponse({"detail": detail}, status_code=403)
    return HTMLResponse(
        f"<h1>403 — not permitted</h1><p>{detail}</p><p><a href='/dashboard'>Back</a></p>",
        status_code=403,
    )