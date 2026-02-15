"""FastAPI route handlers for CivitAI API endpoints."""

from __future__ import annotations

import logging
from enum import Enum
from typing import Annotated, Any

import httpx
from fastapi import APIRouter, Query, Response
from fastapi.responses import JSONResponse

from tensors.config import CIVITAI_API_BASE, load_api_key
from tensors.db import Database

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/civitai", tags=["CivitAI"])


class SortOrder(str, Enum):
    """Sort order options for CivitAI search."""

    most_downloaded = "Most Downloaded"
    highest_rated = "Highest Rated"
    newest = "Newest"


class Period(str, Enum):
    """Time period filter options."""

    all = "AllTime"
    year = "Year"
    month = "Month"
    week = "Week"
    day = "Day"


class NsfwLevel(str, Enum):
    """NSFW content filter levels."""

    none = "None"
    soft = "Soft"
    mature = "Mature"
    x = "X"


class CommercialUse(str, Enum):
    """Commercial use filter options."""

    none = "None"
    image = "Image"
    rent = "Rent"
    sell = "Sell"


def _get_headers(api_key: str | None) -> dict[str, str]:
    """Get headers for CivitAI API requests."""
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


@router.get("/search", response_model=None)
async def search_models(
    query: Annotated[str | None, Query(description="Search query")] = None,
    types: Annotated[str | None, Query(description="Model type (Checkpoint, LORA, etc.)")] = None,
    base_models: Annotated[str | None, Query(alias="baseModels", description="Base model")] = None,
    sort: Annotated[SortOrder, Query(description="Sort order")] = SortOrder.most_downloaded,
    limit: Annotated[int | None, Query(le=100, description="Max results (default: 25)", example=5)] = None,
    period: Annotated[Period | None, Query(description="Time period filter")] = None,
    tag: Annotated[str | None, Query(description="Filter by tag")] = None,
    username: Annotated[str | None, Query(description="Filter by creator username")] = None,
    page: Annotated[int | None, Query(ge=1, description="Page number")] = None,
    nsfw: Annotated[NsfwLevel | None, Query(description="NSFW filter level")] = None,
    sfw: Annotated[bool, Query(description="Exclude NSFW content")] = False,
    commercial: Annotated[CommercialUse | None, Query(description="Commercial use filter")] = None,
) -> dict[str, Any] | Response:
    """Search CivitAI models.

    Supports all CivitAI search parameters including filters for type, base model,
    time period, tags, creator, NSFW level, and commercial use.
    """
    api_key = load_api_key()
    actual_limit = limit if limit is not None else 25

    params: dict[str, Any] = {
        "limit": min(actual_limit, 100),
        "sort": sort.value,
    }

    # Handle NSFW filtering
    if sfw:
        params["nsfw"] = "false"
    elif nsfw:
        params["nsfwLevel"] = nsfw.value
    else:
        params["nsfw"] = "true"  # Default: include all

    if query:
        params["query"] = query
    if types:
        params["types"] = types
    if base_models:
        params["baseModels"] = base_models
    if period:
        params["period"] = period.value
    if tag:
        params["tag"] = tag
    if username:
        params["username"] = username
    if page:
        params["page"] = page
    if commercial:
        params["allowCommercialUse"] = commercial.value

    url = f"{CIVITAI_API_BASE}/models"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, params=params, headers=_get_headers(api_key))
            response.raise_for_status()
            result: dict[str, Any] = response.json()

            # Cache all models from search results
            items = result.get("items", [])
            if items:
                try:
                    with Database() as db:
                        db.init_schema()
                        for model_data in items:
                            db.cache_model(model_data)
                except Exception as e:
                    logger.warning("Failed to cache search results: %s", e)

            return result
    except httpx.HTTPStatusError as e:
        logger.error("CivitAI API error: %s", e.response.status_code)
        return JSONResponse({"error": f"API error: {e.response.status_code}"}, status_code=e.response.status_code)
    except httpx.RequestError as e:
        logger.error("CivitAI request error: %s", e)
        return JSONResponse({"error": f"Request error: {e}"}, status_code=500)


@router.get("/model/{model_id}", response_model=None)
async def get_model(model_id: int) -> dict[str, Any] | Response:
    """Get model details from CivitAI and cache to database."""
    api_key = load_api_key()
    url = f"{CIVITAI_API_BASE}/models/{model_id}"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, headers=_get_headers(api_key))
            response.raise_for_status()
            result: dict[str, Any] = response.json()

            # Cache the model data to database
            try:
                with Database() as db:
                    db.init_schema()
                    db.cache_model(result)
            except Exception as e:
                logger.warning("Failed to cache model %d: %s", model_id, e)

            return result
    except httpx.HTTPStatusError:
        return JSONResponse({"error": "Model not found"}, status_code=404)
    except httpx.RequestError as e:
        return JSONResponse({"error": f"Request error: {e}"}, status_code=500)


def create_civitai_router() -> APIRouter:
    """Return the CivitAI API router."""
    return router
