from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from app.services.sponsor_outreach import (
    DEFAULT_SPONSOR_DATASET_PATH,
    DEFAULT_SPONSOR_PDF_PATH,
    build_outreach_dataset,
    load_outreach_dataset,
    search_leads,
)

router = APIRouter(prefix="/sponsors", tags=["Sponsors"])


@router.post("/rebuild")
def rebuild_sponsor_dataset(
    pdf_path: str = str(DEFAULT_SPONSOR_PDF_PATH),
    output_path: str = str(DEFAULT_SPONSOR_DATASET_PATH),
):
    source = Path(pdf_path)
    target = Path(output_path)
    if not source.exists():
        raise HTTPException(status_code=404, detail=f"Sponsor PDF not found: {source}")

    leads = build_outreach_dataset(source, target)
    return {
        "status": "ok",
        "pdf_path": str(source),
        "output_path": str(target),
        "companies": len(leads),
    }


@router.get("")
def list_sponsors(
    query: str = "",
    city: str = "",
    route: str = "",
    min_score: int = Query(default=0, ge=0, le=100),
    limit: int = Query(default=50, ge=1, le=200),
):
    try:
        leads = load_outreach_dataset()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    results = search_leads(
        leads=leads,
        query=query,
        city=city,
        route=route,
        min_score=min_score,
        limit=limit,
    )
    return {
        "count": len(results),
        "results": [lead.model_dump() for lead in results],
    }


@router.get("/{slug}")
def get_sponsor(slug: str):
    try:
        leads = load_outreach_dataset()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    for lead in leads:
        if lead.slug == slug:
            return lead.model_dump()

    raise HTTPException(status_code=404, detail=f"Sponsor not found: {slug}")
