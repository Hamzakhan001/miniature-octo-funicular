from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Iterable
from urllib.parse import quote_plus

from pydantic import BaseModel, Field


DEFAULT_SPONSOR_PDF_PATH = Path.home() / "Downloads" / "749d56_756baf5b73b2459e8e18f82628d75cef.pdf"
DEFAULT_SPONSOR_DATASET_PATH = Path("data/sponsor_outreach/companies.json")

DEFAULT_COLUMN_STARTS = {
    "organisation_name": 0,
    "town_city": 57,
    "county": 73,
    "type_rating": 98,
    "route": 126,
}

TECH_KEYWORDS = {
    "ai": 30,
    "artificial intelligence": 35,
    "agentic": 25,
    "analytics": 20,
    "app": 15,
    "automation": 20,
    "banking": 10,
    "cloud": 20,
    "consulting": 12,
    "cyber": 20,
    "data": 25,
    "digital": 18,
    "engineering": 18,
    "fintech": 25,
    "games": 8,
    "geospatial": 20,
    "infiniti": 6,
    "innovation": 10,
    "insights": 14,
    "labs": 18,
    "machine learning": 30,
    "ml": 18,
    "platform": 18,
    "robotics": 30,
    "saas": 22,
    "security": 20,
    "software": 28,
    "solutions": 12,
    "systems": 18,
    "tech": 25,
    "technology": 25,
}

LOW_PRIORITY_KEYWORDS = {
    "care": -22,
    "dental": -20,
    "food": -15,
    "garage": -18,
    "grocery": -16,
    "hospitality": -18,
    "hotel": -18,
    "meat": -18,
    "restaurant": -20,
    "retail": -18,
    "store": -16,
    "supermarket": -18,
    "taxi": -18,
}

TECH_TITLES = [
    "Technical Recruiter",
    "Engineering Manager",
    "Head of Engineering",
    "Head of AI",
]

GENERAL_TITLES = [
    "Talent Acquisition",
    "Hiring Manager",
    "Engineering Manager",
    "CTO",
]

KNOWN_ROUTES = [
    "Skilled Worker",
    "Creative Worker",
    "Charity Worker",
    "Government Authorised Exchange",
    "International Sportsperson",
    "Religious Worker",
    "Scale-up Worker",
    "Seasonal Worker",
    "Service Supplier",
    "UK Expansion Worker",
    "Global Business Mobility: Senior or Specialist Worker",
    "Global Business Mobility: Graduate Trainee",
    "Global Business Mobility: UK Expansion Worker",
    "Global Business Mobility: Service Supplier",
    "Global Business Mobility: Secondment Worker",
]


class SponsorEntry(BaseModel):
    organisation_name: str
    town_city: str = ""
    county: str = ""
    type_rating: str = ""
    route: str = ""


class SponsorLead(BaseModel):
    slug: str
    organisation_name: str
    alternate_names: list[str] = Field(default_factory=list)
    towns_cities: list[str] = Field(default_factory=list)
    counties: list[str] = Field(default_factory=list)
    routes: list[str] = Field(default_factory=list)
    type_ratings: list[str] = Field(default_factory=list)
    record_count: int = 0
    fit_score: int = 0
    priority: str
    tags: list[str] = Field(default_factory=list)
    linkedin_company_search_url: str
    linkedin_people_search_url: str
    linkedin_jobs_search_url: str
    recommended_titles: list[str] = Field(default_factory=list)
    outreach_reason: str
    connection_note: str
    follow_up_message: str


def build_outreach_dataset(
    pdf_path: Path = DEFAULT_SPONSOR_PDF_PATH,
    output_path: Path = DEFAULT_SPONSOR_DATASET_PATH,
) -> list[SponsorLead]:
    entries = parse_sponsor_pdf(pdf_path)
    leads = aggregate_sponsor_entries(entries)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps([lead.model_dump() for lead in leads], indent=2),
        encoding="utf-8",
    )
    return leads


def load_outreach_dataset(dataset_path: Path = DEFAULT_SPONSOR_DATASET_PATH) -> list[SponsorLead]:
    if not dataset_path.exists():
        if DEFAULT_SPONSOR_PDF_PATH.exists():
            return build_outreach_dataset(DEFAULT_SPONSOR_PDF_PATH, dataset_path)
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    return [SponsorLead.model_validate(item) for item in data]


def search_leads(
    leads: Iterable[SponsorLead],
    query: str = "",
    city: str = "",
    route: str = "",
    min_score: int = 0,
    limit: int = 50,
) -> list[SponsorLead]:
    query_value = query.strip().lower()
    city_value = city.strip().lower()
    route_value = route.strip().lower()

    filtered: list[SponsorLead] = []
    for lead in leads:
        if query_value and query_value not in _search_blob(lead):
            continue
        if city_value and not any(city_value in item.lower() for item in lead.towns_cities):
            continue
        if route_value and not any(route_value in item.lower() for item in lead.routes):
            continue
        if lead.fit_score < min_score:
            continue
        filtered.append(lead)

    filtered.sort(key=lambda lead: (-lead.fit_score, lead.organisation_name.lower()))
    return filtered[:limit]


def parse_sponsor_pdf(pdf_path: Path) -> list[SponsorEntry]:
    text = _extract_layout_text(pdf_path)
    return _parse_layout_text(text)


def aggregate_sponsor_entries(entries: Iterable[SponsorEntry]) -> list[SponsorLead]:
    grouped: dict[str, dict] = {}

    for entry in entries:
        key = _canonical_company_key(entry.organisation_name)
        current = grouped.setdefault(
            key,
            {
                "organisation_name": entry.organisation_name,
                "alternate_names": set(),
                "towns_cities": set(),
                "counties": set(),
                "routes": set(),
                "type_ratings": set(),
                "record_count": 0,
            },
        )

        current["alternate_names"].add(entry.organisation_name)
        if entry.town_city:
            current["towns_cities"].add(entry.town_city)
        if entry.county:
            current["counties"].add(entry.county)
        if entry.route:
            current["routes"].add(entry.route)
        if entry.type_rating:
            current["type_ratings"].add(entry.type_rating)
        current["record_count"] += 1

    leads: list[SponsorLead] = []
    for key, value in grouped.items():
        organisation_name = _pick_display_name(value["alternate_names"])
        tags = _classify_company(organisation_name)
        fit_score = _score_company(organisation_name, tags)
        towns = sorted(value["towns_cities"])
        routes = sorted(value["routes"])
        recommended_titles = TECH_TITLES if "tech" in tags else GENERAL_TITLES
        reason = _build_outreach_reason(tags, routes, towns)

        leads.append(
            SponsorLead(
                slug=_slugify(organisation_name),
                organisation_name=organisation_name,
                alternate_names=sorted(value["alternate_names"]),
                towns_cities=towns,
                counties=sorted(value["counties"]),
                routes=routes,
                type_ratings=sorted(value["type_ratings"]),
                record_count=value["record_count"],
                fit_score=fit_score,
                priority=_priority_for_score(fit_score),
                tags=tags,
                linkedin_company_search_url=_linkedin_company_search_url(organisation_name),
                linkedin_people_search_url=_linkedin_people_search_url(
                    organisation_name,
                    recommended_titles,
                ),
                linkedin_jobs_search_url=_linkedin_jobs_search_url(organisation_name),
                recommended_titles=recommended_titles,
                outreach_reason=reason,
                connection_note=_build_connection_note(organisation_name),
                follow_up_message=_build_follow_up_message(organisation_name, reason),
            )
        )

    leads.sort(key=lambda lead: (-lead.fit_score, lead.organisation_name.lower()))
    return leads


def _extract_layout_text(pdf_path: Path) -> str:
    pdftotext_path = shutil.which("pdftotext")
    if pdftotext_path:
        result = subprocess.run(
            [pdftotext_path, "-layout", str(pdf_path), "-"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout

    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _parse_layout_text(text: str) -> list[SponsorEntry]:
    starts = _detect_column_starts(text)
    entries: list[SponsorEntry] = []
    blocks = re.split(r"\n\s*\n", text)

    for block in blocks:
        lines = [line.rstrip("\n") for line in block.splitlines() if not _skip_line(line)]
        if not lines:
            continue

        current = defaultdict(str)
        for line in lines:
            fragment = _slice_line(line, starts)
            if not _looks_like_data(fragment):
                continue
            if current and current["type_rating"] and current["route"] and fragment["organisation_name"]:
                entries.append(_build_entry(current))
                current = defaultdict(str)
            for field, value in fragment.items():
                if not value:
                    continue
                current[field] = _join_field(current[field], value)

        if current and current["organisation_name"]:
            entries.append(_build_entry(current))

    return entries


def _detect_column_starts(text: str) -> dict[str, int]:
    for line in text.splitlines():
        normalized = line.lstrip()
        if "Organisation Name" in normalized and "Town/City" in normalized and "Type & Rating" in normalized:
            starts = {
                "organisation_name": normalized.find("Organisation Name"),
                "town_city": normalized.find("Town/City"),
                "county": normalized.find("County"),
                "type_rating": normalized.find("Type & Rating"),
                "route": normalized.find("Route"),
            }
            if all(value >= 0 for value in starts.values()):
                return starts
    return DEFAULT_COLUMN_STARTS.copy()


def _skip_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if stripped.isdigit():
        return True
    if "2025-12-31_-_Worker_and_Temporary_Worker" in stripped:
        return True
    if "Organisation Name" in stripped and "Town/City" in stripped:
        return True
    if stripped in {"fi", "ff", "￼"}:
        return True
    return False


def _slice_line(line: str, starts: dict[str, int]) -> dict[str, str]:
    if line[:starts["town_city"]].strip():
        line = line.lstrip()
    town_start = starts["town_city"]
    county_start = starts["county"]
    rating_start = starts["type_rating"]
    route_start = starts["route"]
    fragment = {
        "organisation_name": line[starts["organisation_name"]:town_start].strip(),
        "town_city": line[town_start:county_start].strip(),
        "county": line[county_start:rating_start].strip(),
        "type_rating": line[rating_start:route_start].strip(),
        "route": line[route_start:].strip(),
    }
    if not fragment["type_rating"] and not fragment["route"]:
        stripped = line.strip()
        if stripped and stripped[0].isalnum():
            fragment["organisation_name"] = stripped
            fragment["town_city"] = ""
            fragment["county"] = ""
    return fragment


def _looks_like_data(fragment: dict[str, str]) -> bool:
    if fragment["type_rating"] and "rating" in fragment["type_rating"].lower():
        return True
    if fragment["organisation_name"] and any(fragment[field] for field in ("town_city", "county", "route")):
        return True
    return bool(fragment["organisation_name"])


def _build_entry(values: dict[str, str]) -> SponsorEntry:
    type_rating, route = _normalize_type_and_route(values["type_rating"], values["route"])
    county = _clean_value(values["county"])
    if len(county) <= 2 and county.isalpha():
        county = ""
    return SponsorEntry(
        organisation_name=_clean_value(values["organisation_name"]),
        town_city=_clean_value(values["town_city"]),
        county=county,
        type_rating=_clean_value(type_rating),
        route=_clean_value(route),
    )


def _clean_value(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip(" ,")
    replacements = {
        "Su olk": "Suffolk",
        "Sta ordshire": "Staffordshire",
        "Fritzovia": "Fitzrovia",
    }
    return replacements.get(value, value)


def _join_field(existing: str, new_value: str) -> str:
    if not existing:
        return new_value
    return f"{existing} {new_value}".strip()


def _normalize_type_and_route(type_rating: str, route: str) -> tuple[str, str]:
    combined = f"{type_rating} {route}".strip()
    normalized_route = route
    normalized_type = type_rating

    for candidate in KNOWN_ROUTES:
        if candidate.lower() in combined.lower():
            normalized_route = candidate
            break

    type_match = re.search(r"(Temporary Worker|Worker)\s+\([A-Za-z ]+rating\)", combined)
    if type_match:
        normalized_type = type_match.group(0)
    elif "rating" in combined.lower():
        if "temporary" in combined.lower():
            normalized_type = "Temporary Worker (A rating)"
        elif "worker" in combined.lower() or "rker" in combined.lower():
            normalized_type = "Worker (A rating)"

    combined_lower = combined.lower()
    route_lower = normalized_route.lower()
    if route_lower.endswith("worker"):
        if "sk" in combined_lower or "illed worker" in combined_lower:
            normalized_route = "Skilled Worker"
        elif "creative" in combined_lower:
            normalized_route = "Creative Worker"

    return normalized_type, normalized_route


def _canonical_company_key(name: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", name.lower())
    cleaned = re.sub(r"\b(limited|ltd|llp|plc|uk)\b", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _pick_display_name(names: set[str]) -> str:
    return sorted(names, key=lambda item: (len(item), item.lower()))[0]


def _score_company(name: str, tags: list[str]) -> int:
    lowered = name.lower()
    score = 20
    for keyword, weight in TECH_KEYWORDS.items():
        if keyword in lowered:
            score += weight
    for keyword, penalty in LOW_PRIORITY_KEYWORDS.items():
        if keyword in lowered:
            score += penalty
    if "tech" in tags:
        score += 10
    return max(0, min(100, score))


def _classify_company(name: str) -> list[str]:
    lowered = name.lower()
    tags: list[str] = []

    tech_matches = [
        keyword
        for keyword in ("ai", "data", "digital", "software", "systems", "tech", "technology", "platform", "cloud")
        if keyword in lowered
    ]
    if tech_matches:
        tags.append("tech")
    tags.extend(tech_matches[:4])

    if any(keyword in lowered for keyword in ("consulting", "solutions", "services")):
        tags.append("services")
    if any(keyword in lowered for keyword in ("bank", "banking", "capital", "fintech")):
        tags.append("finance")
    if any(keyword in lowered for keyword in ("games", "studio", "media")):
        tags.append("product")
    return sorted(set(tags))


def _priority_for_score(score: int) -> str:
    if score >= 70:
        return "high"
    if score >= 45:
        return "medium"
    return "low"


def _build_outreach_reason(tags: list[str], routes: list[str], towns: list[str]) -> str:
    reasons: list[str] = []
    if "tech" in tags:
        reasons.append("The company name suggests a technology or product-oriented business.")
    if any("Skilled Worker" in route for route in routes):
        reasons.append("It is already licensed for the Skilled Worker route.")
    if towns:
        reasons.append(f"It has a listed presence in {towns[0]}.")
    return " ".join(reasons) or "It appears on the licensed sponsor register and is worth reviewing."


def _build_connection_note(company_name: str) -> str:
    note = (
        f"Hi, I noticed {company_name} is a licensed sponsor. "
        "I’m a UK-based applied AI/software engineer with 3+ years in Python, FastAPI, React, and production AI systems. "
        f"I’d value connecting and learning about relevant engineering openings at {company_name}."
    )
    return note[:300]


def _build_follow_up_message(company_name: str, reason: str) -> str:
    return (
        f"Hi, thanks for connecting. I’m exploring software, backend, and applied AI roles with licensed sponsors and "
        f"{company_name} stood out because {reason.lower()} My background is in Python, FastAPI, TypeScript/React, cloud workflows, "
        "and production RAG/LLM systems. If there’s a suitable team or recruiter I should speak with, I’d really appreciate the direction."
    )


def _linkedin_company_search_url(company_name: str) -> str:
    return f"https://www.linkedin.com/search/results/companies/?keywords={quote_plus(company_name)}"


def _linkedin_people_search_url(company_name: str, titles: list[str]) -> str:
    keywords = f'{company_name} {" OR ".join(titles)}'
    return f"https://www.linkedin.com/search/results/people/?keywords={quote_plus(keywords)}"


def _linkedin_jobs_search_url(company_name: str) -> str:
    return f"https://www.linkedin.com/jobs/search/?keywords={quote_plus(company_name)}"


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "company"


def _search_blob(lead: SponsorLead) -> str:
    parts = [
        lead.organisation_name,
        " ".join(lead.alternate_names),
        " ".join(lead.towns_cities),
        " ".join(lead.counties),
        " ".join(lead.routes),
        " ".join(lead.tags),
    ]
    return " ".join(parts).lower()
