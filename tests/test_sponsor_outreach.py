from app.services.sponsor_outreach import _parse_layout_text, aggregate_sponsor_entries


def test_parse_layout_text_handles_wrapped_rows() -> None:
    sample = """
2025-12-31_-_Worker_and_Temporary_Worker

Organisation Name                                     Town/City       County                   Type & Rating              Route

11 Hospitality Limited T/A Holiday Inn Birmingham Airport NEC
                                                      Birmingham                               Worker (A rating)          Skilled Worker
10X Banking Technology Services Ltd                   London                                   Worker (A rating)          Skilled Worker
"""
    entries = _parse_layout_text(sample)

    assert len(entries) == 2
    assert entries[0].organisation_name == "11 Hospitality Limited T/A Holiday Inn Birmingham Airport NEC"
    assert entries[0].town_city == "Birmingham"
    assert entries[0].route == "Skilled Worker"
    assert entries[1].organisation_name == "10X Banking Technology Services Ltd"


def test_aggregate_sponsor_entries_builds_linkedin_metadata() -> None:
    sample = """
Organisation Name                                     Town/City       County                   Type & Rating              Route
10X Banking Technology Services Ltd                   London                                   Worker (A rating)          Skilled Worker
10X Banking Technology Services Ltd                   London                                   Worker (A rating)          Global Business Mobility: Senior or Specialist Worker
"""
    entries = _parse_layout_text(sample)
    leads = aggregate_sponsor_entries(entries)

    assert len(leads) == 1
    lead = leads[0]
    assert lead.fit_score >= 50
    assert "linkedin.com/search/results/people/" in lead.linkedin_people_search_url
    assert "Skilled Worker" in lead.routes
