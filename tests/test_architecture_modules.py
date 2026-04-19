from premise.acquisition import list_sources, search_sources
from premise.conversion import summarize_harmonization_scope
from premise.product_evaluation import describe_ranking_workflow


def test_catalog_lists_precipitation_sources():
    keys = {src.key for src in list_sources(variable="precipitation")}
    assert "chirps" in keys
    assert "era5" in keys


def test_source_search_finds_soil_moisture_product():
    keys = {src.key for src in search_sources("soil moisture")}
    assert "gleam" in keys


def test_harmonization_scope_has_formats_and_steps():
    scope = summarize_harmonization_scope()
    assert "NetCDF" in scope["formats"]
    assert "unit normalization" in scope["standardization"]


def test_ranking_workflow_description_has_steps():
    info = describe_ranking_workflow()
    assert len(info["steps"]) >= 4
