"""Minimal architecture overview for the six-module PREMISE design."""

from premise.acquisition import list_sources
from premise.conversion import summarize_harmonization_scope
from premise.product_evaluation import describe_ranking_workflow

print("Available hydroclimatic source templates:")
for src in list_sources()[:3]:
    print(f"- {src.key}: {src.title} -> {', '.join(src.variables)}")

print("
Harmonization scope:")
print(summarize_harmonization_scope())

print("
Integrated ranking workflow:")
print(describe_ranking_workflow())
