# PREMISE v2.1 module architecture

PREMISE v2.1 is organized around six core application modules plus a product-ranking module.

| Module | Package path | Role |
|---|---|---|
| Data acquisition | `premise.acquisition` | Catalogue and query hydroclimatic data sources and acquisition templates |
| Conversion and harmonization | `premise.conversion` | Convert heterogeneous source files and summarize harmonization requirements |
| Basin preprocessing | `premise.basin` | Clip, subset, resample, rename, and prepare basin-oriented forcing datasets |
| Climate indices | `premise.climate_indices` | Compute hydroclimatic and precipitation-extreme indicators |
| Product evaluation | `premise.product_evaluation` | Compare candidate products against reference data using multiple diagnostics |
| Visualization | `premise.visualization` | Generate maps, diagrams, time-series graphics, and summary plots |
| Product ranking | `premise.product_ranking` | Rank products using multi-criteria decision-support methods |

Legacy top-level modules such as `premise.indices`, `premise.metrics`, and `premise.evaluation` are retained for backward compatibility.
