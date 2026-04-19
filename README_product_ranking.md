# product_ranking module

This module ranks same-type products using evaluation metrics computed in earlier modules.

Supported ranking styles:
- Simple ascending/descending by each metric
- Average-rank aggregation
- Borda aggregation
- TOPSIS with equal / CRITIC / entropy weights
- Weighted-sum with equal / CRITIC / entropy weights
- Consensus ranking across multiple methods

Fusion outputs:
- Mean fusion of top-N products
- Weighted-mean fusion of top-N products

Recommended workflow:
1. Use product_evaluation to generate a product-level metric table.
2. Feed that table into product_ranking.
3. Inspect per-method rankings and consensus results.
4. Select top-N products and generate fused NetCDF for downstream use.
