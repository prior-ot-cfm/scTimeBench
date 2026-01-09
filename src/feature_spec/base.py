"""
Feature specifications module.

This includes a simple registry of feature specifications required for each metric
and what each model can predict. e.g.: predicted gene expression is required
for interpolated Wasserstein distance, and scNODE can provide them.

This also includes the classes which implement these specifications, such as
building trajectory graphs from embeddings.
"""


class FeatureSpec:
    pass
