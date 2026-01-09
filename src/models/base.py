"""
Model Base Class.
"""
from typing import final
from enum import Enum

MODEL_REGISTRY = {}


def register_model(cls):
    """Decorator to register a model class in the MODEL_REGISTRY."""
    MODEL_REGISTRY[cls.__name__] = cls


class FeatureSpec(Enum):
    """Enum for different feature specifications of models, and required features for metrics."""

    CONTINUOUS = "continuous"
    EMBEDDING = "embedding"
    TRAJECTORY = "trajectory"
    GENE_EXPRESSION = "gene_expression"
    GRN_INFERENCE = "grn_inference"


class BaseModel:
    def __init__(self, config):
        self.config = config
        self._check_feature_specs()

    def __init_subclass__(cls):
        """
        Automatically register subclasses in the MODEL_REGISTRY.
        """
        register_model(cls)

    @final
    def _check_feature_specs(self):
        """
        Populate the feature specifications required for the metric.
        """
        self.required_feature_specs = None
        self._populate_feature_specs()
        assert (
            self.required_feature_specs is not None
        ), "Subclasses must define required_feature_specs"

    def _populate_feature_specs(self):
        """
        Subclasses should implement this method to define
        their required feature specifications.
        """
        raise NotImplementedError("Subclasses should implement this method.")
