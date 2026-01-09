"""
Base class for all metrics. They should all implement the eval method, and
depend on the dataset that they belong to.
"""
from config import Config
from typing import final
from models.base import MODEL_REGISTRY

METRIC_REGISTRY = {}


def register_metric(cls):
    """Decorator to register a metric class in the METRIC_REGISTRY."""
    METRIC_REGISTRY[cls.__name__] = cls


# also store a registry of metrics of name to class
class BaseMetric:
    def __init__(self, config: Config):
        self.config = config
        # 1) check the required feature specs match up
        self._check_feature_specs()
        self._init_model()
        self._check_model_feature_specs()
        print("Metric setup complete.")

    def __init_subclass__(cls):
        """
        Automatically register subclasses in the METRIC_REGISTRY.

        This allows for hierarchical structuring of metrics, where
        subclasses are tracked under their parent classes.
        """
        register_metric(cls)

        # we want to clear the submetrics list for each subclass
        cls.submetrics = []

        for base in cls.__bases__:
            if hasattr(base, "submetrics"):
                base.submetrics.append(cls)

    def eval(self, *args, **kwargs):
        raise NotImplementedError("Subclasses should implement this method.")

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

    @final
    def _init_model(self):
        """
        Initialize the model based on the configuration.
        """
        model_name = self.config.model["name"]
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Model {model_name} not found in registry.")
        model_class = MODEL_REGISTRY[model_name]
        self.model = model_class(self.config)

    @final
    def _check_model_feature_specs(self):
        """
        Check if the model's feature specifications satisfy the metric's requirements.
        """
        model_feature_specs = self.model.required_feature_specs
        required_specs = self.required_feature_specs

        for spec in required_specs:
            if spec not in model_feature_specs:
                raise ValueError(
                    f"Model {self.config.model['name']} does not satisfy the required feature spec: {spec}"
                )
