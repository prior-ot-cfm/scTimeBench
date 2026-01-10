"""
Example Random Sampler Model.

This model is very simple, it just memorizes a single random sample from the dataset
per time point, and during inference it returns these samples as the predicted trajectory.

For continuous cases, it simply linearly interpolates between the memorized samples.
"""
from models.base import BaseModel, FeatureSpec


class ExampleRandomSampler(BaseModel):
    def _populate_feature_specs(self):
        """
        Populate the feature specifications required for ExampleRandomSampler model.
        """
        self.required_feature_specs = [
            FeatureSpec.GENE_EXPRESSION,
            FeatureSpec.CONTINUOUS,
            FeatureSpec.TRAJECTORY,
        ]
