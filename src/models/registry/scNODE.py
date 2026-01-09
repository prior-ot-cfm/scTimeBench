"""
scNODE model class.
"""
from models.base import BaseModel, FeatureSpec


class scNODE(BaseModel):
    def _populate_feature_specs(self):
        """
        Populate the feature specifications required for scNODE model.
        """
        self.required_feature_specs = [
            FeatureSpec.GENE_EXPRESSION,
            FeatureSpec.CONTINUOUS,
            FeatureSpec.TRAJECTORY,
        ]
