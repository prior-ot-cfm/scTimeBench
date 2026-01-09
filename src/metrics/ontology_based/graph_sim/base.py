"""
Graph Similarity Metric Base Class
"""
from metrics.base import BaseMetric


class GraphSimMetric(BaseMetric):
    def eval(self, graph_pred):
        """
        The graph similarity metrics we will be using will take in
        """
        for submetric in self.submetrics:
            submetric().eval(graph_pred)

    def populate_feature_specs(self):
        """
        Populate the feature specifications required for graph similarity metrics
        """
        # self.feature_specs =
