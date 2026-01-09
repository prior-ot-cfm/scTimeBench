from metrics.ontology_based.graph_sim.base import GraphSimMetric


class GraphEditDistance(GraphSimMetric):
    def eval(self, graph_pred):
        """
        The graph similarity metrics we will be using will take in
        """
        print("Running GraphEditDistance evaluation on graph prediction")
