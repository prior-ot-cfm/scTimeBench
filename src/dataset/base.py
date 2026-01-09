"""
Base filter for datasets. Every metric will likely require different splits of the
data, so this base class will define the necessary interface for dataset preprocessing.
"""


class BaseDataset:
    def __init__(self, config):
        self.config = config

    def load_data(self):
        """
        Subclasses should implement this method to load the dataset.
        Each dataset might require its own loading mechanism, as well as preprocessing
        mechanisms, but the BaseDatasetFilter should hopefully work on all datasets.
        """
        raise NotImplementedError("Subclasses should implement this method.")


class BaseDatasetFilter:
    def __init__(self, config):
        self.config = config

    def filter(self):
        """
        Subclasses should implement this method to filter and split the dataset
        according to the metric's requirements.
        """
        raise NotImplementedError("Subclasses should implement this method.")
