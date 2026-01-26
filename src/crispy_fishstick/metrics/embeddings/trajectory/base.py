"""
Embedding-based metrics.
"""
from crispy_fishstick.metrics.base import OutputPathName
from crispy_fishstick.metrics.embeddings.base import EmbeddingMetrics
from crispy_fishstick.shared.constants import RequiredOutputColumns, ObservationColumns
from crispy_fishstick.trajectory_infer.base import (
    TrajectoryInferenceMethodFactory,
)
from crispy_fishstick.trajectory_infer.classifier import Classifier
from crispy_fishstick.trajectory_infer.kNN import kNN
from sklearn.neighbors import NearestNeighbors

import logging
import os
import json
import numpy as np
import scanpy as sc


class TrajectoryEmbeddingMetrics(EmbeddingMetrics):
    def _setup_model_output_requirements(self):
        # ** NOTE: must define the following attributes **
        # where we define the output embedding name
        # as well as the required features and outputs
        self.output_path_name = OutputPathName.GRAPH_SIM
        self.required_outputs = [
            RequiredOutputColumns.EMBEDDING,
            RequiredOutputColumns.NEXT_TIMEPOINT_EMBEDDING,
        ]


# now let's create the other metrics such as the gini index or classification entropy, etc.
class ClassificationEntropy(TrajectoryEmbeddingMetrics):
    def _setup_trajectory_inference_model(self):
        # by default we use the classifier trajectory inference model
        logging.debug(
            "Setting up trajectory inference model for classification entropy."
        )

        self.trajectory_infer_model: Classifier = (
            TrajectoryInferenceMethodFactory().get_trajectory_infer_method(
                self.metric_config.get(
                    "trajectory_infer_model", {"name": Classifier.__name__}
                )
            )
        )
        assert isinstance(
            self.trajectory_infer_model, Classifier
        ), "ClassifierEntropy only supports Classifier trajectory inference model."
        self.params["trajectory_infer_model"] = str(self.trajectory_infer_model)

    def _embedding_eval(self, output_path):
        model_output_file = os.path.join(output_path, self.output_path_name.value)
        (
            test_probas,
            accuracy,
            next_tp_probas,
        ) = self.trajectory_infer_model.train_and_predict(model_output_file)

        entropy = -np.sum(
            test_probas * np.log(test_probas + 1e-10), axis=1
        )  # avoid log(0)
        logging.debug(f"Average classification entropy: {np.mean(entropy)}")

        predicted_entropy = -np.sum(
            next_tp_probas * np.log(next_tp_probas + 1e-10), axis=1
        )  # avoid log(0)
        logging.debug(
            f"Average predicted classification entropy: {np.mean(predicted_entropy)}"
        )

        num_classes = test_probas.shape[1]
        normalized_entropy = entropy / np.log(num_classes)

        return json.dumps(
            {
                "avg_entropy": np.mean(entropy).item(),
                "std_entropy": np.std(entropy).item(),
                "avg_normalized_entropy": np.mean(normalized_entropy).item(),
                "std_normalized_entropy": np.std(normalized_entropy).item(),
                "num_classes": num_classes,
                "classifier_accuracy": accuracy,
                "pred_tp_avg_entropy": np.mean(predicted_entropy).item(),
                "pred_tp_avg_normalized_entropy": np.mean(
                    predicted_entropy / np.log(num_classes)
                ).item(),
                "pred_tp_std_entropy": np.std(predicted_entropy).item(),
            }
        )


# now let's create the other metrics such as the gini index or classification entropy, etc.
class EmbeddingGiniIndex(TrajectoryEmbeddingMetrics):
    def _setup_trajectory_inference_model(self):
        # by default we use the classifier trajectory inference model
        logging.debug(
            "Setting up trajectory inference model for classification entropy."
        )

        self.trajectory_infer_model: kNN = (
            TrajectoryInferenceMethodFactory().get_trajectory_infer_method(
                self.metric_config.get("trajectory_infer_model", {"name": kNN.__name__})
            )
        )
        assert isinstance(
            self.trajectory_infer_model, kNN
        ), "Gini Index only supports kNN trajectory inference model."
        self.params["trajectory_infer_model"] = str(self.trajectory_infer_model)

    def _embedding_eval(self, output_path):
        model_output_file = os.path.join(output_path, self.output_path_name.value)
        ann_data = sc.read_h5ad(model_output_file)

        required_obs_columns = [
            ObservationColumns.CELL_TYPE.value,
            ObservationColumns.TIMEPOINT.value,
        ]

        required_obsm_columns = [
            RequiredOutputColumns.EMBEDDING.value,
            RequiredOutputColumns.NEXT_TIMEPOINT_EMBEDDING.value,
        ]

        for col in required_obs_columns:
            if col not in ann_data.obs.columns:
                raise ValueError(
                    f"Predicted graph data must have '{col}' in observation metadata."
                )
        for col in required_obsm_columns:
            if col not in ann_data.obsm.keys():
                raise ValueError(
                    f"Predicted graph data must have '{col}' in observation embeddings."
                )

        embeddings = ann_data.obsm[RequiredOutputColumns.EMBEDDING.value]
        next_timepoint_embeddings = ann_data.obsm[
            RequiredOutputColumns.NEXT_TIMEPOINT_EMBEDDING.value
        ]

        # filter out the nans/timepoints without next embeddings
        next_timepoint_embeddings = next_timepoint_embeddings[
            ~np.isnan(next_timepoint_embeddings).any(axis=1)
        ]

        knn_graph: NearestNeighbors = self.trajectory_infer_model.get_kNN_graph(
            ann_data, output_path
        )

        cell_types = ann_data.obs[ObservationColumns.CELL_TYPE.value].to_numpy()

        # now let's compute the gini index based on the knn graph
        def calc_gini(indices):
            gini_indices = []
            for neighbor_indices in indices:
                neighbor_cell_types = cell_types[neighbor_indices]
                _, counts = np.unique(neighbor_cell_types, return_counts=True)
                proportions = counts / counts.sum()
                gini_index = 1 - np.sum(proportions**2)
                gini_indices.append(gini_index)
            return np.array(gini_indices)

        _, indices = knn_graph.kneighbors(embeddings)
        gini_indices = calc_gini(indices)

        _, pred_indices = knn_graph.kneighbors(next_timepoint_embeddings)
        gini_pred_indices = calc_gini(pred_indices)

        return json.dumps(
            {
                "avg_gini_index": np.mean(gini_indices).item(),
                "std_gini_index": np.std(gini_indices).item(),
                "pred_tp_avg_gini_index": np.mean(gini_pred_indices).item(),
                "pred_tp_std_gini_index": np.std(gini_pred_indices).item(),
            }
        )
