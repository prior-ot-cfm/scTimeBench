"""
Axolotl (https://db.cngb.org/stomics/artista/) dataset, 2022.
"""

from scTimeBench.shared.dataset.base import BaseDataset, ObservationColumns
import scanpy as sc


class ArtistaDataset(BaseDataset):
    def _load_data(self):
        """
        Load the Artista dataset.
        """
        print("Loading Artista dataset...")
        # read from the dataset data_path
        data_path = self.dataset_dict["data_path"]
        self.data = sc.read_h5ad(data_path)

        # given the timepoints of 2DPI, 5DPI, etc.
        # convert to numeric timepoints of 2, 5, etc.
        self.data.obs["timepoint"] = (
            self.data.obs["timepoint"].str.replace("DPI", "").astype(int)
        )

        # rename these columns to standard names
        self.data.obs = self.data.obs.rename(
            columns={
                "Annotation": ObservationColumns.CELL_TYPE.value,
                "timepoint": ObservationColumns.TIMEPOINT.value,
            }
        )

        print(
            f"Cell type counts: {self.data.obs[ObservationColumns.CELL_TYPE.value].value_counts()}"
        )
        print("Artista dataset loaded successfully.")
