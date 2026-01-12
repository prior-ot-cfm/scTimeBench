"""
Database manager using sqlite3.

This module provides a simple interface to interact with an SQLite database,
including the setup of tables for storing:
1. Paths to processed datasets.
2. Paths to model checkpoints.
3. Paths to model predictions.
4. Metric results.
"""

import sqlite3
from config import Config

from dataset.base import BaseDataset
from models.base import BaseModel


class DatabaseManager:
    def __init__(self, config: Config):
        self.conn = sqlite3.connect(config.database_path)
        self._create_tables()

    def _create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS processed_datasets (
                id INTEGER PRIMARY KEY,
                name TEXT,
                config TEXT,
                filters TEXT,
                path TEXT
            )
        """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS model_outputs (
                id INTEGER PRIMARY KEY,
                name TEXT,
                processed_dataset_id INTEGER,
                metadata TEXT,
                path TEXT,
                UNIQUE(name, processed_dataset_id, metadata, path)
            )
        """
        )
        self.conn.commit()

    def insert_processed_dataset(self, dataset: BaseDataset, path):
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO processed_datasets (name, config, filters, path)
            VALUES (?, ?, ?, ?)
        """,
            (
                dataset.get_name(),
                dataset.encode_config(),
                dataset.encode_filters(),
                path,
            ),
        )
        self.conn.commit()

    def get_processed_dataset_path(self, dataset: BaseDataset):
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT path FROM processed_datasets
            WHERE name = ? AND config = ? AND filters = ?
        """,
            (dataset.get_name(), dataset.encode_config(), dataset.encode_filters()),
        )
        result = cursor.fetchone()
        return result[0] if result else None

    def insert_model_output(self, model: BaseModel, output_path: str):
        cursor = self.conn.cursor()

        # first grab the processed_dataset_id
        cursor.execute(
            """
            SELECT id FROM processed_datasets
            WHERE name = ? AND config = ? AND filters = ?
        """,
            (
                model.dataset.get_name(),
                model.dataset.encode_config(),
                model.dataset.encode_filters(),
            ),
        )
        result = cursor.fetchone()
        if result is None:
            raise ValueError("Processed dataset not found in database.")

        processed_dataset_id = result[0]

        cursor.execute(
            """
            INSERT INTO model_outputs (name, processed_dataset_id, metadata, path)
            VALUES (?, ?, ?, ?)
        """,
            (
                model._get_name(),
                processed_dataset_id,
                model._encode_metadata(),
                output_path,
            ),
        )
        self.conn.commit()

    def get_model_output_path(self, model: BaseModel):
        cursor = self.conn.cursor()

        # first grab the processed_dataset_id
        cursor.execute(
            """
            SELECT id FROM processed_datasets
            WHERE name = ? AND config = ? AND filters = ?
        """,
            (
                model.dataset.get_name(),
                model.dataset.encode_config(),
                model.dataset.encode_filters(),
            ),
        )
        result = cursor.fetchone()
        if result is None:
            raise ValueError("Processed dataset not found in database.")

        processed_dataset_id = result[0]

        cursor.execute(
            """
            SELECT path FROM model_outputs
            WHERE name = ? AND processed_dataset_id = ? AND metadata = ?
        """,
            (model._get_name(), processed_dataset_id, model._encode_metadata()),
        )
        result = cursor.fetchone()
        return result[0] if result else None

    def close(self):
        self.conn.close()

    def print_all(self):
        cursor = self.conn.cursor()
        for table in ["processed_datasets", "model_outputs"]:
            print("-" * 100)
            print(f"Contents of table: {table}")
            cursor.execute(f"SELECT * FROM {table}")
            rows = cursor.fetchall()
            for row in rows:
                print(row)
        self.conn.commit()
