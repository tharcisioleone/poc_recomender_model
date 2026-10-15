"""Book Recommendation Pipeline - Base Classes.

Defines the Template Method pattern for all pipeline steps:
    BaseETL:        extract() -> transform() -> load() -> run()
    BaseAnalysis:   extract() -> analyze() -> run()
    BaseMLPipeline: extract() -> prepare() -> train() -> evaluate() -> run()
"""

from abc import ABC, abstractmethod
import json
import logging
from pathlib import Path
from pyspark.sql import SparkSession, DataFrame


class _BaseConfig(ABC):
    """Shared configuration, logging and table-name utilities."""

    def __init__(self, config_path: str, spark: SparkSession = None):
        self.spark = spark or SparkSession.builder.getOrCreate()
        self.config = self._load_config(config_path)
        self.logger = self._setup_logger()

    def _load_config(self, config_path: str) -> dict:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)
        setup_path = path.parent / "00_setup.json"
        if setup_path.exists() and path.name != "00_setup.json":
            with open(setup_path, "r", encoding="utf-8") as f:
                self._setup = json.load(f)
        else:
            self._setup = {}
        return config

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )
            logger.addHandler(handler)
        return logger

    def _table_name(self, layer: str = None) -> str:
        """Build 2-part table name from config (schema.table)."""
        storage = self.config.get("storage", {})
        schema = storage.get("schema", layer or "bronze")
        table = storage.get("table", "unknown")
        return f"{schema}.{table}"

    def _source_table(self, key: str) -> str:
        return self.config.get("source_tables", {}).get(key, "")

    def _storage_table(self, layer: str) -> str:
        """Build table name from nested storage config (storage.{layer}.schema/table)."""
        storage = self.config.get("storage", {})
        layer_cfg = storage.get(layer, {})
        schema = layer_cfg.get("schema", layer)
        table = layer_cfg.get("table", "unknown")
        return f"{schema}.{table}"

    def _ensure_schemas(self) -> None:
        storage = self.config.get("storage", {})
        schemas = set()
        if "schema" in storage:
            schemas.add(storage["schema"])
        for layer in ("bronze", "silver", "gold", "quarantine"):
            if layer in storage and isinstance(storage[layer], dict):
                if "schema" in storage[layer]:
                    schemas.add(storage[layer]["schema"])
        for schema in schemas:
            self.spark.sql(f"CREATE DATABASE IF NOT EXISTS {schema}")

    @abstractmethod
    def run(self):
        pass


class BaseETL(_BaseConfig):
    """Template for ETL pipelines: extract -> transform -> load."""

    @abstractmethod
    def extract(self) -> DataFrame:
        pass

    @abstractmethod
    def transform(self, df: DataFrame) -> DataFrame:
        pass

    @abstractmethod
    def load(self, df: DataFrame) -> None:
        pass

    def run(self) -> DataFrame:
        self.logger.info(f"Pipeline started: {self.__class__.__name__}")
        self._ensure_schemas()
        df = self.extract()
        self.logger.info(f"Extract complete: {df.count()} rows")
        df = self.transform(df)
        self.logger.info(f"Transform complete: {df.count()} rows")
        self.load(df)
        self.logger.info("Load complete")
        return df


class BaseAnalysis(_BaseConfig):
    """Template for analysis/reporting steps (no table writes)."""

    @abstractmethod
    def extract(self) -> dict:
        pass

    @abstractmethod
    def analyze(self, dataframes: dict) -> dict:
        pass

    def run(self) -> dict:
        self.logger.info(f"Analysis started: {self.__class__.__name__}")
        dfs = self.extract()
        results = self.analyze(dfs)
        self.logger.info("Analysis complete")
        return results


class BaseMLPipeline(_BaseConfig):
    """Template for ML pipelines: extract -> prepare -> train -> evaluate."""

    @abstractmethod
    def extract(self) -> DataFrame:
        pass

    @abstractmethod
    def prepare(self, df: DataFrame) -> tuple:
        pass

    @abstractmethod
    def train(self, train_df: DataFrame):
        pass

    @abstractmethod
    def evaluate(self, model, test_df: DataFrame) -> dict:
        pass

    def run(self) -> dict:
        self.logger.info(f"ML Pipeline started: {self.__class__.__name__}")
        df = self.extract()
        self.logger.info(f"Data loaded: {df.count()} rows")
        train_df, test_df = self.prepare(df)
        self.logger.info(f"Prepared: {train_df.count()} train, {test_df.count()} test")
        model = self.train(train_df)
        self.logger.info("Training complete")
        metrics = self.evaluate(model, test_df)
        self.logger.info(f"Evaluation complete: {metrics}")
        return {"model": model, "metrics": metrics}
