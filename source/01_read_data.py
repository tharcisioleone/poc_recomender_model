"""Ingestion: CSV files to Bronze layer."""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from source.base import BaseETL


class ReadData(BaseETL):
    """Reads Kaggle Book-Crossing CSVs into bronze Delta tables."""

    def __init__(self, config_path: str, spark: SparkSession = None):
        super().__init__(config_path, spark)
        self.source_cfg = self.config["source"]
        self.bronze_cfg = self.config["storage"]["bronze"]
        self._dataframes: dict[str, DataFrame] = {}

    def extract(self) -> DataFrame:
        data_path = self._setup.get("data_path", self.source_cfg.get("data_path", ""))
        sep = self.source_cfg["separator"]
        encoding = self.source_cfg["encoding"]

        for key, filename in self.source_cfg["files"].items():
            raw_path = f"{data_path}/{filename}"
            path = f"file:{raw_path}" if raw_path.startswith("/Workspace") else raw_path
            self.logger.info(f"Reading {filename}")
            df = (
                self.spark.read
                .option("header", "true")
                .option("sep", sep)
                .option("encoding", encoding)
                .option("quote", '"')
                .option("escape", '"')
                .option("multiLine", "true")
                .csv(path)
            )
            df = df.withColumn("_source_file", F.lit(filename))
            df = df.withColumn("_ingestion_timestamp", F.current_timestamp())
            self._dataframes[key] = df
            self.logger.info(f"{filename}: {df.count()} rows, {len(df.columns)} columns")

        return self._dataframes.get("books")

    def transform(self, df: DataFrame) -> DataFrame:
        import re
        for col_name in df.columns:
            clean = re.sub(r"[^a-zA-Z0-9_]", "_", col_name)
            if clean != col_name:
                df = df.withColumnRenamed(col_name, clean)
        return df

    def load(self, df: DataFrame) -> None:
        mode = self.config["storage"]["mode"]
        tables = self.bronze_cfg["tables"]
        schema = self.bronze_cfg["schema"]

        for key, table_name in tables.items():
            if key in self._dataframes:
                full_name = f"{schema}.{table_name}"
                self.logger.info(f"Writing {full_name}")
                self._dataframes[key].write.mode(mode).option("overwriteSchema", "true").saveAsTable(full_name)

    def run(self) -> dict[str, DataFrame]:
        self.logger.info(f"Pipeline started: {self.__class__.__name__}")
        self._ensure_schemas()
        self.extract()
        for key, df in self._dataframes.items():
            self._dataframes[key] = self.transform(df)
        self.load(None)
        self.logger.info("Load complete: new")
        return self._dataframes
