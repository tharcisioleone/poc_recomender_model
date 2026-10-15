"""Transformation: Bronze to Silver layer."""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from source.base import BaseETL


class DataTransformation(BaseETL):
    """Cleans, standardizes and validates bronze data into silver tables."""

    def __init__(self, config_path: str, spark: SparkSession = None):
        super().__init__(config_path, spark)
        self.quality_rules = self.config.get("data_quality", {})
        self.col_mapping = self.config.get("column_mapping", {})
        self._dataframes: dict[str, DataFrame] = {}

    def _rename_columns(self, df: DataFrame, mapping: dict) -> DataFrame:
        for old_name, new_name in mapping.items():
            if old_name in df.columns:
                df = df.withColumnRenamed(old_name, new_name)
        return df

    def _transform_books(self, df: DataFrame) -> DataFrame:
        df = self._rename_columns(df, self.col_mapping.get("books", {}))
        df = df.withColumn(
            "year_of_publication",
            F.expr("try_cast(year_of_publication AS int)")
        )
        df = df.filter(
            F.col("year_of_publication").between(
                self.quality_rules["min_year"],
                self.quality_rules["max_year"],
            )
        )
        df = df.filter(F.col("isbn").isNotNull() & (F.trim(F.col("isbn")) != ""))
        df = df.dropDuplicates(["isbn"])
        df = df.withColumn("title", F.trim(F.col("title")))
        df = df.withColumn("author", F.trim(F.col("author")))
        df = df.withColumn("publisher", F.trim(F.col("publisher")))
        return df

    def _transform_users(self, df: DataFrame) -> DataFrame:
        df = self._rename_columns(df, self.col_mapping.get("users", {}))
        df = df.withColumn("user_id", F.expr("try_cast(user_id AS int)"))
        df = df.withColumn("age", F.expr("try_cast(age AS int)"))
        df = df.withColumn(
            "age",
            F.when(
                F.col("age").between(
                    self.quality_rules["min_age"],
                    self.quality_rules["max_age"],
                ),
                F.col("age"),
            ),
        )
        # Parse location: "city, state, country"
        df = df.withColumn("_loc_parts", F.split(F.col("location"), ",\\s*"))
        df = df.withColumn("city", F.trim(F.try_element_at(F.col("_loc_parts"), F.lit(1))))
        df = df.withColumn("state", F.trim(F.try_element_at(F.col("_loc_parts"), F.lit(2))))
        df = df.withColumn("country", F.trim(F.try_element_at(F.col("_loc_parts"), F.lit(3))))
        df = df.drop("_loc_parts")
        df = df.filter(F.col("user_id").isNotNull())
        df = df.dropDuplicates(["user_id"])
        return df

    def _transform_ratings(self, df: DataFrame) -> DataFrame:
        df = self._rename_columns(df, self.col_mapping.get("ratings", {}))
        df = df.withColumn("user_id", F.expr("try_cast(user_id AS int)"))
        df = df.withColumn("book_rating", F.expr("try_cast(book_rating AS int)"))
        df = df.filter(
            F.col("book_rating").between(
                self.quality_rules["min_rating"],
                self.quality_rules["max_rating"],
            )
        )
        df = df.filter(F.col("isbn").isNotNull() & F.col("user_id").isNotNull())
        return df

    def extract(self) -> DataFrame:
        source_tables = self.config.get("source_tables", {})
        for key, table in source_tables.items():
            self.logger.info(f"Reading {table}")
            self._dataframes[key] = self.spark.table(table)
        return self._dataframes.get("books")

    def transform(self, df: DataFrame) -> DataFrame:
        transform_map = {
            "books": self._transform_books,
            "users": self._transform_users,
            "ratings": self._transform_ratings,
        }
        for key, transform_fn in transform_map.items():
            if key in self._dataframes:
                before = self._dataframes[key].count()
                self._dataframes[key] = transform_fn(self._dataframes[key])
                after = self._dataframes[key].count()
                self.logger.info(f"{key}: {before} -> {after} rows")
        return self._dataframes.get("books")

    def load(self, df: DataFrame) -> None:
        silver_cfg = self.config["storage"]["silver"]
        mode = self.config["storage"]["mode"]
        schema = silver_cfg["schema"]

        for key, table_name in silver_cfg["tables"].items():
            if key in self._dataframes:
                full_name = f"{schema}.{table_name}"
                self.logger.info(f"Writing {full_name}")
                self._dataframes[key].write.mode(mode).saveAsTable(full_name)

    def run(self) -> dict[str, DataFrame]:
        self.logger.info(f"Pipeline started: {self.__class__.__name__}")
        self._ensure_schemas()
        self.extract()
        self.transform(None)
        self.load(None)
        self.logger.info("Load complete")
        return self._dataframes
