"""Data Merge: Silver to Gold layer with aggregations."""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from source.base import BaseETL


class DataMerge(BaseETL):
    """Joins silver tables into gold layer with pre-computed aggregations."""

    def __init__(self, config_path: str, spark: SparkSession = None):
        super().__init__(config_path, spark)
        self._aggregations: dict[str, DataFrame] = {}

    def extract(self) -> DataFrame:
        meta_cols = ["_source_file", "_ingestion_timestamp"]
        ratings = self.spark.table(self._source_table("ratings")).drop(*meta_cols)
        books = self.spark.table(self._source_table("books")).drop(*meta_cols)
        users = self.spark.table(self._source_table("users")).drop(*meta_cols)

        # Only explicit ratings (> 0)
        explicit_ratings = ratings.filter(F.col("book_rating") > 0)

        join_keys = self.config["join_keys"]
        df = (
            explicit_ratings
            .join(books, on=join_keys["ratings_books"], how="inner")
            .join(users, on=join_keys["ratings_users"], how="inner")
        )
        return df

    def transform(self, df: DataFrame) -> DataFrame:
        # Books summary
        books_summary = (
            df.groupBy("isbn", "title", "author", "publisher", "year_of_publication")
            .agg(
                F.count("book_rating").alias("rating_count"),
                F.avg("book_rating").alias("avg_rating"),
                F.min("book_rating").alias("min_rating"),
                F.max("book_rating").alias("max_rating"),
                F.stddev("book_rating").alias("stddev_rating"),
                F.countDistinct("user_id").alias("unique_raters"),
            )
            .withColumn("avg_rating", F.round("avg_rating", 2))
            .withColumn("stddev_rating", F.round("stddev_rating", 2))
        )

        # Authors summary
        authors_summary = (
            df.groupBy("author")
            .agg(
                F.count("book_rating").alias("total_ratings"),
                F.avg("book_rating").alias("avg_rating"),
                F.countDistinct("isbn").alias("book_count"),
                F.countDistinct("user_id").alias("unique_raters"),
                F.countDistinct("publisher").alias("publisher_count"),
                F.min("year_of_publication").alias("first_publication"),
                F.max("year_of_publication").alias("last_publication"),
            )
            .withColumn("avg_rating", F.round("avg_rating", 2))
        )

        self._aggregations["books_summary"] = books_summary
        self._aggregations["authors_summary"] = authors_summary
        return df

    def load(self, df: DataFrame) -> None:
        mode = self.config["storage"]["mode"]

        full_name = self._table_name()
        self.logger.info(f"Writing {full_name}")
        df.write.mode(mode).saveAsTable(full_name)

        agg_cfg = self.config.get("aggregations", {})
        for key, agg_df in self._aggregations.items():
            if key in agg_cfg:
                schema = agg_cfg[key]["schema"]
                table = agg_cfg[key]["table"]
                full_name = f"{schema}.{table}"
                self.logger.info(f"Writing aggregation: {full_name}")
                agg_df.write.mode(mode).saveAsTable(full_name)
