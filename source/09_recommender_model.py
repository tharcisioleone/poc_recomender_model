"""Recommender System: ALS collaborative filtering with MLflow tracking."""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from source.base import BaseMLPipeline


class RecommenderModel(BaseMLPipeline):
    """ALS-based book recommender with MLflow experiment tracking."""

    def __init__(self, config_path: str, spark: SparkSession = None):
        super().__init__(config_path, spark)
        self.model_cfg = self.config.get("model", {})
        self.mlflow_cfg = self.config.get("mlflow", {})
        self._model = None
        self._books_df = None
        scale = self.model_cfg.get("rating_scale", {})
        self._rating_min = float(scale.get("min", 1))
        self._rating_max = float(scale.get("max", 10))

    def _clip_rating(self, col: str):
        """Clip ALS scores to the valid rating scale."""
        return F.least(
            F.greatest(F.col(col), F.lit(self._rating_min)),
            F.lit(self._rating_max),
        )

    def extract(self) -> DataFrame:
        book_ratings = self.spark.table(self._source_table("book_ratings"))
        self._books_df = book_ratings.select(
            "isbn", "title", "author"
        ).distinct()
        return book_ratings

    def _build_user_item_index(self, df: DataFrame) -> DataFrame:
        """Map ISBN strings to integer indices for ALS."""
        from pyspark.ml.feature import StringIndexer

        isbn_indexer = StringIndexer(
            inputCol="isbn", outputCol="isbn_index", handleInvalid="keep"
        )
        isbn_model = isbn_indexer.fit(df)
        df = isbn_model.transform(df)
        self._isbn_index_model = isbn_model

        df = df.withColumn("isbn_index", F.col("isbn_index").cast("int"))
        df = df.withColumn("user_id", F.col("user_id").cast("int"))
        df = df.withColumn("book_rating", F.col("book_rating").cast("float"))
        return df

    def prepare(self, df: DataFrame) -> tuple:
        df = self._build_user_item_index(df)
        split_ratio = self.model_cfg["train_test_split"]
        seed = self.model_cfg["seed"]
        train_df, test_df = df.randomSplit([split_ratio, 1 - split_ratio], seed=seed)
        return train_df, test_df

    def train(self, train_df: DataFrame):
        import mlflow

        params = self.model_cfg["params"]
        als = ALS(
            maxIter=params["max_iter"],
            regParam=params["reg_param"],
            rank=params["rank"],
            userCol="user_id",
            itemCol="isbn_index",
            ratingCol="book_rating",
            coldStartStrategy=params["cold_start_strategy"],
            seed=self.model_cfg["seed"],
        )

        mlflow.set_experiment(self.mlflow_cfg["experiment_name"])
        with mlflow.start_run(run_name=self.mlflow_cfg["run_name"]):
            self._model = als.fit(train_df)
            mlflow.log_params(params)
            mlflow.spark.log_model(self._model, "als_model")
            self.logger.info("Model trained and logged to MLflow")

        return self._model

    def evaluate(self, model, test_df: DataFrame) -> dict:
        import mlflow

        predictions = model.transform(test_df).withColumn(
            "prediction", self._clip_rating("prediction")
        )
        evaluator_rmse = RegressionEvaluator(
            metricName="rmse", labelCol="book_rating", predictionCol="prediction"
        )
        evaluator_mae = RegressionEvaluator(
            metricName="mae", labelCol="book_rating", predictionCol="prediction"
        )

        rmse = evaluator_rmse.evaluate(predictions)
        mae = evaluator_mae.evaluate(predictions)
        metrics = {"rmse": round(rmse, 4), "mae": round(mae, 4)}

        with mlflow.start_run(run_name=self.mlflow_cfg["run_name"]):
            mlflow.log_metrics(metrics)

        self._predictions = predictions
        return metrics

    def generate_recommendations(self, n: int = None) -> DataFrame:
        """Generate top-N recommendations per user."""
        top_n = n or self.model_cfg["top_n_recommendations"]
        recs = self._model.recommendForAllUsers(top_n)

        recs_exploded = recs.select(
            F.col("user_id"),
            F.explode("recommendations").alias("rec"),
        ).select(
            F.col("user_id"),
            F.col("rec.isbn_index").alias("isbn_index"),
            F.col("rec.rating").alias("predicted_rating"),
        )

        # Map isbn_index back to ISBN
        index_to_isbn = (
            self._isbn_index_model
            .transform(self._books_df.select("isbn"))
            .select("isbn", F.col("isbn_index").cast("int").alias("isbn_index"))
        )
        recs_with_isbn = recs_exploded.join(index_to_isbn, on="isbn_index", how="inner")
        recs_final = recs_with_isbn.join(
            self._books_df.select("isbn", "title", "author"),
            on="isbn", how="inner",
        )
        return recs_final.select(
            "user_id", "isbn", "title", "author",
            F.round(self._clip_rating("predicted_rating"), 2)
            .cast("float").alias("predicted_rating"),
        )

    def run(self) -> dict:
        result = super().run()

        recs_df = self.generate_recommendations()
        full_name = self._table_name()
        self._ensure_schemas()
        recs_df.write.mode("overwrite").saveAsTable(full_name)
        self.logger.info(f"Recommendations saved: {full_name}")

        result["recommendations"] = recs_df
        return result
