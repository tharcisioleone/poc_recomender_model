"""ML Performance: model evaluation visualizations."""

import os
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from source.base import BaseAnalysis


class MLPerformance(BaseAnalysis):
    """Generates Plotly charts for recommender model evaluation."""

    def __init__(self, config_path: str, spark: SparkSession = None):
        super().__init__(config_path, spark)
        self.report_cfg = self.config.get("report", {})
        self.chart_cfg = self.config.get("charts", {})
        self._figures = []

    def extract(self) -> dict[str, DataFrame]:
        source_tables = self.config.get("source_tables", {})
        return {key: self.spark.table(tbl) for key, tbl in source_tables.items()}

    def _model_comparison(self) -> "Figure":
        """Compare RMSE and MAE across model runs from MLflow."""
        import mlflow
        import plotly.graph_objects as go

        comparison_cfg = self.config.get("comparison", {})
        models = comparison_cfg.get("models", [])
        experiment_name = self.config["model"]["experiment_name"]
        experiment = mlflow.get_experiment_by_name(experiment_name)

        labels, rmse_vals, mae_vals = [], [], []
        for model in models:
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=(
                    f"tags.mlflow.runName = '{model['run_name']}' "
                    "and metrics.rmse > 0"
                ),
                order_by=["start_time DESC"],
                max_results=1,
            )
            if not runs.empty:
                rmse = runs.iloc[0].get("metrics.rmse", 0)
                mae = runs.iloc[0].get("metrics.mae", 0)
                labels.append(model["label"])
                rmse_vals.append(round(rmse, 4))
                mae_vals.append(round(mae, 4))
                self.logger.info(
                    f"{model['label']}: RMSE={rmse:.4f}, MAE={mae:.4f}"
                )

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="RMSE", x=labels, y=rmse_vals,
            text=[f"{v:.4f}" for v in rmse_vals], textposition="auto",
        ))
        fig.add_trace(go.Bar(
            name="MAE", x=labels, y=mae_vals,
            text=[f"{v:.4f}" for v in mae_vals], textposition="auto",
        ))

        # Compute improvement if two models
        subtitle = ""
        if len(rmse_vals) == 2:
            rmse_pct = (rmse_vals[0] - rmse_vals[1]) / rmse_vals[0] * 100
            mae_pct = (mae_vals[0] - mae_vals[1]) / mae_vals[0] * 100
            subtitle = (
                f"<br><sub>Improvement: RMSE {rmse_pct:+.1f}%, "
                f"MAE {mae_pct:+.1f}%</sub>"
            )

        fig.update_layout(
            title=f"Model Comparison: RMSE and MAE{subtitle}",
            barmode="group",
            yaxis_title="Error",
            template=self.chart_cfg["template"],
            height=450,
        )
        return fig

    def _prediction_vs_actual(self, book_ratings: DataFrame, recommendations: DataFrame) -> "Figure":
        import plotly.express as px

        # Sample users who have recommendations
        rec_users = recommendations.select("user_id").distinct().limit(1000)
        actual = (
            book_ratings
            .join(rec_users, on="user_id", how="inner")
            .groupBy("user_id")
            .agg(F.avg("book_rating").alias("avg_actual_rating"))
        )
        predicted = (
            recommendations
            .join(rec_users, on="user_id", how="inner")
            .groupBy("user_id")
            .agg(F.avg("predicted_rating").alias("avg_predicted_rating"))
        )
        merged = actual.join(predicted, on="user_id", how="inner").toPandas()

        fig = px.scatter(
            merged, x="avg_actual_rating", y="avg_predicted_rating",
            title="Predicted vs Actual Average Ratings (per User)",
            labels={
                "avg_actual_rating": "Avg Actual Rating",
                "avg_predicted_rating": "Avg Predicted Rating",
            },
            opacity=0.5,
            template=self.chart_cfg["template"],
        )
        fig.add_shape(type="line", x0=0, y0=0, x1=10, y1=10, line=dict(dash="dash", color="red"))
        fig.update_layout(height=500)
        return fig

    def _rating_distribution_comparison(self, book_ratings: DataFrame, recommendations: DataFrame) -> "Figure":
        import plotly.graph_objects as go

        actual_dist = (
            book_ratings
            .groupBy("book_rating")
            .count()
            .orderBy("book_rating")
            .toPandas()
        )
        pred_dist = (
            recommendations
            .withColumn("predicted_bucket", F.round("predicted_rating").cast("int"))
            .groupBy("predicted_bucket")
            .count()
            .orderBy("predicted_bucket")
            .toPandas()
        )

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=actual_dist["book_rating"], y=actual_dist["count"],
            name="Actual Ratings", opacity=0.7,
        ))
        fig.add_trace(go.Bar(
            x=pred_dist["predicted_bucket"], y=pred_dist["count"],
            name="Predicted Ratings", opacity=0.7,
        ))
        fig.update_layout(
            title="Distribution: Actual vs Predicted Ratings",
            xaxis_title="Rating", yaxis_title="Count",
            barmode="overlay", height=450,
            template=self.chart_cfg["template"],
        )
        return fig

    def _top_recommended_books(self, recommendations: DataFrame) -> "Figure":
        import plotly.express as px

        top_recs = (
            recommendations
            .groupBy("isbn", "title", "author")
            .agg(
                F.count("*").alias("times_recommended"),
                F.avg("predicted_rating").alias("avg_predicted_rating"),
            )
            .orderBy(F.col("times_recommended").desc())
            .limit(25)
            .toPandas()
        )
        top_recs["title"] = top_recs["title"].str[:20]
        fig = px.bar(
            top_recs, x="title", y="times_recommended",
            title="Top 25 Most Frequently Recommended Books",
            labels={"times_recommended": "Times Recommended", "title": "Book"},
            color="avg_predicted_rating", color_continuous_scale="RdYlGn",
            template=self.chart_cfg["template"],
        )
        fig.update_layout(xaxis={"categoryorder": "total descending"}, height=500)
        return fig

    def _coverage_metrics(self, book_ratings: DataFrame, recommendations: DataFrame) -> "Figure":
        import plotly.graph_objects as go

        total_books = book_ratings.select("isbn").distinct().count()
        recommended_books = recommendations.select("isbn").distinct().count()
        total_users = book_ratings.select("user_id").distinct().count()
        users_with_recs = recommendations.select("user_id").distinct().count()

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["Book Coverage", "User Coverage"],
            y=[
                round(recommended_books / total_books * 100, 1),
                round(users_with_recs / total_users * 100, 1),
            ],
            text=[
                f"{recommended_books}/{total_books}",
                f"{users_with_recs}/{total_users}",
            ],
            textposition="auto",
        ))
        fig.update_layout(
            title="Recommendation Coverage",
            yaxis_title="Coverage (%)", height=400,
            template=self.chart_cfg["template"],
        )
        return fig

    def _save_html_report(self) -> str:
        reports_dir = self._setup.get("reports_path", "/tmp")
        output_file = self.report_cfg["output_file"]
        output_path = f"{reports_dir}/{output_file}"
        title = self.report_cfg["title"]
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        html_parts = [
            f"<html><head><title>{title}</title>",
            '<meta charset="utf-8">',
            "<style>body{font-family:Arial,sans-serif;margin:40px;}",
            ".chart{margin-bottom:40px;}</style>",
            f"</head><body><h1>{title}</h1>",
        ]
        for fig in self._figures:
            html_parts.append('<div class="chart">')
            html_parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))
            html_parts.append("</div>")
        html_parts.append("</body></html>")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(html_parts))
        return output_path

    def analyze(self, dataframes: dict[str, DataFrame]) -> dict:
        book_ratings = dataframes["book_ratings"]
        recommendations = dataframes["recommendations"]

        self._figures = [
            self._model_comparison(),
            self._prediction_vs_actual(book_ratings, recommendations),
            self._rating_distribution_comparison(book_ratings, recommendations),
            self._top_recommended_books(recommendations),
            self._coverage_metrics(book_ratings, recommendations),
        ]

        report_path = self._save_html_report()
        self.logger.info(f"ML performance report saved: {report_path}")

        return {"figures": self._figures, "report_path": report_path}
