"""Data Visualization: interactive Plotly charts saved as HTML report."""

import os
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from source.base import BaseAnalysis


class DataVisualization(BaseAnalysis):
    """Creates interactive Plotly visualizations from gold-layer data."""

    def __init__(self, config_path: str, spark: SparkSession = None):
        super().__init__(config_path, spark)
        self.chart_cfg = self.config.get("charts", {})
        self.report_cfg = self.config.get("report", {})
        self._figures = []
        self._defaults = {
            "top_n": 10,
            "min_ratings": 50,
            "min_ratings_per_group": 10,
            "title_max_chars": 50,
            "template": "plotly_white",
        }

    def extract(self) -> dict[str, DataFrame]:
        source_tables = self.config.get("source_tables", {})
        return {key: self.spark.table(tbl) for key, tbl in source_tables.items()}

    def _resolve_year_range(self, books_summary: DataFrame) -> None:
        row = books_summary.select(
            F.min("year_of_publication").alias("min_y"),
            F.max("year_of_publication").alias("max_y"),
        ).first()
        self._min_year = self.chart_cfg.get("min_year_manual", row["min_y"])
        self._max_year = self.chart_cfg.get("max_year_manual", row["max_y"])

    def _top_popular_books(self, books_summary: DataFrame) -> "Figure":
        import plotly.express as px

        top_n = self.chart_cfg.get("top_n", self._defaults["top_n"])
        min_ratings = self.chart_cfg.get("min_ratings", self._defaults["min_ratings"])
        max_chars = self.chart_cfg.get("title_max_chars", self._defaults["title_max_chars"])
        pdf = (
            books_summary
            .filter(F.col("rating_count") >= min_ratings)
            .filter(F.col("year_of_publication").between(self._min_year, self._max_year))
            .orderBy(F.col("rating_count").desc())
            .limit(top_n)
            .toPandas()
        )
        pdf["short_title"] = (
            pdf["title"].str[:max_chars]
            + pdf["title"].str[max_chars:].apply(lambda s: "..." if s else "")
        )
        fig = px.bar(
            pdf, x="rating_count", y="short_title", orientation="h",
            title=f"Top {top_n} Most Popular Books (min. {min_ratings} ratings, {self._min_year}\u2013{self._max_year})",
            labels={"rating_count": "Number of Ratings", "short_title": "Book Title"},
            color="avg_rating", color_continuous_scale="RdYlGn",
            hover_data={"short_title": False, "title": True, "author": True},
            template=self.chart_cfg.get("template", self._defaults["template"]),
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=500)
        return fig

    def _top_rated_authors(self, authors_summary: DataFrame) -> "Figure":
        import plotly.express as px

        top_n = self.chart_cfg.get("top_n", self._defaults["top_n"])
        min_ratings = self.chart_cfg.get("min_ratings", self._defaults["min_ratings"])
        pdf = (
            authors_summary
            .filter(F.col("total_ratings") >= min_ratings)
            .orderBy(F.col("avg_rating").desc())
            .limit(top_n)
            .toPandas()
        )
        fig = px.bar(
            pdf, x="avg_rating", y="author", orientation="h",
            title=f"Top {top_n} Best-Rated Authors (min. {min_ratings} ratings)",
            labels={"avg_rating": "Average Rating", "author": "Author"},
            color="total_ratings", color_continuous_scale="Blues",
            template=self.chart_cfg.get("template", self._defaults["template"]),
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=500)
        return fig

    def _authors_by_year_publisher(self, book_ratings: DataFrame) -> "Figure":
        import plotly.express as px

        top_n = self.chart_cfg.get("top_n", self._defaults["top_n"])
        min_rc = self.chart_cfg.get("min_ratings_per_group", self._defaults["min_ratings_per_group"])
        pdf = (
            book_ratings
            .filter(F.col("book_rating") > 0)
            .filter(F.col("year_of_publication").between(self._min_year, self._max_year))
            .groupBy("author", "year_of_publication", "publisher")
            .agg(
                F.avg("book_rating").alias("avg_rating"),
                F.count("book_rating").alias("rating_count"),
            )
            .filter(F.col("rating_count") >= min_rc)
            .orderBy(F.col("avg_rating").desc())
            .limit(top_n)
            .toPandas()
        )
        fig = px.scatter(
            pdf, x="year_of_publication", y="avg_rating",
            size="rating_count", color="author", hover_data=["publisher"],
            title=f"Top {top_n} Author-Publisher Combos ({self._min_year}\u2013{self._max_year}, min. {min_rc} ratings/group)",
            labels={
                "avg_rating": "Average Rating",
                "year_of_publication": "Year of Publication",
                "rating_count": "Number of Ratings",
            },
            template=self.chart_cfg.get("template", self._defaults["template"]),
        )
        fig.update_layout(height=500)
        return fig

    def _rating_distribution(self, book_ratings: DataFrame) -> "Figure":
        import plotly.express as px

        pdf = (
            book_ratings
            .filter(F.col("book_rating") > 0)
            .filter(F.col("year_of_publication").between(self._min_year, self._max_year))
            .groupBy("book_rating")
            .count()
            .orderBy("book_rating")
            .toPandas()
        )
        fig = px.bar(
            pdf, x="book_rating", y="count",
            title=f"Distribution of Book Ratings ({self._min_year}\u2013{self._max_year}, Explicit Only)",
            labels={"book_rating": "Rating", "count": "Count"},
            template=self.chart_cfg.get("template", self._defaults["template"]),
        )
        fig.update_layout(height=400)
        return fig

    def _publications_by_year(self, books_summary: DataFrame) -> "Figure":
        import plotly.express as px

        pdf = (
            books_summary
            .filter(
                F.col("year_of_publication").between(self._min_year, self._max_year)
            )
            .groupBy("year_of_publication")
            .agg(F.count("*").alias("book_count"))
            .orderBy("year_of_publication")
            .toPandas()
        )
        fig = px.line(
            pdf, x="year_of_publication", y="book_count",
            title=f"Number of Rated Books by Year of Publication ({self._min_year}\u2013{self._max_year})",
            labels={"year_of_publication": "Year", "book_count": "Books"},
            template=self.chart_cfg.get("template", self._defaults["template"]),
        )
        fig.update_layout(height=400)
        return fig

    def _top_books_by_decade(self, books_summary: DataFrame) -> "Figure":
        import plotly.express as px

        top_n = self.chart_cfg.get("top_n", self._defaults["top_n"])
        min_ratings = self.chart_cfg.get("min_ratings", self._defaults["min_ratings"])
        max_chars = self.chart_cfg.get("title_max_chars", self._defaults["title_max_chars"])
        tpl = self.chart_cfg.get("template", self._defaults["template"])

        pdf = (
            books_summary
            .filter(F.col("rating_count") >= min_ratings)
            .filter(F.col("year_of_publication").between(self._min_year, self._max_year))
            .orderBy(F.col("avg_rating").desc())
            .limit(top_n)
            .toPandas()
        )
        pdf["short_title"] = (
            pdf["title"].str[:max_chars]
            + pdf["title"].str[max_chars:].apply(lambda s: "..." if s else "")
        )

        fig = px.bar(
            pdf, x="short_title", y="avg_rating",
            title=f"Top {top_n} Books by Avg Rating (min. {min_ratings} ratings, {self._min_year}\u2013{self._max_year})",
            labels={"avg_rating": "Average Rating", "short_title": "Book Title"},
            hover_data={"short_title": False, "title": True,
                        "author": True, "year_of_publication": True,
                        "rating_count": True},
            template=tpl,
        )
        fig.update_layout(
            xaxis={"categoryorder": "total descending", "tickangle": -45},
            height=550, margin={"b": 150},
        )
        return fig

    def _save_html_report(self) -> str:
        reports_dir = self._setup.get("reports_path", "/tmp")
        output_file = self.report_cfg["output_file"]
        output_path = f"{reports_dir}/{output_file}"
        title = self.report_cfg["title"]
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        manual = "min_year_manual" in self.chart_cfg
        year_src = "manual override" if manual else "dataset min/max"
        cfg, d = self.chart_cfg, self._defaults
        rows = [
            ("top_n", cfg.get("top_n", d["top_n"])),
            ("min_ratings", cfg.get("min_ratings", d["min_ratings"])),
            ("min_ratings_per_group", cfg.get("min_ratings_per_group", d["min_ratings_per_group"])),
            ("title_max_chars", cfg.get("title_max_chars", d["title_max_chars"])),
            ("min_year (resolved)", self._min_year),
            ("max_year (resolved)", self._max_year),
            ("year range source", year_src),
        ]
        tr = "".join(
            f"<tr><td style='padding:2px 12px 2px 0'>{k}</td><td>{v}</td></tr>"
            for k, v in rows
        )
        params_html = (
            "<div style='background:#f5f5f5;padding:16px 20px;"
            "border-radius:6px;margin-bottom:30px;font-size:14px'>"
            "<strong>Chart parameters</strong> "
            "<span style='color:#888'>(config/08_data_visualization.json)</span>"
            f"<table style='margin-top:8px;border-collapse:collapse'>{tr}</table>"
            "</div>"
        )

        html_parts = [
            "<html><head>",
            f"<title>{title}</title>",
            '<meta charset="utf-8">',
            "<style>body{font-family:Arial,sans-serif;margin:40px;}",
            ".chart{margin-bottom:40px;}</style>",
            f"</head><body><h1>{title}</h1>",
            params_html,
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
        books_summary = dataframes.get("books_summary")
        authors_summary = dataframes.get("authors_summary")
        book_ratings = dataframes.get("book_ratings")

        self._resolve_year_range(books_summary)

        self._figures = [
            self._top_popular_books(books_summary),
            self._top_rated_authors(authors_summary),
            self._authors_by_year_publisher(book_ratings),
            self._rating_distribution(book_ratings),
            self._publications_by_year(books_summary),
            self._top_books_by_decade(books_summary),
        ]

        report_path = self._save_html_report()
        self.logger.info(f"HTML report saved: {report_path}")

        return {"figures": self._figures, "report_path": report_path}
