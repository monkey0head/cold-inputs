import logging
from functools import cached_property
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.semantic_ids._quantizer import Quantizer
from src.semantic_ids.evaluation import metrics
from src.semantic_ids.evaluation import plot
from src.semantic_ids.evaluation.knn import evaluate_prefix_knn
from src.trackers import ExperimentTracker


class SemanticIdsEvaluator:
    """Evaluates and logs quality metrics for semantic IDs."""

    def __init__(
        self,
        sids: torch.Tensor,
        embeddings: np.ndarray,
        codebook_sizes: list[int],
        quantizer: Optional[Quantizer] = None,
        experiment: Optional[ExperimentTracker] = None,
        categories: Optional[pd.DataFrame] = None,
        log_to_console: bool = True,
        log_to_experiment: bool = True,
        compute_entropy: bool = True,
        compute_distances: bool = True,
        compute_normalized_distances: bool = False,
        compute_knn_metrics: bool = True,
        knn_kwargs: dict | None = None,
        compute_prefix_purity: bool = True,
        compute_prefix_clustering: bool = True,
        plot_figures: bool = True,
    ):
        """
        Args:
            sids: Tensor of shape (N, L) containing semantic IDs.
            embeddings: Float32 array of shape (N, D) with original embeddings.
            codebook_sizes: List of codebook sizes for each level.
            quantizer: Optional fitted quantizer for reconstruction.
            experiment: Optional ExperimentTracker for logging.
            categories: Optional DataFrame of shape (N, C) with ground-truth
                category labels per item, aligned row-by-row with ``sids``.
                Each column is one level (e.g. cat1, cat2, ...). Cells may be
                NaN for missing categories. Used by prefix-purity and prefix-
                clustering metrics.
            log_to_console: Whether to log metrics via logging.info.
            log_to_experiment: Whether to log metrics via ExperimentTracker.
            compute_entropy: Whether to compute entropy metrics.
            compute_distances: Whether to compute intra-prefix distances.
            compute_normalized_distances: Whether to compute normalized intra-prefix distances.
            compute_knn_metrics: Whether to compute KNN-based metrics.
            knn_kwargs: Additional arguments for KNN metrics (e.g. k, num_query_samples, metric).
            compute_prefix_purity: Whether to compute prefix category purity
                (purity / entropy / gini per prefix). Requires ``categories``.
            compute_prefix_clustering: Whether to compute clustering metrics
                (V-measure / NMI / ARI). Requires ``categories``.
            plot_figures: Whether to plot and log figures.
        """

        self.sids = sids
        self.embeddings = torch.as_tensor(embeddings, device=sids.device)
        self.codebook_sizes = codebook_sizes
        self.quantizer = quantizer
        self.experiment = experiment
        self.categories = categories

        self.log_to_console = log_to_console
        self.log_to_experiment = log_to_experiment
        self.compute_entropy = compute_entropy
        self.compute_distances = compute_distances
        self.compute_normalized_distances = compute_normalized_distances
        self.compute_knn_metrics = compute_knn_metrics
        self.knn_kwargs = knn_kwargs or {}
        self.compute_prefix_purity = compute_prefix_purity
        self.compute_prefix_clustering = compute_prefix_clustering
        self.plot_figures = plot_figures

    def run_and_log_all(self):
        """Main entry point. Runs the full analysis pipeline."""

        self._log_basic_stats()
        self._log_distribution_stats()

        if self.compute_entropy:
            self._log_entropy()

        if self.quantizer is not None:
            self._log_reconstruction()

        if self.compute_distances or self.compute_normalized_distances:
            self._log_distances()

        if self.compute_knn_metrics:
            self._log_knn_metrics()

        if self.categories is not None and (
            self.compute_prefix_purity or self.compute_prefix_clustering
        ):
            self._log_prefix_categories()

        if self.plot_figures:
            self._plot_and_log_figures()

    def _log_metric(self, name: str, value):
        """Helper to log a metric to console and/or experiment tracker."""

        if self.log_to_console:
            logging.info(f"{name}: {value}")
        if self.log_to_experiment and self.experiment is not None:
            self.experiment.log_scalar(name, value)

    def _log_metrics(self, title: str, values: dict, prefix: str = ""):
        """Helper to log multiple metrics as a dictionary to console and/or experiment tracker."""

        if self.log_to_console:
            logging.info(f"{title}: {values}")
        if self.log_to_experiment and self.experiment is not None:
            # Some metrics are undefined for certain levels (e.g. KNN precision at a
            # level where every query has bucket size 1). The tracker rejects None.
            filtered = {k: v for k, v in values.items() if v is not None}
            if prefix:
                tracker_values = {f"{prefix}/{k}": v for k, v in filtered.items()}
            else:
                tracker_values = filtered
            self.experiment.log_scalars(tracker_values)

    def _log_table(self, title: str, df: pd.DataFrame, series: str | None = None):
        """Helper to log a pandas DataFrame as a table to console and/or experiment tracker."""

        if self.log_to_experiment and self.experiment is not None:
            self.experiment.log_table(df, title=title, series=series)

    def _log_figure(self, name: str, fig: plt.Figure, series: str | None = None):
        """Helper to log a figure to console (display) and/or experiment tracker."""

        if self.log_to_experiment and self.experiment is not None:
            self.experiment.log_figure(fig, title=name, series=series)

        if self.log_to_console:
            try:
                from IPython.display import display
                display(fig)
            except ImportError:
                pass

        plt.close(fig)

    @cached_property
    def codes_distribution(self) -> dict[str, np.ndarray]:
        return metrics.get_codes_distribution(self.sids, codebook_widths=self.codebook_sizes)

    @cached_property
    def prefix_distribution(self) -> dict[str, np.ndarray]:
        return metrics.get_prefix_distribution(self.sids)

    def _log_basic_stats(self):
        """Logs collision rate, unique prefixes, and codebook utilization."""

        # Collision rate
        collision_rate = metrics.get_collision_rate(self.sids)
        self._log_metrics("collision rate", {"collision_rate": collision_rate})

        # Unique prefixes
        unique_prefixes = metrics.count_unique_prefixes(self.sids)
        self._log_metrics("unique prefixes", unique_prefixes, prefix="unique_prefixes")

        # Codebook utilization (absolute counts)
        utilization_counts = metrics.get_codebook_utilization(self.sids)
        self._log_metrics("codebook utilization (counts)", utilization_counts, prefix="codes_used")

        # Codebook utilization (normalized fractions)
        utilization_fractions = metrics.get_codebook_utilization(
            self.sids, codebook_widths=self.codebook_sizes
        )
        self._log_metrics("codebook utilization (fractions)", utilization_fractions, prefix="codebook_utilization")

        # Prefixes density
        prefixes_density = metrics.prefixes_density(self.sids, codebook_widths=self.codebook_sizes)
        self._log_metrics("prefixes density", {"prefixes_density": prefixes_density})

        # Collision load
        collision_load = metrics.get_collision_load(self.sids)
        self._log_metrics("collision load", {"collision_load": collision_load})

        # Log all as a table
        df_basic = pd.DataFrame({
            "unique_prefixes": unique_prefixes,
            "utilization_counts": utilization_counts,
            "utilization_fractions": utilization_fractions
        })
        self._log_table("Basic level stats", df_basic, series="basic_stats")

    def _log_distribution_stats(self):
        """Logs min, max, median, std, and quantiles for code and prefix distributions."""

        code_stats = self.compute_array_stats(self.codes_distribution)            
        for level, stats in code_stats.items():
            self._log_metrics(f"code usage stats {level}", stats, prefix=f"code_usage_{level}")
        df = pd.DataFrame.from_dict(code_stats, orient='index')
        self._log_table("Code usage", df, series="stats")

        prefix_stats = self.compute_array_stats(self.prefix_distribution)
        for level, stats in prefix_stats.items():
            self._log_metrics(f"prefix usage stats {level}", stats, prefix=f"prefix_usage_{level}")
        df = pd.DataFrame.from_dict(prefix_stats, orient='index')
        self._log_table("Prefix usage", df, series="stats")

        # Collisions stats
        last_level = list(self.prefix_distribution.keys())[-1]
        collisions_array = self.prefix_distribution[last_level] - 1
        num_prefixes_with_collisions = int(np.sum(collisions_array > 0))
        total_prefixes = len(collisions_array)
        if num_prefixes_with_collisions == 0:
            # Zero-collision quantizers (e.g. balanced_kmeans):
            # compute_array_stats filters > 0 and would skip the empty entry.
            collision_stats = {k: 0.0 for k in (
                "min", "max", "mean", "std", "q10", "q25", "median", "q75", "q90"
            )}
        else:
            collision_stats_dict = self.compute_array_stats({"collisions": collisions_array})
            collision_stats = collision_stats_dict["collisions"]
        collision_stats["count"] = num_prefixes_with_collisions
        collision_stats["fraction"] = float(num_prefixes_with_collisions / total_prefixes) if total_prefixes > 0 else 0.0
        # log collisions stats
        self._log_metrics("collision stats", collision_stats, prefix="collision_stats")
        df_collisions = pd.DataFrame.from_dict({"collisions": collision_stats}, orient='index')
        self._log_table("Collisions", df_collisions, series="stats")

    def compute_array_stats(self, level_to_arrays: dict):
        """Helper to compute min, max, mean, std, and quantiles for a dictionary of arrays.

        Returns:
            dict mapping level to a dictionary of computed statistics.
        """
        level_to_stats = {}
        for level, array in level_to_arrays.items():
            if len(array) == 0:
                continue

            # Convert to numpy array and filter out zeros
            # (zeros in counts mean unused codes; zeros in distances mean prefixes of size 1)
            array = np.array(array)
            valid_array = array[array > 0]
            if len(valid_array) == 0:
                continue

            stats = {
                "min": float(np.min(valid_array)),
                "max": float(np.max(valid_array)),
                "mean": float(np.mean(valid_array)),
                "std": float(np.std(valid_array)),
                "q10": float(np.percentile(valid_array, 10)),
                "q25": float(np.percentile(valid_array, 25)),
                "median": float(np.median(valid_array)),
                "q75": float(np.percentile(valid_array, 75)),
                "q90": float(np.percentile(valid_array, 90)),
            }
            level_to_stats[level] = stats

        return level_to_stats

    def _log_entropy(self):
        """Logs entropy of code and prefix distributions."""

        # Unnormalized
        code_entropy = metrics.get_entropy_from_counts(self.codes_distribution, normalize=False)
        self._log_metrics("entropy codes", code_entropy, prefix="entropy_codes")

        prefix_entropy = metrics.get_entropy_from_counts(self.prefix_distribution, normalize=False)
        self._log_metrics("entropy prefixes", prefix_entropy, prefix="entropy_prefixes")

        # Normalized
        code_entropy_norm = metrics.get_entropy_from_counts(self.codes_distribution, normalize=True)
        self._log_metrics("normalized entropy codes", code_entropy_norm, prefix="entropy_codes_normed")

        prefix_entropy_norm = metrics.get_entropy_from_counts(self.prefix_distribution, normalize=True)
        self._log_metrics("normalized entropy prefixes", prefix_entropy_norm, prefix="entropy_prefixes_normed")

        # Log all as a table
        df_entropy = pd.DataFrame({
            "codes_entropy": code_entropy,
            "prefixes_entropy": prefix_entropy,
            "codes_entropy_normalized": code_entropy_norm,
            "prefixes_entropy_normalized": prefix_entropy_norm
        })
        self._log_table("Entropy", df_entropy, series="stats")

    def _log_reconstruction(self):
        """Logs reconstruction quality metrics."""

        embed_norm = torch.norm(self.embeddings, dim=1).mean().item()
        embeddings_np = self.embeddings.cpu().numpy()
        rec_embeddings = self.quantizer.reconstruct(embeddings_np)

        if rec_embeddings is not None:
            reconstruction_error, explained_variance = metrics.eval_reconstruction_quality(
                self.embeddings, torch.as_tensor(rec_embeddings, device=self.embeddings.device)
            )
            stats = {
                "error": reconstruction_error,
                "explained_variance": explained_variance,
                "embedding_norm": embed_norm,
            }
            self._log_metrics("reconstruction quality", stats, prefix="reconstruction")

        rec_per_level = self.quantizer.reconstruct_per_level(embeddings_np)
        if rec_per_level is not None:
            per_level_stats = {}
            for level, rec_level in enumerate(rec_per_level, start=1):
                error_level, variance_level = metrics.eval_reconstruction_quality(
                    self.embeddings, torch.as_tensor(rec_level, device=self.embeddings.device)
                )
                level_stats = {
                    "error": error_level,
                    "explained_variance": variance_level,
                }
                self._log_metrics(
                    f"reconstruction quality l{level}",
                    level_stats,
                    prefix=f"reconstruction_l{level}",
                )
                per_level_stats[f"l_{level}"] = level_stats
            df = pd.DataFrame.from_dict(per_level_stats, orient="index")
            self._log_table("Reconstruction by level", df, series="stats")

    def _log_distances(self):
        """Logs intra-prefix distances."""

        if self.compute_distances:
            logging.info("Computing intra-prefix distances...")
            self.distances = metrics.get_intra_prefix_distances(self.sids, self.embeddings)  # Cache for plotting
            dist_stats = self.compute_array_stats(self.distances)
            for level, stats in dist_stats.items():
                self._log_metrics(f"intra_prefix_dist stats {level}", stats, prefix=f"intra_prefix_dist_{level}")
            df = pd.DataFrame.from_dict(dist_stats, orient='index')
            self._log_table("Intra-prefix distances", df, series="stats")

        if self.compute_normalized_distances:
            logging.info("Computing normalized intra-prefix distances...")
            self.normalized_distances = metrics.get_normalized_intra_prefix_distances(self.sids, self.embeddings)  # Cache for plotting
            norm_dist_stats = self.compute_array_stats(self.normalized_distances)
            for level, stats in norm_dist_stats.items():
                self._log_metrics(f"normalized_intra_prefix_dist stats {level}", stats, prefix=f"intra_prefix_dist_normed_{level}")
            df = pd.DataFrame.from_dict(norm_dist_stats, orient='index')
            self._log_table("Normalized intra-prefix distances", df, series="stats")

    def _log_knn_metrics(self):
        """Logs KNN-based metrics for prefix evaluation."""

        logging.info("Computing KNN-based metrics...")

        knn_metrics = evaluate_prefix_knn(
            semantic_ids=self.sids,
            embeddings=self.embeddings,
            **self.knn_kwargs
        )

        df_metrics = {}
        for metric_name, values in knn_metrics.items():
            if isinstance(values, dict):
                self._log_metrics(f"KNN {metric_name}", values, prefix=f"knn_{metric_name}")
                df_metrics[metric_name] = values
            else:
                self._log_metric(f"knn_{metric_name}", values)

        if df_metrics:
            df = pd.DataFrame(df_metrics)
            self._log_table("KNN-based", df, series="metrics")

    def _log_prefix_categories(self):
        """Logs prefix purity and prefix-vs-category clustering metrics."""

        if self.compute_prefix_purity:
            logging.info("Computing prefix category purity...")
            purity = metrics.get_prefix_category_purity(self.sids, self.categories)
            self._log_category_metrics(
                purity, table_name="Prefix purity", scalar_prefix="prefix_purity"
            )

        if self.compute_prefix_clustering:
            logging.info("Computing prefix clustering metrics...")
            clustering = metrics.get_prefix_clustering_metrics(self.sids, self.categories)
            self._log_category_metrics(
                clustering, table_name="Prefix clustering", scalar_prefix="prefix_clustering"
            )

    def _log_category_metrics(
        self,
        per_level: dict[str, dict[str, dict[str, float]]],
        table_name: str,
        scalar_prefix: str,
    ):
        """Helper that flattens {level -> {cat -> {metric -> value}}} and logs it."""

        rows = []
        for level, level_stats in per_level.items():
            for cat, stats in level_stats.items():
                self._log_metrics(
                    f"{table_name} {level}/{cat}",
                    stats,
                    prefix=f"{scalar_prefix}_{level}_{cat}",
                )
                rows.append({"sid_level": level, "category": cat, **stats})

        if rows:
            df = pd.DataFrame(rows).set_index(["sid_level", "category"])
            self._log_table(table_name, df, series="categories")

    def _plot_and_log_figures(self):
        """Plots and logs distributions and distance histograms."""

        logging.info("Plotting figures...")

        # 1. Codes distribution (bar charts)
        fig_codes = plot.plot_codes_distribution(
            self.codes_distribution,
            title="Codes Distribution"
        )
        self._log_figure("Code usage", fig_codes, series="distributions")
        fig_codes = plot.plot_codes_distribution(
            {key: np.random.permutation(array) for key, array in self.codes_distribution.items()},  # to avoid artifacts of quantiation e.g. for matryoshka_fsq
            title="Codes Distribution permuted"
        )
        self._log_figure("Code usage", fig_codes, series="distributions_permuted")

        # 2. Codes distribution (histograms of frequencies)
        fig_codes_hist = plot.plot_level_histograms(
            self.codes_distribution, 
            xlabel="Code Frequency", 
            ylabel="Number of Codes",
            title="Code usage histogram"
        )
        self._log_figure("Code usage", fig_codes_hist, series="histrogram")

        # 3. Prefix distribution
        fig_prefixes = plot.plot_level_histograms(
            self.prefix_distribution, 
            xlabel="Prefix Frequency", 
            ylabel="Number of Prefixes", 
            log_y=False,
            title="Prefix Frequencies Histogram"
        )
        self._log_figure("Prefix usage", fig_prefixes, series="histrogram")

        # 3.1. Collision distribution
        last_level = list(self.prefix_distribution.keys())[-1]
        collision_distribution = {last_level: self.prefix_distribution[last_level] - 1}
        fig_collisions = plot.plot_level_histograms(
            collision_distribution,
            xlabel="Collisions per ID",
            ylabel="Number of IDs",
            log_y=False,
            exclude_lower_than=1,
            title="Collisions Histogram"
        )
        self._log_figure("Collisions", fig_collisions, series="histrogram")

        # 4. Intra-prefix distances
        if hasattr(self, 'distances'):
            fig_dist = plot.plot_level_histograms(
                self.distances,
                xlabel="Intra-Prefix Distance",
                ylabel="Frequency",
                exclude_lower_than=1e-8,
                title="Intra-Prefix Distances Histogram"
            )
            self._log_figure("Intra-prefix distances", fig_dist, series='histrogram')

        # 5. Normalized intra-prefix distances
        if hasattr(self, 'normalized_distances'):
            fig_norm_dist = plot.plot_level_histograms(
                self.normalized_distances,
                xlabel="Normalized Intra-Prefix Distance",
                ylabel="Frequency",
                exclude_lower_than=1e-8,
                title="Normalized Intra-Prefix Distances Histogram"
            )
            self._log_figure("Normalized intra-prefix distances", fig_norm_dist, series='histrogram')
