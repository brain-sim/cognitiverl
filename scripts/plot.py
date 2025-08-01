import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tyro
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")

# Set plotting style
plt.style.use("default")
sns.set_palette("Set2")
plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 11,
        "figure.titlesize": 16,
        "lines.linewidth": 2.5,
        "grid.alpha": 0.3,
    }
)


@dataclass
class Config:
    """Configuration for multi-algorithm comparison plotting."""

    csv_files: List[str]
    algorithm_names: Optional[List[str]] = None
    output_prefix: str = "approach_avoidance"
    title: str = "Success Rate over Training Steps"
    environment: str = "Approach Avoidance"
    min_seeds: int = 3
    interpolate_missing: bool = True
    filter_outlier: bool = False
    figsize: Tuple[int, int] = (12, 8)
    verbose: bool = True


class OutlierDetector:
    """Handles outlier detection and removal for seeds."""

    def __init__(self, config: Config):
        self.config = config

    def _log(self, message: str, level: str = "INFO"):
        if self.config.verbose:
            print(f"[{level}] {message}")

    def detect_outlier_seeds(
        self, melted_data: pd.DataFrame, algorithm_name: str
    ) -> List[str]:
        """
        Detect outlier seeds using 1-sigma rule from mean.
        Only removes seeds that are more than 1 standard deviation below the mean.
        """
        if not self.config.filter_outlier:
            return []

        self._log(f"Detecting outliers for {algorithm_name}")

        # Debug: Show all unique seeds
        all_seeds = melted_data["Seed"].unique()
        self._log(f"All seeds found: {list(all_seeds)} (total: {len(all_seeds)})")

        if len(all_seeds) < self.config.min_seeds:
            self._log(
                f"Not enough total seeds ({len(all_seeds)} < {self.config.min_seeds})"
            )
            return []

        # Get final step performance for each seed
        final_step = melted_data["Step"].max()
        final_data = melted_data[melted_data["Step"] == final_step]

        # Get average performance for each seed at final step
        seed_performance = (
            final_data.groupby("Seed")["Success_Rate"].mean().reset_index()
        )

        self._log(f"Seed performances at final step ({final_step}):")
        for _, row in seed_performance.iterrows():
            self._log(f"  {row['Seed']}: {row['Success_Rate']:.3f}")

        if len(seed_performance) < self.config.min_seeds:
            self._log(
                f"Not enough seeds with final data ({len(seed_performance)} < {self.config.min_seeds})"
            )
            return []

        # Calculate mean and standard deviation
        success_rates = seed_performance["Success_Rate"].values

        if len(success_rates) <= 3:
            self._log("Too few seeds for reliable outlier detection (≤3), skipping")
            return []

        mean_rate = np.mean(success_rates)
        std_rate = np.std(success_rates)

        # One sigma threshold (only for poor performers)
        lower_threshold = mean_rate - std_rate

        self._log("Performance statistics:")
        self._log(f"  Mean: {mean_rate:.3f}")
        self._log(f"  Std: {std_rate:.3f}")
        self._log(f"  Lower threshold (mean - 1σ): {lower_threshold:.3f}")

        # Find outliers (only below threshold)
        outlier_info = []

        for _, row in seed_performance.iterrows():
            seed_name = row["Seed"]
            performance = row["Success_Rate"]

            if performance < lower_threshold:
                outlier_info.append((performance, seed_name))
                self._log(
                    f"  Outlier detected - {seed_name}: {performance:.3f} (< {lower_threshold:.3f})"
                )

        if not outlier_info:
            self._log("No outliers detected using 1-sigma rule")
            return []

        # Sort by performance (worst first)
        outlier_info.sort(key=lambda x: x[0])

        total_seeds = len(all_seeds)
        total_outliers = len(outlier_info)
        max_removable = total_seeds - self.config.min_seeds

        if max_removable <= 0:
            self._log(
                f"Cannot remove any seeds (would go below minimum of {self.config.min_seeds})"
            )
            return []

        # Remove worst outliers up to the limit
        outliers_to_remove = min(total_outliers, max_removable)
        seeds_to_remove = [seed for _, seed in outlier_info[:outliers_to_remove]]

        self._log("Outlier removal plan:")
        self._log(f"  Total seeds: {total_seeds}")
        self._log(f"  Outliers detected: {total_outliers}")
        self._log(f"  Maximum removable: {max_removable}")
        self._log(f"  Will remove: {outliers_to_remove} worst outliers")

        for i, (perf, seed) in enumerate(outlier_info):
            action = "REMOVE" if i < outliers_to_remove else "KEEP"
            self._log(f"    {action}: {seed} (performance: {perf:.3f})")

        return seeds_to_remove

    def remove_outlier_seeds(
        self, melted_data: pd.DataFrame, outlier_seeds: List[str]
    ) -> pd.DataFrame:
        """Remove specified outlier seeds from the melted data."""
        if not outlier_seeds:
            return melted_data

        original_count = len(melted_data)
        original_seeds = len(melted_data["Seed"].unique())

        cleaned_data = melted_data[~melted_data["Seed"].isin(outlier_seeds)].copy()
        removed_count = original_count - len(cleaned_data)

        remaining_seeds = len(cleaned_data["Seed"].unique())
        self._log(
            f"Removed {removed_count} data points from {len(outlier_seeds)} outlier seeds"
        )
        self._log(f"Seeds: {original_seeds} → {remaining_seeds}")

        return cleaned_data


class CSVProcessor:
    """Handles CSV file processing and cleaning."""

    def __init__(self, config: Config):
        self.config = config

    def _log(self, message: str, level: str = "INFO"):
        """Simple logging function."""
        if self.config.verbose:
            print(f"[{level}] {message}")

    def _extract_seed_columns(self, df: pd.DataFrame) -> List[str]:
        """Extract success rate columns (excluding MIN/MAX)."""
        return [
            col
            for col in df.columns
            if "metrics/success_rate" in col and "MIN" not in col and "MAX" not in col
        ]

    def _get_algorithm_name(self, csv_file: str, index: int) -> str:
        """Get algorithm name from config or derive from filename."""
        if self.config.algorithm_names and index < len(self.config.algorithm_names):
            return self.config.algorithm_names[index]
        return Path(csv_file).stem.upper()

    def clean_csv(self, csv_file: str, algorithm_name: str) -> pd.DataFrame:
        """Clean a single CSV file to have complete data for all seeds."""
        self._log(f"Processing {algorithm_name} ({csv_file})")

        if not os.path.exists(csv_file):
            self._log(f"File not found: {csv_file}", "ERROR")
            return pd.DataFrame()

        try:
            df = pd.read_csv(csv_file)
            self._log(f"Loaded CSV with shape: {df.shape}")
        except Exception as e:
            self._log(f"Error loading {csv_file}: {e}", "ERROR")
            return pd.DataFrame()

        # Extract seed columns
        seed_cols = self._extract_seed_columns(df)
        if not seed_cols:
            self._log(f"No seed columns found in {csv_file}", "ERROR")
            return pd.DataFrame()

        self._log(f"Found {len(seed_cols)} seed columns")

        # Create clean dataframe
        clean_df = df[["Step"] + seed_cols].copy()

        # Convert to numeric, replacing empty strings with NaN
        for col in seed_cols:
            clean_df[col] = pd.to_numeric(
                clean_df[col].replace("", np.nan), errors="coerce"
            )

            # Check for invalid success rates
            valid_mask = (clean_df[col] >= 0) & (clean_df[col] <= 1)
            invalid_count = (~valid_mask & clean_df[col].notna()).sum()

            if invalid_count > 0:
                self._log(f"WARNING: {col} has {invalid_count} invalid success rates")
                invalid_values = clean_df[~valid_mask & clean_df[col].notna()][
                    col
                ].unique()[:5]
                self._log(f"  Examples: {invalid_values}")

                # Clip to valid range
                clean_df[col] = clean_df[col].clip(0, 1)

        # Find complete rows (all seeds have data)
        complete_mask = clean_df[seed_cols].notna().all(axis=1)
        complete_data = clean_df[complete_mask].copy()

        self._log(f"Complete rows: {len(complete_data)}")

        # Interpolate if needed and enabled
        if len(complete_data) == 0 and self.config.interpolate_missing:
            self._log("No complete rows found, attempting interpolation...")
            complete_data = self._interpolate_missing_data(clean_df, seed_cols)

        return complete_data.sort_values("Step").reset_index(drop=True)

    def _interpolate_missing_data(
        self, df: pd.DataFrame, seed_cols: List[str]
    ) -> pd.DataFrame:
        """Interpolate missing data for seeds."""
        # Find common step range
        step_ranges = []
        for col in seed_cols:
            seed_data = df[["Step", col]].dropna()
            if len(seed_data) >= 2:  # Need at least 2 points for interpolation
                step_ranges.append((seed_data["Step"].min(), seed_data["Step"].max()))

        if not step_ranges:
            self._log("Insufficient data for interpolation", "WARNING")
            return pd.DataFrame()

        # Use intersection of all ranges
        min_step = max(r[0] for r in step_ranges)
        max_step = min(r[1] for r in step_ranges)

        # Create target steps from available data
        all_steps = set()
        for col in seed_cols:
            seed_data = df[["Step", col]].dropna()
            valid_steps = seed_data[
                (seed_data["Step"] >= min_step) & (seed_data["Step"] <= max_step)
            ]["Step"]
            all_steps.update(valid_steps)

        target_steps = sorted(all_steps)
        if len(target_steps) < 2:
            return pd.DataFrame()

        # Interpolate each seed
        result_df = pd.DataFrame({"Step": target_steps})
        for col in seed_cols:
            seed_data = df[["Step", col]].dropna()
            if len(seed_data) >= 2:
                # Filter to valid range
                seed_data = seed_data[
                    (seed_data["Step"] >= min_step) & (seed_data["Step"] <= max_step)
                ]
                if len(seed_data) >= 2:
                    interpolated = np.interp(
                        target_steps, seed_data["Step"], seed_data[col]
                    )
                    result_df[col] = interpolated

        # Keep only rows where all seeds have data
        complete_mask = result_df[seed_cols].notna().all(axis=1)
        return result_df[complete_mask].copy()


class DataAligner:
    """Handles alignment of multiple datasets."""

    def __init__(self, config: Config):
        self.config = config

    def _log(self, message: str, level: str = "INFO"):
        if self.config.verbose:
            print(f"[{level}] {message}")

    def find_target_rows(self, cleaned_data: Dict[str, pd.DataFrame]) -> int:
        """Find the minimum number of rows across all datasets."""
        row_counts = {name: len(df) for name, df in cleaned_data.items() if len(df) > 0}

        if not row_counts:
            return 0

        target_rows = min(row_counts.values())
        target_algo = min(row_counts, key=row_counts.get)

        self._log(f"Target: {target_algo} has minimum rows ({target_rows})")
        return target_rows

    def _validate_and_clean_data(
        self, df: pd.DataFrame, algo_name: str
    ) -> pd.DataFrame:
        """Validate and clean data to ensure success rates are between 0 and 1."""
        step_col = "Step"
        seed_cols = [col for col in df.columns if col != step_col]

        # Check for invalid values
        for col in seed_cols:
            invalid_mask = (df[col] < 0) | (df[col] > 1) | df[col].isna()
            invalid_count = invalid_mask.sum()

            if invalid_count > 0:
                self._log(
                    f"WARNING: {algo_name} has {invalid_count} invalid values in {col}"
                )
                # Show some examples
                invalid_values = df[invalid_mask][col].unique()[:5]
                self._log(f"  Example invalid values: {invalid_values}")

                # Clip values to [0, 1] range
                df[col] = df[col].clip(0, 1)

        return df

    def _downsample_with_running_average(
        self, df: pd.DataFrame, target_rows: int, algo_name: str
    ) -> pd.DataFrame:
        """
        Downsample using running averages with proper validation.
        Ensures success rates stay between 0 and 1.
        """
        current_rows = len(df)

        if current_rows <= target_rows:
            return self._validate_and_clean_data(df, algo_name)

        # Validate input data first
        df = self._validate_and_clean_data(df, algo_name)

        # Get the step column and seed columns
        step_col = "Step"
        seed_cols = [col for col in df.columns if col != step_col]

        self._log(f"Downsampling {algo_name}: {current_rows} → {target_rows} rows")
        self._log(
            f"  Original step range: {df[step_col].min():,} to {df[step_col].max():,}"
        )

        # Sort by step to ensure proper ordering
        df_sorted = df.sort_values(step_col).reset_index(drop=True)

        # Create target steps (evenly spaced)
        min_step = df_sorted[step_col].min()
        max_step = df_sorted[step_col].max()
        target_steps = np.linspace(min_step, max_step, target_rows)

        self._log(
            f"  Target step range: {target_steps[0]:,.0f} to {target_steps[-1]:,.0f}"
        )

        downsampled_data = []

        for i, target_step in enumerate(target_steps):
            # Use a more conservative window size
            step_interval = (
                (max_step - min_step) / (target_rows - 1) if target_rows > 1 else 0
            )
            window_size = step_interval * 0.75  # Use 75% of interval as window

            # Define window around target step
            window_start = target_step - window_size
            window_end = target_step + window_size

            # Get data points within the window
            window_mask = (df_sorted[step_col] >= window_start) & (
                df_sorted[step_col] <= window_end
            )
            window_data = df_sorted[window_mask]

            if len(window_data) == 0:
                # If no data in window, find the closest point(s)
                distances = np.abs(df_sorted[step_col] - target_step)
                closest_idx = distances.idxmin()

                # Get a small neighborhood around the closest point
                start_idx = max(0, closest_idx - 1)
                end_idx = min(len(df_sorted), closest_idx + 2)
                window_data = df_sorted.iloc[start_idx:end_idx]

            # Calculate running average for this window
            avg_row = {step_col: target_step}

            for col in seed_cols:
                # Average the values in the window for this seed column
                col_values = window_data[col].dropna()
                if len(col_values) > 0:
                    avg_value = col_values.mean()
                    # Ensure the averaged value is still valid
                    avg_value = np.clip(avg_value, 0, 1)
                    avg_row[col] = avg_value
                else:
                    # Fallback: use nearest neighbor
                    distances = np.abs(df_sorted[step_col] - target_step)
                    closest_idx = distances.idxmin()
                    avg_row[col] = np.clip(df_sorted.iloc[closest_idx][col], 0, 1)

            downsampled_data.append(avg_row)

        # Create the downsampled dataframe
        downsampled_df = pd.DataFrame(downsampled_data)

        # Final validation
        downsampled_df = self._validate_and_clean_data(
            downsampled_df, f"{algo_name}_downsampled"
        )

        # Log some statistics to verify
        for col in seed_cols:
            col_min = downsampled_df[col].min()
            col_max = downsampled_df[col].max()
            self._log(f"  {col}: range [{col_min:.3f}, {col_max:.3f}]")

            if col_max > 1.0:
                self._log(f"  ERROR: {col} has values > 1.0, clipping to 1.0")
                downsampled_df[col] = downsampled_df[col].clip(0, 1)

        return downsampled_df

    def downsample_all(
        self, cleaned_data: Dict[str, pd.DataFrame], target_rows: int
    ) -> Dict[str, pd.DataFrame]:
        """Downsample all datasets to target number of rows with validation."""
        aligned_data = {}

        for algo_name, df in cleaned_data.items():
            if len(df) == 0:
                continue

            current_rows = len(df)

            if current_rows <= target_rows:
                # Still validate even if not downsampling
                validated_df = self._validate_and_clean_data(df, algo_name)
                aligned_data[algo_name] = validated_df
                self._log(f"{algo_name}: Keeping all {current_rows} rows (validated)")
            else:
                # Use running average downsampling with validation
                downsampled = self._downsample_with_running_average(
                    df, target_rows, algo_name
                )
                aligned_data[algo_name] = downsampled
                self._log(f"{algo_name}: Downsampled with running average (validated)")

        return aligned_data


class DataTransformer:
    """Transforms data for plotting."""

    @staticmethod
    def to_long_format(df: pd.DataFrame, algorithm_name: str) -> pd.DataFrame:
        """Convert wide format data to long format for seaborn."""
        # Extract seed columns
        seed_cols = [col for col in df.columns if col != "Step"]

        # FIXED: Rename columns for clarity - improved seed naming logic
        rename_map = {"Step": "Step"}
        for i, col in enumerate(seed_cols):
            if "__" in col:
                # Extract seed number from column name
                parts = col.split("__")

                # Look for the seed number in the appropriate part
                seed_id = None

                # For columns like: "ppo_continuous_action-Spot-Nav-Avoid-v0__ppo_continuous_action__4 - metrics/success_rate"
                # The seed number is in parts[2]: "4 - metrics/success_rate"
                if len(parts) >= 3:
                    # Extract number from the last part before " - metrics"
                    seed_part = parts[2].split(" - ")[0].split("-")[0].strip()
                    if seed_part.isdigit():
                        seed_id = seed_part

                # Fallback: search all parts for any digits
                if seed_id is None:
                    for part in parts:
                        numbers = re.findall(r"\d+", part)
                        if numbers:
                            seed_id = numbers[-1]  # Take the last number found
                            break

                # Final fallback: use sequential numbering
                if seed_id is None:
                    seed_id = str(i + 1)

                rename_map[col] = f"Seed_{seed_id}"
            else:
                rename_map[col] = f"Seed_{i + 1}"

        df_renamed = df.rename(columns=rename_map)

        # Debug: Show the renaming
        print(f"Column renaming for {algorithm_name}:")
        for old, new in rename_map.items():
            if old != "Step":
                print(f"  {old} → {new}")

        # Melt to long format
        melted = pd.melt(
            df_renamed, id_vars=["Step"], var_name="Seed", value_name="Success_Rate"
        )

        # Add metadata
        melted["Algorithm"] = algorithm_name
        melted["Step_M"] = melted["Step"] / 1_000_000

        # Clean data
        melted = melted.dropna()
        melted["Success_Rate"] = pd.to_numeric(melted["Success_Rate"], errors="coerce")
        melted = melted.dropna()

        # Debug: Show unique seeds after processing
        unique_seeds = melted["Seed"].unique()
        print(
            f"Unique seeds after processing: {list(unique_seeds)} (total: {len(unique_seeds)})"
        )

        return melted


class PlotGenerator:
    """Generates comparison plots."""

    def __init__(self, config: Config):
        self.config = config

    def _bounded_ci_estimator(self, data):
        """Custom estimator that returns bounded confidence intervals."""
        if len(data) == 0:
            return 0, 0, 0

        mean_val = np.mean(data)

        if len(data) == 1:
            return mean_val, mean_val, mean_val

        # Calculate 95% CI using t-distribution
        n = len(data)
        se = scipy_stats.sem(data)
        t_val = scipy_stats.t.ppf(0.975, n - 1)  # 95% CI

        ci_lower = max(0, mean_val - t_val * se)  # Bound to 0
        ci_upper = min(1, mean_val + t_val * se)  # Bound to 1

        return mean_val, ci_lower, ci_upper

    def create_comparison_plot(self, combined_data: pd.DataFrame) -> None:
        """Create and save the comparison plot."""
        fig, ax = plt.subplots(figsize=self.config.figsize)

        # Create the plot manually to have better control over confidence intervals
        algorithms = combined_data["Algorithm"].unique()

        for algo in algorithms:
            algo_data = combined_data[combined_data["Algorithm"] == algo]

            # Group by step and calculate bounded statistics
            step_stats = []
            for step in sorted(algo_data["Step_M"].unique()):
                step_data = algo_data[algo_data["Step_M"] == step]["Success_Rate"]
                mean_val, ci_lower, ci_upper = self._bounded_ci_estimator(step_data)
                step_stats.append(
                    {
                        "Step_M": step,
                        "mean": mean_val,
                        "ci_lower": ci_lower,
                        "ci_upper": ci_upper,
                    }
                )

            step_df = pd.DataFrame(step_stats)

            # Plot the line with bounded error bars
            ax.plot(
                step_df["Step_M"],
                step_df["mean"],
                label=algo,
                linewidth=2.5,
                marker="o",
                markersize=4,
                markeredgewidth=0.5,
                markeredgecolor="white",
                alpha=0.8,
            )

            # Add bounded confidence interval as fill
            ax.fill_between(
                step_df["Step_M"], step_df["ci_lower"], step_df["ci_upper"], alpha=0.2
            )

        # Styling
        self._style_plot(ax, combined_data)
        self._add_statistics_box(ax, combined_data)

        plt.tight_layout()

        # Save plots
        outlier_suffix = "_filtered" if self.config.filter_outlier else ""
        png_file = f"{self.config.output_prefix}{outlier_suffix}.png"
        pdf_file = f"{self.config.output_prefix}{outlier_suffix}.pdf"

        plt.savefig(png_file, dpi=300, bbox_inches="tight")
        plt.savefig(pdf_file, bbox_inches="tight")
        plt.show()

        print(f"✓ Plots saved: {png_file}, {pdf_file}")

    def _calculate_bounded_ci(self, values, confidence=0.95):
        """Calculate confidence intervals that respect [0,1] bounds for success rates."""
        if len(values) <= 1:
            val = values[0] if len(values) == 1 else 0
            return val, val, val

        mean_val = np.mean(values)

        # Calculate confidence interval using t-distribution
        alpha = 1 - confidence
        n = len(values)
        se = scipy_stats.sem(values)
        t_val = scipy_stats.t.ppf(1 - alpha / 2, n - 1)

        # Raw confidence interval
        ci_lower = mean_val - t_val * se
        ci_upper = mean_val + t_val * se

        # Bound the confidence interval to [0, 1]
        ci_lower_bounded = max(0, ci_lower)
        ci_upper_bounded = min(1, ci_upper)

        return mean_val, ci_lower_bounded, ci_upper_bounded

    def _style_plot(self, ax, data: pd.DataFrame) -> None:
        """Apply styling to the plot."""
        outlier_text = " (Outliers Filtered)" if self.config.filter_outlier else ""

        ax.set_xlabel("Training Steps (Millions)", fontsize=14, fontweight="bold")
        ax.set_ylabel("Success Rate", fontsize=14, fontweight="bold")
        ax.set_title(
            f"{self.config.title} on {self.config.environment}{outlier_text}\n"
            "Success Rate with 95% Confidence Intervals",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )

        ax.grid(True, alpha=0.3, linestyle="--")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))

        # Set limits with padding
        x_range = data["Step_M"].max() - data["Step_M"].min()
        y_range = data["Success_Rate"].max() - data["Success_Rate"].min()

        ax.set_xlim(
            data["Step_M"].min() - x_range * 0.02, data["Step_M"].max() + x_range * 0.02
        )
        ax.set_ylim(
            max(0, data["Success_Rate"].min() - y_range * 0.05),
            min(1, data["Success_Rate"].max() + y_range * 0.05),  # Cap at 100%
        )

        # Legend
        legend = ax.legend(
            title="Algorithm",
            loc="lower right",
            frameon=True,
            fancybox=True,
            shadow=True,
            title_fontsize=12,
        )
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_alpha(0.9)

    def _add_statistics_box(self, ax, data: pd.DataFrame) -> None:
        """Add final performance statistics box with proper bounded confidence intervals."""
        stats_lines = []

        for algorithm in data["Algorithm"].unique():
            algo_data = data[data["Algorithm"] == algorithm]
            final_step = algo_data["Step"].max()
            final_values = algo_data[algo_data["Step"] == final_step]["Success_Rate"]

            if len(final_values) > 0:
                # Calculate bounded confidence intervals
                mean_val, ci_lower, ci_upper = self._calculate_bounded_ci(
                    final_values, 0.95
                )
                n_seeds = len(final_values)

                # Show the actual range instead of mean ± std
                stats_lines.append(
                    f"{algorithm}: {mean_val:.1%} [{ci_lower:.1%}-{ci_upper:.1%}] (n={n_seeds})"
                )

        if stats_lines:
            outlier_note = " (Outliers Filtered)" if self.config.filter_outlier else ""
            stats_text = (
                f"Final Performance{outlier_note}:\n"
                + "\n".join(stats_lines)
                + "\n(95% CI)"
            )
            ax.text(
                0.02,
                0.98,
                stats_text,
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
                fontsize=10,
            )


def main(config: Config) -> None:
    """Main processing pipeline."""
    print("=" * 60)
    print("MULTI-ALGORITHM COMPARISON PIPELINE")
    print("=" * 60)

    # Validate inputs
    if not config.csv_files:
        print("No CSV files provided!")
        return

    # Initialize components
    processor = CSVProcessor(config)
    aligner = DataAligner(config)
    transformer = DataTransformer()
    outlier_detector = OutlierDetector(config)
    plotter = PlotGenerator(config)

    # Phase 1: Clean all CSV files
    print("\nPHASE 1: CLEANING CSV FILES")
    print("-" * 40)

    cleaned_data = {}
    for i, csv_file in enumerate(config.csv_files):
        algo_name = processor._get_algorithm_name(csv_file, i)
        cleaned_df = processor.clean_csv(csv_file, algo_name)

        if len(cleaned_df) > 0:
            cleaned_data[algo_name] = cleaned_df
            print(f"✓ {algo_name}: {len(cleaned_df)} clean rows")
        else:
            print(f"✗ {algo_name}: No usable data")

    if not cleaned_data:
        print("No valid data found in any CSV files!")
        return

    # Phase 2: Find target and align
    print("\nPHASE 2: ALIGNING DATASETS")
    print("-" * 40)

    target_rows = aligner.find_target_rows(cleaned_data)
    aligned_data = aligner.downsample_all(cleaned_data, target_rows)

    # Phase 3: Transform for plotting
    print("\nPHASE 3: PREPARING DATA FOR PLOTTING")
    print("-" * 40)

    all_data = []
    for algo_name, df in aligned_data.items():
        melted = transformer.to_long_format(df, algo_name)

        # Apply outlier filtering if enabled
        if config.filter_outlier:
            print(f"\nOutlier filtering for {algo_name}:")
            outlier_seeds = outlier_detector.detect_outlier_seeds(melted, algo_name)
            melted = outlier_detector.remove_outlier_seeds(melted, outlier_seeds)

        # Only add if we still have data after filtering
        if len(melted) > 0:
            all_data.append(melted)
            n_seeds = len(melted["Seed"].unique())
            n_steps = len(melted["Step"].unique())
            print(
                f"✓ {algo_name}: {n_seeds} seeds × {n_steps} steps = {len(melted)} points"
            )
        else:
            print(f"✗ {algo_name}: No data remaining after outlier filtering")

    if not all_data:
        print("No data available for plotting!")
        return

    # Combine all data
    combined_data = pd.concat(all_data, ignore_index=True)
    print(f"\nCombined dataset: {combined_data.shape}")
    print(f"Algorithms: {list(combined_data['Algorithm'].unique())}")

    # Phase 4: Generate plot
    print("\nPHASE 4: GENERATING PLOT")
    print("-" * 40)

    plotter.create_comparison_plot(combined_data)

    outlier_status = (
        "WITH outlier filtering"
        if config.filter_outlier
        else "WITHOUT outlier filtering"
    )
    print(f"\n✓ Processing complete {outlier_status}! Target rows: {target_rows}")


if __name__ == "__main__":
    config = tyro.cli(Config)
    main(config)
