import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from pathlib import Path
import numpy as np
from collections import defaultdict


def parse_filename(filename):
    """Extract learner type and epoch from filename."""
    # Pattern: p_p_{learner}_learner_epoch_{epoch}_evaluation.xlsx
    pattern = r"p_p_(.+?)_learner_epoch_(\d+)_evaluation\.xlsx"
    match = re.match(pattern, filename)

    if match:
        learner_type = match.group(1)
        epoch = int(match.group(2))
        return learner_type, epoch
    return None, None


def load_all_results(results_dir="results/nuria_evaluation"):
    """Load all evaluation results from Excel files."""
    results_data = []

    if not os.path.exists(results_dir):
        print(f"Directory {results_dir} not found!")
        return pd.DataFrame()

    excel_files = [f for f in os.listdir(results_dir) if f.endswith(".xlsx")]
    print(f"Found {len(excel_files)} Excel files")

    for filename in excel_files:
        learner_type, epoch = parse_filename(filename)

        if learner_type is None or epoch is None:
            print(f"Skipping file with unexpected name format: {filename}")
            continue

        filepath = os.path.join(results_dir, filename)

        try:
            # Load per-word results
            df_results = pd.read_excel(filepath, sheet_name="Per_Word_Results")
            df_summary = pd.read_excel(filepath, sheet_name="Dataset_Summary")

            # Add metadata
            df_results["learner_type"] = learner_type
            df_results["epoch"] = epoch
            df_results["filename"] = filename

            results_data.append(df_results)
            print(
                f"Loaded {filename}: {learner_type} learner, epoch {epoch}, {len(df_results)} words"
            )

        except Exception as e:
            print(f"Error loading {filename}: {e}")

    if results_data:
        return pd.concat(results_data, ignore_index=True)
    else:
        return pd.DataFrame()


def calculate_performance_metrics(df):
    """Calculate aggregated performance metrics by learner, epoch, and language."""
    # Group by learner_type, epoch, and language
    grouped = df.groupby(["learner_type", "epoch", "language"])

    metrics = []

    for (learner, epoch, language), group in grouped:
        metric_dict = {
            "learner_type": learner,
            "epoch": epoch,
            "language": language,
            "n_words": len(group),
            "word_accuracy": group["word_exact_match"].mean(),
            "phoneme_accuracy": group["overall_phoneme_accuracy"].mean(),
            "feature_accuracy": group["overall_feature_accuracy"].mean(),
            "sequence_length_match": group["sequence_length_match"].mean(),
            "avg_edit_distance": group["edit_distance"].mean(),
            "avg_confidence": (
                group["avg_max_prob"].mean()
                if "avg_max_prob" in group.columns
                else np.nan
            ),
            "avg_entropy": (
                group["avg_entropy"].mean()
                if "avg_entropy" in group.columns
                else np.nan
            ),
        }

        # Calculate phoneme-level accuracies (average across available phoneme positions)
        phoneme_cols = [
            col
            for col in group.columns
            if col.startswith("phoneme_") and col.endswith("_accuracy")
        ]
        if phoneme_cols:
            phoneme_accuracies = []
            for col in phoneme_cols:
                valid_values = group[col].dropna()
                if len(valid_values) > 0:
                    phoneme_accuracies.append(valid_values.mean())
            metric_dict["avg_phoneme_position_accuracy"] = (
                np.mean(phoneme_accuracies) if phoneme_accuracies else np.nan
            )

        metrics.append(metric_dict)

    return pd.DataFrame(metrics)


def create_performance_plots(metrics_df, save_dir="results/plots"):
    """Create comprehensive performance plots."""
    # Create output directory
    os.makedirs(save_dir, exist_ok=True)

    # Define colors and line styles
    learner_colors = {
        "bilingual": "#1f77b4",  # Blue
        "english_pred": "#ff7f0e",  # Orange
        "spanish_pred": "#2ca02c",  # Green
    }

    language_styles = {"EN": "-", "ES": "--"}  # Solid line  # Dashed line

    # Set up the plotting style
    plt.style.use("default")
    sns.set_palette("husl")

    # Key metrics to plot
    metrics_to_plot = [
        ("word_accuracy", "Word Accuracy", "Proportion of words with exact matches"),
        ("phoneme_accuracy", "Phoneme Accuracy", "Average phoneme-level accuracy"),
        ("feature_accuracy", "Feature Accuracy", "Average feature-level accuracy"),
        (
            "avg_edit_distance",
            "Edit Distance",
            "Average edit distance (lower is better)",
        ),
    ]

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    for idx, (metric, title, description) in enumerate(metrics_to_plot):
        ax = axes[idx]

        # Plot each learner/language combination
        for learner in metrics_df["learner_type"].unique():
            learner_data = metrics_df[metrics_df["learner_type"] == learner]

            for language in ["EN", "ES"]:
                lang_data = learner_data[learner_data["language"] == language]

                if len(lang_data) > 0:
                    # Sort by epoch for proper line plotting
                    lang_data = lang_data.sort_values("epoch")

                    label = f"{learner.replace('_', ' ').title()} - {language}"
                    ax.plot(
                        lang_data["epoch"] + 1,
                        lang_data[metric],
                        color=learner_colors[learner],
                        linestyle=language_styles[language],
                        marker="o",
                        linewidth=2,
                        markersize=6,
                        label=label,
                    )

        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.set_title(f"{title} Over Time\n{description}")
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        # Special handling for edit distance (lower is better)
        if metric == "avg_edit_distance":
            ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "learner_performance_comparison.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    # Create a summary table
    create_summary_table(metrics_df, save_dir)


def create_summary_table(metrics_df, save_dir):
    """Create a summary table showing final performance for each learner/language."""
    # Get the latest epoch for each learner
    latest_metrics = metrics_df.loc[
        metrics_df.groupby(["learner_type", "language"])["epoch"].idxmax()
    ]

    # Pivot to create a comparison table
    summary_cols = [
        "word_accuracy",
        "phoneme_accuracy",
        "feature_accuracy",
        "avg_edit_distance",
    ]

    print("\n" + "=" * 80)
    print("FINAL PERFORMANCE SUMMARY (Latest Epoch)")
    print("=" * 80)

    for metric in summary_cols:
        print(f"\n{metric.replace('_', ' ').title()}:")
        print("-" * 40)

        pivot_table = latest_metrics.pivot(
            index="learner_type", columns="language", values=metric
        )
        print(pivot_table.round(4))

    # Save detailed summary to CSV
    summary_file = os.path.join(save_dir, "performance_summary.csv")
    latest_metrics.to_csv(summary_file, index=False)
    print(f"\nDetailed summary saved to: {summary_file}")


def create_detailed_analysis(metrics_df, save_dir="results/plots"):
    """Create additional detailed analysis plots."""

    # 1. Performance improvement over time (delta from first epoch)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    metrics_to_analyze = [
        "word_accuracy",
        "phoneme_accuracy",
        "feature_accuracy",
        "avg_edit_distance",
    ]

    for idx, metric in enumerate(metrics_to_analyze):
        ax = axes[idx]

        for learner in metrics_df["learner_type"].unique():
            for language in ["EN", "ES"]:
                data = metrics_df[
                    (metrics_df["learner_type"] == learner)
                    & (metrics_df["language"] == language)
                ].sort_values("epoch")

                if len(data) > 1:
                    # Calculate improvement from first epoch
                    baseline = data[metric].iloc[0]
                    if metric == "avg_edit_distance":
                        # For edit distance, improvement means reduction
                        improvement = baseline - data[metric]
                    else:
                        # For accuracy metrics, improvement means increase
                        improvement = data[metric] - baseline

                    label = f"{learner.replace('_', ' ').title()} - {language}"
                    ax.plot(
                        data["epoch"] + 1,
                        improvement,
                        label=label,
                        marker="o",
                        linewidth=2,
                    )

        ax.set_xlabel("Epoch")
        ax.set_ylabel(f'{metric.replace("_", " ").title()} Change from Baseline')
        ax.set_title(f'Learning Progress: {metric.replace("_", " ").title()}')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "learning_progress.png"), dpi=300, bbox_inches="tight"
    )
    plt.show()


def main():
    """Main function to run the analysis."""
    print("Loading evaluation results...")

    # Load all results
    all_results = load_all_results()

    if all_results.empty:
        print("No results loaded. Check your file paths and formats.")
        return

    print(f"\nLoaded {len(all_results)} total evaluations")
    print(f"Learner types: {sorted(all_results['learner_type'].unique())}")
    print(f"Epochs: {sorted(all_results['epoch'].unique())}")
    print(f"Languages: {sorted(all_results['language'].unique())}")

    # Calculate performance metrics
    print("\nCalculating performance metrics...")
    metrics_df = calculate_performance_metrics(all_results)

    print(
        f"Calculated metrics for {len(metrics_df)} learner/epoch/language combinations"
    )

    # Create plots
    print("\nCreating performance plots...")
    create_performance_plots(metrics_df)

    # Create detailed analysis
    print("\nCreating detailed analysis...")
    create_detailed_analysis(metrics_df)

    print("\nAnalysis complete! Check the results/plots directory for output files.")


if __name__ == "__main__":
    main()
