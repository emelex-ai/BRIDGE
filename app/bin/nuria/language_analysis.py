import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")


def load_evaluation_results(excel_path="p2p_english_learner_evaluation_epoch_69.xlsx"):
    """Load the evaluation results from the Excel file."""
    try:
        # Load the per-word results sheet
        df_results = pd.read_excel(excel_path, sheet_name="Per_Word_Results")
        df_summary = pd.read_excel(excel_path, sheet_name="Dataset_Summary")
        print(f"Loaded evaluation results: {len(df_results)} words evaluated")
        return df_results, df_summary
    except FileNotFoundError:
        print(f"Error: Could not find {excel_path}")
        return None, None
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return None, None


def load_multilingual_dataset(
    csv_path="data/nuria_pretraining/p_p_english_pred_learner.csv",
):
    """Load the original multilingual dataset."""
    try:
        df_dataset = pd.read_csv(csv_path)
        print(f"Loaded multilingual dataset: {len(df_dataset)} words")
        print("Language distribution:")
        print(df_dataset["language"].value_counts())
        return df_dataset
    except FileNotFoundError:
        print(f"Error: Could not find {csv_path}")
        return None
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None


def merge_results_with_languages(df_results, df_dataset):
    """Merge evaluation results with language information."""
    # Normalize word text for matching (lowercase, strip whitespace)
    df_results["word_normalized"] = df_results["word_text"].str.lower().str.strip()
    df_dataset["word_normalized"] = df_dataset["word_raw"].str.lower().str.strip()

    # Merge on normalized word text
    df_merged = df_results.merge(
        df_dataset[["word_normalized", "language"]], on="word_normalized", how="left"
    )

    # Check for unmatched words
    unmatched = df_merged["language"].isna().sum()
    if unmatched > 0:
        print(
            f"Warning: {unmatched} words from results could not be matched with language info"
        )
        print("Sample unmatched words:")
        print(df_merged[df_merged["language"].isna()]["word_text"].head(10).tolist())

    # Remove unmatched words for clean analysis
    df_merged = df_merged.dropna(subset=["language"])

    print(f"Successfully merged {len(df_merged)} words with language information")
    print("Language distribution in merged data:")
    print(df_merged["language"].value_counts())

    return df_merged


def calculate_language_statistics(df_merged):
    """Calculate comprehensive statistics by language."""

    # Core performance metrics
    core_metrics = [
        "word_exact_match",
        "overall_feature_accuracy",
        "overall_phoneme_accuracy",
        "edit_distance",
        "sequence_length_match",
    ]

    # Confidence metrics
    confidence_metrics = [
        "avg_max_prob",
        "min_max_prob",
        "max_max_prob",
        "avg_entropy",
        "min_entropy",
        "max_entropy",
    ]

    # Length metrics
    length_metrics = ["target_length", "generated_length"]

    all_metrics = core_metrics + confidence_metrics + length_metrics

    # Calculate statistics by language
    stats_by_language = {}

    for language in df_merged["language"].unique():
        lang_data = df_merged[df_merged["language"] == language]

        stats = {"count": len(lang_data), "metrics": {}}

        for metric in all_metrics:
            if metric in lang_data.columns:
                stats["metrics"][metric] = {
                    "mean": lang_data[metric].mean(),
                    "std": lang_data[metric].std(),
                    "median": lang_data[metric].median(),
                    "min": lang_data[metric].min(),
                    "max": lang_data[metric].max(),
                }

        stats_by_language[language] = stats

    return stats_by_language


def calculate_feature_level_statistics(df_merged):
    """Calculate feature-level accuracy statistics by language."""
    feature_stats = {}

    # Find all feature accuracy columns
    feature_cols = [
        col
        for col in df_merged.columns
        if col.startswith("feature_") and col.endswith("_accuracy")
    ]

    for language in df_merged["language"].unique():
        lang_data = df_merged[df_merged["language"] == language]

        feature_accuracies = []
        for col in feature_cols:
            if col in lang_data.columns:
                # Remove NaN values for this feature
                feature_values = lang_data[col].dropna()
                if len(feature_values) > 0:
                    feature_accuracies.append(feature_values.mean())

        if feature_accuracies:
            feature_stats[language] = {
                "mean_feature_accuracy": np.mean(feature_accuracies),
                "std_feature_accuracy": np.std(feature_accuracies),
                "median_feature_accuracy": np.median(feature_accuracies),
                "features_evaluated": len(feature_accuracies),
            }

    return feature_stats


def print_comparative_summary(stats_by_language):
    """Print a comprehensive comparative summary."""

    print("\n" + "=" * 80)
    print("COMPARATIVE PERFORMANCE ANALYSIS BY LANGUAGE")
    print("=" * 80)

    languages = list(stats_by_language.keys())

    # Sample sizes
    print("\nSAMPLE SIZES:")
    for lang in languages:
        print(f"{lang.upper()}: {stats_by_language[lang]['count']} words")

    # Core performance metrics
    print("\nCORE PERFORMANCE METRICS:")
    print("-" * 50)

    core_metrics = [
        ("word_exact_match", "Word Exact Match Rate", "{:.4f}"),
        ("overall_feature_accuracy", "Feature Accuracy", "{:.4f}"),
        ("overall_phoneme_accuracy", "Phoneme Accuracy", "{:.4f}"),
        ("edit_distance", "Edit Distance", "{:.2f}"),
        ("sequence_length_match", "Length Match Rate", "{:.4f}"),
    ]

    for metric_key, metric_name, fmt in core_metrics:
        print(f"\n{metric_name}:")
        for lang in languages:
            if metric_key in stats_by_language[lang]["metrics"]:
                mean_val = stats_by_language[lang]["metrics"][metric_key]["mean"]
                std_val = stats_by_language[lang]["metrics"][metric_key]["std"]
                print(f"  {lang.upper()}: {fmt.format(mean_val)} (±{std_val:.4f})")

    # Confidence metrics
    print("\nCONFIDENCE METRICS:")
    print("-" * 50)

    confidence_metrics = [
        ("avg_max_prob", "Average Max Probability", "{:.4f}"),
        ("avg_entropy", "Average Entropy", "{:.4f}"),
    ]

    for metric_key, metric_name, fmt in confidence_metrics:
        print(f"\n{metric_name}:")
        for lang in languages:
            if metric_key in stats_by_language[lang]["metrics"]:
                mean_val = stats_by_language[lang]["metrics"][metric_key]["mean"]
                std_val = stats_by_language[lang]["metrics"][metric_key]["std"]
                print(f"  {lang.upper()}: {fmt.format(mean_val)} (±{std_val:.4f})")

    # Length analysis
    print("\nSEQUENCE LENGTH ANALYSIS:")
    print("-" * 50)

    for lang in languages:
        target_len = stats_by_language[lang]["metrics"]["target_length"]["mean"]
        generated_len = stats_by_language[lang]["metrics"]["generated_length"]["mean"]
        print(f"\n{lang.upper()}:")
        print(f"  Average target length: {target_len:.2f}")
        print(f"  Average generated length: {generated_len:.2f}")
        print(f"  Length difference: {abs(target_len - generated_len):.2f}")


def create_comparison_plots(df_merged, output_dir="language_analysis_plots"):
    """Create visualization plots comparing languages."""

    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)

    # Set up the plotting style
    plt.style.use("default")
    sns.set_palette("husl")

    # 1. Core metrics comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(
        "Model Performance Comparison by Language", fontsize=16, fontweight="bold"
    )

    core_metrics = [
        ("word_exact_match", "Word Exact Match Rate"),
        ("overall_feature_accuracy", "Feature Accuracy"),
        ("overall_phoneme_accuracy", "Phoneme Accuracy"),
        ("edit_distance", "Edit Distance"),
    ]

    for i, (metric, title) in enumerate(core_metrics):
        ax = axes[i // 2, i % 2]
        sns.boxplot(data=df_merged, x="language", y=metric, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Language")

        # Add mean values as text
        for lang in df_merged["language"].unique():
            mean_val = df_merged[df_merged["language"] == lang][metric].mean()
            ax.text(
                list(df_merged["language"].unique()).index(lang),
                mean_val,
                f"{mean_val:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/core_metrics_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # 2. Length analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    sns.boxplot(data=df_merged, x="language", y="target_length", ax=ax1)
    ax1.set_title("Target Sequence Length by Language")
    ax1.set_ylabel("Sequence Length")

    sns.boxplot(data=df_merged, x="language", y="generated_length", ax=ax2)
    ax2.set_title("Generated Sequence Length by Language")
    ax2.set_ylabel("Sequence Length")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/length_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Confidence metrics
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.boxplot(data=df_merged, x="language", y="avg_max_prob", ax=axes[0])
    axes[0].set_title("Average Maximum Probability")
    axes[0].set_ylabel("Probability")

    sns.boxplot(data=df_merged, x="language", y="avg_entropy", ax=axes[1])
    axes[1].set_title("Average Entropy")
    axes[1].set_ylabel("Entropy")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/confidence_metrics.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Plots saved to {output_dir}/")


def perform_statistical_tests(df_merged):
    """Perform statistical tests to compare languages."""
    from scipy import stats

    print("\nSTATISTICAL SIGNIFICANCE TESTS:")
    print("-" * 50)

    languages = df_merged["language"].unique()
    if len(languages) != 2:
        print("Statistical tests require exactly 2 languages")
        return

    lang1, lang2 = languages
    lang1_data = df_merged[df_merged["language"] == lang1]
    lang2_data = df_merged[df_merged["language"] == lang2]

    test_metrics = [
        "word_exact_match",
        "overall_feature_accuracy",
        "overall_phoneme_accuracy",
        "edit_distance",
    ]

    for metric in test_metrics:
        if metric in df_merged.columns:
            # Mann-Whitney U test (non-parametric)
            statistic, p_value = stats.mannwhitneyu(
                lang1_data[metric].dropna(),
                lang2_data[metric].dropna(),
                alternative="two-sided",
            )

            # Effect size (Cohen's d approximation)
            mean1 = lang1_data[metric].mean()
            mean2 = lang2_data[metric].mean()
            std_pooled = np.sqrt(
                (lang1_data[metric].var() + lang2_data[metric].var()) / 2
            )
            effect_size = abs(mean1 - mean2) / std_pooled if std_pooled > 0 else 0

            significance = (
                "***"
                if p_value < 0.001
                else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
            )

            print(f"\n{metric}:")
            print(f"  {lang1.upper()}: {mean1:.4f}")
            print(f"  {lang2.upper()}: {mean2:.4f}")
            print(f"  p-value: {p_value:.6f} {significance}")
            print(f"  Effect size: {effect_size:.3f}")


def save_detailed_results(
    df_merged,
    stats_by_language,
    feature_stats,
    output_file="language_analysis_detailed.xlsx",
):
    """Save detailed results to Excel file."""

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        # Save merged data
        df_merged.to_excel(writer, sheet_name="All_Results_with_Language", index=False)

        # Create summary statistics sheet
        summary_data = []
        for lang in stats_by_language.keys():
            for metric, stats in stats_by_language[lang]["metrics"].items():
                summary_data.append(
                    {
                        "Language": lang,
                        "Metric": metric,
                        "Count": stats_by_language[lang]["count"],
                        "Mean": stats["mean"],
                        "Std": stats["std"],
                        "Median": stats["median"],
                        "Min": stats["min"],
                        "Max": stats["max"],
                    }
                )

        df_summary = pd.DataFrame(summary_data)
        df_summary.to_excel(writer, sheet_name="Summary_Statistics", index=False)

        # Feature-level statistics
        if feature_stats:
            feature_data = []
            for lang, stats in feature_stats.items():
                feature_data.append({"Language": lang, **stats})
            df_features = pd.DataFrame(feature_data)
            df_features.to_excel(writer, sheet_name="Feature_Statistics", index=False)

    print(f"Detailed results saved to {output_file}")


def main():
    """Main analysis pipeline."""
    print("MULTILINGUAL MODEL PERFORMANCE ANALYSIS")
    print("=" * 50)

    # Load data
    df_results, df_summary = load_evaluation_results()
    if df_results is None:
        return

    df_dataset = load_multilingual_dataset()
    if df_dataset is None:
        return

    # Merge results with language information
    df_merged = merge_results_with_languages(df_results, df_dataset)
    if len(df_merged) == 0:
        print("Error: No words could be matched between results and dataset")
        return

    # Calculate statistics
    stats_by_language = calculate_language_statistics(df_merged)
    feature_stats = calculate_feature_level_statistics(df_merged)

    # Print comparative summary
    print_comparative_summary(stats_by_language)

    # Perform statistical tests
    perform_statistical_tests(df_merged)

    # Create visualizations
    create_comparison_plots(df_merged)

    # Save detailed results
    save_detailed_results(df_merged, stats_by_language, feature_stats)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("Check the generated plots and Excel file for detailed results.")
    print("=" * 80)


if __name__ == "__main__":
    main()
