"""
Lab 02: Compute Covariance and Correlation Matrices for the Iris Dataset.

This script loads the Iris dataset, computes pairwise covariance and
correlation for numerical flower measurements, and visualizes correlations
as a heatmap.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_PATH = SCRIPT_DIR / "Iris.csv"
HEATMAP_OUTPUT_PATH = SCRIPT_DIR / "correlation_heatmap.png"

# Flower measurement columns used for statistical analysis
NUMERICAL_FEATURES = [
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm",
]


def load_dataset(file_path: Path) -> pd.DataFrame:
    """Load the Iris dataset from a CSV file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {file_path}")

    return pd.read_csv(file_path)


def extract_numerical_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return only the numerical feature columns required for analysis."""
    missing_columns = set(NUMERICAL_FEATURES) - set(dataframe.columns)
    if missing_columns:
        raise ValueError(f"Missing expected columns: {sorted(missing_columns)}")

    return dataframe[NUMERICAL_FEATURES]


def compute_covariance_matrix(features: pd.DataFrame) -> pd.DataFrame:
    """Compute the covariance matrix for the selected numerical features."""
    return features.cov()


def compute_correlation_matrix(features: pd.DataFrame) -> pd.DataFrame:
    """Compute the Pearson correlation matrix for the selected features."""
    return features.corr()


def print_matrix(title: str, matrix: pd.DataFrame) -> None:
    """Print a matrix with a formatted heading."""
    separator = "=" * len(title)
    print(f"\n{separator}")
    print(title)
    print(separator)
    print(matrix.round(4).to_string())


def _get_unique_feature_pairs(
    correlation_matrix: pd.DataFrame,
) -> list[tuple[str, str, float]]:
    """Return unique feature pairs with their correlation coefficients."""
    feature_pairs = []
    columns = correlation_matrix.columns

    for index, feature_a in enumerate(columns):
        for feature_b in columns[index + 1 :]:
            correlation_value = correlation_matrix.loc[feature_a, feature_b]
            feature_pairs.append((feature_a, feature_b, correlation_value))

    return feature_pairs


def _describe_correlation_strength(correlation_value: float) -> str:
    """Classify correlation strength based on the absolute coefficient."""
    absolute_value = abs(correlation_value)

    if absolute_value >= 0.8:
        return "very strong"
    if absolute_value >= 0.6:
        return "strong"
    if absolute_value >= 0.4:
        return "moderate"
    if absolute_value >= 0.2:
        return "weak"
    return "very weak"


def print_observations(correlation_matrix: pd.DataFrame) -> None:
    """Print key findings from the correlation analysis."""
    feature_pairs = _get_unique_feature_pairs(correlation_matrix)

    strongest_pair = max(feature_pairs, key=lambda pair: abs(pair[2]))
    weakest_pair = min(feature_pairs, key=lambda pair: abs(pair[2]))

    petal_length = "PetalLengthCm"
    petal_width = "PetalWidthCm"
    sepal_length = "SepalLengthCm"
    sepal_width = "SepalWidthCm"

    petal_correlation = correlation_matrix.loc[petal_length, petal_width]
    sepal_petal_length = correlation_matrix.loc[sepal_length, petal_length]
    sepal_petal_width = correlation_matrix.loc[sepal_length, petal_width]
    sepal_width_petal_length = correlation_matrix.loc[sepal_width, petal_length]
    sepal_width_petal_width = correlation_matrix.loc[sepal_width, petal_width]

    title = "OBSERVATIONS"
    separator = "=" * len(title)

    print(f"\n{separator}")
    print(title)
    print(separator)

    observations = [
        (
            "1. Strongest Relationship",
            [
                (
                    f"{strongest_pair[0]} and {strongest_pair[1]} show a "
                    f"{_describe_correlation_strength(strongest_pair[2])} "
                    f"{'positive' if strongest_pair[2] > 0 else 'negative'} "
                    f"correlation (r = {strongest_pair[2]:.4f})."
                ),
                (
                    "Petal measurements are closely related, meaning flowers "
                    "with longer petals usually have wider petals as well."
                ),
            ],
        ),
        (
            "2. Sepal Length and Petal Features",
            [
                (
                    f"{sepal_length} is strongly positively correlated with "
                    f"{petal_length} (r = {sepal_petal_length:.4f}) and "
                    f"{petal_width} (r = {sepal_petal_width:.4f})."
                ),
                (
                    "This indicates that larger flowers tend to have both "
                    "longer sepals and larger petals."
                ),
            ],
        ),
        (
            "3. Sepal Width Behavior",
            [
                (
                    f"{sepal_width} shows a moderate negative correlation with "
                    f"{petal_length} (r = {sepal_width_petal_length:.4f}) and "
                    f"{petal_width} (r = {sepal_width_petal_width:.4f})."
                ),
                (
                    "Flowers with wider sepals are generally associated with "
                    "smaller petal measurements."
                ),
            ],
        ),
        (
            "4. Weakest Relationship",
            [
                (
                    f"{weakest_pair[0]} and {weakest_pair[1]} have a "
                    f"{_describe_correlation_strength(weakest_pair[2])} "
                    f"{'positive' if weakest_pair[2] > 0 else 'negative'} "
                    f"correlation (r = {weakest_pair[2]:.4f})."
                ),
                (
                    "Sepal length and sepal width vary almost independently, "
                    "so sepal size alone is not enough to describe overall flower size."
                ),
            ],
        ),
        (
            "5. Practical Implications for Machine Learning",
            [
                (
                    f"{petal_length} and {petal_width} provide overlapping "
                    f"information (r = {petal_correlation:.4f}), so using both "
                    "may introduce redundancy in a model."
                ),
                (
                    "Correlation analysis helps in exploratory data analysis, "
                    "feature selection, and detecting multicollinearity before "
                    "training machine learning models."
                ),
            ],
        ),
    ]

    for heading, points in observations:
        print(f"\n{heading}")
        for point in points:
            print(f"   - {point}")


def save_correlation_heatmap(
    correlation_matrix: pd.DataFrame,
    output_path: Path,
) -> None:
    """Create and save a correlation heatmap as a PNG image."""
    plt.figure(figsize=(8, 6))

    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Correlation Coefficient"},
    )

    plt.title("Iris Dataset - Correlation Heatmap", fontsize=14, pad=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nCorrelation heatmap saved to: {output_path}")


def main() -> None:
    """Run the covariance and correlation analysis pipeline."""
    print("Loading Iris dataset...")
    iris_data = load_dataset(DATASET_PATH)

    print(f"Dataset shape: {iris_data.shape[0]} rows x {iris_data.shape[1]} columns")

    numerical_features = extract_numerical_features(iris_data)
    print(f"Selected numerical features: {', '.join(NUMERICAL_FEATURES)}")

    covariance_matrix = compute_covariance_matrix(numerical_features)
    correlation_matrix = compute_correlation_matrix(numerical_features)

    print_matrix("COVARIANCE MATRIX", covariance_matrix)
    print_matrix("CORRELATION MATRIX", correlation_matrix)
    print_observations(correlation_matrix)

    save_correlation_heatmap(correlation_matrix, HEATMAP_OUTPUT_PATH)


if __name__ == "__main__":
    main()
