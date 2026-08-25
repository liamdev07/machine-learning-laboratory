"""
Streamlit application for covariance and correlation matrix analysis.

This module provides a polished web-based UI for uploading datasets, previewing
data, detecting numerical features, computing covariance and correlation
matrices, visualizing correlations, and generating automated observations.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Covariance & Correlation Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

PREVIEW_ROW_COUNT = 10
MATRIX_DECIMAL_PLACES = 4
HEATMAP_ANNOTATION_FORMAT = ".2f"

# Identifier-style columns excluded from statistical analysis
EXCLUDED_IDENTIFIER_COLUMNS = {"id"}


# ---------------------------------------------------------------------------
# Custom Styling
# ---------------------------------------------------------------------------


def inject_custom_styles() -> None:
    """Inject custom CSS for a cleaner, more professional UI."""
    st.markdown(
        """
        <style>
            .main-title {
                font-size: 2.2rem;
                font-weight: 700;
                color: #1f2937;
                margin-bottom: 0.25rem;
            }
            .main-subtitle {
                font-size: 1.05rem;
                color: #6b7280;
                margin-bottom: 1.5rem;
            }
            .section-header {
                background: linear-gradient(90deg, #f8fafc 0%, #ffffff 100%);
                border-left: 4px solid #2563eb;
                padding: 0.85rem 1rem;
                border-radius: 0.5rem;
                margin: 1.5rem 0 1rem 0;
            }
            .section-title {
                font-size: 1.35rem;
                font-weight: 650;
                color: #111827;
                margin: 0;
            }
            .section-caption {
                font-size: 0.95rem;
                color: #6b7280;
                margin: 0.35rem 0 0 0;
            }
            .feature-chip {
                display: inline-block;
                background-color: #eff6ff;
                color: #1d4ed8;
                padding: 0.35rem 0.75rem;
                border-radius: 999px;
                font-size: 0.9rem;
                margin: 0.15rem 0.35rem 0.15rem 0;
                border: 1px solid #bfdbfe;
            }
            div[data-testid="stMetric"] {
                background-color: #f8fafc;
                border: 1px solid #e5e7eb;
                padding: 0.75rem 1rem;
                border-radius: 0.75rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(title: str, caption: str = "") -> None:
    """Render a styled section header with optional caption."""
    caption_html = (
        f'<p class="section-caption">{caption}</p>' if caption else ""
    )
    st.markdown(
        f"""
        <div class="section-header">
            <p class="section-title">{title}</p>
            {caption_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    """Render the application sidebar with guidance and context."""
    with st.sidebar:
        st.markdown("## 📊 Correlation Analyzer")
        st.markdown(
            "An interactive tool for exploring **covariance** and **correlation** "
            "relationships in tabular datasets."
        )
        st.divider()
        st.markdown("### How to Use")
        st.markdown(
            """
            1. Upload a CSV file using the uploader.
            2. Review the dataset preview and detected features.
            3. Explore the covariance and correlation matrices.
            4. Interpret the heatmap and automated observations.
            """
        )
        st.divider()
        st.markdown("### Analysis Notes")
        st.info(
            "Identifier columns such as **Id** are automatically excluded "
            "from numerical analysis to avoid misleading statistical results.",
            icon="ℹ️",
        )
        st.divider()
        st.caption("Lab 02 · M.Tech AI & Data Science")
        st.caption("Covariance & Correlation Matrix Practical")


# ---------------------------------------------------------------------------
# Data Processing
# ---------------------------------------------------------------------------


def load_dataset(uploaded_file) -> pd.DataFrame:
    """Load an uploaded CSV file into a pandas DataFrame."""
    return pd.read_csv(uploaded_file)


def get_numerical_columns(dataframe: pd.DataFrame) -> list[str]:
    """Return numerical columns suitable for statistical analysis."""
    numerical_columns = dataframe.select_dtypes(include="number").columns.tolist()

    return [
        column_name
        for column_name in numerical_columns
        if column_name.lower() not in EXCLUDED_IDENTIFIER_COLUMNS
    ]


def get_excluded_identifier_columns(dataframe: pd.DataFrame) -> list[str]:
    """Return identifier columns that were excluded from analysis."""
    numerical_columns = dataframe.select_dtypes(include="number").columns.tolist()

    return [
        column_name
        for column_name in numerical_columns
        if column_name.lower() in EXCLUDED_IDENTIFIER_COLUMNS
    ]


def extract_numerical_features(
    dataframe: pd.DataFrame,
    numerical_columns: list[str],
) -> pd.DataFrame:
    """Return a DataFrame containing only the selected numerical columns."""
    return dataframe[numerical_columns]


def compute_covariance_matrix(features: pd.DataFrame) -> pd.DataFrame:
    """Compute the covariance matrix for numerical features."""
    return features.cov()


def compute_correlation_matrix(features: pd.DataFrame) -> pd.DataFrame:
    """Compute the Pearson correlation matrix for numerical features."""
    return features.corr()


# ---------------------------------------------------------------------------
# Matrix Styling
# ---------------------------------------------------------------------------


def style_covariance_matrix(matrix: pd.DataFrame) -> pd.io.formats.style.Styler:
    """Apply readable styling to the covariance matrix."""
    format_string = f"{{:.{MATRIX_DECIMAL_PLACES}f}}"

    return (
        matrix.round(MATRIX_DECIMAL_PLACES)
        .style.format(format_string)
        .background_gradient(cmap="Blues", axis=None)
        .set_properties(
            **{
                "text-align": "center",
                "font-size": "13px",
                "padding": "8px",
            }
        )
        .set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [
                        ("text-align", "center"),
                        ("font-weight", "600"),
                        ("background-color", "#f3f4f6"),
                    ],
                }
            ]
        )
    )


def style_correlation_matrix(matrix: pd.DataFrame) -> pd.io.formats.style.Styler:
    """Apply readable styling to the correlation matrix."""
    format_string = f"{{:.{MATRIX_DECIMAL_PLACES}f}}"

    return (
        matrix.round(MATRIX_DECIMAL_PLACES)
        .style.format(format_string)
        .background_gradient(cmap="coolwarm", vmin=-1, vmax=1, axis=None)
        .set_properties(
            **{
                "text-align": "center",
                "font-size": "13px",
                "padding": "8px",
            }
        )
        .set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [
                        ("text-align", "center"),
                        ("font-weight", "600"),
                        ("background-color", "#f3f4f6"),
                    ],
                }
            ]
        )
    )


# ---------------------------------------------------------------------------
# UI Components
# ---------------------------------------------------------------------------


def render_page_header() -> None:
    """Display the main page header."""
    st.markdown(
        '<p class="main-title">Covariance & Correlation Matrix Analyzer</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="main-subtitle">Upload a CSV dataset to explore statistical '
        "relationships between numerical features with interactive matrices, "
        "visualizations, and automated insights.</p>",
        unsafe_allow_html=True,
    )


def render_file_uploader():
    """Render the CSV file uploader widget."""
    render_section_header(
        "Upload Dataset",
        "Select a CSV file to begin the analysis workflow.",
    )

    return st.file_uploader(
        label="Choose a CSV file",
        type=["csv"],
        help="Supported format: comma-separated values (.csv).",
        label_visibility="collapsed",
    )


def render_dataset_metrics(
    dataframe: pd.DataFrame,
    analysis_columns: list[str],
) -> None:
    """Display high-level dataset metrics."""
    metric_columns = st.columns(3)

    metric_columns[0].metric("Rows", f"{dataframe.shape[0]:,}")
    metric_columns[1].metric("Total Columns", dataframe.shape[1])
    metric_columns[2].metric("Analysis Features", len(analysis_columns))


def render_dataset_info(dataframe: pd.DataFrame) -> None:
    """Display basic dataset information."""
    render_section_header(
        "Dataset Overview",
        "Summary information for the uploaded dataset.",
    )

    st.markdown("**All Column Names**")
    st.code(", ".join(dataframe.columns), language=None)


def render_data_preview(dataframe: pd.DataFrame) -> None:
    """Display the first few rows of the uploaded dataset."""
    render_section_header(
        "Data Preview",
        f"First {PREVIEW_ROW_COUNT} rows of the uploaded dataset.",
    )

    st.dataframe(
        dataframe.head(PREVIEW_ROW_COUNT),
        use_container_width=True,
        hide_index=True,
    )


def render_numerical_columns(
    analysis_columns: list[str],
    excluded_columns: list[str],
) -> None:
    """Display detected numerical columns used for analysis."""
    render_section_header(
        "Numerical Features for Analysis",
        "Only meaningful numerical columns are used for matrix computation.",
    )

    if analysis_columns:
        feature_chips = "".join(
            f'<span class="feature-chip">{column_name}</span>'
            for column_name in analysis_columns
        )
        st.markdown(feature_chips, unsafe_allow_html=True)
    else:
        st.warning(
            "No suitable numerical columns were found for analysis.",
            icon="⚠️",
        )

    if excluded_columns:
        st.info(
            "Excluded identifier column(s): "
            + ", ".join(f"`{column_name}`" for column_name in excluded_columns),
            icon="ℹ️",
        )


def render_matrix(
    title: str,
    styled_matrix: pd.io.formats.style.Styler,
    *,
    show_header: bool = True,
) -> None:
    """Display a styled matrix with an optional section heading."""
    if show_header:
        st.markdown(f"### {title}")

    st.dataframe(
        styled_matrix,
        use_container_width=True,
    )


def create_correlation_heatmap(
    correlation_matrix: pd.DataFrame,
) -> plt.Figure:
    """Create a polished seaborn correlation heatmap figure."""
    feature_count = len(correlation_matrix.columns)
    figure_size = max(7, feature_count * 1.35)

    sns.set_theme(style="white")
    figure, axis = plt.subplots(figsize=(figure_size, figure_size * 0.9))

    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=HEATMAP_ANNOTATION_FORMAT,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.8,
        linecolor="#ffffff",
        cbar_kws={"label": "Correlation Coefficient", "shrink": 0.85},
        ax=axis,
    )

    axis.set_title(
        "Correlation Heatmap",
        fontsize=15,
        fontweight="bold",
        pad=14,
    )
    axis.tick_params(axis="x", labelrotation=35)
    axis.tick_params(axis="y", labelrotation=0)
    figure.tight_layout()

    return figure


def render_correlation_heatmap(
    correlation_matrix: pd.DataFrame,
    *,
    show_header: bool = True,
) -> None:
    """Display the correlation heatmap in the Streamlit UI."""
    if show_header:
        st.markdown("### Correlation Heatmap")
        st.caption(
            "Visual overview of correlation strength across all analysis features."
        )

    figure = create_correlation_heatmap(correlation_matrix)
    st.pyplot(figure, use_container_width=True)
    plt.close(figure)


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------


def get_unique_feature_pairs(
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


def describe_correlation_strength(correlation_value: float) -> str:
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


def format_feature_pair_observation(
    feature_a: str,
    feature_b: str,
    correlation_value: float,
) -> str:
    """Format a readable sentence for a feature pair correlation."""
    direction = "positive" if correlation_value > 0 else "negative"
    strength = describe_correlation_strength(correlation_value)

    return (
        f"**{feature_a}** and **{feature_b}** show a {strength} {direction} "
        f"correlation (r = {correlation_value:.{MATRIX_DECIMAL_PLACES}f})."
    )


def generate_observations(
    correlation_matrix: pd.DataFrame,
) -> list[dict[str, list[str] | str]]:
    """Generate observation sections dynamically from the correlation matrix."""
    feature_pairs = get_unique_feature_pairs(correlation_matrix)

    if not feature_pairs:
        return [
            {
                "title": "Insufficient Data for Observations",
                "icon": "⚠️",
                "tone": "warning",
                "points": [
                    "At least two numerical columns are required to analyze "
                    "relationships between features."
                ],
            }
        ]

    positive_pairs = [pair for pair in feature_pairs if pair[2] > 0]
    negative_pairs = [pair for pair in feature_pairs if pair[2] < 0]
    weak_pairs = [pair for pair in feature_pairs if abs(pair[2]) < 0.2]
    high_correlation_pairs = [pair for pair in feature_pairs if abs(pair[2]) >= 0.8]

    observations: list[dict[str, list[str] | str]] = []

    if positive_pairs:
        strongest_positive = max(positive_pairs, key=lambda pair: pair[2])
        observations.append(
            {
                "title": "Strongest Positive Correlation",
                "icon": "📈",
                "tone": "success",
                "points": [
                    format_feature_pair_observation(*strongest_positive),
                    (
                        "When one feature increases, the other feature also "
                        "tends to increase."
                    ),
                ],
            }
        )
    else:
        observations.append(
            {
                "title": "Strongest Positive Correlation",
                "icon": "📈",
                "tone": "info",
                "points": [
                    "No positive correlations were found between the analysis features."
                ],
            }
        )

    if negative_pairs:
        strongest_negative = min(negative_pairs, key=lambda pair: pair[2])
        observations.append(
            {
                "title": "Strongest Negative Correlation",
                "icon": "📉",
                "tone": "info",
                "points": [
                    format_feature_pair_observation(*strongest_negative),
                    (
                        "An increase in one feature is associated with a "
                        "decrease in the other."
                    ),
                ],
            }
        )
    else:
        observations.append(
            {
                "title": "Strongest Negative Correlation",
                "icon": "📉",
                "tone": "info",
                "points": [
                    "No negative correlations were found between the analysis features."
                ],
            }
        )

    weakest_pair = min(feature_pairs, key=lambda pair: abs(pair[2]))
    weak_relationship_points = [
        format_feature_pair_observation(*weakest_pair),
        (
            "These features show little linear dependence and may contribute "
            "independent information to a machine learning model."
        ),
    ]

    if weak_pairs:
        weak_pair_names = [
            f"`{pair[0]}` and `{pair[1]}` (r = {pair[2]:.{MATRIX_DECIMAL_PLACES}f})"
            for pair in sorted(weak_pairs, key=lambda pair: abs(pair[2]))
        ]
        weak_relationship_points.append(
            "Other weak relationships (|r| < 0.2): "
            + ", ".join(weak_pair_names)
            + "."
        )

    observations.append(
        {
            "title": "Weak Relationships",
            "icon": "🔍",
            "tone": "info",
            "points": weak_relationship_points,
        }
    )

    ml_implications = [
        (
            "Correlation analysis supports exploratory data analysis by revealing "
            "how features move together before model building."
        ),
        (
            "Highly correlated features may carry redundant information and should "
            "be reviewed during feature selection or dimensionality reduction."
        ),
    ]

    if high_correlation_pairs:
        redundant_pairs = [
            f"`{pair[0]}` and `{pair[1]}` (r = {pair[2]:.{MATRIX_DECIMAL_PLACES}f})"
            for pair in sorted(
                high_correlation_pairs,
                key=lambda pair: abs(pair[2]),
                reverse=True,
            )
        ]
        ml_implications.insert(
            0,
            (
                "Very strong correlations (|r| ≥ 0.8) were found between "
                + ", ".join(redundant_pairs)
                + ". Using all of these features together may introduce redundancy "
                "or multicollinearity."
            ),
        )
    else:
        ml_implications.insert(
            0,
            (
                "No very strong correlations (|r| ≥ 0.8) were detected, suggesting "
                "a lower immediate risk of severe feature redundancy."
            ),
        )

    observations.append(
        {
            "title": "Practical Machine Learning Implications",
            "icon": "🤖",
            "tone": "success",
            "points": ml_implications,
        }
    )

    return observations


def render_observation_card(observation: dict[str, list[str] | str]) -> None:
    """Render a single observation card with visual hierarchy."""
    observation_text = "\n\n".join(
        f"- {point}" for point in observation["points"]
    )
    tone = observation.get("tone", "info")
    message = f"**{observation['title']}**\n\n{observation_text}"

    if tone == "success":
        st.success(message, icon=str(observation["icon"]))
    elif tone == "warning":
        st.warning(message, icon=str(observation["icon"]))
    else:
        st.info(message, icon=str(observation["icon"]))


def render_observations(
    correlation_matrix: pd.DataFrame,
    *,
    show_header: bool = True,
) -> None:
    """Display automated observations derived from the correlation matrix."""
    if show_header:
        st.markdown("### Automated Observations")
        st.caption("Key insights generated dynamically from the correlation matrix.")

    observations = generate_observations(correlation_matrix)

    for observation in observations:
        render_observation_card(observation)


# ---------------------------------------------------------------------------
# Analysis Workflow
# ---------------------------------------------------------------------------


def render_matrices(numerical_features: pd.DataFrame) -> None:
    """Compute and display covariance and correlation analysis results."""
    render_section_header(
        "Statistical Analysis",
        "Covariance and correlation results computed using the selected numerical features.",
    )

    covariance_matrix = compute_covariance_matrix(numerical_features)
    correlation_matrix = compute_correlation_matrix(numerical_features)

    matrix_tabs = st.tabs(
        ["Covariance Matrix", "Correlation Matrix", "Heatmap", "Observations"]
    )

    with matrix_tabs[0]:
        render_matrix(
            "Covariance Matrix",
            style_covariance_matrix(covariance_matrix),
        )

    with matrix_tabs[1]:
        render_matrix(
            "Correlation Matrix",
            style_correlation_matrix(correlation_matrix),
        )

    with matrix_tabs[2]:
        render_correlation_heatmap(correlation_matrix)

    with matrix_tabs[3]:
        render_observations(correlation_matrix)


def render_uploaded_dataset(uploaded_file) -> None:
    """Process and display information for an uploaded CSV file."""
    st.success(
        f"Dataset loaded successfully: **{uploaded_file.name}**",
        icon="✅",
    )

    try:
        dataset = load_dataset(uploaded_file)
    except Exception as error:
        st.error(f"Unable to read the uploaded CSV file: {error}", icon="❌")
        return

    analysis_columns = get_numerical_columns(dataset)
    excluded_columns = get_excluded_identifier_columns(dataset)

    st.divider()
    render_dataset_metrics(dataset, analysis_columns)
    st.divider()
    render_dataset_info(dataset)
    st.divider()
    render_data_preview(dataset)
    st.divider()
    render_numerical_columns(analysis_columns, excluded_columns)

    if len(analysis_columns) >= 2:
        numerical_features = extract_numerical_features(dataset, analysis_columns)
        st.divider()
        render_matrices(numerical_features)
    elif len(analysis_columns) == 1:
        st.warning(
            "At least two numerical features are required to compute covariance "
            "and correlation matrices.",
            icon="⚠️",
        )
    else:
        st.warning(
            "No valid numerical features available for statistical analysis.",
            icon="⚠️",
        )


def render_empty_state() -> None:
    """Display guidance when no dataset has been uploaded."""
    st.info(
        "Upload a CSV file to preview your data and generate covariance and "
        "correlation insights.",
        icon="📁",
    )


def main() -> None:
    """Run the Streamlit application."""
    inject_custom_styles()
    render_sidebar()

    render_page_header()
    st.divider()

    uploaded_file = render_file_uploader()

    if uploaded_file is not None:
        st.divider()
        render_uploaded_dataset(uploaded_file)
    else:
        render_empty_state()


if __name__ == "__main__":
    main()
