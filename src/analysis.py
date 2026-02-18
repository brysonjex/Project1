from __future__ import annotations

from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "Grad Program Exit Survey Data.xlsx"
OUTPUT_DIR = BASE_DIR / "outputs"
TABLE_PATH = OUTPUT_DIR / "ranking_table.csv"
PLOT_PATH = OUTPUT_DIR / "ranking_plot.png"


def load_data(path: Path) -> pd.DataFrame:
    """Load survey data and remove metadata rows from Qualtrics exports."""
    df = pd.read_excel(path)

    # Qualtrics exports often include an "ImportId" metadata row right after headers.
    importid_mask = df.apply(
        lambda row: row.astype(str).str.contains(r"\{\"ImportId\"", regex=True, na=False).mean() > 0.5,
        axis=1,
    )
    df = df.loc[~importid_mask].copy()

    return df


def _coerce_year_series(series: pd.Series) -> pd.Series:
    """Convert a candidate year/date column into a numeric year series."""
    # Native datetime columns.
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.dt.year

    # Try plain numeric years first.
    numeric = pd.to_numeric(series, errors="coerce")
    if not numeric.dropna().empty:
        is_year_like = numeric.between(2000, 2100)
        if is_year_like.mean() > 0.5:
            return numeric

        # Excel date serial numbers (e.g., 45037) -> year.
        serial_like = numeric.between(30000, 70000)
        if serial_like.mean() > 0.5:
            as_dt = pd.to_datetime(numeric, unit="D", origin="1899-12-30", errors="coerce")
            return as_dt.dt.year

    # Try parsing strings as datetimes.
    parsed_dt = pd.to_datetime(series, errors="coerce")
    return parsed_dt.dt.year


def _find_year_column(df: pd.DataFrame) -> tuple[str, pd.Series]:
    """Identify the survey year column using column name and value heuristics."""
    preferred_patterns = [
        r"\byear\b",
        r"recorded\s*date",
        r"survey\s*date",
        r"end\s*date",
        r"start\s*date",
        r"date",
    ]

    ordered_candidates: list[str] = []
    for pattern in preferred_patterns:
        ordered_candidates.extend(
            [
                col
                for col in df.columns
                if col not in ordered_candidates
                and re.search(pattern, str(col), flags=re.IGNORECASE)
            ]
        )

    # Fallback: test all columns if none match by name.
    if not ordered_candidates:
        ordered_candidates = list(df.columns)

    best_col: str | None = None
    best_years: pd.Series | None = None
    best_score = -1.0

    for col in ordered_candidates:
        years = _coerce_year_series(df[col])
        valid = years.dropna()
        if valid.empty:
            continue

        score = float(valid.between(2000, 2100).mean())
        if score > best_score:
            best_score = score
            best_col = str(col)
            best_years = years

    if best_col is None or best_years is None or best_score < 0.5:
        raise ValueError("Unable to identify a usable year/date column in the dataset.")

    return best_col, best_years


def filter_year(df: pd.DataFrame) -> tuple[pd.DataFrame, int, str]:
    """Filter to the most recent survey year without hardcoding the year."""
    year_col, year_values = _find_year_column(df)

    if year_values.dropna().empty:
        raise ValueError(f"Year column '{year_col}' does not contain usable year values.")

    selected_year = int(year_values.max())
    filtered_df = df.loc[year_values == selected_year].copy()

    if filtered_df.empty:
        raise ValueError(f"No rows found for selected year {selected_year}.")

    return filtered_df, selected_year, year_col


def _extract_program_name(column_name: str) -> str:
    """Shorten long Qualtrics ranking headers to a readable course/program label."""
    # Typical segment: "... - ACC 6510 Financial Audit - Rank"
    if " - " in column_name:
        parts = [part.strip() for part in column_name.split(" - ")]
        if len(parts) >= 2 and parts[-1].lower() == "rank":
            return parts[-2]
    return column_name


def _select_rating_columns(df: pd.DataFrame, year_col: str) -> list[str]:
    """Select columns that represent numeric course/program rating or preference ranks."""
    selected: list[str] = []

    for col in df.columns:
        if col == year_col:
            continue

        name = str(col)
        lowered = name.lower()

        # Favor explicit rank/rating fields from survey exports.
        is_rating_label = ("rank" in lowered) or ("rating" in lowered) or ("beneficial" in lowered)
        if not is_rating_label:
            continue

        series = pd.to_numeric(df[col], errors="coerce")
        valid = series.dropna()
        if valid.empty:
            continue

        # Numeric preference/rating columns should have multiple values and be in a small bounded range.
        if valid.nunique() < 2:
            continue
        if valid.between(1, 25).mean() < 0.9:
            continue

        selected.append(col)

    if not selected:
        raise ValueError("No rating columns identified after filtering.")

    return selected


def compute_rankings(df: pd.DataFrame, year_col: str) -> pd.DataFrame:
    """Compute mean rating and response count for each program/course."""
    rating_columns = _select_rating_columns(df, year_col)

    records: list[dict[str, object]] = []
    for col in rating_columns:
        series = pd.to_numeric(df[col], errors="coerce")
        response_count = int(series.notna().sum())
        if response_count == 0:
            continue

        records.append(
            {
                "Program/Course Name": _extract_program_name(str(col)),
                "Mean Rating": float(series.mean(skipna=True)),
                "Response Count": response_count,
            }
        )

    rankings = pd.DataFrame(records)
    if rankings.empty:
        raise ValueError("No valid ratings were available to rank.")

    # Merge duplicate course labels that appear across multiple question blocks.
    rankings = (
        rankings.groupby("Program/Course Name", as_index=False)
        .agg({"Mean Rating": "mean", "Response Count": "sum"})
        .sort_values(by=["Mean Rating", "Response Count", "Program/Course Name"], ascending=[False, False, True])
        .reset_index(drop=True)
    )

    rankings.insert(0, "Rank", rankings.index + 1)
    rankings["Mean Rating"] = rankings["Mean Rating"].round(3)
    return rankings


def create_plot(rankings: pd.DataFrame, year: int, output_path: Path) -> None:
    """Create a horizontal bar chart of course/program mean ratings."""
    plot_df = rankings.sort_values("Mean Rating", ascending=True)

    fig, ax = plt.subplots(figsize=(11, max(6, len(plot_df) * 0.45)))
    bars = ax.barh(plot_df["Program/Course Name"], plot_df["Mean Rating"], color="#2E6F9E")

    ax.set_title(f"MAcc Exit Survey Program Rankings – {year}")
    ax.set_xlabel("Mean rating")
    ax.set_ylabel("Program/Course")
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    xmax = float(plot_df["Mean Rating"].max())
    ax.set_xlim(0, xmax * 1.15)

    for bar in bars:
        width = bar.get_width()
        ax.text(width + (xmax * 0.01), bar.get_y() + bar.get_height() / 2, f"{width:.2f}", va="center", fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data(DATA_PATH)
    year_df, selected_year, year_col = filter_year(df)
    rankings = compute_rankings(year_df, year_col)

    rankings.to_csv(TABLE_PATH, index=False)
    create_plot(rankings, selected_year, PLOT_PATH)

    print(f"Year selected: {selected_year}")
    print(f"Year column used: {year_col}")
    print(f"Number of programs ranked: {len(rankings)}")
    print(f"Ranking table saved to: {TABLE_PATH}")
    print(f"Ranking plot saved to: {PLOT_PATH}")


if __name__ == "__main__":
    main()
