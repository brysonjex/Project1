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
    """Load survey data from the Excel workbook."""
    return pd.read_excel(path)


def _find_year_column(df: pd.DataFrame) -> str:
    """Identify the survey year column using name and value heuristics."""
    name_candidates = [
        c
        for c in df.columns
        if re.search(r"year|survey.*year|academic.*year", str(c), flags=re.IGNORECASE)
    ]

    if name_candidates:
        return name_candidates[0]

    for col in df.columns:
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if values.empty:
            continue

        if values.between(2000, 2100).mean() > 0.8 and values.nunique() > 1:
            return col

    raise ValueError("Unable to identify a year column in the dataset.")


def filter_year(df: pd.DataFrame) -> tuple[pd.DataFrame, int, str]:
    """Filter to the most recent survey year without hardcoding the year."""
    year_col = _find_year_column(df)
    year_values = pd.to_numeric(df[year_col], errors="coerce")

    if year_values.dropna().empty:
        raise ValueError(f"Year column '{year_col}' does not contain numeric values.")

    selected_year = int(year_values.max())
    filtered_df = df.loc[year_values == selected_year].copy()
    return filtered_df, selected_year, year_col


def _select_rating_columns(df: pd.DataFrame, year_col: str) -> list[str]:
    """Select likely rating columns and exclude metadata or demographic fields."""
    exclude_keywords = {
        "id",
        "uid",
        "email",
        "name",
        "timestamp",
        "time",
        "date",
        "age",
        "gender",
        "sex",
        "ethnicity",
        "race",
        "comment",
        "feedback",
        "text",
        "open",
        "cohort",
        "section",
        "semester",
        "term",
        "gpa",
        "major",
        "minor",
        "program",
        "degree",
        "year",
    }

    numeric_cols = []
    for col in df.columns:
        if col == year_col:
            continue

        series = pd.to_numeric(df[col], errors="coerce")
        valid_ratio = series.notna().mean()
        if valid_ratio < 0.3:
            continue

        col_name = str(col).lower()
        if any(keyword in col_name for keyword in exclude_keywords):
            continue

        # Keep columns that look like Likert-style responses.
        valid_values = series.dropna()
        if valid_values.empty:
            continue

        # Common rating scales are bounded and have repeated values.
        if valid_values.nunique() < 2:
            continue

        if valid_values.between(0, 10).mean() < 0.8:
            continue

        numeric_cols.append(col)

    if not numeric_cols:
        raise ValueError("No rating columns identified after filtering.")

    return numeric_cols


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
                "Program/Course Name": str(col),
                "Mean Rating": float(series.mean(skipna=True)),
                "Response Count": response_count,
            }
        )

    rankings = pd.DataFrame(records)
    if rankings.empty:
        raise ValueError("No valid ratings were available to rank.")

    rankings = rankings.sort_values(
        by=["Mean Rating", "Response Count", "Program/Course Name"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    rankings.insert(0, "Rank", rankings.index + 1)
    return rankings


def create_plot(rankings: pd.DataFrame, year: int, output_path: Path) -> None:
    """Create a horizontal bar chart of course/program mean ratings."""
    plot_df = rankings.sort_values("Mean Rating", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(6, len(plot_df) * 0.4)))
    bars = ax.barh(plot_df["Program/Course Name"], plot_df["Mean Rating"], color="#2E6F9E")

    ax.set_title(f"MAcc Exit Survey Program Rankings – {year}")
    ax.set_xlabel("Mean Rating")
    ax.set_ylabel("Program/Course")
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height() / 2, f"{width:.2f}", va="center", fontsize=8)

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
    print(f"Number of programs ranked: {len(rankings)}")
    print(f"Ranking table saved to: {TABLE_PATH}")
    print(f"Ranking plot saved to: {PLOT_PATH}")


if __name__ == "__main__":
    main()
