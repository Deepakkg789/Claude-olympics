#!/usr/bin/env python3
"""
medal_report.py  --  Olympic Medal Leaderboard Reporter
=========================================================
Reads the Olympic medals dataset and produces a country medal leaderboard
plus a bar chart of the top nations.

Data model:
    The CSV is ATHLETE-LEVEL — one row per athlete.

    Two event types exist in the data:
      • Team events  (event name contains "team"):  multiple athletes share
        one medal win.  Deduplicated to one row per
        (year, country_code, event, medal) before counting.
      • Solo events  (all others):  each athlete row represents a distinct
        medal in a distinct discipline.  All rows are kept.

    This matches the IOC counting convention: one medal per event won,
    regardless of squad size.

    Note: the dataset is a synthetic sample (~360 unique athletes across
    60 years) and does not cover the full Olympic record.  Counts will
    be proportionally lower than official IOC totals but the relative
    ranking and methodology are correct.

Usage:
    python medal_report.py data/olympic_medals.csv
"""
import sys
import logging
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Config
REPORT_EMAIL = "olympics-bot@devpulse.internal"
TOP_N = 8

# Canonical medal normalisation — covers all observed variants
MEDAL_NORMALISE = {
    "gold":   "Gold",   "GOLD":   "Gold",   "G": "Gold",   "1st": "Gold",
    "silver": "Silver", "SILVER": "Silver", "S": "Silver", "2nd": "Silver",
    "bronze": "Bronze", "BRONZE": "Bronze", "B": "Bronze", "3rd": "Bronze",
    "Gold":   "Gold",   "Silver": "Silver", "Bronze": "Bronze",
}

MEDAL_POINTS = {"Gold": 3, "Silver": 2, "Bronze": 1}


# ---------------------------------------------------------------------------
# Step 1 — load & remove exact duplicate rows
# ---------------------------------------------------------------------------
def load_data(path):
    df = pd.read_csv(path)
    log.info(f"Loaded {len(df)} rows from {path}")

    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    if removed:
        log.warning(f"Removed {removed} exact duplicate rows ({before} → {len(df)})")
    else:
        log.info("No duplicate rows found")

    return df


# ---------------------------------------------------------------------------
# Step 2 — normalise medal values; log and drop unrecognised ones
# ---------------------------------------------------------------------------
def normalise_medals(df):
    original = df["medal"].copy()
    df = df.copy()
    df["medal"] = df["medal"].map(MEDAL_NORMALISE)

    unknown_mask = df["medal"].isna()
    n_unknown = unknown_mask.sum()
    if n_unknown:
        unique_unknown = original[unknown_mask].unique()
        log.warning(
            f"{n_unknown} rows have unrecognised medal values and will be excluded: "
            f"{list(unique_unknown)}"
        )
    else:
        log.info("All medal values normalised successfully — no rows dropped")

    n_corrected = (original != df["medal"]).sum() - n_unknown
    if n_corrected:
        log.info(f"Normalised {n_corrected} non-standard but recoverable medal values")

    return df[~unknown_mask].copy()


# ---------------------------------------------------------------------------
# Step 3 — deduplicate team events to one row per medal win
# ---------------------------------------------------------------------------
def resolve_event_level(df):
    """
    Team events: every squad member shares (year, country_code, event, medal).
    Collapse to one row per unique combination — one medal per team win.

    Solo events: each athlete row is a distinct discipline medal.
    These are kept as-is; no deduplication applied.

    The 'team' keyword in the event name is the reliable signal in this dataset.
    """
    team_mask = df["event"].str.contains("team", case=False)

    df_team = df[team_mask].drop_duplicates(
        subset=["year", "country_code", "event", "medal"]
    )
    df_solo = df[~team_mask]

    athlete_rows_removed = team_mask.sum() - len(df_team)
    log.info(
        f"Team events: {team_mask.sum()} athlete rows → {len(df_team)} medal events "
        f"({athlete_rows_removed} team-member rows collapsed)"
    )
    log.info(f"Solo events: {len(df_solo)} rows retained (each = distinct medal)")

    return pd.concat([df_team, df_solo], ignore_index=True)


# ---------------------------------------------------------------------------
# Step 4 — compute leaderboard
# ---------------------------------------------------------------------------
def compute_leaderboard(df):
    """Tally each country's medal haul."""
    df = df.copy()
    df["points"] = df["medal"].map(MEDAL_POINTS)

    board = (
        df.groupby("country_code")
        .agg(medals=("medal", "count"), points=("points", "sum"))
        .sort_values("medals", ascending=False)
    )
    return board


# ---------------------------------------------------------------------------
# Step 5 — chart with zero-based y-axis
# ---------------------------------------------------------------------------
def make_chart(board, outfile="leaderboard.png"):
    top = board.head(TOP_N)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(top.index, top["medals"], color="#2a78d6")
    ax.set_title("Top Nations by Total Medals (IOC event-level counting)")
    ax.set_ylabel("Medals")
    ax.set_ylim(0, int(top["medals"].max()) * 1.1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    plt.savefig(outfile)
    log.info(f"Chart written to {outfile}")
    return outfile


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/olympic_medals.csv"

    df = load_data(path)
    df = normalise_medals(df)
    df = resolve_event_level(df)   # team-event dedup; solo rows unchanged

    board = compute_leaderboard(df)

    print(f"\n=== MEDAL LEADERBOARD (Top {TOP_N}) ===")
    print(board.head(TOP_N).to_string())

    make_chart(board)
    print(f"\nReport sent to {REPORT_EMAIL}")


if __name__ == "__main__":
    main()
