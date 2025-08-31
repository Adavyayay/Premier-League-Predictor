# pl_outcome_pipeline.py
# Premier League outcome predictor (modular, no classes)

from __future__ import annotations
import os
from pathlib import Path
from io import StringIO

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# ----------------------------- I/O & DATA -----------------------------

def ensure_data_dir(dirname: str = "data") -> Path:
    p = Path(dirname)
    p.mkdir(parents=True, exist_ok=True)
    return p


def download_season(url: str, season: str) -> pd.DataFrame:
    """Fetch one season CSV from football-data.co.uk and tag with `Season`."""
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))
    df["Season"] = season
    return df


def download_all_seasons(season_urls: dict[str, str]) -> pd.DataFrame:
    """Download and concatenate multiple seasons (keep ALL columns = raw)."""
    frames = []
    print("Downloading seasons:")
    for season, url in season_urls.items():
        try:
            print(f"  • {season} …")
            df = download_season(url, season)
            frames.append(df)
            print(f"    {len(df)} rows")
        except Exception as e:
            print(f"    !! failed: {e}")
    if not frames:
        raise RuntimeError("No data downloaded.")
    raw = pd.concat(frames, ignore_index=True)
    print(f"Total raw rows: {len(raw)}")
    return raw


def save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)
    print(f"Saved: {path}  ({len(df)} rows)")


# ---------------------- CLEANING & BASE TABLE -------------------------

RENAME_MAP = {
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "FTR": "result",
    "Date": "date",
    "Season": "season",
}

def make_base_table(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Build the minimal supervised table from RAW:
    columns: season, date, home_team, away_team, home_goals, away_goals, result
    """
    # keep original columns safe
    df = raw.copy()

    # standardize names where present
    existing = {src: dst for src, dst in RENAME_MAP.items() if src in df.columns}
    df = df.rename(columns=existing)

    # parse dates (football-data mixes formats)
    if "date" in df.columns:
        try:
            df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
        except Exception:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # keep only rows that have required fields
    needed = ["home_team", "away_team", "home_goals", "away_goals", "result"]
    base = df.dropna(subset=[c for c in needed if c in df.columns]).copy()

    # guard: restrict to H/D/A labels
    base = base[base["result"].isin(["H", "D", "A"])]

    # fill/ensure season text
    if "season" not in base.columns:
        base["season"] = "unknown"

    # order
    sort_cols = [c for c in ["season", "date"] if c in base.columns]
    if sort_cols:
        base = base.sort_values(sort_cols).reset_index(drop=True)

    return base[["season", "date", "home_team", "away_team", "home_goals", "away_goals", "result"]]


# ------------------------- FEATURE ENGINEERING ------------------------

def get_team_history(df: pd.DataFrame, team: str, idx: int, games: int = 5) -> pd.DataFrame:
    """Previous N matches for a team before index `idx` (chronological slice)."""
    past = df.iloc[:idx]
    team_matches = past[(past["home_team"] == team) | (past["away_team"] == team)]
    return team_matches.tail(games)


def calculate_team_stats(history: pd.DataFrame, team: str) -> dict:
    """Points, goals for/against, simple strength & form from a team's recent history."""
    if history.empty:
        return {"strength": 50.0, "form": 1.0, "goals": 1.5, "goals_against": 1.5}

    goals_scored = 0
    goals_conceded = 0
    points = 0

    for _, m in history.iterrows():
        if m["home_team"] == team:
            goals_scored += m["home_goals"]
            goals_conceded += m["away_goals"]
            if m["result"] == "H":
                points += 3
            elif m["result"] == "D":
                points += 1
        else:
            goals_scored += m["away_goals"]
            goals_conceded += m["home_goals"]
            if m["result"] == "A":
                points += 3
            elif m["result"] == "D":
                points += 1

    n = len(history)
    gpg = goals_scored / n
    gcpg = goals_conceded / n
    ppg = points / n  # 0..3

    strength = 50 + 15 * (ppg - 1.5)  # ~[7.5..92.5]
    form = 0.5 + (ppg / 3.0) * 1.5    # ~[0.5..2.0]

    return {"strength": float(strength), "form": float(form), "goals": float(gpg), "goals_against": float(gcpg)}


def add_match_features(base: pd.DataFrame, games: int = 5) -> pd.DataFrame:
    """
    Compute rolling features for each match from each team's previous N games.
    Keeps original target `result`.
    """
    if base.empty:
        return base

    df = base.sort_values(["season", "date"]).reset_index(drop=True).copy()

    # priors / bias
    df["home_team_strength"] = 50.0
    df["away_team_strength"] = 50.0
    df["home_recent_form"] = 1.0
    df["away_recent_form"] = 1.0
    df["home_goals_avg"] = 1.5
    df["away_goals_avg"] = 1.5
    df["home_goals_conceded_avg"] = 1.5
    df["away_goals_conceded_avg"] = 1.5
    df["home_advantage"] = 1.0

    for i, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        home_hist = get_team_history(df, home, i, games)
        away_hist = get_team_history(df, away, i, games)

        hs = calculate_team_stats(home_hist, home)
        as_ = calculate_team_stats(away_hist, away)

        df.loc[i, "home_team_strength"] = hs["strength"]
        df.loc[i, "away_team_strength"] = as_["strength"]
        df.loc[i, "home_recent_form"] = hs["form"]
        df.loc[i, "away_recent_form"] = as_["form"]
        df.loc[i, "home_goals_avg"] = hs["goals"]
        df.loc[i, "away_goals_avg"] = as_["goals"]
        df.loc[i, "home_goals_conceded_avg"] = hs["goals_against"]
        df.loc[i, "away_goals_conceded_avg"] = as_["goals_against"]

    print(f"Features computed for {len(df)} matches (window={games})")
    return df


# ---------------------------- MODELING --------------------------------

FEATURES = [
    "home_team_strength",
    "away_team_strength",
    "home_recent_form",
    "away_recent_form",
    "home_goals_avg",
    "away_goals_avg",
    "home_goals_conceded_avg",
    "away_goals_conceded_avg",
    "home_advantage",
]

def prepare_xy(processed: pd.DataFrame):
    data = processed.dropna(subset=FEATURES + ["result"]).copy()
    X = data[FEATURES]
    y = data["result"]
    return X, y


def train_random_forest(X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=99, stratify=y)
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(Xtr, ytr)
    preds = model.predict(Xte)
    acc = accuracy_score(yte, preds)
    print(f"\nAccuracy: {acc:.3f} (on {len(Xte)} test matches)")
    print(classification_report(yte, preds, digits=3))
    print("Feature importances:")
    for name, imp in sorted(zip(FEATURES, model.feature_importances_), key=lambda x: x[1], reverse=True):
        print(f"  {name:<28} {imp:.3f}")
    return model


def predict_match(model: RandomForestClassifier, processed: pd.DataFrame, home_team: str, away_team: str, games: int = 5):
    # recompute latest features so history includes all prior games
    enhanced = add_match_features(processed[["season","date","home_team","away_team","home_goals","away_goals","result"]], games=games)

    # build a single-row feature vector using the latest stats for each team
    home_hist = get_team_history(enhanced, home_team, len(enhanced), games=games)
    away_hist = get_team_history(enhanced, away_team, len(enhanced), games=games)
    hs = calculate_team_stats(home_hist, home_team)
    as_ = calculate_team_stats(away_hist, away_team)

    row = pd.DataFrame({
        "home_team_strength": [hs["strength"]],
        "away_team_strength": [as_["strength"]],
        "home_recent_form": [hs["form"]],
        "away_recent_form": [as_["form"]],
        "home_goals_avg": [hs["goals"]],
        "away_goals_avg": [as_["goals"]],
        "home_goals_conceded_avg": [hs["goals_against"]],
        "away_goals_conceded_avg": [as_["goals_against"]],
        "home_advantage": [1.0],
    })
    pred = model.predict(row)[0]
    probs = model.predict_proba(row)[0]
    classes = model.classes_

    label = {"H": f"{home_team} Win", "D": "Draw", "A": f"{away_team} Win"}[pred]
    print(f"\nPrediction: {label}")
    print("Probabilities:")
    for c, p in zip(classes, probs):
        txt = {"H": f"{home_team} Win", "D": "Draw", "A": f"{away_team} Win"}[c]
        print(f"  {txt:<18} {p:.3f}")

    return pred, dict(zip(classes, probs))


# ------------------------------ MAIN ----------------------------------

def main():
    data_dir = ensure_data_dir("data")

    # 1) seasons to pull
    season_urls = {
        "2022-23": "https://www.football-data.co.uk/mmz4281/2223/E0.csv",
        "2023-24": "https://www.football-data.co.uk/mmz4281/2324/E0.csv",
        "2024-25": "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
    }

    # 2) download & save RAW (all columns)
    raw = download_all_seasons(season_urls)
    save_csv(raw, data_dir / "premier_league_raw.csv")

    # 3) build base supervised table
    base = make_base_table(raw)

    # 4) feature engineering → save PROCESSED
    processed = add_match_features(base, games=5)
    # Optional: drop the first few rows that only contain priors
    processed_to_save = processed.iloc[10:].reset_index(drop=True)
    save_csv(processed_to_save, data_dir / "premier_league_processed.csv")

    # 5) training
    X, y = prepare_xy(processed_to_save)
    model = train_random_forest(X, y)

    # 6) quick demo prediction
    predict_match(model, processed, home_team="Liverpool", away_team="Bournemouth", games=5)


if __name__ == "__main__":
    main()
