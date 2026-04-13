"""
Toronto Shelter System Flow integration.

Loads monthly shelter flow data (2018–2026) and produces:
  1) Annual aggregates (`load_flow`)
  2) Calibration helper for SNA aggregates (`calibrate_agg_df`)
  3) Region-year enrichment helper (`enrich_region_year`)
"""

import io
from pathlib import Path
from typing import Union, Optional

import numpy as np
import pandas as pd
import requests

CKAN_BASE = "https://ckan0.cf.opendata.inter.prod-toronto.ca"
PACKAGE_ID = "toronto-shelter-system-flow"
DATE_COL = "date(mmm-yy)"
POP_COL = "population_group"
ALL_POP = "All Population"
CHRONIC_POP = "Chronic"
YOUTH_POP = "Youth"
INDIGENOUS_POP = "Indigenous"

OCCUPANCY_GLOBS = [
    "Daily shelter occupancy *.csv",
    "daily-shelter-overnight-service-occupancy-capacity-*.csv",
]


def _parse_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format="%y-%b", errors="coerce")


def fetch_flow_from_api() -> pd.DataFrame:
    resp = requests.get(
        f"{CKAN_BASE}/api/3/action/package_show",
        params={"id": PACKAGE_ID},
        timeout=30,
    )
    resp.raise_for_status()
    resources = resp.json()["result"]["resources"]

    for res in resources:
        if res.get("datastore_active"):
            url = f"{CKAN_BASE}/datastore/dump/{res['id']}"
            print(f"  [flow] Downloading from datastore: {url}")
            raw = requests.get(url, timeout=60).text
            return pd.read_csv(io.StringIO(raw))

    for res in resources:
        url = res.get("url", "")
        if url.lower().endswith(".csv"):
            print(f"  [flow] Downloading CSV: {url}")
            raw = requests.get(url, timeout=60).text
            return pd.read_csv(io.StringIO(raw))

    raise ValueError("No usable resource found in shelter flow package.")


def load_flow_csv(source) -> pd.DataFrame:
    if isinstance(source, bytes):
        return pd.read_csv(io.BytesIO(source))
    return pd.read_csv(source)


def _parse_occupancy_date(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.notna().any():
        return parsed
    return pd.to_datetime(series, format="%m/%d/%Y", errors="coerce")


def _annualize_occupancy(df: pd.DataFrame) -> pd.DataFrame:
    date_col = "OCCUPANCY_DATE" if "OCCUPANCY_DATE" in df.columns else None
    if date_col is None:
        return pd.DataFrame()

    work = df.copy()
    work["_date"] = _parse_occupancy_date(work[date_col])
    work = work.dropna(subset=["_date"])
    if work.empty:
        return pd.DataFrame()

    if "SERVICE_USER_COUNT" in work.columns:
        people = pd.to_numeric(work["SERVICE_USER_COUNT"], errors="coerce").fillna(0)
    elif "OCCUPANCY" in work.columns:
        people = pd.to_numeric(work["OCCUPANCY"], errors="coerce").fillna(0)
    else:
        beds = pd.to_numeric(work.get("OCCUPIED_BEDS", 0), errors="coerce").fillna(0)
        rooms = pd.to_numeric(work.get("OCCUPIED_ROOMS", 0), errors="coerce").fillna(0)
        people = beds + rooms

    if "CAPACITY" in work.columns:
        cap = pd.to_numeric(work["CAPACITY"], errors="coerce").fillna(0)
    else:
        bed_cap = pd.to_numeric(work.get("CAPACITY_ACTUAL_BED", 0), errors="coerce").fillna(0)
        room_cap = pd.to_numeric(work.get("CAPACITY_ACTUAL_ROOM", 0), errors="coerce").fillna(0)
        cap = bed_cap + room_cap

    daily = pd.DataFrame({
        "date": work["_date"].dt.normalize(),
        "active": people,
        "capacity": cap,
    }).groupby("date", as_index=False).sum()
    daily["year"] = daily["date"].dt.year

    yearly = daily.groupby("year", as_index=True).agg(
        actively_homeless=("active", "mean"),
        occupancy_capacity=("capacity", "mean"),
    )
    yearly["occupancy_rate"] = yearly["actively_homeless"] / yearly["occupancy_capacity"].replace(0, np.nan)
    yearly["newly_identified"] = np.nan
    return yearly


def load_occupancy_yearly(local_dir: Union[str, Path] = "source_data") -> pd.DataFrame:
    base = Path(local_dir)
    if not base.exists():
        return pd.DataFrame()

    files: list = []
    for pattern in OCCUPANCY_GLOBS:
        files.extend(sorted(base.glob(pattern)))

    if not files:
        return pd.DataFrame()

    frames = []
    for path in files:
        try:
            frames.append(_annualize_occupancy(pd.read_csv(path)))
        except Exception:
            continue

    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()

    merged = pd.concat(frames).groupby(level=0).mean().sort_index()
    print(f"  [flow] Loaded occupancy/capacity annualized data from {len(files)} file(s).")
    return merged


def load_flow(local_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    occupancy_yearly = load_occupancy_yearly("source_data")

    if local_path is not None:
        raw_df = load_flow_csv(local_path)
        print(f"  [flow] Loaded from local file: {local_path}")
    else:
        raw_df = fetch_flow_from_api()
        print("  [flow] Loaded from API.")

    if DATE_COL not in raw_df.columns or POP_COL not in raw_df.columns:
        fallback = _annualize_occupancy(raw_df)
        if not fallback.empty:
            return fallback.clip(lower=0)
        return occupancy_yearly.clip(lower=0) if not occupancy_yearly.empty else pd.DataFrame()

    raw_df[DATE_COL] = _parse_date(raw_df[DATE_COL])
    raw_df["year"] = raw_df[DATE_COL].dt.year
    raw_df = raw_df.dropna(subset=[DATE_COL])

    all_pop = raw_df[raw_df[POP_COL] == ALL_POP].copy()

    gender_total = (
        all_pop["gender_male"].fillna(0)
        + all_pop["gender_female"].fillna(0)
        + all_pop["gender_transgender,non-binary_or_two_spirit"].fillna(0)
    ).replace(0, np.nan)

    all_pop["_pct_male"] = all_pop["gender_male"] / gender_total
    all_pop["_pct_female"] = all_pop["gender_female"] / gender_total
    all_pop["_pct_tnb"] = all_pop["gender_transgender,non-binary_or_two_spirit"] / gender_total

    age_cols = ["ageunder16", "age16-24", "age25-34", "age35-44", "age45-54", "age55-64", "age65over"]
    age_total = all_pop[age_cols].fillna(0).sum(axis=1).replace(0, np.nan)
    for col in age_cols:
        all_pop[f"_pct_{col}"] = all_pop[col].fillna(0) / age_total

    midpoints = {
        "ageunder16": 10,
        "age16-24": 20,
        "age25-34": 29.5,
        "age35-44": 39.5,
        "age45-54": 49.5,
        "age55-64": 59.5,
        "age65over": 70,
    }
    all_pop["_age_avg"] = sum(all_pop[f"_pct_{col}"].fillna(0) * mid for col, mid in midpoints.items())

    yearly_all = all_pop.groupby("year").agg(
        actively_homeless=("actively_homeless", "mean"),
        newly_identified=("newly_identified", "sum"),
        pct_male_flow=("_pct_male", "mean"),
        pct_female_flow=("_pct_female", "mean"),
        pct_tnb_flow=("_pct_tnb", "mean"),
        age_avg_flow=("_age_avg", "mean"),
        age_pct_under16=("_pct_ageunder16", "mean"),
        age_pct_16_24=("_pct_age16-24", "mean"),
        age_pct_25_34=("_pct_age25-34", "mean"),
        age_pct_35_44=("_pct_age35-44", "mean"),
        age_pct_45_54=("_pct_age45-54", "mean"),
        age_pct_55_64=("_pct_age55-64", "mean"),
        age_pct_65over=("_pct_age65over", "mean"),
    )

    for subpop, col_name in [
        (CHRONIC_POP, "pct_chronic_flow"),
        (YOUTH_POP, "pct_youth_flow"),
        (INDIGENOUS_POP, "pct_indigenous_flow"),
    ]:
        sub = raw_df[raw_df[POP_COL] == subpop][["year", DATE_COL, "actively_homeless"]].copy()
        merged = all_pop[["year", DATE_COL, "actively_homeless"]].merge(
            sub, on=[DATE_COL, "year"], suffixes=("_all", "_sub"), how="left"
        )
        merged["_pct"] = merged["actively_homeless_sub"] / merged["actively_homeless_all"].replace(0, np.nan)
        yearly_sub = merged.groupby("year")["_pct"].mean().rename(col_name)
        yearly_all = yearly_all.join(yearly_sub, how="left")

    yearly_all = yearly_all.clip(lower=0)

    if not occupancy_yearly.empty:
        yearly_all = yearly_all.join(
            occupancy_yearly[["occupancy_capacity", "occupancy_rate"]],
            how="outer",
        )
        overlap = yearly_all.index.intersection(occupancy_yearly.index)
        if len(overlap) > 0:
            yearly_all.loc[overlap, "actively_homeless"] = occupancy_yearly.loc[overlap, "actively_homeless"]

    return yearly_all


def calibrate_agg_df(agg_df: pd.DataFrame, flow_yearly: pd.DataFrame) -> pd.DataFrame:
    agg = agg_df.copy()
    flow = flow_yearly.copy()

    flow_first_year = int(flow.index.min())
    all_years = sorted(agg.index.tolist())

    overwrite_map = {
        "total_surveyed": "actively_homeless",
        "pct_male": "pct_male_flow",
        "pct_female": "pct_female_flow",
        "pct_trans_nonbinary": "pct_tnb_flow",
        "age_avg": "age_avg_flow",
        "pct_chronic": "pct_chronic_flow",
        "pct_indigenous": "pct_indigenous_flow",
    }

    for agg_col in overwrite_map:
        if agg_col in agg.columns:
            agg[agg_col] = pd.to_numeric(agg[agg_col], errors="coerce").astype(float)

    for yr in all_years:
        if yr in flow.index:
            for agg_col, flow_col in overwrite_map.items():
                if flow_col in flow.columns and agg_col in agg.columns:
                    val = flow.loc[yr, flow_col]
                    if pd.notna(val):
                        agg.loc[yr, agg_col] = val

            if "age_pct_under16" in flow.columns and "age_pct_16_24" in flow.columns:
                under16 = pd.to_numeric(flow.loc[[yr], "age_pct_under16"], errors="coerce").iloc[0]
                age16_24 = pd.to_numeric(flow.loc[[yr], "age_pct_16_24"], errors="coerce").iloc[0]
                pct_youth = float(under16 + age16_24)
                agg.loc[yr, "pct_youth"] = float(np.clip(pct_youth, 0.01, 0.99))

    pre_flow_years = [yr for yr in all_years if yr < flow_first_year]
    if pre_flow_years:
        slope_window = sorted([yr for yr in flow.index if yr <= flow_first_year + 2])
        for agg_col, flow_col in overwrite_map.items():
            if flow_col not in flow.columns or agg_col not in agg.columns:
                continue

            flow_vals = flow.loc[slope_window, flow_col].dropna()
            if len(flow_vals) < 2:
                continue

            xs = np.array(flow_vals.index, dtype=float)
            ys = np.asarray(flow_vals.values, dtype=float)
            slope = np.polyfit(xs, ys, 1)[0]
            anchor_val = float(pd.to_numeric(flow.loc[[flow_first_year], flow_col], errors="coerce").iloc[0])

            for yr in pre_flow_years:
                extrapolated = anchor_val + slope * (yr - flow_first_year)
                if agg_col.startswith("pct_"):
                    extrapolated = float(np.clip(extrapolated, 0.01, 0.99))
                elif agg_col == "age_avg":
                    extrapolated = float(np.clip(extrapolated, 15, 85))
                agg.loc[yr, agg_col] = extrapolated

        if "age_pct_under16" in flow.columns and "age_pct_16_24" in flow.columns:
            anchor_under16 = pd.to_numeric(flow.loc[[flow_first_year], "age_pct_under16"], errors="coerce").iloc[0]
            anchor_16_24 = pd.to_numeric(flow.loc[[flow_first_year], "age_pct_16_24"], errors="coerce").iloc[0]
            anchor_youth = float(anchor_under16 + anchor_16_24)
            for yr in pre_flow_years:
                agg.loc[yr, "pct_youth"] = float(np.clip(anchor_youth, 0.01, 0.99))

    agg["total_surveyed"] = agg["total_surveyed"].round().astype(int)
    return agg


def enrich_region_year(ry: pd.DataFrame, flow_yearly: pd.DataFrame) -> pd.DataFrame:
    flow = flow_yearly[["actively_homeless", "newly_identified"]].copy()
    flow["flow_growth_rate"] = flow["actively_homeless"].pct_change().fillna(0)

    ry = ry.merge(
        flow.rename(columns={"actively_homeless": "flow_active_homeless"}),
        on="year",
        how="left",
    )

    mean_growth = ry["flow_growth_rate"].dropna().mean()
    ry["flow_growth_rate"] = ry["flow_growth_rate"].fillna(mean_growth)
    ry["newly_identified"] = ry["newly_identified"].fillna(ry["newly_identified"].median())
    ry["flow_active_homeless"] = ry["flow_active_homeless"].ffill().bfill().fillna(ry["flow_active_homeless"].median())
    return ry
