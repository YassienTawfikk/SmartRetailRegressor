# Data Handling
import pandas as pd
import numpy as np

# Paths
from src.__00__paths import curated_data_dir, processed_data_dir, raw_data_dir
import shutil
from pathlib import Path

# Datasets Source
import kagglehub


def download_dataset():
    data_items = [
        raw_data_dir / "features.csv",
        raw_data_dir / "stores.csv",
        raw_data_dir / "test.csv",
        raw_data_dir / "train.csv"
    ]

    if all(item.exists() for item in data_items):
        print("✔️ Data already downloaded")
    else:
        dataset_path = Path(kagglehub.dataset_download("aslanahmedov/walmart-sales-forecast"))

        if not dataset_path.exists():
            raise FileNotFoundError("Dataset not found")

        for item in dataset_path.iterdir():
            target = raw_data_dir / item.name
            shutil.copy2(item, target)

        print("✔️ Data downloaded successfully")


def load_dataset(file_name):
    return pd.read_csv(file_name)


def merge_datasets(df1, df2, on):
    return df1.merge(df2, on=on, how="left")


def add_date_features(df, date_col="Date", clip_weeks=8):
    out = df.copy()
    d = pd.to_datetime(out[date_col]).dt.tz_localize(None).dt.normalize()

    # Core calendar
    out["time_index"] = ((d - d.min()).dt.days // 7).astype(int)

    woy = d.dt.isocalendar().week.astype(int)
    out["woy_sin"] = np.sin(2 * np.pi * woy / 52.1775)
    out["woy_cos"] = np.cos(2 * np.pi * woy / 52.1775)

    mon = d.dt.month
    out["mon_sin"] = np.sin(2 * np.pi * mon / 12)
    out["mon_cos"] = np.cos(2 * np.pi * mon / 12)

    out["is_month_start"] = d.dt.is_month_start.astype(int)
    out["is_month_end"] = d.dt.is_month_end.astype(int)
    out["is_quarter_end"] = d.dt.is_quarter_end.astype(int)
    out["is_year_end"] = d.dt.is_year_end.astype(int)

    out["week_of_month"] = (((d.dt.day - 1) // 7) + 1).clip(1, 5).astype(int)

    # US retail events & distances (weeks)
    from datetime import date, timedelta
    from calendar import monthrange

    def nth_weekday_of_month(y, m, weekday, n):
        first = date(y, m, 1)
        shift = (weekday - first.weekday()) % 7
        return first + timedelta(days=shift + 7 * (n - 1))

    def last_weekday_of_month(y, m, weekday):
        last_day = monthrange(y, m)[1]
        last = date(y, m, last_day)
        shift = (last.weekday() - weekday) % 7
        return last - timedelta(days=shift)

    def easter_sunday(y):
        a = y % 19
        b = y // 100;
        c = y % 100
        d0 = b // 4;
        e0 = b % 4
        f = (b + 8) // 25
        g = (b - f + 1) // 3
        h = (19 * a + b - d0 - g + 15) % 30
        i = c // 4;
        k = c % 4
        l = (32 + 2 * e0 + 2 * i - h - k) % 7
        m0 = (a + 11 * h + 22 * l) // 451
        month = (h + l - 7 * m0 + 114) // 31
        day = ((h + l - 7 * m0 + 114) % 31) + 1
        return date(y, month, day)

    def event_dates(y):
        tg = nth_weekday_of_month(y, 11, 3, 4)
        bf = tg + timedelta(days=1)
        xp = date(y, 12, 23)
        es = easter_sunday(y)
        md = last_weekday_of_month(y, 5, 0)
        j4 = date(y, 7, 4)
        ld = nth_weekday_of_month(y, 9, 0, 1)
        sb = nth_weekday_of_month(y, 2, 6, 1)
        return {"thanksgiving": tg, "black_friday": bf, "xmas_peak": xp,
                "easter": es, "memorial_day": md, "july4": j4,
                "labor_day": ld, "super_bowl": sb}

    years = d.dt.year.unique()
    year_to_events = {int(y): event_dates(int(y)) for y in years}

    def clipped_weeks(series, m):
        s = series.astype(int)
        return s.where(s.abs() <= m, 0)

    events = ["thanksgiving", "black_friday", "xmas_peak", "easter",
              "memorial_day", "july4", "labor_day", "super_bowl"]

    for ev in events:
        ev_dates = d.dt.year.map(lambda y: year_to_events[int(y)][ev])
        dist_w = (d.dt.date - ev_dates).map(lambda td: td.days // 7)
        out[f"dist_{ev}_wk"] = clipped_weeks(pd.Series(dist_w, index=out.index), clip_weeks).astype(int)

    out["is_black_friday_wk"] = (out["dist_black_friday_wk"] == 0).astype(int)
    out["is_thanksgiving_wk"] = (out["dist_thanksgiving_wk"] == 0).astype(int)
    out["is_xmas_peak_wk"] = (out["dist_xmas_peak_wk"] == 0).astype(int)

    y = d.dt.year.astype(str)
    bts_start = pd.to_datetime(y + "-07-15")
    bts_end = pd.to_datetime(y + "-09-10")
    out["is_back_to_school"] = ((d >= bts_start) & (d <= bts_end)).astype(int)

    return out


def reorder_data_frame(df):
    # Step 1: Sort rows by Date (chronologically)
    df = df.sort_values("Date").reset_index(drop=True)

    # Step 2: Reorder the columns as desired
    desired_order = [
        "Date",
        "Store", "Type_A", "Type_B", "Type_C", "Size",  # Store-related
        "Dept", "IsHoliday",  # Base data
        "time_index",  # Trend
        "woy_sin", "woy_cos",  # Week of year (cyclic)
        "mon_sin", "mon_cos",  # Month (cyclic)
        "week_of_month",  # In-month position
        "is_month_start", "is_month_end", "is_quarter_end", "is_year_end",  # Calendar flags
        "dist_thanksgiving_wk", "dist_black_friday_wk", "dist_xmas_peak_wk",
        "dist_easter_wk", "dist_memorial_day_wk", "dist_july4_wk", "dist_labor_day_wk", "dist_super_bowl_wk",
        # Event distances
        "is_black_friday_wk", "is_thanksgiving_wk", "is_xmas_peak_wk", "is_back_to_school",  # Event flags
        "Weekly_Sales"  # Label
    ]

    # Step 3: Reorder columns
    return df[desired_order]


def split_data(df, onDate="2012-04-13"):
    # Step 1: Safely parse mixed-format or inconsistent dates
    df["Date"] = pd.to_datetime(
        df["Date"],
        dayfirst=True,
        format='mixed',
        errors='coerce'  # turn invalid dates into NaT (not crash)
    )

    # Step 2: Drop rows with unparseable (missing) dates
    df = df.dropna(subset=["Date"])

    # Step 3: Sort by Date (chronological order)
    df = df.sort_values("Date").reset_index(drop=True)

    # Step 4: Define split date for 80/20 time-aware split
    split_date = pd.to_datetime(onDate)

    # Step 5: Split into train and validation
    train_df = df[df["Date"] <= split_date]
    valid_df = df[df["Date"] > split_date]

    cols_to_drop = ["Date", "Store"]

    train_df = train_df.drop(columns=cols_to_drop)
    valid_df = valid_df.drop(columns=cols_to_drop)

    print(f"📊 Train Percentage: {len(train_df) / len(df):.2%}")
    return train_df, valid_df


def save_data(df, file_name):
    df.to_csv(file_name, index=False)
    print(f"✔️ Saved {'/'.join(file_name.parts[-3:])}")
