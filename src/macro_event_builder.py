"""

Builds a clean US macro-event calendar for EquityBERT.

Output columns:
    datetime   — timezone-aware timestamp (America/New_York)
    event_type — one of CPI, PPI, NFP, FOMC

Two data sources:
    1. FRED vintage dates (fredapi)  → actual first-release dates for
       CPI (CPIAUCSL), PPI (PPIACO), NFP (PAYEMS)
    2. Federal Reserve FOMC calendar → hardcoded statement release dates
       (2019-01-01 to 2026-03-31)

No impact column — my dataset.py already handles use_event_impact=False
by forcing the token to 0.

    Get a free API key from https://fred.stlouisfed.org/docs/api/api_key.html
"""
import pandas as pd
from fredapi import Fred
import os
from dotenv import load_dotenv
load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY")
START = "2019-01-01"
END   = "2026-03-31"

# Standard release times (Eastern):
#   BLS releases (CPI, PPI, NFP) → 08:30 ET
#   FOMC statement               → 14:00 ET
BLS_RELEASE_HOUR  = 8
BLS_RELEASE_MIN   = 30
FOMC_RELEASE_HOUR = 14
FOMC_RELEASE_MIN  = 0


# Get actual release dates from FRED vintage data 
def get_fred_release_dates(series_id: str, event_type: str,
                           fred: Fred, start: str, end: str) -> pd.DataFrame:
    """
    Uses FRED's vintage / all-releases data to extract the ACTUAL
    first-publication date for each observation of a series.

    fredapi.get_series_all_releases() returns a DataFrame with columns:
        date           — the observation period (e.g. 2023-01-01 for Jan CPI)
        realtime_start — the date the value was first published (= release day)
        value          — the data value

    We group by observation 'date' and take the earliest 'realtime_start'
    to get the initial release date (ignoring revisions).
    """
    df = fred.get_series_all_releases(series_id)
    df = df.reset_index()  # 'date' becomes a column

    # Keep only first release per observation period
    first_releases = (
        df.sort_values("realtime_start")
        .groupby("date", as_index=False)
        .first()
    )

    # Filter to our date range (release date within range)
    first_releases = first_releases[
        (first_releases["realtime_start"] >= pd.Timestamp(start)) &
        (first_releases["realtime_start"] <= pd.Timestamp(end))
    ].copy()

    first_releases["event_type"] = event_type
    first_releases = first_releases.rename(
        columns={"realtime_start": "release_date"}
    )

    return first_releases[["release_date", "event_type"]]


# ── 2. FOMC statement dates (hardcoded from federalreserve.gov) ────────
# Source: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
# These are the dates the FOMC statement was released (2nd day of meeting).
# Emergency / unscheduled meetings in 2020 are included.
FOMC_STATEMENT_DATES = [
    # 2019
    "2019-01-30", "2019-03-20", "2019-05-01", "2019-06-19",
    "2019-07-31", "2019-09-18", "2019-10-30", "2019-12-11",
    # 2020 (includes emergency cuts on Mar 3 and Mar 15)
    "2020-01-29", "2020-03-03", "2020-03-15", "2020-04-29",
    "2020-06-10", "2020-07-29", "2020-09-16", "2020-11-05",
    "2020-12-16",
    # 2021
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16",
    "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
    # 2022
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15",
    "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
    # 2023
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14",
    "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    # 2024
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    # 2026 (through March)
    "2026-01-28", "2026-03-18",
]


# Build the combined calendar 
def main():
    fred = Fred(api_key=FRED_API_KEY)

    print("Fetching CPI release dates from FRED...")
    cpi_dates = get_fred_release_dates("CPIAUCSL", "CPI", fred, START, END)
    print(f"  → {len(cpi_dates)} CPI releases")

    print("Fetching PPI release dates from FRED...")
    ppi_dates = get_fred_release_dates("PPIACO", "PPI", fred, START, END)
    print(f"  → {len(ppi_dates)} PPI releases")

    print("Fetching NFP release dates from FRED...")
    nfp_dates = get_fred_release_dates("PAYEMS", "NFP", fred, START, END)
    print(f"  → {len(nfp_dates)} NFP releases")

    # Combine BLS events and add 08:30 ET release time
    bls_events = pd.concat([cpi_dates, ppi_dates, nfp_dates], ignore_index=True)
    bls_events["datetime"] = (
        pd.to_datetime(bls_events["release_date"])
        + pd.Timedelta(hours=BLS_RELEASE_HOUR, minutes=BLS_RELEASE_MIN)
    )
    bls_events["datetime"] = (
        bls_events["datetime"]
        .dt.tz_localize("America/New_York")
    )

    # FOMC events at 14:00 ET
    fomc_events = pd.DataFrame({
        "release_date": pd.to_datetime(FOMC_STATEMENT_DATES),
        "event_type": "FOMC",
    })
    # Filter to date range
    fomc_events = fomc_events[
        (fomc_events["release_date"] >= pd.Timestamp(START)) &
        (fomc_events["release_date"] <= pd.Timestamp(END))
    ].copy()

    fomc_events["datetime"] = (
        pd.to_datetime(fomc_events["release_date"])
        + pd.Timedelta(hours=FOMC_RELEASE_HOUR, minutes=FOMC_RELEASE_MIN)
    )
    fomc_events["datetime"] = (
        fomc_events["datetime"]
        .dt.tz_localize("America/New_York")
    )

    print(f"  → {len(fomc_events)} FOMC statements")

    #  Combine all events 
    all_events = pd.concat(
        [bls_events, fomc_events], ignore_index=True
    )
    all_events = (
        all_events[["datetime", "event_type"]]
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    all_events.to_csv("NEW_macro_events_us.csv", index=False)
    all_events.to_parquet("NEW_macro_events_us.parquet", index=False)

    print(f"\nTotal events: {len(all_events)}")
    print(f"\nEvent type counts:")
    print(all_events["event_type"].value_counts().to_string())
    print(f"\nDate range: {all_events['datetime'].min()} → {all_events['datetime'].max()}")
    print(f"\nFirst 10 events:")
    print(all_events.head(10).to_string())
    print(f"\nLast 10 events:")
    print(all_events.tail(10).to_string())
    print("\nSaved: NEW_macro_events_us.csv, NEW_macro_events_us.parquet")


if __name__ == "__main__":
    main()