# Video Link: https://www.youtube.com/watch?v=7PohQZBEQKY

import os
from datetime import date

import requests
from dotenv import load_dotenv
from supabase import create_client

LATITUDE = 35.4676
LONGITUDE = -97.5164


def get_client():
    load_dotenv()
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url:
        raise ValueError("missing Supabase URL. Please set in the .env file.")
    if not key:
        raise ValueError("missing Supabase key. Please set in the .env file.")
    return create_client(url, key)


# Step 1: Extract

def extract_weather():
    response = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max",
            "timezone": "auto",
        },
    )
    response.raise_for_status()
    data = response.json()

    print("Status:", response.status_code)
    print("Location:", data["latitude"], data["longitude"])
    print("Days returned:", len(data["daily"]["time"]))
    return data


# Step 2: Transform

def transform(data):
    daily = data["daily"]
    records = []

    for i in range(len(daily["time"])):
        records.append(
            {
                "date": daily["time"][i],
                "temperature_2m_max": daily["temperature_2m_max"][i],
                "temperature_2m_min": daily["temperature_2m_min"][i],
                "precipitation_sum": daily["precipitation_sum"][i],
                "wind_speed_10m_max": daily["wind_speed_10m_max"][i],
            }
        )

    print("First record:", records[0])
    print("Last record:", records[-1])

    # I expected 365 records for a full year because 2023 was not a leap year, and I got 365.
    # If the numbers differed it could mean the archive is missing days for this location or the
    # date range got clipped. A day can also come back with a null value even when the row is there.

    return records


# Step 3: Load

def load(supabase, records):
    before = supabase.table("weather_raw").select("date", count="exact").execute()
    print("Row count before load:", before.count)

    response = (
        supabase.table("weather_raw")
        .upsert(records, on_conflict="date")
        .execute()
    )
    print("Upserted", len(response.data), "rows into weather_raw")

    after = supabase.table("weather_raw").select("date", count="exact").execute()
    print("Row count after load:", after.count)

    # Running the script a second time does not change the row count. That is idempotency. Because
    # date is the primary key and I am using upsert, a repeat run updates the rows that are already
    # there instead of adding duplicates, so the pipeline is safe to run as many times as I want.


# Step 4: Verify

def verify(supabase):
    total = supabase.table("weather_raw").select("date", count="exact").execute()
    print("Total rows:", total.count)

    earliest = supabase.table("weather_raw").select("date").order("date").limit(1).execute()
    latest = supabase.table("weather_raw").select("date").order("date", desc=True).limit(1).execute()
    print("Earliest date:", earliest.data[0]["date"])
    print("Latest date:", latest.data[0]["date"])

    target = "2023-07-04"
    exact = supabase.table("weather_raw").select("*").eq("date", target).execute()

    if exact.data:
        print(target, ":", exact.data[0])
    else:
        # Find the closest row before and after the target, then keep the nearer one.
        before = (
            supabase.table("weather_raw")
            .select("*")
            .lte("date", target)
            .order("date", desc=True)
            .limit(1)
            .execute()
        )
        after = (
            supabase.table("weather_raw")
            .select("*")
            .gte("date", target)
            .order("date")
            .limit(1)
            .execute()
        )

        target_date = date.fromisoformat(target)
        candidates = before.data + after.data
        nearest = min(
            candidates,
            key=lambda row: abs((date.fromisoformat(row["date"]) - target_date).days),
        )
        print(target, "not found. Nearest date is", nearest["date"], ":", nearest)


supabase = get_client()
data = extract_weather()
records = transform(data)
load(supabase, records)
verify(supabase)