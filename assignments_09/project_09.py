# Video link: PASTE YOUR LINK HERE

import os

import requests
from dotenv import load_dotenv
from supabase import create_client, Client


def get_client() -> Client:
    """Load Supabase credentials from .env and return a connected client."""
    load_dotenv()

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    if not url:
        raise ValueError("SUPABASE_URL is missing. Add it to your .env file.")
    if not key:
        raise ValueError("SUPABASE_KEY is missing. Add it to your .env file.")

    return create_client(url, key)


# Step 1: Extract

CITY = "Oklahoma City"
LATITUDE = 35.4676
LONGITUDE = -97.5164

def extract_weather():
    """Fetch 2023 daily weather from the Open-Meteo historical archive API."""
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

    print("--- Extract ---")
    print("Status:", response.status_code)
    print("City:", CITY)
    print("Location:", data["latitude"], data["longitude"])
    print("Units:", data["daily_units"])
    print("Days returned:", len(data["daily"]["time"]))
    return data


# Step 2: Transform

def transform(data):
    """Turn the columnar API response into a list of row dictionaries."""
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

    print("\n--- Transform ---")
    print("First record:", records[0])

    # I expected 365 records because 2023 was not a leap year, and I got 365.
    # If the count came back lower it could mean the archive is missing days for
    # this location, or the date range got clipped somewhere. A day can also come
    # back with a null value for one variable even when the row itself is there,
    # so the count can look right while individual values are missing.

    return records


# Step 3: Load

def load(supabase, records):
    """Upsert all records into weather_raw."""
    response = (
        supabase.table("weather_raw")
        .upsert(records, on_conflict="date")
        .execute()
    )
    print("\n--- Load ---")
    print(f"Upserted {len(response.data)} rows into weather_raw")

    # Running the script a second time upserts the same 365 dates and the row count
    # in weather_raw does not change. That is idempotency. Because date is the primary
    # key and I am using upsert instead of insert, a repeat run updates the rows that
    # are already there rather than adding duplicates or erroring out. That makes the
    # pipeline safe to re-run after a failure or to backfill a range I already loaded.

    return response.data


# Step 4: Verify

def verify(supabase):
    """Check what actually landed in the table."""
    print("\n--- Verify ---")

    # 1. Total rows (weather_raw)
    total = supabase.table("weather_raw").select("date", count="exact").execute()
    print("Total rows:", total.count) 

    # 2. Earliest and latest dates, by sorting and taking one row from each end.
    earliest = supabase.table("weather_raw").select("date").order("date").limit(1).execute()
    latest = supabase.table("weather_raw").select("date").order("date", desc=True).limit(1).execute()
    print("Earliest date:", earliest.data[0]["date"])
    print("Latest date:", latest.data[0]["date"])

    # 3. One specific day.
    july4 = supabase.table("weather_raw").select("*").eq("date", "2023-07-04").execute()
    if july4.data:
        print("2023-07-04:", july4.data[0])
    else:
        print("2023-07-04 not found in the table.")


if __name__ == "__main__":
    supabase = get_client()
    data = extract_weather()
    records = transform(data)
    load(supabase, records)
    verify(supabase)