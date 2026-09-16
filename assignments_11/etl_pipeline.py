# Video Link: PASTE YOUR VIDEO LINK HERE

import json
import os

import joblib
import pandas as pd
import requests
from dotenv import load_dotenv
from openai import OpenAI
from prefect import flow, get_run_logger, task
from supabase import create_client

load_dotenv()

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

with open("models/weather_classifier_metadata.json") as f:
    metadata = json.load(f)
    FEATURES = metadata["feature_names"]

LATITUDE = 47.6062
LONGITUDE = -122.3321

SYSTEM_PROMPT = (
    "You write running recommendations for a daily weather app. You will receive one day's "
    "weather conditions and a machine learning model's prediction about whether that day is "
    "good for running. "
    "Always agree with the model's prediction - your job is to explain it, not to second-guess it. "
    "Write exactly one sentence stating whether the day is good for running and why. "
    "Be direct, practical, and specific to the conditions. "
    "Do not use bullet points, headers, or phrases like 'Based on the data'."
)

# extract

@task(name="extract_weather", retries=2, retry_delay_seconds=10)
def extract() -> list:
    """Fetch 2023 daily weather for Seattle from the Open-Meteo archive."""
    logger = get_run_logger()

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": "2023-01-01",
        "end_date": "2023-12-31",
        "daily": FEATURES,
        "timezone": "auto",
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    payload = response.json()
    if "daily" not in payload:
        raise ValueError(f"Unexpected API response, no 'daily' block: {list(payload)}")

    daily = payload["daily"]
    records = [
        {
            "date": daily["time"][i],
            "temperature_2m_max": daily["temperature_2m_max"][i],
            "temperature_2m_min": daily["temperature_2m_min"][i],
            "precipitation_sum": daily["precipitation_sum"][i],
            "wind_speed_10m_max": daily["wind_speed_10m_max"][i],
        }
        for i in range(len(daily["time"]))
    ]

    logger.info(f"Open-Meteo returned {len(records)} days, {records[0]['date']} to {records[-1]['date']}")
    print(f"Extracted {len(records)} daily records from Open-Meteo")
    return records


@task(name="load_raw", retries=2, retry_delay_seconds=5)
def load_raw(records: list) -> None:
    """Upsert raw weather rows into weather_raw."""
    logger = get_run_logger()

    response = (
        supabase.table("weather_raw")
        .upsert(records, on_conflict="date")
        .execute()
    )
    written = len(response.data)
    if written != len(records):
        logger.warning(f"Partial write: sent {len(records)} records, database returned {written}")

    logger.info(f"weather_raw upsert: {written} of {len(records)} records written")
    print(f"Upserted {written} rows into weather_raw")


@task(name="transform")
def transform(raw_records: list) -> list:
    """Classify unprocessed days with the saved model, then write one LLM sentence each."""
    logger = get_run_logger()

    # --- Incremental check ---
    # Fetch the dates already in weather_enriched and skip them.
    enriched = supabase.table("weather_enriched").select("date").execute().data
    already_done = {row["date"] for row in enriched}
    to_process = [row for row in raw_records if row["date"] not in already_done]

    logger.info(f"Incremental check: {len(to_process)} to process, {len(already_done)} skipped")
    print(f"Records to transform: {len(to_process)} (skipping {len(already_done)} already enriched)")

    if not to_process:
        print("All records already enriched - nothing to do.")
        return []

    # --- ML classify ---
    missing = [f for f in FEATURES if f not in to_process[0]]
    if missing:
        raise ValueError(f"Records are missing features the model expects: {missing}")

    clf = joblib.load("models/weather_classifier.pkl")
    X = pd.DataFrame(to_process)[FEATURES]

    predictions = clf.predict(X)
    probabilities = clf.predict_proba(X)

    enrichment_records = [
        {
            "date": to_process[i]["date"],
            "good_for_running": bool(predictions[i]),
            "confidence": round(float(probabilities[i].max()), 4),
        }
        for i in range(len(to_process))
    ]

    good_days = sum(1 for r in enrichment_records if r["good_for_running"])
    print(f"ML classification complete. Good days: {good_days} / {len(enrichment_records)}")

    # --- LLM enrich ---
    fallbacks = 0
    for i, record in enumerate(enrichment_records):
        raw_row = to_process[i]
        prediction_text = "good for running" if record["good_for_running"] else "not ideal for running"
        user_message = (
            f"Date: {raw_row['date']}\n"
            f"High: {raw_row['temperature_2m_max']}°C, Low: {raw_row['temperature_2m_min']}°C\n"
            f"Precipitation: {raw_row['precipitation_sum']} mm\n"
            f"Max wind speed: {raw_row['wind_speed_10m_max']} km/h\n"
            f"Model prediction: {prediction_text} (confidence: {record['confidence']:.0%})"
        )
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=100,
            )
            summary = (response.choices[0].message.content or "").strip()
            if not summary:
                raise ValueError("empty completion")
            record["llm_summary"] = summary
        except Exception as e:
            logger.error(f"LLM failed on {record['date']}: {type(e).__name__}: {e}")
            print(f"  LLM error on {record['date']}: {type(e).__name__}: {e}")
            record["llm_summary"] = "Recommendation unavailable."
            fallbacks += 1

        if (i + 1) % 50 == 0:
            print(f"  LLM enriched {i + 1} / {len(enrichment_records)} records")

    logger.info(f"LLM enrichment done: {len(enrichment_records) - fallbacks} written, {fallbacks} fallbacks")
    print(f"Transform complete: {len(enrichment_records)} records enriched ({fallbacks} fallbacks)")
    return enrichment_records


# load_enriched 
@task(name="load_enriched", retries=2, retry_delay_seconds=5)
def load_enriched(enrichment_records: list) -> None:
    """Upsert enriched rows into weather_enriched."""
    logger = get_run_logger()

    if not enrichment_records:
        print("No new enrichment records to load.")
        return

    response = (
        supabase.table("weather_enriched")
        .upsert(enrichment_records, on_conflict="date")
        .execute()
    )
    written = len(response.data)
    if written != len(enrichment_records):
        logger.warning(f"Partial write: sent {len(enrichment_records)} records, database returned {written}")

    logger.info(f"weather_enriched upsert: {written} of {len(enrichment_records)} records written")
    print(f"Upserted {written} rows into weather_enriched")

# flow
@flow(name="weather-etl", log_prints=True, description="Extract 2023 Seattle weather, classify it, enrich it with an LLM, load to Supabase.")
def etl_pipeline():
    raw_records = extract()
    load_raw(raw_records)
    enrichment_records = transform(raw_records)
    load_enriched(enrichment_records)
    print("Pipeline complete.")


if __name__ == "__main__":
    etl_pipeline()