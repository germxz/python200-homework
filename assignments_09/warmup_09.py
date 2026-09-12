import os
from datetime import date

from dotenv import load_dotenv
from supabase import create_client


# --- Supabase Connection ---


# Connection Q1

# Supabase-py needs my Supabase URL and key to connect to my project. You can find the URL in the supabase dashboard overview page right under the project name.
# The key can be found within the settings page under the API keys section. This key should never be hardcoded into the code because it
# can be exploited by anyone who comes across the key.

# Connection Q2

def get_client():           # loads supabase URL and key from .env file and creates a supabase client

    load_dotenv()
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url:
        raise ValueError("missing Supabase URL. Please set in the .env file.")
    if not key:
        raise ValueError("missing Supabase key. Please set in the .env file.")
    return create_client(url, key)


# Connection Q3

# Row level security (RLS) allows an admin to control access of who can read and write to specific rows in the database.
# It is disabled for this project because I will be accessing the database and making changes. If I were to enable RLS, I
# would not be able to edit this table and add rows, because with no policies written every insert and query gets rejected.
# For this project I will keep RLS disabled so my script has full access to the table. An application that would be good to
# keep RLS on would be an application that has several users with different tiers of access to the database. For example, an
# application with free and paid users where they get different access to the database based on their subscription level.

# --- Supabase-py CRUD ---

# CRUD Q1

day = date.today().isoformat()

supabase = get_client()


def insert_test_record(supabase):
    record = {
        "date": day,
        "temperature_2m_max": 31.5,
        "temperature_2m_min": 19.2,
        "precipitation_sum": 0.0,
        "wind_speed_10m_max": 18.4,
    }
    response = supabase.table("weather_raw").insert(record).execute()
    return response


insert_test_record(supabase)


# If I run this function twice, I will get an error message telling me that the
# date already exists, meaning that I can only make one entry per day because it's the primary key.
# To ensure that I can run this function multiple times, I can change supabase.table("weather_raw").insert(record).execute()
# to use upsert instead of insert. Upsert will insert the record if it doesn't exist, and update it if it does exist.

# CRUD Q2

def get_records_by_date_range(supabase, start, end):
    response = (
        supabase.table("weather_raw")
        .select("*")
        .gte("date", start)
        .lte("date", end)
        .execute()
    )
    return response.data


# The range uses the same date the Q1 record was inserted with, so it always contains that row.
rows = get_records_by_date_range(supabase, day, day)
print(rows)


# CRUD Q3


# Starting with the .insert method, it will try to add a new row to the table if the primary key
# doesn't exist already. In this case, the primary key is the date column. If the key is already
# present, the function will throw an error. The .upsert method will add the row if the primary key
# is new and update the existing row if that key already has values, so it does not error on a repeat.
# I would use insert if I wanted to add a new soccer player to a team's roster. The primary key will
# be unique for the new player and there won't be a case where he is logged in two instances at a
# time. I would use .upsert to load a player's stats after each match, since I don't know ahead of
# time whether that match is already recorded, and upsert handles both cases in one call.


def safe_upsert(supabase, records):
    response = (
        supabase.table("weather_raw")
        .upsert(records, on_conflict="date")
        .execute()
    )
    print(len(response.data), "rows affected by upsert")
    return response.data


records = [
    {"date": "2026-09-05", "temperature_2m_max": 29.0, "temperature_2m_min": 17.5,
     "precipitation_sum": 0.0, "wind_speed_10m_max": 14.0},
    {"date": "2026-09-06", "temperature_2m_max": 30.2, "temperature_2m_min": 18.1,
     "precipitation_sum": 2.5, "wind_speed_10m_max": 16.3},
]

safe_upsert(supabase, records)


# Idempotency


# Idempotency Q1

# Idempotency is important in web applications because it ensures that a request can be made
# multiple times without changing the result beyond the initial application. This is important for
# ensuring that a request can be retried if it fails, without causing unintended side effects. For example, if a user
# submits a form to create a new account, and the request fails due to a network error, the user can retry the
# request without creating multiple accounts. In this case, the request is idempotent because
# it will only create one account regardless of how many times it is submitted.