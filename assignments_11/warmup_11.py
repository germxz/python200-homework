from prefect import task, get_run_logger

# Prefect Orchestration

# Prefect Q1

# A @task is a single unit of work in a pipeline, usually something that touches the outside world like a data
# transformation, an API call, or a database write. Prefect tracks each task run on its own, so it gets its own
# state, its own logs, and its own retries. A @flow is the container that calls those tasks and sets the order
# they run in, and it is what shows up as one pipeline run in the UI. I would not use @task on my
# celsius_to_fahrenheit function because it is not bound to fail. All it takes is known numbers and some math,
# which will never fail unless the function itself has a bug. It has no external forces that will affect it, so
# retries would never do anything and it would only clutter the UI with a task run that tells me nothing.


# Prefect Q2

# A bare decorator with no function under it is a SyntaxError, so it is
# commented out here. This is the line the question asked for:

# @task(name="call_api", retries=3, retry_delay_seconds=30)


# Prefect Q3

# If the pipeline fails at "transform", the Prefect UI dashboard shows the run marked Failed. I click into that
# run, where I can see transform is the task that broke, and that load_enriched never ran because of it.
#
# Clicking the failed task gives me several tabs. Details has the state message, run ID, and timing, but the
# Logs tab is where the real answer is: the exception (e.g. KeyError: 'temperature_2m_max') and the traceback
# with the file and line number in my own code.


# Production Patterns


# Production Q1

# raise_for_status() will stop further execution if something goes wrong in a task. It checks the status code
# and raises an HTTPError on any 4xx or 5xx response, so instead of running on garbled data the pipeline raises
# and stops when something breaks. Printing an error with response.status_code does not raise anything, so the
# task keeps running and passes the corrupted data downstream.
#
# On a 500: with raise_for_status() the task is marked Failed and the downstream tasks never run (NotReady).
# With the print, the task is marked Completed and the downstream tasks run anyway on bad data.


# Production Q2

# upsert with on_conflict="date" is what makes the re-run safe. load_raw already loaded those dates before
# transform crashed, so on the second run the database updates the existing rows instead of treating them as
# new. That makes load_raw idempotent, meaning it is safe to run as many times as I need.
#
# With plain insert, the second run would hit a unique constraint violation on the first date that is already
# in the table, because date is the primary key. Postgres rejects it and raises a duplicate key error, so
# load_raw fails before writing anything and the pipeline never reaches the transform fix I was trying to test.
# I would have to delete the rows from the first partial run by hand before every re-run, which is exactly the
# manual cleanup upsert exists to avoid.


# Production Q3

@task
def load_enriched(enrichment_records):
    logger = get_run_logger()
    logger.info(f"upserted {len(enrichment_records)} enrichment records")


# Production Q4

# The incremental check looks at what has already been enriched and filters down to just the new records before
# the ML and LLM steps run. That is what makes the task idempotent: re-running it does not redo work that is
# already done, so the output is the same whether I run it once or five times.
#
# If I removed it and ran on all 365 records every time, the first cost is money. Every record is one API call
# to the model, so a re-run after a crash pays for all 365 again instead of the handful that are actually
# missing, and on a daily schedule that is 365 calls a day to regenerate sentences I already have.
#
# The second cost is time. The LLM loop is sequential and waits on each call, so a full pass takes minutes
# instead of seconds, which risks hitting the API rate limit or timing out the task.
#
# The third is data correctness, and that is the one that actually worries me. LLM output is not deterministic,
# so re-processing a record I already enriched gives a different sentence, and the upsert overwrites the good
# historical value with the new one. My source data never changed but my enriched table quietly did.