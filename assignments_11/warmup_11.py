from prefect import task, get_run_logger

# Prefect Orchestration

# Prefect Q1

# A @task in prefect is used for classifying tools like MLdata transformation or calling API's. A @task will execute individually within a @flow without getting blocked. 
# On the other hand, a @flow will wait for the pipeline witihin it to execute before allowing the next @flow to run. @flow is used for stuctural integrity as it classifies
# tasks within a certain department. I would not use a @task or @flow for my farenheight_to_celcius function because it is not bound to fail. All it takes is known numbers 
# and run some math which will never fail nunless the function itself has a bug. It has no external forces that will affect the math.



# Prefect Q2

# A bare decorator with no function under it is a SyntaxError, so it is
# commented out here. This is the line the question asked for:

# @task(name="call_api", retries=3, retry_delay_seconds=30)





# Prefect Q3

# If the pipeline fails at "transform", the Prefect UI dashboard shows the run
# marked Failed. I click into that run, where I can see transform is the task
# that broke (and that load_enriched never ran because of it).

# Clicking the failed task gives me several tabs. Details has the state message,
# run ID, and timing, but the Logs tab is where the real answer is: the
# exception (e.g. KeyError: 'series is not deined') and the traceback with the file and
# line number in my own code.


# Production Patterns


# Production Q1

# Raise for status will stop further execution if something goes wrong in a task. Instead of still running the garbled data, the pipeline 
# will raise the error and stop when something breaks. On the other hand, printing an error with response status code will not raise an error
# and stil keep the pipeline running and using the corrupted data. On a 500: with raise_for_status() the task is marked Failed and the downstream tasks never run (NotReady);
# with the print, the task is marked Completed and the downstream tasks run anyway on bad data.


# Production Q2

# upsert with on_conflict="date" keeps the re-run from duplicating rows. load_raw already loaded those dates before transform crashed, 
# so on the second run the database updates the existing rows instead of adding copies. That makes load_raw idempotent, meaning it's 
# safe to run as many times as I need. With insert, every re-run would append the same dates again, so transform would run on duplicated 
# data and produce inflated counts and skewed averages. I'd have to delete the duplicates by hand before each re-run.

# Production Q3

@task
def load_enriched(enrichment_records):
    logger = get_run_logger()
    logger.info(f"upserted {len(enrichment_records)} enrichment records")

# Production Q4


# The incremental check looks at what's already been enriched and filters down to just the new records before the
# ML and LLM steps run. That's what makes the task idempotent: re-running it doesn't redo work that's already done, 
# so the output is the same whether I run it once or five times. If I were to remove incremental processing on all
# 365 records, I would spend more money on API calls on AI models or rack up some kind of API bill which is wasting 
# the tokens on corrupted data when the pipeline fails in some parts.# It would also make every run take as long as
# a full backfill, turning something that finishes in seconds into minutes, which risks hitting the API rate limit or
# timing out the task. And since LLM output isn't deterministic, re-processing the same record can give a different answer
# each time, so the upsert overwrites good historical values with new ones and my data quietly changes even though
# the source never did.