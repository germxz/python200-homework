import time

# ML vs LLM Pipelines

# ML/LLM Q1

# An ML classifier uses mathematical operations to classify data based on features, while an LLM uses deep learning
# techniques to understand and generate human-like text based on context and patterns in language. The LLM does it's thing to
# output text that is coherent and contextually relevant, while the ML classifier outputs a label or category based on the
# input features. In other words, an ML classifier will output a label, while an LLM will output text outputs.

# Each one does what it does because of how it was built. My classifier was trained on labeled weather data, so along with
# the good/skip label it hands me a probability I can actually trust and use. It also gives me the same answer every
# time for the same input, and it costs nothing to run. The LLM is not trained on my weather data at all. It's good at
# the recommendation because writing a sentence that fits the conditions is a language problem, not a math problem.

# If I tried swapping these tool's tasks with each other, the ML classifier would not be able to generate coherent text,
# and the LLM would not be able to classify data accurately. The classifier literally can't write a recommendation. The
# only thing it can output is one of the labels it was trained on, it has no words. And if I let the LLM make the binary
# call, I'd get a different answer on different runs for the same day, any confidence number it gave me would just be
# text it made up instead of a real probability, and I'd be paying for every single prediction.


# ML/LLM Q2

# Converting a date string like "2023-07-04" to day-of-week
# A: Deterministic code, because there is exactly one right answer and datetime already knows it.

# Classifying a job posting as "entry-level", "mid-level", or "senior" based on freeform text
# A: LLM, because the level is buried in messy human writing and I don't have a labeled training set to build a model on.

# Predicting customer churn given 15 numeric features and a labeled training dataset
# A: ML, because numeric features plus labeled data is exactly the setup a trained model is built for.

# Normalizing inconsistent city names ("NYC", "New York City", "New York, NY") to a canonical form
# A: Deterministic code with a lookup table, because the common variants are already known and a lookup is free,
# repeatable, and easy to audit. I'd only send the weird leftovers I've never seen before to an LLM.

# Summing a column of revenue figures
# A: Deterministic code, because math has one right answer and I don't want a model guessing at it.


# ML/LLM Q3

# Incremental processing is a technique where data is processed in small chunks when needed. When we update
# a database with new data, we can use incremental processing to only process the new data instead of reprocessing
# the entire dataset. In this pipeline that means checking which dates are already sitting in weather_enriched and only
# running the classifier and the LLM on the dates that aren't there yet.
#
# This is more efficient and saves time and resources. In contrast, batch processing involves processing all the data at
# once, which can be time-consuming and resource-intensive. Every run gets more expensive as the dataset grows. By day 365
# you're reprocessing the entire year just to add one new day's data. That's wasted compute and time that scales badly.
# On cost specifically, 365 records means 365 paid API calls every single run, so if I run the script ten times while I'm
# debugging it that's 3,650 calls to produce one year of summaries.
#
# Due to me using the .upsert method, the pipeline is idempotent. This means that the data will stay the same if I run the
# pipeline multiple times. But that only protects the shape of the table, not the content. The LLM is non-deterministic,
# so re-processing a date I already did rewrites that row's summary with a slightly different sentence. A row I already
# looked at and thought was fine quietly changes for no reason, which makes the table hard to trust.


# Prompt Deaign

# Prompt Q1

alt_SYSTEM_PROMPT= (
    "You are writing a two-sentence running recommendation for a daily weather summary app. "
    "You will receive weather conditions for a single day and a machine learning prediction "
    "about whether the day is good for running. "
    "Write exactly two sentences — one sentence stating a prediction, and the second sentence explaining the reasoning. "
    " Be direct, practical, and specific to the conditions. "
    "Do not use bullet points, headers, or phrases like 'Based on the data'."
)

# To accommodate two sentences, I would change the validation logic from checking
# for exactly one sentence-ending punctuation mark to exactly two. However, this only
# checks sentence count, not content. It wouldn't catch a response where both sentences
# were reasoning and neither stated a prediction. A stricter check could look for keywords
# or split the response into sentences and inspect each one separately to confirm the first
# is a prediction and the second is reasoning.

# Prompt Q2

def call_with_retry(client, messages, max_retries=3):
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=200,
            )
        except Exception as e:
            print(f"Attempt {attempt + 1} of {max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
    return None

# I'd use this in a production pipeline because API calls fail for reasons that fix themselves. It could be rate limits, timeouts,
# a server hiccup, or a technical difficulty. Without a retry, one bad second kills a run that's already 200 records deep. Returning 
# None instead of raising an error lets the code that called it drop in a fallback string and keep going, so one dead record doesn't cost me the
# other 364.