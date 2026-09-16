## Reflection

    The pipeline did not run cleanly on the first try. My first run completed but the transform task

skipped every record, because weather_enriched was still full from Week 10, so I used SQL to clear
both tables before pulling data for a new city. Then I hit an interpreter problem: Prefect was
installed in two places, and the server I started from my venv wrote a schema version the other
install could not read, so the server would not boot until I deleted prefect.db and ran everything
from one Python. Once that was sorted, the Prefect UI showed all four tasks Completed with no
retries across two clean runs, and the transform logs let me watch the LLM progress fifty records
at a time.
One summary that stood out was [2-14-23]: "[Today is good for running due to mild temperatures,
minimal precipitation, and a manageable wind speed, making for comfortable outdoor exercise conditions.]".
It was interesting because although it may have rained, the model classifies the day as good for running.
I still would run on a slightly rainy day though. It reads well, but my classifier was trained on Oklahoma
City and the wind threshold was tuned for that climate, so applying it to Seattle means the model is drawing
the line in the wrong place and the LLM explains that call confidently either way. If I deployed this on a
daily schedule I would fetch only the previous day instead of all 365 records, and add an alert on the fallback
count so a silent run of "Recommendation unavailable." rows does not look like a successful run.
