import pandas as pd
import numpy as np
from prefect import task, flow

#---pipelines question 2 ---
arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

@task
def create_series(arr):
    return pd.Series(arr, name="values")

@task 
def clean_data(series):
    return series.dropna()
    
@task
def summarize_data(series):
    return {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0]
    }
    

@flow
def pipeline_flow():
    series = create_series(arr)
    cleaned_series = clean_data(series)
    summary = summarize_data(cleaned_series)
    return summary

if __name__ == "__main__":
    result = pipeline_flow()
    for key, value in result.items():
        print(f"{key}: {value}")

#--- Reflection Question 1: Is Prefect worth the overhead for this pipeline?
#--- For a simple three-step pipeline like this, Prefect is probably more overhead than
#--- necessary. Plain Python functions already work fine here, and adding @task/@flow
#--- decorators, the Prefect runtime, and run tracking adds complexity without much
#--- benefit at this small scale.

#--- Reflection Question 2: In what scenarios would Prefect be useful?
#--- Prefect becomes valuable for larger workflows: data ingestion pipelines that run
#--- on a schedule, ETL processes with many dependent steps, and machine learning
#--- workflows where you need automatic retries on failure, logging, monitoring, and
#--- orchestration of tasks that depend on each other.