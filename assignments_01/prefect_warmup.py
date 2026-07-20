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

#---Prefect might be too much overhead for this simple pipeline since it isn't much to keep track of, but it can be useful for more complex workflows with dependencies and scheduling.
#---Some scenarios that prefect might be useful include data ingestion pipelines, ETL processes, and machine learning workflows where tasks need to be orchestrated and monitored.