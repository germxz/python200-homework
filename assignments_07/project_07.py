from pathlib import Path
import pandas as pd
from scipy.stats import pearsonr
from smolagents import CodeAgent, OpenAIServerModel, tool
import os
from dotenv import load_dotenv
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, avoids tkinter thread errors


load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")
df = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "assignments_01" / "outputs" / "merged_happiness.csv"
FALLBACK_DIR = PROJECT_ROOT / "assignments" / "resources" / "happiness_project"


#Task 1

#Tool 1
@tool
def load_happiness_data() -> dict:
    """Load the World Happiness dataset into memory.

    Loads the merged CSV from DATA_PATH if it exists. Otherwise, falls back to
    loading and merging all yearly CSV files found in the happiness_project
    resources folder.

    Returns:
        A dict with "shape" (rows, columns) and "columns" (list of column names).
    """
    global df

    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
    elif FALLBACK_DIR.exists():
        yearly_dfs = []
        for csv_file in sorted(FALLBACK_DIR.glob("*.csv")):
            yearly_dfs.append(pd.read_csv(csv_file))
        if yearly_dfs:
            df = pd.concat(yearly_dfs, ignore_index=True)
        else:
            raise FileNotFoundError(f"No CSV files found in fallback directory: {FALLBACK_DIR}")
    else:
        raise FileNotFoundError(
            f"Data file not found at {DATA_PATH} and fallback directory not found at {FALLBACK_DIR}."
        )

    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
    }


#Tool 2
@tool
def summarize_column(column: str) -> dict:
    """Return descriptive statistics for a single column in the loaded dataset.

    Computes count, mean, standard deviation, min, max, and quartiles
    (25%, 50%, 75%) for the specified column, using pandas' built-in
    describe() method. Use this tool when the user wants a quick
    statistical overview of one column (e.g. "what's the average GDP?"
    or "summarize the population column").

    Args:
        column: The name of the column to summarize. Must be a numeric
            column present in the currently loaded dataset.

    Returns:
        A dictionary of summary statistics (count, mean, std, min, 25%,
        50%, 75%, max) as returned by pandas' describe(). If no dataset
        is loaded or the column does not exist, returns a dictionary
        with a single "error" key describing the problem.
    """
    if df is None:
        return {"error": "No data loaded. Call load_happiness_data first."}
    if column not in df.columns:
        return {"error": f"Column '{column}' not found in dataset."}

    return df[column].describe().to_dict()


# Tool 3
@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """Compute the Pearson correlation coefficient and p-value between two numeric columns.

    Use this tool when the user asks whether two variables are related
    or move together (e.g. "is there a correlation between GDP and
    life expectancy?"). The Pearson r measures the strength and
    direction of a linear relationship, ranging from -1 (perfect
    negative) to 1 (perfect positive), and the p-value indicates
    whether that correlation is statistically significant.

    Args:
        col1: The name of the first numeric column.
        col2: The name of the second numeric column.

    Returns:
        A dictionary with the keys "col1", "col2", "pearson_r", and
        "p_value" (the latter two rounded to 4 decimal places). If no
        dataset is loaded, either column is missing, or either column
        is non-numeric, returns a dictionary with a single "error" key
        describing the problem.
    """
    if df is None:
        return {"error": "No data loaded. Call load_happiness_data first."}
    if col1 not in df.columns or col2 not in df.columns:
        return {"error": f"One or both columns not found: '{col1}', '{col2}'."}
    if not pd.api.types.is_numeric_dtype(df[col1]) or not pd.api.types.is_numeric_dtype(df[col2]):
        return {"error": f"Both columns must be numeric: '{col1}', '{col2}'."}

    valid = df[[col1, col2]].dropna()
    r, p = pearsonr(valid[col1], valid[col2])

    return {
        "col1": col1,
        "col2": col2,
        "pearson_r": round(r, 4),
        "p_value": round(p, 4),
    }


#Tool 4
@tool
def get_top_n_countries(column: str, year: int, n: int = 5) -> dict:
    """Return the top N countries ranked by a given column for a specific year.

    Filters the dataset to the specified year, sorts the remaining rows
    by the given column in descending order, and returns the top N
    countries. Use this tool when the user asks for rankings or
    leaderboards (e.g. "which 5 countries had the highest CO2 emissions
    in 2020?" or "top 10 countries by GDP in 2015").

    Args:
        column: The name of the numeric column to rank countries by.
        year: The year to filter the dataset on before ranking.
        n: The number of top countries to return. Defaults to 5.

    Returns:
        A list of dictionaries, each containing "Country" and the
        requested column's value, sorted in descending order by that
        column, limited to the top n entries. If no dataset is loaded,
        the column or year is invalid, or no data exists for that year,
        returns a dictionary with a single "error" key describing the
        problem.
    """
    if df is None:
        return {"error": "No data loaded. Call load_happiness_data first."}
    if column not in df.columns:
        return {"error": f"Column '{column}' not found in dataset."}
    if "Year" not in df.columns:
        return {"error": "Dataset has no 'Year' column."}
    if "Country" not in df.columns:
        return {"error": "Dataset has no 'Country' column."}

    subset = df[df["Year"] == year]
    if subset.empty:
        return {"error": f"No data found for year {year}."}

    top = subset.sort_values(by=column, ascending=False).head(n)
    return top[["Country", column]].to_dict(orient="records")


# Task 2 initiate agent

model = OpenAIServerModel(api_key=api_key, model_id="gpt-4o-mini")

SYSTEM_PROMPT = """
You are a data analyst assistant for the World Happiness dataset.
Use the available tools for loading data, summarizing columns, computing correlations,
and ranking countries. Write Python code directly only when the tools are not sufficient
(for example, when creating custom plots or computing something the tools don't cover).
Be concise and student-friendly in your responses.
"""

agent = CodeAgent(
    tools=[load_happiness_data, summarize_column, compute_correlation, get_top_n_countries],
    model=model,
    instructions=SYSTEM_PROMPT,
    additional_authorized_imports=["pandas", "matplotlib.pyplot", "scipy.stats"],
    max_steps=8,
)


# Task 3 run guided queries

queries = [
    "Load the happiness data and tell me its shape and column names.",
    "Summarize the happiness_score column.",
    "What is the correlation between gdp_per_capita and happiness_score? Is it statistically significant?",
    "Show me the top 5 happiest countries in 2020.",
    "Plot happiness_score over the years as a line chart, with one line per region. Save the plot to outputs/happiness_by_region.png.",
]

for query in queries:
    print(f"\n--- Query: {query} ---")
    if "Plot" in query:
        # The plotting task needs row-level data (year x region), which none
        # of the four tools expose. Passing the real dataframe into the
        # sandbox lets the agent write correct plotting code against real
        # data instead of fabricating values.
        response = agent.run(query, reset=False, additional_args={"df": df})
    else:
        response = agent.run(query, reset=False)
    print(response)


# Task 4: Your own questions

# My query 1
my_query_1 = "What is the correlation between social support and happiness score, and is it stronger or weaker than the GDP correlation?"
response_1 = agent.run(my_query_1, reset=False)
print(f"\n--- Query: {my_query_1} ---")
print(response_1)
# Comment: This should trigger a tool call (compute_correlation). Since
# reset=False keeps prior conversation context, the agent may reuse the
# earlier GDP/happiness_score correlation result instead of recomputing it.

# My query 2
my_query_2 = "Which region had the biggest increase in average happiness score between 2015 and 2024?"
print(f"\n--- Query: {my_query_2} ---")
response_2 = agent.run(my_query_2, reset=False, additional_args={"df": df})
print(response_2)

# No tool computes a groupby + year-over-year comparison across
# regions, so this should force the agent to write its own pandas code
# (groupby on Region/Year, filter to 2015 and 2024, compute the difference)
# rather than call one of the four tools. The real dataframe is passed in
# via additional_args for the same reason it was needed for the plot query.