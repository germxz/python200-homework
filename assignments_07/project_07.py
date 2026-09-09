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

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

# My Week 1 merged file stores columns as "Happiness score" / "GDP per capita" /
# "Regional indicator", but every query in this assignment asks for
# happiness_score, gdp_per_capita and region. Without this one rename step the
# tools return {"error": "column not found"} for all five required queries, so
# this is the minimum needed to make the assignment's own queries work.
# (The raw 2024 file also calls the target column "Ladder score".)
COLUMN_RENAMES = {
    "regional indicator": "region",
    "ladder score": "happiness_score",
}


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of frame with the project-wide snake_case column schema."""
    renamed = {}
    for col in frame.columns:
        key = col.strip().lower()
        renamed[col] = COLUMN_RENAMES.get(key, key.replace(" ", "_"))
    return frame.rename(columns=renamed)


#Task 1

#Tool 1
@tool
def load_happiness_data() -> dict:
    """Load the World Happiness dataset into memory.

    Loads the merged CSV from assignments_01/outputs/merged_happiness.csv if
    it exists. Otherwise, falls back to loading and merging all yearly CSV
    files found in assignments/resources/happiness_project/ (those raw files
    are semicolon-separated, use commas as decimal marks, and carry the year
    only in the filename). Column names are lowercased with underscores, so the
    dataset exposes "country", "region", "year", "happiness_score",
    "gdp_per_capita", "social_support", and so on.

    If you need row-level data that these tools do not expose (for example, to
    build a custom plot), the loaded DataFrame is handed to you directly as the
    variable `df`. Use it. Never invent or simulate rows.

    Returns:
        A dict (not a DataFrame) with "shape" (tuple of rows, columns) and
        "columns" (list of column names).
    """
    global df

    if DATA_PATH.exists():
        df = _normalize_columns(pd.read_csv(DATA_PATH))
    elif FALLBACK_DIR.exists():
        yearly_dfs = []
        for csv_file in sorted(FALLBACK_DIR.glob("world_happiness_*.csv")):
            yearly = pd.read_csv(csv_file, sep=";", decimal=",")
            yearly = _normalize_columns(yearly)
            # The year is only in the filename, not inside the raw yearly files.
            yearly["year"] = int(csv_file.stem.split("_")[-1])
            yearly_dfs.append(yearly)
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
    statistical overview of one column (e.g. "summarize the
    happiness_score column" or "what's the average gdp_per_capita?").

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
    or move together (e.g. "is there a correlation between
    gdp_per_capita and happiness_score?"). The Pearson r measures the strength and
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
        A list of dictionaries, each containing "country" and the
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
    if "year" not in df.columns:
        return {"error": "Dataset has no 'year' column."}
    if "country" not in df.columns:
        return {"error": "Dataset has no 'country' column."}

    subset = df[df["year"] == year]
    if subset.empty:
        return {"error": f"No data found for year {year}."}

    top = subset.sort_values(by=column, ascending=False).head(n)
    return top[["country", column]].to_dict(orient="records")


# Task 2: initiate agent

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


# Task 3: guided query sequence
queries = [
    "Load the happiness data and tell me its shape and column names.",
    "Summarize the happiness_score column.",
    "What is the correlation between gdp_per_capita and happiness_score? Is it statistically significant?",
    "Show me the top 5 happiest countries in 2020.",
    "Plot happiness_score over the years as a line chart, with one line per region. Save the plot to outputs/happiness_by_region.png.",
]

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Query 5 needs the actual rows to draw one line per region, and no tool
    # returns rows. Passing the loaded DataFrame in as `df` gives the agent the
    # real data to write its plotting code against. Without this it has nothing
    # to plot -- which is exactly when it starts inventing numbers.
    for query in queries:
        print(f"\n--- Query: {query} ---")
        response = agent.run(query, reset=False, additional_args={"df": df})
        print(response)

    # Task 4: additional queries
    my_query_1 = "What is the correlation between social support and happiness score, and is it stronger or weaker than the GDP correlation?"
    response_1 = agent.run(my_query_1, reset=False, additional_args={"df": df})
    print(f"\n--- Query: {my_query_1} ---")
    print(response_1)
    # Comment: TOOL USE (wrapped in one line of code). The agent called
    # compute_correlation("social_support", "happiness_score") and compared the
    # result to the GDP number it still had in context from Query 3.
    # r = 0.7439 for social support vs r = 0.6313 for GDP, so social support is
    # the stronger correlate. reset=False is what made that comparison possible.

    my_query_2 = "Which region had the biggest increase in average happiness score between 2015 and 2024?"
    response_2 = agent.run(my_query_2, reset=False, additional_args={"df": df})
    print(f"\n--- Query: {my_query_2} ---")
    print(response_2)
    # Comment: CODE GENERATION. No tool computes a year-over-year regional
    # comparison, so the agent wrote its own pandas against `df` (filter to 2015
    # and 2024, groupby region, mean, subtract, take the max). Answer:
    # Central and Eastern Europe, +0.6627.


# --- Reflection ---
#
# 1. In Query 3, how did the agent communicate whether the correlation was statistically
#    significant? Did it use the p-value correctly? What threshold did it apply?
#
#    It called compute_correlation("gdp_per_capita", "happiness_score"), got back
#    {'pearson_r': 0.6313, 'p_value': 0.0}, and then returned:
#        {"pearson_r": 0.6313, "p_value": 0.0, "statistically_significant": True}
#
#    The conclusion is correct, but note HOW it got there: it never wrote the
#    threshold down. There is no `p_value < 0.05` anywhere in its generated code
#    -- it read p = 0.0 and then hardcoded the boolean True. The alpha = 0.05
#    convention was applied implicitly, by the model, not by the code.
#
#    With p = 0.0 that shortcut happens to land on the right answer. But it means
#    the "statistically_significant" flag is an assertion rather than a
#    computation: if the p-value had come back as 0.06, nothing in the code would
#    have forced it to say False. That is an argument for putting the threshold
#    inside the tool, where it gets applied the same way every time, instead of
#    leaving it to the model to remember.
#
# 2. Did any of the agent's responses surprise you — either by being more capable than
#    you expected, or less? Describe one specific example.
#
#    More capable than I expected, on Query 5. No tool returns rows and no tool
#    plots anything, so the agent had to do the whole thing itself -- and in a
#    single step it wrote:
#        df.groupby(['year', 'region'])['happiness_score'].mean().unstack()
#    then plotted it, titled it, added a legend, and saved it to the exact path
#    I asked for. The .unstack() is the clever part: it is what turns the grouped
#    result into one column per region, which is precisely what "one line per
#    region" requires. I did not tell it to reshape the data that way.
#
#    What surprised me less pleasantly is how much that capability depends on
#    having real data in reach. The tools deliberately return summaries, not
#    rows, so I had to hand the DataFrame to the agent explicitly via
#    additional_args={"df": df}. Query 5 only works because of that one
#    parameter -- otherwise the agent is asked to plot data it cannot see.
#
# 3. What one additional tool would make this agent meaningfully more useful?
#    Describe what it would do and what kind of question it would help the agent answer.
#    (You do not need to implement it.)
#
#    get_grouped_average(group_by: str, value_column: str, year: int | None = None)
#    returning a dict of group -> mean value: average happiness_score per region,
#    optionally filtered to one year.
#
#    Every question in this assignment that forced the agent to write its own
#    pandas was this same shape -- Query 5's chart (mean happiness_score by region
#    and year) and my second custom query (which region gained the most between
#    2015 and 2024) both reduce to a grouped average. That is the one clear gap
#    the four existing tools leave.
#
#    It would also move the aggregation into reviewed code. Right now the agent
#    re-derives the groupby from scratch on every run, so the numbers depend on
#    whatever it wrote that time; behind a tool the calculation would be the same
#    every time and I could unit-test it.