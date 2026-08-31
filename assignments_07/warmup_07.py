from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path
import os
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend: a GUI plt.show() would block the script
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from smolagents import ToolCallingAgent, OpenAIServerModel, tool
from smolagents import CodeAgent

# Plots are written here so Q8's "did the dots actually come out green?" is
# something we can open and check, rather than something we take the agent's
# word for.
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"



if load_dotenv():
    print('Successfully loaded environment variables from .env')
else:
    print('Warning: could not load environment variables from .env')

client = OpenAI()
print('OpenAI client created.')


# --- Lesson 02 ---

# Q1

def celsius_to_fahrenheit(celsius: float) -> str:
    """Convert a Celsius temperature to Fahrenheit and return it as a formatted string."""
    fahrenheit = (celsius * 9 / 5) + 32
    return f"{celsius}°C is {fahrenheit}°F"


# Flat schema shape, matching the get_current_time schema style from the lesson
# (no outer 'type': 'function' / 'function': {...} wrapper here).
celsius_to_fahrenheit_schema = {
    'name': 'celsius_to_fahrenheit',
    'description': 'Convert a Celsius temperature to Fahrenheit and return it as a formatted string.',
    'parameters': {
        'type': 'object',
        'properties': {
            'celsius': {
                'type': 'number',
                'description': 'The temperature in Celsius to convert.',
            },
        },
        'required': ['celsius'],
    },
}

print(celsius_to_fahrenheit(0))
print(celsius_to_fahrenheit(100))
print(celsius_to_fahrenheit(-40))


# Q2
# Calling run_agent with "Convert 100 degrees Celsius to Fahrenheit" will not trigger a tool call because this agent is dedicated to getting the current time.
# This run_agent function doesn't have anything to do with temperature conversion.

# I think that there will be 2 api calls to answer this query since the first llm has to run in order to access the tools and another to work with the tool.
# I predict that this may be the case even if the prompt is incompatible with the task.


def get_current_time() -> str:
    '''Return the current local time as a formatted string.'''
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


# Flat schema shape (matches the lesson's get_current_time schema exactly).
get_current_time_schema = {
    'name': 'get_current_time',
    'description': 'Returns the current local time as a string.',
    'parameters': {
        'type': 'object',
        'properties': {},
        'required': [],
    },
}

# The OpenAI Chat Completions API requires each tool wrapped as
# {'type': 'function', 'function': {...}}, so we wrap the flat schemas here,
# right before they're used in the API call. The schemas themselves stay flat,
# matching the lesson's shape.
tools = [{'type': 'function', 'function': get_current_time_schema}]

def run_agent(user_prompt: str) -> str:
    '''Run a minimal ReAct-style agent for a single user prompt.'''

    SYSTEM_PROMPT = '''You are a simple assistant that can tell the current time.
                     Use the tool get_current_time whenever a user asks about the time.'''

    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': user_prompt},
    ]

    first_response = client.chat.completions.create(
        model='gpt-4.1-mini',
        messages=messages,
        tools=tools,
        tool_choice='auto',
    )

    print("First response received from model...")
    print(first_response)
    first_message = first_response.choices[0].message

    messages.append(
        {
            'role': 'assistant',
            'content': first_message.content,
            'tool_calls': first_message.tool_calls,
        }
    )

    if first_message.tool_calls:
        print("Agentic mode engaged...")
        for tool_call in first_message.tool_calls:
            function_name = tool_call.function.name
            if function_name == 'get_current_time':
                tool_result = get_current_time()
            else:
                tool_result = f'Error: unknown tool {function_name}.'

            print('Tool called:', function_name)
            print('Tool result:', tool_result)

            messages.append(
                {
                    'role': 'tool',
                    'tool_call_id': tool_call.id,
                    'name': function_name,
                    'content': tool_result,
                }
            )

        second_response = client.chat.completions.create(
            model='gpt-4.1-mini',
            messages=messages,
        )
        print("Second response received from model...")
        print(second_response)

        final_message = second_response.choices[0].message
        return final_message.content or ''
    else:
        print("No tools needed....")

    return first_message.content or ''


ans = run_agent("Convert 100 degrees Celsius to Fahrenheit")
print(ans)

# That first version only made one API call. The first response had no tool calls,
# so the code went straight to the else branch and returned the model's direct
# answer instead of calling the second API endpoint.


# Q3

# Wrap both flat schemas for the API call, same pattern as above.
tools = [
    {'type': 'function', 'function': get_current_time_schema},
    {'type': 'function', 'function': celsius_to_fahrenheit_schema},
]


def run_agent(user_prompt: str) -> str:
    '''Run a minimal ReAct-style agent for a single user prompt (now with two tools).'''

    SYSTEM_PROMPT = '''You are a simple assistant that can tell the current time
                     and convert Celsius temperatures to Fahrenheit.
                     Use get_current_time when asked about the time.
                     Use celsius_to_fahrenheit when asked to convert a temperature.'''

    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': user_prompt},
    ]

    first_response = client.chat.completions.create(
        model='gpt-4.1-mini',
        messages=messages,
        tools=tools,
        tool_choice='auto',
    )

    print("First response received from model...")
    print(first_response)
    first_message = first_response.choices[0].message

    messages.append(
        {
            'role': 'assistant',
            'content': first_message.content,
            'tool_calls': first_message.tool_calls,
        }
    )

    if first_message.tool_calls:
        print("Agentic mode engaged...")
        for tool_call in first_message.tool_calls:
            function_name = tool_call.function.name

            if function_name == 'get_current_time':
                tool_result = get_current_time()
            elif function_name == 'celsius_to_fahrenheit':
                args = json.loads(tool_call.function.arguments)
                tool_result = celsius_to_fahrenheit(args['celsius'])
            else:
                tool_result = f'Error: unknown tool {function_name}.'

            print('Tool called:', function_name)
            print('Tool result:', tool_result)

            messages.append(
                {
                    'role': 'tool',
                    'tool_call_id': tool_call.id,
                    'name': function_name,
                    'content': tool_result,
                }
            )

        second_response = client.chat.completions.create(
            model='gpt-4.1-mini',
            messages=messages,
        )
        print("Second response received from model...")
        print(second_response)

        final_message = second_response.choices[0].message
        return final_message.content or ''
    else:
        print("No tools needed....")

    return first_message.content or ''


response_a = run_agent("What is 37 degrees Celsius in Fahrenheit?")
print("Response A:", response_a)
# Tool called: celsius_to_fahrenheit. The prompt directly matches this tool's
# description, so the model requested it instead of computing the conversion itself.

response_b = run_agent("What is the boiling point of water in plain English?")
print("Response B:", response_b)
# No tool called. Neither tool's description matches a general-knowledge question
# like this one, so the model answered directly from its own training knowledge.


# --- Lesson 03 Multi tool agent ---

# Q4

class CsvManager:
    def __init__(self, resources_dir: Path):
        self.resources_dir = resources_dir
        self.df = None
        self.csv_name = None

    # --- Small internal helpers --------------------------------------

    def _normalize_csv_name(self, filename: str) -> str:
        if not filename.lower().endswith(".csv"):
            return filename + ".csv"
        return filename

    def _available_csv_files(self) -> list[str]:
        if not self.resources_dir.exists():
            return []
        return sorted(
            [
                p.name
                for p in self.resources_dir.iterdir()
                if p.is_file() and p.suffix.lower() == ".csv"
            ]
        )

    def _ensure_loaded(self):
        if self.df is None:
            files = self._available_csv_files()
            example = files[0] if files else "your_file.csv"
            return {
                "error": (
                    "No CSV is loaded yet. First load one from resources/. "
                    f"For example: load_csv '{example}'."
                )
            }
        return None

    # --- Tools (public methods) --------------------------------------

    def list_csv_files(self):
        """
        List available CSV files in resources/.
        """
        files = self._available_csv_files()
        if not files:
            return {
                "message": (
                    "No CSV files found in resources/. "
                    "Create a resources/ folder and put one or more .csv files inside it."
                ),
                "files": [],
            }
        return {"files": files}

    def load_csv(self, filename: str):
        """
        Load a CSV file from resources/ and make it the active dataset.

        filename can be "bike_commute" or "bike_commute.csv".
        """
        filename = self._normalize_csv_name(filename)
        path = self.resources_dir / filename

        if not path.exists():
            return {
                "error": f"Could not find '{filename}' in resources/.",
                "available_files": self._available_csv_files(),
            }

        self.df = pd.read_csv(path)
        self.csv_name = filename

        return {
            "message": f"Loaded {filename} with shape {self.df.shape}.",
            "columns": self.df.columns.tolist(),
        }

    def get_columns(self):
        """
        Return column names for the currently loaded CSV.
        """
        error = self._ensure_loaded()
        if error:
            return error
        return self.df.columns.tolist()

    def summarize_columns(self, columns: list[str] | None = None):
        """
        Return basic summary stats for one or more columns.

        If columns is None, summarize all columns.
        Uses pandas.describe(include="all") to stay simple and readable.
        """
        error = self._ensure_loaded()
        if error:
            return error

        if columns is None:
            data = self.df
        else:
            missing = [c for c in columns if c not in self.df.columns]
            if missing:
                return {"error": f"These columns are not in the data: {missing}"}
            data = self.df[columns]

        summary = data.describe(include="all").transpose().round(3)
        return summary.to_dict()

    def describe_column(self, column: str):
        """
        Simple summary for a single column using pandas.describe().
        """
        error = self._ensure_loaded()
        if error:
            return error

        if column not in self.df.columns:
            return {"error": f"'{column}' is not a column. Options: {self.df.columns.tolist()}"}

        s = self.df[column]
        summary = s.describe().to_dict()

        cleaned = {}
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                cleaned[key] = round(value, 3)
            else:
                cleaned[key] = value

        return cleaned

    def compute_correlation(self, col1: str, col2: str):
        """
        Compute the Pearson correlation between two columns in the loaded DataFrame.
        Returns the correlation coefficient and p-value.

        Explicitly guards against the "no CSV loaded" case, as required, rather
        than relying only on the shared _ensure_loaded() helper.
        """
        if self.df is None:
            return {"error": "No CSV loaded. Please load a CSV first."}

        if col1 not in self.df.columns or col2 not in self.df.columns:
            return {"error": f"One or both columns not found. Options: {self.df.columns.tolist()}"}

        r, p_value = pearsonr(self.df[col1], self.df[col2])

        return {
            "col1": col1,
            "col2": col2,
            "pearson_r": round(r, 4),
            "p_value": round(p_value, 4),
        }

    def _save_figure(self, stem: str) -> str:
        """Save the current figure into outputs/ and close it. Returns the path."""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUTPUT_DIR / f"{stem}.png"
        plt.savefig(path, dpi=100, bbox_inches="tight")
        plt.close()
        return str(path)

    def plot_data(self, y: str, x: str | None = None, plot_type: str = "line",
                  color: str | None = None):
        """
        Plot from the active CSV.

        - If x is None: plot y vs row index.
        - If x is provided: plot y vs x.
        - color optionally sets the line/marker color, e.g. "green".

        Q8 asks whether each agent actually produced *green* dots. Originally
        this tool had no color parameter at all, so neither agent could have
        satisfied that request no matter how well it reasoned -- the honest
        comparison the question asks for wasn't even possible. Adding `color`
        here makes it possible, and saving the figure (instead of just calling
        plt.show()) makes the result checkable after the fact.
        """
        error = self._ensure_loaded()
        if error:
            return error

        if plot_type not in ["scatter", "line"]:
            return "Error: I can only do 'scatter' or 'line'."

        if y not in self.df.columns:
            return f"Error: column '{y}' is not in {self.df.columns.tolist()}"

        if x == y:
            x = None

        if plot_type == "scatter" and x is None:
            return "Error: scatter plots need both x and y columns."

        title_csv = self.csv_name or "current CSV"
        color_kwargs = {"color": color} if color else {}
        color_note = color if color else "matplotlib default"

        if x is None:
            ax = self.df[y].plot(kind="line", **color_kwargs)
            ax.set_title(f"{title_csv} | Line plot: {y} vs row index")
            saved = self._save_figure(f"line_{y}_vs_index")
            return (f"Plotted {y} vs row index as a line plot "
                    f"(color: {color_note}). Saved to {saved}.")

        if x not in self.df.columns:
            return f"Error: column '{x}' is not in {self.df.columns.tolist()}"

        ax = self.df.plot(x=x, y=y, kind=plot_type, **color_kwargs)
        ax.set_title(f"{title_csv} | {plot_type.title()} plot: {y} vs {x}")
        saved = self._save_figure(f"{plot_type}_{y}_vs_{x}")

        return (f"Plotted {y} vs {x} as a {plot_type} "
                f"(color: {color_note}). Saved to {saved}.")


print("Class defined")


# --- csv_manager instance (must exist before node_tools references its methods) ---
csv_manager = CsvManager(resources_dir=Path("resources"))

node_tools = {
    "list_csv_files": csv_manager.list_csv_files,
    "load_csv": csv_manager.load_csv,
    "get_columns": csv_manager.get_columns,
    "summarize_columns": csv_manager.summarize_columns,
    "describe_column": csv_manager.describe_column,
    "compute_correlation": csv_manager.compute_correlation,
    "plot_data": csv_manager.plot_data,
}

tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "list_csv_files",
            "description": "List available CSV files in the resources/ folder.",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_csv",
            "description": "Load a CSV file from the resources/ folder and make it the active dataset.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "CSV filename in resources/, e.g. 'bike_commute.csv'.",
                    }
                },
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_columns",
            "description": "Get the column names of the currently loaded CSV.",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_columns",
            "description": "Show basic summary statistics for columns (uses pandas.describe).",
            "parameters": {
                "type": "object",
                "properties": {
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of column names. If omitted, summarize all columns.",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_column",
            "description": "Show basic summary statistics for a single column (uses pandas.describe).",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "Column name to describe.",
                    }
                },
                "required": ["column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_correlation",
            "description": "Compute the Pearson correlation coefficient and p-value between two numeric columns in the loaded CSV.",
            "parameters": {
                "type": "object",
                "properties": {
                    "col1": {"type": "string", "description": "First column name."},
                    "col2": {"type": "string", "description": "Second column name."},
                },
                "required": ["col1", "col2"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_data",
            "description": "Plot data from the active CSV. If only y is provided, plot y vs row index.",
            "parameters": {
                "type": "object",
                "properties": {
                    "y": {"type": "string", "description": "Column name for y-axis."},
                    "x": {"type": "string", "description": "Optional column name for x-axis."},
                    "plot_type": {
                        "type": "string",
                        "enum": ["scatter", "line"],
                        "description": "Type of plot to create.",
                    },
                    "color": {
                        "type": "string",
                        "description": "Optional color for the line or dots, e.g. 'green'.",
                    },
                },
                "required": ["y"],
            },
        },
    },
]

# Q5
# (run_agent_cycle was previously defined twice — once here and again identically
# further down. The duplicate has been removed; this is the single lesson-flow
# version used for the rest of the warmup.)

def run_agent_cycle(messages, user_text, max_tool_rounds=5):
    """
    Run through one react-agent loop using a simple tool-using agent.
    `messages` parameter will usually just contain a system prompt,
    and then user text will be appended.

    The loop has three main steps:

    REASON:
      - Call the model with the conversation so far.
      - The model either replies normally, or asks to call a tool from tool set.

    ACT:
      - If tools are requested, run the Python functions

    OBSERVE:
      - Append each requested tool result back into the LLMs conversation history.
      - On the next iteration, the model reads those tool call results and determines
        whether it has reached the goal.

    Stop condition:
      - If the model returns an assistant message with no tool calls, this is the
        final answer for this react cycle, this implies that reasoning alone without
        tool calls was enough.
      - max_tool_rounds is a safety cap to prevent infinite loops.
    """
    messages.append({"role": "user", "content": user_text})

    def observe_tool_result(tool_call_id, result):
        """
        Return a tool's return value as a message that can be appended to the
        LLMs conversation history. The model will read this tool output on the next
        REASON step.
        """
        content = json.dumps(result, default=str) if not isinstance(result, str) else result
        tool_message = {"role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": content,}
        return tool_message

    for loop_idx in range(max_tool_rounds):
        # REASON: call the model
        # Here it will make use of any previous tool outputs it appended ("observed")
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            tools=tools_schema,
        )

        msg = response.choices[0].message

        # Append the assistant message to the conversation history.
        # Use a plain dict so `messages` stays simple and inspectable.
        assistant_entry = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            assistant_entry["tool_calls"] = [tc.model_dump() for tc in msg.tool_calls]
        messages.append(assistant_entry)

        # No tool calls means the model is answering directly.
        if not msg.tool_calls:
            return msg.content

        # ACT + OBSERVE: run each tool call, then append its result.
        # Note there may be multiple tool calls
        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments or "{}")

            print(f"ACT: {name}({tool_args})")

            fn = node_tools.get(name)
            if fn is None:
                result = {"error": f"Tool '{name}' not found."}
            else:
                try:
                    result = fn(**tool_args) if tool_args else fn()
                except Exception as e:
                    print(f"Tool error in {name}: {type(e).__name__}: {e}")
                    result = {"error": f"Tool '{name}' failed: {type(e).__name__}: {e}"}

            # OBSERVE: append the tool result back into the conversation history.
            messages.append(observe_tool_result(tool_call.id, result))

            # After appending information about all tool outputs, we loop back and REASON again.

    return "I hit the tool-round limit. Try a simpler request."


SYSTEM_PROMPT = """You are a helpful data analysis assistant. You have access to tools
for listing, loading, and analyzing CSV files. Use the tools whenever you need to
inspect or compute something about the data."""


# This is the same scenario from the lesson that used to stall on the tool-round
# limit because there was no correlation tool. Once compute_correlation exists,
# the same request finishes normally.
print("\n" + "=" * 70)
print("Q5: correlation scenario that previously hit the tool-round limit")
print("=" * 70)

messages = [{"role": "system", "content": SYSTEM_PROMPT}]

result = run_agent_cycle(
    messages,
    "Load bike_commute.csv and compute the correlation between avg_traffic_density and avg_speed_kmh.",
)
print("\nQ5 final answer:", result)


# Q6

# This is the actual ReAct loop, recorded in order:
#
#   "system"                — the instruction prompt
#   "user"                  — the goal from the user
#   "assistant" (tool_call) — the model decides it needs load_csv
#   "tool"                  — the result from loading the CSV, fed back in
#   "assistant" (tool_call) — now it asks for compute_correlation
#   "tool"                  — the correlation result, fed back in
#   "assistant" (final)     — the final answer in plain English
#
# The important point is that the model never computed the correlation itself;
# it only asked for tools and used the returned values.

print("\n" + "=" * 70)
print("Q6: full message history after Q5 (the complete ReAct loop)")
print("=" * 70)
print(json.dumps(messages, indent=2, default=str))
print(f"\n(total messages: {len(messages)}; "
      f"roles in order: {[m['role'] for m in messages]})")


# --- Lesson 04: smolagents ---

# Q7

from smolagents import tool

@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """
    Compute the Pearson correlation coefficient and p-value between two numeric
    columns in the currently loaded CSV.

    Args:
        col1: The name of the first column to correlate.
        col2: The name of the second column to correlate.
    """
    return csv_manager.compute_correlation(col1, col2)


print(compute_correlation.description)

# smolagents generates the tool schema for me from the function signature and
# docstring. In Q4, I had to write that schema by hand: the name, description,
# parameters object, each argument type, and the required fields.
#
# With smolagents, I only need:
#   1. Type hints on each parameter (`col1: str`, `col2: str`) — these become the
#      JSON schema types.
#   2. A docstring with an "Args:" section — that becomes the argument descriptions.
#   3. A top-level docstring summary — that becomes the tool description.
#
# So the information is still the same, but smolagents pulls it out of Python
# code instead of making me duplicate it in a separate JSON dict.

# Q8


@tool
def list_csv_files() -> dict:
    """List available CSV files in resources/.

    Returns:
        A dict with a "files" list, or a message if none are found.
    """
    return csv_manager.list_csv_files()

@tool
def load_csv(filename: str) -> dict:
    """Load a CSV file from resources/ and make it the active dataset.

    Args:
        filename: CSV filename in resources/. You can pass "bike_commute" or "bike_commute.csv".

    Returns:
        A dict with a status message and column names, or an error dict.
    """
    return csv_manager.load_csv(filename)

@tool
def get_columns() -> list[str] | dict:
    """Return column names for the currently loaded CSV.

    Returns:
        A list of column names, or an error dict if no CSV is loaded.
    """
    return csv_manager.get_columns()

@tool
def summarize_columns(columns: list[str] | None = None) -> dict:
    """Return summary stats for selected columns (or all columns).

    Args:
        columns: Column names to summarize. If None, summarizes all columns.

    Returns:
        A dict of summary statistics (from pandas.describe), or an error dict.
    """
    return csv_manager.summarize_columns(columns)

@tool
def describe_column(column: str) -> dict:
    """Describe a single column (basic stats) for the requested column.

    Args:
        column: The name of the column to describe.

    Returns:
        A dict of basic stats for the column, or an error dict.
    """
    return csv_manager.describe_column(column)

@tool
def plot_data(y: str, x: str | None = None, plot_type: str = "line",
              color: str | None = None) -> str | dict:
    """Plot from the active CSV and save the figure to outputs/.

    Args:
        y: Column name to plot on the y-axis.
        x: Column name to plot on the x-axis. If None, use row index.
        plot_type: "line" or "scatter". Scatter requires x and y.
        color: Optional color for the line or dots, e.g. "green". If omitted,
            matplotlib's default color is used.

    Returns:
        A short success message string naming the color used and the saved
        file path, or an error dict/string.
    """
    return csv_manager.plot_data(y=y, x=x, plot_type=plot_type, color=color)



TOOLS = [
    list_csv_files,
    load_csv,
    get_columns,
    summarize_columns,
    describe_column,
    compute_correlation,   # from Q7
    plot_data,
]

model_to_use = "gpt-4o-mini"
model = OpenAIServerModel(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_id=model_to_use,
)

SYSTEM_PROMPT = (
    "You are a small data assistant to help analyze files stored in resources/. "
    "Use the available tools to do any work requested (do not guess). "
    "Report only what the tools actually did -- if you could not satisfy part of "
    "the request, say so plainly instead of claiming success. "
    "Keep answers short and student-friendly."
)

# The CodeAgent gets one extra sentence the ToolCallingAgent cannot act on:
# permission to write its own matplotlib when the tools fall short. Without this
# the CodeAgent just calls plot_data like the tool agent does, and the two
# agents look identical -- which hides the very difference Q8 is asking about.
CODE_AGENT_PROMPT = SYSTEM_PROMPT + (
    " You can also write your own Python. If a request asks for styling or "
    "analysis the tools do not expose, write matplotlib/pandas code yourself "
    "using the real CSV in resources/ rather than forcing the request into a "
    "tool that cannot do it. Save any figure you create into outputs/."
)

tool_agent = ToolCallingAgent(
    tools=TOOLS,
    model=model,
    instructions=SYSTEM_PROMPT,
)

code_agent = CodeAgent(
    tools=TOOLS,
    model=model,
    instructions=CODE_AGENT_PROMPT,
    additional_authorized_imports=["pandas", "matplotlib.pyplot", "scipy.stats"],
    max_steps=8,
)

prompt = "Load bike_commute.csv. Plot avg_heart_rate vs duration_min as a scatter plot with green dots."

print("\n" + "=" * 70)
print("Q8: same prompt, two agent types")
print("=" * 70)

print("\n--- ToolCallingAgent ---")
response_tool = tool_agent.run(prompt)

print("\n--- CodeAgent ---")
response_code = code_agent.run(prompt, additional_args={"csv_manager": csv_manager})

print("\nTool agent response:", response_tool)
print("Code agent response:", response_code)


# --- Q8 results (observed, not predicted) ---
#
# I kept both runs because the first one was the real lesson.
#
# FIRST RUN: before plot_data had a `color` parameter:
#   Neither agent could actually do "green dots," because no tool exposed color.
#   The difference was in how they handled that mismatch:
#     - ToolCallingAgent called plot_data(y=..., x=..., plot_type="scatter") and
#       then claimed it had made green dots. That was a hallucination.
#     - CodeAgent made the same call but did not claim the color.
#   The failure mode here was that a missing capability got described as success.
#
# SECOND RUN — after adding `color` to plot_data and its schema:
#   1. What each agent actually produced:
#      - ToolCallingAgent: three distinct tool calls, one per step —
#          load_csv("bike_commute.csv"), plot_data(..., color="green"), and final_answer(...)
#      - CodeAgent: a single Python step that included a few intermediate calls,
#        then a plot call with color="green".
#      - The dots were indeed green. I checked the saved PNG instead of trusting
#        the agent output.
#      - Both agents write to the same filename, so the second run overwrites the
#        first one. The saved file on disk is the later run.
#
#   2. What this tells us:
#      Once the tool can express the request, the two agents start to converge.
#      The CodeAgent still has more room to compose steps, but the ToolCallingAgent
#      stays more predictable and easier to audit because each action is a named,
#      schema-checked call.


# Q9

# 1. A task better suited to a ToolCallingAgent: a customer-facing order status
#    and ETA lookup ("where is order #48213, and when does it arrive?").
#
#    The property that makes it a good fit is that the action space is CLOSED
#    and ENUMERABLE. There is exactly one legal operation — look up one order id
#    against the orders API — and every valid request is that same call with a
#    different parameter. Nothing about the task benefits from composing steps or
#    computing something new.
#
#    When the action space is closed, the JSON schema becomes a real constraint
#    rather than a suggestion: the model can only emit a well-formed call with an
#    order id, so it cannot invent a query, join against a table it shouldn't
#    touch, or read a different customer's record. Every action is logged as a
#    named call with typed arguments, which is exactly what you need when the
#    output is shown to a customer and may have to be audited later. A CodeAgent
#    would buy flexibility the task never uses, and pay for it in reviewability.
#
# 2. A meaningful risk unique to a CodeAgent: ARBITRARY CODE EXECUTION.
#
#    A ToolCallingAgent can only ask for functions I wrote and registered; the
#    worst it can do is call one of my functions with bad arguments. A CodeAgent
#    generates Python that is then actually executed in my process, with whatever
#    filesystem, network, and library reach that process has — code that was
#    never written by a developer, never reviewed, and never tested. The failure
#    isn't just a wrong answer; it's a wrong *action* (an overwritten file, an
#    unintended request) taken before anyone can inspect it.
#
#    I hit a concrete version of this in the project half of this assignment. The
#    tools returned {"error": "column not found"} because the tool names and the
#    dataset's column names disagreed. Instead of surfacing that failure, the
#    agent wrote code that fabricated the data with random.uniform(), plotted the
#    fake numbers, labeled the chart "Simulated Happiness Score," and reported
#    success. A ToolCallingAgent literally cannot do that — it has no tool named
#    "make up plausible data," so it would have had to return the error. The code
#    path let a data-quality failure turn silently into a confident wrong answer.