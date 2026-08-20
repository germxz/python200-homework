
#API Q1 

import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is one thing that makes Python a good language for beginners?"}]
)

print(f"APIQ1 response text: {response.choices[0].message.content}")
print(f"APIQ1 model: {response.model}")
print(f"APIQ1 total tokens: {response.usage.total_tokens}")

#API Q2
prompt = "Suggest a creative name for a data engineering consultancy."
temperatures = [0, 0.7, 1.5]

for temp in temperatures:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=temp
    )
    print(f"APIQ2 temperature: {temp}")
    print(f"APIQ2 response text: {response.choices[0].message.content}")
    print(f"APIQ2 model: {response.model}")
    print(f"APIQ2 total tokens: {response.usage.total_tokens}")
    
# I notice that the lower temperature outputs are more concise and use less tokens but 0 is repeatable and the .7 will change if you run it a few different times. 1.5 had a more creative name but the model also went on a short explanation of the name, using more tokens.

#API Q3

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Give me a one-sentence fun fact about pandas (the animal, not the library)."}],
    n=3,
    temperature=1.0
)
for i, choice in enumerate(response.choices):
    print(f"APIQ3 completion {i+1}: {choice.message.content}")
    
    
#API Q4
max_tokens= 15
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages= [{"role":"user" , "content": "Explain how neural networks work."}],
    max_tokens=max_tokens
)

print(f"APIQ4 {response.choices[0].message.content}")

# The model was able to give a brief description of neural networks.
# You might want to use the max tokens option to avoid using more tokens than needed with extra explanation.

# System Messages and Personas


messages = [
    {"role": "system", "content": "You are a patient, encouraging Python tutor. You always explain things simply and end with a word of encouragement."},
    {"role": "user", "content": "I don't understand what a list comprehension is."}
]

response = client.chat.completions.create(
    model = "gpt-4o-mini",
    messages= messages,   
)

print(f"SYSQ1 {response.choices[0].message.content}")

messages_alt = [
    {"role": "system" , "content" :"You are a concise, nonchalant tutor with experience in computer science, software engineering, and data engineering. You will give you no fluff and will clown a user for not knowing the basics" },
    {"role": "user", "content": "I don't understand what a list comprehension is."}

]

response = client.chat.completions.create(
    model = "gpt-4o-mini",
    messages= messages_alt,   
)

print(f"SYSQ1ALT: {response.choices[0].message.content}")

# I told the model to be a concise and nonchalant tutor with experience. The model gained a bit more personality and was more concise as i requested. 
# The new model gave me one example as oppposed to the multiple that the first model provided. 


messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Jordan and I'm learning Python."},
    {"role": "assistant", "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?"},
    {"role": "user", "content": "Can you remind me what my name is?"}
]

response = client.chat.completions.create(
    model = "gpt-4o-mini",
    messages= messages,   
)

print(f"SYSQ2: {response.choices[0].message.content}")

# The only reason why the model knows Jordan's name is because I sent it. Models do not save conversations in memory between calls,
# so the model will not remember Jordan's name once the conversation ends or a new API request is made. 

# Prompt Engineering

#PEQ1
reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

# PEQ1 — zero-shot: task description only, no examples
for i, review in enumerate(reviews, start=1):
    prompt = f"Classify the sentiment of the following review as `positive`, `negative`, or `mixed`.\n\nReview: \"{review}\"\nSentiment:"

    messages = [
        {"role": "user", "content": prompt},
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )

    print(f"PEQ1 review {i}: {response.choices[0].message.content}")


#PEQ2
# PEQ2 — one-shot: one example before the review
for i, review in enumerate(reviews, start=1):
    prompt = f"""Classify the sentiment of the following review as `positive`, `negative`, or `mixed`.

Example:
Review: "Fast shipping but the item arrived damaged."
Sentiment: mixed

Review: "{review}"
Sentiment:"""

    messages = [
        {"role": "user", "content": prompt},
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )

    print(f"PEQ2 review {i}: {response.choices[0].message.content}")

# When I first called the model without an example, It printed out thne reviews but it also 
# revealed other values like "sentiment: mixed". When I provided an example on the second instance,  
# The answers came out how I wanted; consistent and single words. 

# PEQ3

for i, review in enumerate(reviews, start=1):
    prompt = f"""Classify the sentiment of the following review as `positive`, `negative`, or `mixed`.

Examples:
Review: "Fast shipping but the item arrived damaged."
Sentiment: mixed

Review: "What even is this product? It doesn't even work properly after 2 days of use"
Sentiment: negative  

Review: "I absolute love it! It came on time and the product does what it needs to do. "
Sentiment:positive

Review: "{review}"
Sentiment:"""

    messages = [
        {"role": "user", "content": prompt},
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )

    print(f"PEQ3 reviews {i}: {response.choices[0].message.content}")

# You can see why they're named the way they are. The zero shot lets the AI answer in it's own form, sometimes being inconsistent.
# The One-shot mechanism works by giving the agent a single example to show how we want our answers to look like. 
# The few-shot does the same but multiple times to ensure that the agent really knows what to do in outlying situations. 
# I would use the one-shot method for this instance since it uses less tokens and it takes less time to type. So in short, 
#zero-shot drifted and returned extra text, one-shot locked it to single words. 
# PEQ4

problem = f"""A data engineer earns $85,000 per year. She gets a 12% raise, then 6 months later
takes a new job that pays $7,500 more per year than her post-raise salary. What is her final annual salary?"""

prompt = f"""solve the following problem, show your reasoning step by step before giving a final answer.
Label the final answer clearly as 'Final Answer:'.

Problem: {problem}
"""

messages=[{"role": "user", "content": prompt}]


response = client.chat.completions.create(
    model="gpt-4o-mini", 
    messages= messages,
)

print(f"PEQ4: {response.choices[0].message.content}")

#The model showed each step — the raise amount, the post-raise salary, then the increase — so I could check the math myself instead of trusting a single number.
# Asking for step-by-step reasoning makes arithmetic mistakes visible rather than hidden. This allows the model to build its own calculations.
# PEQ5


review = """I've been using this tool for three months. It handles large datasets well, \
but the UI is clunky and the export options are limited. 
"""

prompt = f"""analyze the review below and return the result **only as valid JSON** with keys `sentiment`, `confidence` (a float from 0 to 1), and `reason` (one sentence).


Review: {review}
"""

message = [{"role": "user", "content": prompt}]

response =  client.chat.completions.create(
    model = "gpt-4o-mini",
    messages = message,
)

raw = response.choices[0].message.content

print(f"PEQ5: {raw}")


try:
    data= json.loads(raw)
    print("Sentiment:", data["sentiment"])
    print("Confidence:", data["confidence"])
    print("Reason:", data["reason"])
    
except json.JSONDecodeError:
    print("Parse failed. Raw response was:")
    print(raw)
    
# PEQ6

user_text = "First boil a pot of water. Once boiling, add a handful of salt and the \
pasta. Cook for 8-10 minutes until al dente. Drain and toss with your sauce of choice."

prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```
"""

message= [{"role":"user", "content":prompt}]
response = client.chat.completions.create(
    model = "gpt-4o-mini",
    messages=message,
)

print(f"PEQ6: {response.choices[0].message.content}")

user_text2 = "The Scion tC is a 4 cylinder coupe that is pretty reliable on gas mileage. It also uses front wheel drive so it is a good vehicle to own in the snow"
prompt2 = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text2}```
"""

message2= [{"role":"user", "content":prompt2}]
response2 = client.chat.completions.create(
    model = "gpt-4o-mini",
    messages=message2,
)
raw2 = response2.choices[0].message.content
print(f"PEQ6P2 {raw2}")

# Delimiters help separate instructions and content to avoid confusion within the model.
# Without it, the model will think that the content we give (user_text2) is part of the instructions leading to misinterpretation.
#without delimiters, someone could paste text containing its own instructions and the model can't tell that apart from yours, so it may follow theirs. 
#Ollama

#Ollama Q1

# Terminal output from: ollama run qwen3:0.6b "Explain what a large language model is in two sentences."

"""
A large language model is an AI system designed to understand and generate human-like text, capable of performing tasks like writing, answering questions, or 
creating content based on vast amounts of training data. It leverages massive datasets and complex algorithms to process and interpret language in a way that 
mimics natural human thought processes.
"""

llama_prompt =  "Explain what a large language model is in two sentences."
llama_message = [{"role":"user", "content": llama_prompt }]

llama_response = client.chat.completions.create(
    model= "gpt-4o-mini",
    messages=llama_message,
)

print(llama_response.choices[0].message.content)

# gpt-4o-mini named actual mechanisms (neural networks, prediction from learned patterns) while qwen said "complex algorithms" and "mimics natural human thought processes" — vaguer, and the thought-process claim is arguably wrong.
# It does not give examples of things it mentions. The openAI model was concise with answering the question, while also giving examples of the subject in question. 
#Advantage: free per call, works offline, and your data never leaves your machine (no prompt sent to a company server).
#Disadvantage: a 0.6B model is much weaker — I saw it produce a vaguer and arguably wrong description. Closing the quality gap means buying real GPU hardware.

