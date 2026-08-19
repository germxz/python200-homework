from dotenv import load_dotenv
from openai import OpenAI
import json
load_dotenv()

# task 1
client = OpenAI()


def get_completion(messages, model="gpt-4o-mini", temperature=0.7):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400
    )
    return response.choices[0].message.content


SYSTEM_PROMPT = """
You are a resume and cover letter coach who asks before you rewrite. You help career
changers whose past work doesn't obviously map to the jobs they're applying for. When a
user's bullet point is vague, you ask what they actually did — numbers, scale, outcomes —
before rewriting it, because you'd rather get one specific detail than invent a
plausible-sounding one.

Your rules:
- Stay focused on job application materials. If the user asks about something else, tell
  them that's outside what you help with and steer back to their application.
- Always remind the user to review and edit your output before submitting it anywhere.
- Acknowledge that you may not know the user's specific industry norms, and that they
  should use their own judgment.
"""

# Deliberate choice: I told the coach to ask for details before rewriting so it wouldn't invent accomplishments I don't actually have

#Task 2

bullets = [
    "Helped customers with their problems",
    "Made reports for the management team",
    "Worked with a team to finish the project on time"
]
def rewrite_bullets(bullets: list[str]) -> list[dict]:
    # Format the bullets into a delimited block
    bullet_text = "\n".join(f"- {b}" for b in bullets)

    prompt = f"""
    You are a professional resume coach helping a career changer.
    Rewrite each resume bullet point below to be more specific, results-oriented, and compelling.
    Use strong action verbs. Do not invent facts that aren't implied by the original.

    Return ONLY a valid JSON list. Each item should have two keys:
    "original" (the original bullet) and "improved" (your rewritten version).
    
    Respond ONLY with valid JSON, no other text.

    Bullet points:
    ```
    {bullet_text}
    ```
    """

    messages = [{"role": "user", "content": prompt}]
    # Your code here: call get_completion(), parse the JSON, and return the result

    raw = get_completion(messages)
    try:
        cleaned = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(cleaned)       
        for item in data:
            print("Original:",item["original"])          
            print("Improved:",item["improved"])          
            print()              
        return data
    except json.JSONDecodeError:
        print("Could not parse JSON. Raw response:")
        print(raw)
        return []
    
rewrite_bullets(bullets)

# These bullets are weak because they do not explain how the employee helped, 
# what kind of reports, and what kind of project he was working on. The model 
# suggested adding information about how he helped the company through 
# each review using descriptive references.


# Task 3 Cover letter generator

job_title = "Junior Data Engineer"
background = "Five years of experience as a middle school math teacher; recently completed \
a Python course and built data pipelines using Prefect and Pandas."

def generate_cover_letter(job_title: str, background: str) -> str:
    prompt = f"""
    You write strong cover letter opening paragraphs for career changers.
    The paragraph should be 3-5 sentences: confident, specific, and free of clichés.

    Here are two examples of the style and tone you should match:

    Example 1:
    Role: Data Analyst at a healthcare nonprofit
    Background: Seven years as a registered nurse, recently completed a data analytics bootcamp.
    Opening: After seven years as a registered nurse, I've spent my career making decisions
    under pressure using incomplete information — which turns out to be excellent training for
    data analysis. I recently completed a data analytics program where I built dashboards
    tracking patient outcomes across departments. I'm excited to bring that combination of
    clinical context and technical skill to [Company]'s mission-driven work.

    Example 2:
    Role: Junior Software Engineer at a fintech startup
    Background: Ten years in retail banking operations, self-taught Python developer for two years.
    Opening: I spent a decade on the operations side of banking, watching technology decisions
    get made by people who had never processed a wire transfer or resolved a failed ACH batch.
    That frustration turned into curiosity, and two years of self-teaching Python later, I'm
    ready to be on the other side of those decisions. I'm applying to [Company] because your
    work on payment infrastructure is exactly where my domain expertise and new technical skills
    intersect.

    Now write an opening paragraph for this person:
    Role: {job_title}
    Background: {background}
    Opening:
    """

    messages = [{"role": "user", "content": prompt}]
    # Your code here: call get_completion() and return the result
    
    raw = get_completion(messages)
    return raw

print(generate_cover_letter(job_title, background))


# I chose multiple examples because the few-shot method helps increase reliabilty in outputs as the
# model will reference the given examples.

# Task 4 moderation check 

def is_safe(text: str) -> bool:
    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text
    )
    flagged = result.results[0].flagged

    if flagged:
        print("I can't work with that phrasing. Could you rephrase your request?")
        return False
    return True


# Tests
print("Safe input:", is_safe("Can you help me rewrite my resume bullet points?"))
print("Flagged input:", is_safe("you're so useless i want to break you, set you on fire, and throw you at a government building "))