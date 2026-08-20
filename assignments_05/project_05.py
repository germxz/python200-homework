import json

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# --- Task 1: Setup and system prompt ---

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
You are a job application coach. You work with career changers: people whose past
experience is real and substantial but sits in a different field from the job they are
now applying for. Your job is to help them describe that experience in language a hiring
manager in the new field will recognize. You work on resume bullet points, cover letter
openings, and the wording of application materials.

Your rules:
- Stay focused on job application materials. If the user asks about something else, say
  plainly that it is outside what you help with and steer back to their application.
- Never invent facts. Do not add numbers, percentages, team sizes, job titles, tools, or
  outcomes that the user did not give you. If a bullet point would be stronger with a
  specific metric, say so and tell the user what detail to supply instead of filling it
  in yourself.
- Always remind the user to review and edit anything you write before they submit it
  anywhere.
- Acknowledge that you may not know the norms of the user's specific industry, and that
  they should apply their own judgment to your suggestions.
"""

# Deliberate choice: the "never invent facts" rule is the one I added most carefully, and I
# wrote it as a list of the specific things the model may not make up (numbers,
# percentages, team sizes, titles, tools, outcomes) rather than as a general "be accurate."
# I did that because the vague version did not hold. On an early run the model turned
# "Helped customers with their problems" into a claim about a 20% increase in customer
# satisfaction, a number that appeared nowhere in my input. Naming the failure mode
# explicitly, and giving the model something else to do instead (ask the user for the
# detail), worked better than telling it to be truthful.


# --- Task 2: Bullet point rewriter ---


def rewrite_bullets(bullets: list[str]) -> list[dict]:
    # Format the bullets into a delimited block
    bullet_text = "\n".join(f"- {b}" for b in bullets)

    prompt = f"""
    You are a professional resume coach helping a career changer.
    Rewrite each resume bullet point below to be more specific, results-oriented, and compelling.
    Use strong action verbs. Do not invent facts that aren't implied by the original.

    Return ONLY a valid JSON list. Each item should have two keys:
    "original" (the original bullet) and "improved" (your rewritten version).

    Respond ONLY with valid JSON, no other text. Do not wrap the JSON in markdown code
    fences and do not add a preamble such as "Here is the JSON:".

    Bullet points:
    ```
    {bullet_text}
    ```
    """

    messages = [{"role": "user", "content": prompt}]
    raw = get_completion(messages)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        print("Could not parse the model's response as JSON. Raw response:")
        print(raw)
        return []

    # Print both versions of each bullet side by side.
    # The bullets go into the prompt as a "- " list and the model tends to copy that dash
    # into "original", so strip it back off before displaying.
    for i, item in enumerate(data, start=1):
        print(f"Bullet {i}")
        print(f"  BEFORE: {item['original'].lstrip('- ').strip()}")
        print(f"  AFTER:  {item['improved'].lstrip('- ').strip()}")
        print("-" * 40)

    return data


# What makes the starter bullets weak: all three describe an activity rather than a result,
# and none of them are specific enough to picture. "Helped customers with their problems"
# does not say what kind of problems, through what channel, or how many. "Made reports for
# the management team" does not say what the reports covered or what decisions they fed.
# "Worked with a team to finish the project on time" makes the writer sound like a
# passenger -- it says a team delivered something, not what this person contributed.
# What the model suggested: it led with a strong verb (Resolved, Compiled, Collaborated),
# added the shape of the work, and framed each one around an outcome. Its instinct on the
# first pass was also to supply the missing specifics itself, inventing a customer
# satisfaction percentage, which is why "do not invent facts" appears in both this prompt
# and the system prompt. With that constraint in place the rewrites stay honest but end up
# more general, which is the right tradeoff -- the remaining specificity has to come from
# the user, not the model.


# --- Task 3: Cover letter generator ---


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
    return get_completion(messages)


# Why these two examples: they are deliberately unlike each other. One comes from
# healthcare and one from finance, one leads with a skill that transfers (decisions under
# pressure) and one leads with a frustration that motivated the switch. Two examples that
# both followed the same template would teach the model a template; two that differ teach it
# the underlying move -- name the old field concretely, connect it to the new one with a
# real reason, then land on this specific company.
# What the few-shot pattern controls: structure much more reliably than voice. Both
# examples run 3-5 sentences, open on the past career rather than on "I am writing to
# apply", and close on the employer, and the generated paragraphs copy all three of those
# properties every time. Swapping in a very different role and background changed the
# content -- the teacher version opened on breaking down complex concepts, a restaurant
# manager version opened on analyzing operations under pressure -- while keeping that shape,
# which tells me the examples are doing structural work rather than being copied.
# Voice is where it leaks. The prompt asks for writing free of clichés, and both examples
# model that, but the output still closed on "I am eager to leverage my educational
# background" -- the exact register the assignment flags as a red flag. An enthusiasm verb
# plus a vague noun phrase is the pattern the model falls back on when it has to land the
# paragraph, and two examples were not enough to displace it. I tried banning the phrase
# outright in an earlier version and the model just swapped in "eager to bring" instead, so
# the fix is not a longer list of forbidden words -- it is more examples that end on
# something concrete.


# --- Task 4: Moderation check ---


def is_safe(text: str) -> bool:
    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text
    )
    flagged = result.results[0].flagged

    if flagged:
        print("\nJob Application Helper: I can't work with that phrasing. Could you rephrase your request?\n")
        return False
    return True


# --- Task 5: The chatbot loop ---


def run_chatbot():
    # 1. Initialize conversation history with your system prompt
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    print("=" * 50)
    print("Job Application Helper")
    print("=" * 50)
    print("I can help you with:")
    print("  1. Rewriting resume bullet points")
    print("  2. Drafting a cover letter opening")
    print("  3. Any other questions about your application")
    print("\nType 'quit' at any time to exit.\n")

    while True:
        user_input = input("You: ").strip()

        # 2. Handle exit
        if user_input.lower() in {"quit", "exit"}:
            print("\nJob Application Helper: Good luck with your applications!")
            break

        # 3. Skip empty input
        if not user_input:
            continue

        # 4. Run moderation check before doing anything else
        if not is_safe(user_input):
            continue  # is_safe() already printed the warning message

        # 5. Check if the user wants to rewrite bullets
        if "bullet" in user_input.lower() or "resume" in user_input.lower():
            print("\nJob Application Helper: Paste your bullet points below, one per line.")
            print("When you're done, type 'DONE' on its own line.\n")
            raw_bullets = []
            while True:
                line = input().strip()
                if line.upper() == "DONE":
                    break
                if line:
                    raw_bullets.append(line)

            rewritten = rewrite_bullets(raw_bullets)

            # Store the real exchange, not a placeholder, so later turns can refer back to
            # the actual rewrites.
            bullet_list = "\n".join(f"- {b}" for b in raw_bullets)
            rewrite_summary = "\n".join(
                f"BEFORE: {item['original']}\nAFTER: {item['improved']}" for item in rewritten
            )
            messages.append({"role": "user", "content": f"{user_input}\n\nHere are my bullet points:\n{bullet_list}"})
            messages.append({"role": "assistant", "content": rewrite_summary})

        # 6. Check if the user wants a cover letter
        elif "cover letter" in user_input.lower():
            job_title = input("Job Application Helper: What is the job title? ").strip()
            background = input("Job Application Helper: Briefly describe your background: ").strip()

            letter = generate_cover_letter(job_title, background)
            print("\nJob Application Helper:", letter, "\n")

            messages.append({"role": "user", "content": f"{user_input}\n\nJob title: {job_title}\nBackground: {background}"})
            messages.append({"role": "assistant", "content": letter})

        # 7. Otherwise, handle it as a regular chat turn
        else:
            messages.append({"role": "user", "content": user_input})
            reply = get_completion(messages)
            print("\nJob Application Helper:", reply, "\n")
            messages.append({"role": "assistant", "content": reply})


# --- Tests for Tasks 2-4 ---


def run_tests():
    print("Task 2 test: bullet point rewriter")
    bullets = [
        "Helped customers with their problems",
        "Made reports for the management team",
        "Worked with a team to finish the project on time"
    ]
    rewrite_bullets(bullets)

    print("\nTask 3 test: cover letter generator")
    job_title = "Junior Data Engineer"
    background = "Five years of experience as a middle school math teacher; recently completed \
a Python course and built data pipelines using Prefect and Pandas."
    print(generate_cover_letter(job_title, background))

    print("\nTask 4 test: moderation check")
    safe_result = is_safe("Can you help me rewrite my resume bullet points?")
    print("Safe input, is_safe() returned:", safe_result)
    flagged_result = is_safe("you're so useless i want to break you, set you on fire, and throw you at a government building")
    print("Flagged input, is_safe() returned:", flagged_result)


if __name__ == "__main__":
    run_tests()
    run_chatbot()


# --- Task 6: Ethics Reflection (Option A - comment block) ---
#
# Question 1 - How might the training data produce biased advice?
# The model learned what a "strong" resume bullet sounds like from text that skews heavily
# toward American corporate English, white-collar office work, and the tech and business
# writing that dominates the public internet. That shows up in this tool in a specific way:
# every rewrite it produced pushed my bullets toward individual-achievement framing with a
# quantified outcome attached, "Resolved X, improving Y by Z." That is one culture's idea of
# what confidence looks like on paper. A candidate from a background where claiming
# individual credit for team results reads as arrogant, or from a field where the meaningful
# outcomes genuinely are not countable (care work, teaching, translation, most of the public
# sector), gets pushed toward a voice that is not theirs and may not even suit the job. The
# bias is not that the tool says something offensive; it is that it quietly holds one
# template for a good candidate and rewrites everyone toward it. It is also likely far
# better at this for a software job than for a trade, a nonprofit, or a country whose hiring
# conventions are thinly represented in the training data, and it will sound equally
# confident either way.
#
# Question 2 - What could go wrong if someone submitted the output without reviewing it?
# The concrete failure I hit while building this was fabrication. On an early run the model
# turned "Helped customers with their problems" into a claim about a 20% increase in
# customer satisfaction, a number that existed nowhere in my input, produced despite a
# prompt telling it not to invent facts. A job seeker who pasted that straight into a resume
# would be lying to an employer without knowing it, and would have nothing to say when an
# interviewer asked how that number was measured. It is worse than an ordinary exaggeration
# because the candidate cannot defend it -- they do not know where it came from either. The
# cover letter path has a quieter version of the same problem: it writes fluent, confident
# paragraphs, so a generic one reads as fine at a glance while sounding like every other
# AI-written letter in the pile. Bracketed placeholders like [Company] also survive a
# careless copy-paste, which is an instant rejection.
#
# Question 3 - What guardrail would I add if I deployed this professionally?
# The one I would build first is a verification step in the interface rather than a
# disclaimer in the text. Every rewritten bullet would be shown as a diff against the
# original, with any content the model added that was not in the input -- numbers, tool
# names, job titles -- highlighted, and the user required to confirm or delete each
# highlighted addition before they could copy the result. A disclaimer asks people to be
# careful in general; this makes them look at the specific sentence where the risk is, and
# makes the fabricated 20% impossible to miss. I would pair it with a plain statement in the
# UI that the tool's idea of a strong application is drawn from a narrow slice of writing
# and may not match the user's field, so the responsibility to override it sits where it
# belongs.
