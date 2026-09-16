from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
import os

import logging  
logging.getLogger("pypdf").setLevel(logging.ERROR) # remove logs 

from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
from llama_index.llms.openai import OpenAI



if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")
    
    
# --- RAG Concepts ---

#Concepts Q1
#SCENARIO A: RAG. The policy library is hundreds of PDFs that change every quarter, and the assistant has
# to answer from whatever the current versions say. RAG re-reads the live document store at query time, so
# each quarter's updates are picked up automatically without retraining. Prompt engineering can't hold that
# much text, and fine-tuning would bake in stale policies that go out of date every three months.

#SCENARIO B: Fine-tuning. The startup has 3,000 in-house examples of a specific brand voice that barely
# appears online, so the goal is to change how the model writes, not to give it facts to look up. Training on
# those examples bakes the style into the model itself; RAG and prompt engineering can supply content but
# can't reliably reshape the voice the way fine-tuning on the examples does.


#SCENARIO C: Prompt engineering. You only need to ask an LLM a few questions over a single report, so you
# can paste the report straight into the prompt and ask (one-shot or few-shot). RAG is overkill here: it
# means building a database and embeddings for what already fits in one prompt. Fine-tuning is wrong too:
# it is expensive, needs a training dataset, and bakes in knowledge you only need once, not repeatedly.

#Concepts Q2

#   Why is a confidently wrong answer more harmful than one that says "I am not sure"? Give one example of a real situation where
#    a confident hallucination could cause harm

# A confident wrong answer goes unchecked and gets built on: you keep stacking assumptions and estimates
# on top of one bad fact. You can get a long way into a research project before realizing a statistic was
# hallucinated, then have to backtrack and redo the work. A model that says "I'm not sure" flags the gap up
# front, so you can double-check or find another source before it poisons everything downstream.
#
# Real example: a paralegal asks an AI for case citations for a court filing. A confident hallucination
# invents real-sounding but fake cases; if it had said "I'm not sure," the paralegal would have looked them
# up manually. Lawyers have actually been sanctioned for filing fabricated citations.
#
# Tone matters too: a confident tone signals reliability, so users lower their guard and skip verifying.
# The same wrong fact does far more damage wrapped in confident language than in a hedge.

#Concepts Q3

# steps = [
#     "Extract text from source documents",       # read the raw text out of each source file
#     "Split text into chunks",                   # break long docs into smaller passages
#     "Convert text chunks into embeddings",      # turn each chunk into a vector of its meaning
#     "Receive the user's query",                 # the user asks a question
#     "Embed the user's query",                   # turn the query into a vector the same way
#     "Retrieve the most relevant chunks",        # find the chunks whose vectors are closest to the query
#     "Inject retrieved chunks into the prompt",  # paste those chunks into the prompt as context
#     "Generate a response from the LLM",         # the LLM answers using the injected context
# ]
# The first three steps build the index once, ahead of time; the last five run on every query.

# --- Keyword RAG ---


import string
from pathlib import Path

def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]

#Keyword Q1:


query = "What are your hours on weekends?"

documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}

result = simple_keyword_retrieval(query, documents= documents, verbose=True)
best_result_name = result[0][0]
print(f"Name of document: {best_result_name} ")

# Selected document: loyalty.txt (printed above). This is a three-way tie at overlap=1:
#   hours.txt matches "weekends", hiring.txt matches "your", loyalty.txt matches "your".
# "your" is not in the stopword list, so hiring.txt and loyalty.txt tie with the actually-correct hours.txt.
# scores.sort(reverse=True) breaks the tie by name in reverse-alphabetical order, so "loyalty.txt" wins.
# Adding "your" to the stopwords drops hiring/loyalty to 0 and leaves hours.txt as the clear match.



#Keyword Q2:

query = "Do you have anything without caffeine?"

result2 = simple_keyword_retrieval(query=query,documents=documents, verbose=True)
print(f"Selected document: {result2[0][0]}")



# Selected document: None found - all four scored 0 overlap, so keyword RAG returned no match (got it wrong).

# Keyword RAG got this wrong: menu.txt is actually the relevant document
# (it lists espresso, lattes, cold brew, and oat/almond milk), but none of
# the exact words in the query ("caffeine", "without", "anything") show up
# in menu.txt's text, so the function couldn't find any overlap.

# Semantic RAG would do better here, since it compares the meaning  of the
# query and documents (via embeddings/cosine similarity) rather than exact
# word matches - it could recognize that "anything without caffeine" is
# still related to a menu of caffeinated drinks even without shared words.


#Keyword Q3:


query = "How do I sign up for rewards?"

# Prediction (before running): None found. None of the content words ("sign", "up", "rewards") appear
# literally in any document, and some query words ("how", "do", "i") are stopwords or don't match anyway.



result3= simple_keyword_retrieval(query= query, documents=documents, verbose=True)
print(result3)

# My prediction was correct and I wasn't too surprised. The model did not match any of the words within
# the documents hence, it came to no conclusion. This is because the model will look for exact
# word matches and none of these words match with any doc. If we want to have a more broad
# input availability, they should use semantic RAG to match words to their synonyms.


# ---Semantic RAG Concepts ---

#Semantic Q1

# 1. vector embeddings transform unstructured, "chaotic" data into ordered numerical
#    arrays (vectors).  These arrays are stored in a vector database, which organizes 
#    them in a multi-dimensional space where mathematical proximity equals semantic meaning. 

# 2. The score that is closest to 1 will be the more meaningful chunk. That means that the embedding 
#   of 0.85 has more similarities by word definition to the chunk than the 0.30 chunk.


# 3. Semantic RAG can find a chunk even if the exact word doesn't appear in documents because the model will use a neural
#   network to embed the word's meanings. This allows the retrieval system to compare it to similar embeddings and how similar they are by 
#   using cosine similarity.




#Semantic Q2

#       | Feature                 | Keyword RAG                    | Semantic RAG                          |
#       |-------------------------|--------------------------------|---------------------------------------|
#       | What is compared?       | Exact word overlap             | Vector embeddings of query and text   |
#       | What is retrieved?      | Full document                  | The most similar chunks               |
#       | Can it handle synonyms? | No                             | Yes (matches meaning, not exact words)|
#       | Storage format          | Plain-text dictionary          | Vector database                       |
#       | Relevance score         | Number of overlapping keywords | Cosine similarity between embeddings   |

# ---Llama Index ---
 


pdf_directory = Path(__file__).parent / "brightleaf_pdfs"
docs = SimpleDirectoryReader(input_dir=str(pdf_directory)).load_data()

#LLama Index  Q1

questions = [    "What employee benefits does BrightLeaf offer?",
    "What are BrightLeaf's security policies?",
]

index = VectorStoreIndex.from_documents(docs)

query_engine = index.as_query_engine(similarity_top_k=3)

for q in questions: 
    print(f"\nQ: {q}")
    response = query_engine.query(q)
    print("A:", response)
    
    for node_with_score in response.source_nodes:
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print(f"Text Snippet: {node_with_score.node.get_content()[:150]}...")
        print("-" * 30)



# LlamaIndex Q1 comments:
#
# Query 1 ("What employee benefits does BrightLeaf offer?"):
# - The retrieved chunks were relevant. The top chunk (score 0.9088) was the
#   actual intro of the benefits policy doc, so the highest-scoring match was
#   exactly right.
# - The response was confident and specific, not hedgy. It listed concrete
#   benefits (medical, vision, wellness, life/disability/retirement, parental
#   leave, professional development, mentorship) instead of saying something
#   vague like "based on the context" or "I'm not sure."
# - Something a little unexpected: the 2nd and 3rd chunks (security policy doc,
#   company overview/mission doc) still scored fairly high (~0.81) even though
#   neither is actually about benefits. They didn't end up messing up the final
#   answer, but it shows similarity_top_k=3 can still pull in some chunks that
#   aren't really relevant, just because they share similar corporate language.
#
# Query 2 ("What are BrightLeaf's security policies?"):
# - Retrieved chunks were relevant here too. The top chunk (0.8814) was pulled
#   directly from the security policy document.
# - The response was confident and detailed, listing specific policies (MFA,
#   VPN, 90-day credential rotation, encryption, NIST 800-61, ISO 27001, etc.)
#   with no hedging language.
# - Same pattern as query 1: the benefits doc and mission/overview doc chunks
#   showed up as the 2nd and 3rd matches even though they aren't about security.
#   My guess is all of BrightLeaf's docs share a similar corporate tone/style,
#   so they end up with a moderately high baseline similarity to almost any
#   BrightLeaf-related query, even when they're not the right document.


#LLama Index Q2


question = "What employee benefits does BrightLeaf offer?"

for k in [1, 5]:
    query_engine = index.as_query_engine(similarity_top_k=k)
    print(f"\n--- similarity_top_k={k} ---")

    response = query_engine.query(question)
    print(question)
    print(f"Answer: {response}")

    for node_with_score in response.source_nodes:
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print(f"Text Snippet: {node_with_score.node.get_content()[:150]}...")
        print("-" * 30)


# When comparing both outputs from k=[1,5] you can see that it did make a difference in
# output. K=5 cause the output to be more descriptive with its company programs. This was due to it exctracting 5 chunks 
# from the text. K=1 produced a very similar output but it wasn't as descriptive and in-depth as k=5. K=1 used the 
# same first chunk as k=5 but only the first one so it is expected to have a slightly more  generalized output. 
#
# Is more retrieved context always better? No. k=5 was more descriptive here, but pulling more chunks also
# pulls in less-relevant ones (the security/mission docs scored ~0.81 on a benefits query). More context
# means more tokens, higher cost, and more room for the model to drift onto tangential material. The right
# k depends on the question: broad questions benefit from more chunks, precise ones do better with fewer.




#LLama Index Q3
question1= "Does the company have an inclusive environment and do they have any discounts for national coffee day?"

query_engine = index.as_query_engine()

response1= query_engine.query(question1)

print(F"llama Q3: {question1}")
print(f"A: {response1}")

for node_with_score in response1.source_nodes:
    print(f"Similarity Score: {node_with_score.score:.4f}")
    print(f"Text Snippet: {node_with_score.node.get_content()[:150]}...")
    print("-" * 30)
    
# I expected an answer from the first sentence since companies usually mention their inclusivity practices.
# I didn't know what to expect from the second sentence asking about dscounts for national coffee day. It was
# a sentence not related to the company and it responded that there is no mention of "coffee day discounts". 
# I think the system handled this query well. It managed to acknowledge it's lack of understanding in that area
# because of absence of the business's documentation for any discounts. 

# If there's something I must change, it would be the simimlarity score threshold to have a consistent "I'm not sure" 
# output for when inputs start getting vague. This will ensure that the system doesn't try to answer 
#something it can't answer fully correctly.
#
# Failure mode, stated plainly: the retriever returns its best available chunks even when none are strongly
# relevant, and the LLM then answers confidently from weak context instead of refusing. Low similarity
# scores are the warning sign, but nothing in the default setup acts on them.


#LLama Index Q4

q = "What employee benefits does BrightLeaf offer?"
q1 = "When is the CEO's birthday?"


llm = OpenAI(model="gpt-4o-mini")
faithfulness = FaithfulnessEvaluator(llm=llm)
relevancy =  RelevancyEvaluator(llm=llm)

query_engine = index.as_query_engine(similarity_top_k=3)
response = query_engine.query(q)

print(f"\nQ: {q}")
print("A:", response)

faithfulness_result = faithfulness.evaluate_response(response=response)
relevancy_result = relevancy.evaluate_response(query=q, response=response)


print(f"Faithfulness score: {faithfulness_result.score}")
print(f"Relevancy Score: {relevancy_result.score}")
print("-" * 30)

# QUERY 2

query_engine = index.as_query_engine(similarity_top_k=3)
response1 = query_engine.query(q1)

print(f"\nQ: {q1}")
print("A:", response1)

faithfulness_result = faithfulness.evaluate_response(response=response1)
relevancy_result = relevancy.evaluate_response(query=q1, response=response1)


print(f"Faithfulness score: {faithfulness_result.score}")
print(f"Relevancy Score: {relevancy_result.score}")
print("-" * 30)


# [1] What faithfulness 1.0 vs 0.0 means:
# A faithfulness score of 1.0 means the response is fully supported by the retrieved
# source context - every claim in the answer can be traced back to the documents,
# with no fabricated or unsupported information. A score of 0.0 means the response
# contains claims that are NOT backed up by the retrieved context, whether that's
# because the model hallucinated an answer or, as in this case, because the response
# itself (an admission of not knowing) has nothing in the retrieved chunks to
# ground it against.


# [2] Relevancy vs faithfulness:
# Relevancy measures whether the response actually addresses the question that was
# asked. Faithfulness checks "is this response backed up by the source documents?"
# while relevancy checks "does this response answer the actual query?" A response
# could theoretically be faithful (accurately describing retrieved context) without
# being relevant (if that context doesn't answer the question), or vice versa.
#

# [3] Did the scores match expectations?
# Yes - Query 1 scored 1.0/1.0 and Query 2 scored 0.0/0.0. For Query 1, the
# benefits question, BrightLeaf's documents clearly contained relevant, detailed
# information, so the response was both grounded in the source material and
# directly answered the question. For Query 2, the CEO's birthday question, there
# was no such information in the documents. Interestingly, the model responded
# correctly by saying it didn't have that information rather than making something
# up - but the evaluators still scored both metrics as 0.0. This shows a limitation
# of these evaluators: an honest "I don't know" response, which is the *desired*
# behavior for an out-of-scope query, still scores as a failure because there's no
# supporting context to evaluate faithfulness against, and the non-answer isn't
# treated as "relevant" to the literal question asked.


# [4] What LLM-as-a-judge means and why simple metrics fall short:
# LLM-as-a-judge means using a separate LLM call to read the query, the response, 
# and the retrieved context, then assess qualities like faithfulness and relevancy that
# don't have a single "correct" string to match against. Simple accuracy metrics (like exact 
# match or keyword overlap) don't work well for RAG evaluation because responses are open-ended natural
# language - two answers can be worded completely differently but be equally correct, or worded
# similarly but differ in whether they're actually grounded in the source material.
# An LLM judge can reason about meaning and groundedness in a way that string-
# matching metrics cannot.