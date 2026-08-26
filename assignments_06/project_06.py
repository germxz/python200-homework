from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

import logging

logging.getLogger("httpx").setLevel(logging.WARNING)


# step 1

docs_dir = Path("groundwork_docs")
assert docs_dir.exists(), f"Document directory not found: {docs_dir}"

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")


# Step 2
documents= SimpleDirectoryReader(str(docs_dir)).load_data()

print(f"Loaded {len(documents)} documents")
for doc in documents:
    print(f"-  {doc.metadata['file_name']}")
    
    
 # step 3
    
index= VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine(similarity_top_k=3)
print("Index built successfully. Ready to answer questions.")

#Step 4 


questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]

for q in questions:
    print(f"Question: {q}")
    response = query_engine.query(q)
    print(f"Answer: {response}")
    
    top_node = response.source_nodes[0]
    
    print(f"Document Name:{top_node.node.metadata['file_name']} ")
    print(f"Similarity Score: {top_node.score}")
    print(f" Text Snippet: {top_node.get_content()[:200]}...")
    print("-"*30)
        
# After running all 5 queries, most responses were confident and accurate - the answers
# were concise and the retrieved document matched what the question was actually asking
# about. The one exception was the dairy-free milk question: it pulled from
# seasonal_specials.txt (a limited-time lemonade description) instead of a more
# complete source like the menu or FAQ, and gave a vague, generalized answer without
# citing which options existed. What surprised me was that the model still sounded
# fully confident even though the answer was incomplete and weakly grounded. When I increased the 
# snippet word count, I saw that the documentation does refer to dairy free options but it 
# Doesn't describe which drinks are dairy free.


vague_q = "What drink would you recommend for someone who just got out of a rough breakup?"

query_engine = index.as_query_engine(similarity_top_k=3)
response = query_engine.query(vague_q)

print(f"Question: {vague_q}")
print(f"Answer: {response}")

for node in response.source_nodes:
    print(f"Document Name: {node.node.metadata['file_name']}")
    print(f"Similarity Score: {node.score:.4f}")
    print(f"Text Snippet: {node.node.get_content()[:200]}...")
    print("-" * 30)
    
    
    
# I asked what drinks does it recommend after a breakup. I supposed this would be hard because it involves human emotions
# and recommendations, and they're likely not accounted for in our groundwork_docs index.\
#It seems like the problem was that the output chunks only had a similarity score range of 0.70 to 0.7331 which is low compared to more correct 
#outputs with scores of 0.77 and up. At this threshold, the LLM will still try to generate an answer that to pertain to recommendations.
# The output was simply the name of the drink and nothing more. No explanation as to why that drink was chosen. 
# What I would change is rasising the similarity score threshold so lower confidence answers aren't answered so confidently. Perhaps a comment saying "I can't give you a confident answer"
# when it is below the new threshold.




#Step 6: reflection

