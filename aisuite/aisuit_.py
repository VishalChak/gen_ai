!pip install aisuite[all] -
!pip install llama-index llama-index-core llama-parse llama_index.embeddings.huggingface -q
!pip install llama-index-llms-anthropic -q

OPENAI_API_KEY = ""
CLAUDE_API_KEY = ""
LLAMAPARSE_API_KEY = ""

import os
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
os.environ['ANTHROPIC_API_KEY'] = CLAUDE_API_KEY

!wget "https://d18rn0p25nwr6d.cloudfront.net/CIK-0001018724/c7c14359-36fa-40c3-b3ca-5bf7f3fa0b96.pdf" -O amzn_2023_10k.pdf

from llama_parse import LlamaParse
import nest_asyncio;
nest_asyncio.apply()
pdf_name = "amzn_2023_10k.pdf"
parser = LlamaParse(api_key = LLAMAPARSE_API_KEY, result_type = "markdown", gpt4o_mode = False)
documents = parser.load_data(pdf_name)

print("Documents:" , len(documents))

### sentenceSplitter###
from llama_index.core.node_parser import SentenceSplitter
splitter = SentenceSplitter(chunk_size = 1024)
nodes = splitter.get_nodes_from_documents(documents)
print("Nodes: " , len(nodes))


### Vector index###
from llama_index.core import VectorStoreIndex
embed_model = "local:BAAI/bge-small-en-v1.5"
vector_index = VectorStoreIndex(nodes, embed_model = embed_model)

from llama_index.core.indices.vector_store.retrievers import VectorIndexRetriever
from llama_index.core.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index.core import get_response_synthesizer


# retriever = VectorIndexRetriever(
#     index = vector_index,
#     similarrity_top_k = 3,
# )
# ## bulld query engine##

# query_engine = RetrieverQueryEngine(retriever = retriever)
# response = query_engine.query("What was the net income in 2003")


 def get_context(query_engine,user_query):
     # query_engine = RetrieverQueryEngine(
     #
     retriever=retriever, response_synthesizer=get_response_synthesizer()
     # )
     response = query_engine.query(user_query)
     context=""
     for node in response.source_nodes:
         context += node.text
         context += "\n" + "="*100 + "\n"
     return context



def create_messages(query_engine, user_query, system_content = "You are an expert in financial analysis."):
    context = get_context(query_engine,user_query)
    messages = [{"role": "system", "content": system_content},]
    user_content = f"""Based on the following context, answer the question:\n\nContext:{context}\n\nQuestion:{user_query}\n\nAnswer:"""
    messages.append( {"role": "user", "content": user_content} )
    return messages

def call_llm_aisuite(llms, user_query):
    messages = create_messages(query_engine, user_query)
    for llm_aisuite in llms:
        print(f"LLM: {llm_aisuite}")
        response = client.chat.completions.create(model=llm_aisuite,messages=messages)
        print(response.choices[0].message.content)
        print("\n\n")



# import aisuite as ai
# import time
# user_query= "What was the total lease cost for the year ended December 31, 2023?"
# client = ai.Client()
# llms = [
#     "anthropic:claude-3-5-sonnet-20241022",
#      "openai:gpt-4o",
# ]
# call_llm_aisuite(llms, user_query)