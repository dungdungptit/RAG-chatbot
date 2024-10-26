import streamlit as st

st.set_page_config(page_title="Langchain Chatbot", page_icon="💬", layout="wide")

st.header("Chatbot Implementations with Langchain")
st.write(
    """
[![view source code ](https://img.shields.io/badge/GitHub%20Repository-gray?logo=github)](https://github.com/shashankdeshpande/langchain-chatbot)
[![linkedin ](https://img.shields.io/badge/Shashank%20Deshpande-blue?logo=linkedin&color=gray)](https://www.linkedin.com/in/shashank-deshpande/)
![Visitors](https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Flangchain-chatbot.streamlit.app&label=Visitors&labelColor=%235d5d5d&countColor=%231e7ebf&style=flat)
"""
)

st.image("./assets/RAG.png", caption="RAG methods")

st.write(
    """
# RAG (RAG)
"""
)
st.image("./assets/rag_indexing.png", caption="RAG indexing")
st.image("./assets/rag_generation.png", caption="RAG generation")

st.write(
    """
# RAG Fusion (RAG Fusion)
"""
)
st.image("./assets/rag_fusion.png", caption="RAG Fusion")

st.write(
    """
# Rank RAG (Rank RAG)
"""
)
st.image("./assets/rank_rag.png", caption="Rank RAG")

st.write(
    """
# Corrective RAG (CRAG)

Corrective-RAG (CRAG) is a strategy for RAG that incorperates self-reflection / self-grading on retrieved documents. 

In the paper [here](https://arxiv.org/pdf/2401.15884.pdf), a few steps are taken:

* If at least one document exceeds the threshold for relevance, then it proceeds to generation
* Before generation, it performns knowledge refinement
* This paritions the document into "knowledge strips"
* It grades each strip, and filters our irrelevant ones
* If all documents fall below the relevance threshold or if the grader is unsure, then the framework seeks an additional datasource
* It will use web search to supplement retrieval
 
We will implement some of these ideas from scratch using [LangGraph](https://python.langchain.com/docs/langgraph):

* Let's skip the knowledge refinement phase as a first pass. This can be added back as a node, if desired. 
* If *any* documents are irrelevant, let's opt to supplement retrieval with web search. 
* We'll use [Tavily Search](https://python.langchain.com/docs/integrations/tools/tavily_search) for web search.
* Let's use query re-writing to optimize the query for web search.

"""
)

st.image("./assets/crag.png", caption="RAG CRAG")

st.write(
    """
# Self RAG

Self-RAG is a strategy for RAG that incorperates self-reflection / self-grading on retrieved documents and generations. 

In the [paper](https://arxiv.org/abs/2310.11511), a few decisions are made:

1. Should I retrieve from retriever, `R` -

* Input: `x (question)` OR `x (question)`, `y (generation)`
* Decides when to retrieve `D` chunks with `R`
* Output: `yes, no, continue`

2. Are the retrieved passages `D` relevant to the question `x` -

* * Input: (`x (question)`, `d (chunk)`) for `d` in `D`
* `d` provides useful information to solve `x`
* Output: `relevant, irrelevant`

3. Are the LLM generation from each chunk in `D` is relevant to the chunk (hallucinations, etc)  -

* Input: `x (question)`, `d (chunk)`,  `y (generation)` for `d` in `D`
* All of the verification-worthy statements in `y (generation)` are supported by `d`
* Output: `{fully supported, partially supported, no support`

4. The LLM generation from each chunk in `D` is a useful response to `x (question)` -

* Input: `x (question)`, `y (generation)` for `d` in `D`
* `y (generation)` is a useful response to `x (question)`.
* Output: `{5, 4, 3, 2, 1}`

We will implement some of these ideas from scratch using [LangGraph](https://python.langchain.com/docs/langgraph).

"""
)

st.image("./assets/self_rag.png", caption="RAG Self-RAG")


st.write(
    """
Langchain is a powerful framework designed to streamline the development of applications using Language Models (LLMs). It provides a comprehensive integration of various components, simplifying the process of assembling them to create robust applications.

Leveraging the power of Langchain, the creation of chatbots becomes effortless. Here are a few examples of chatbot implementations catering to different use cases:

- **Basic Chatbot**: Engage in interactive conversations with the LLM.
- **Context aware chatbot**: A chatbot that remembers previous conversations and provides responses accordingly.
- **Chatbot with Internet Access**: An internet-enabled chatbot capable of answering user queries about recent events.
- **Chat with your documents**: Empower the chatbot with the ability to access custom documents, enabling it to provide answers to user queries based on the referenced information.
- **Chat with SQL database**: Enable the chatbot to interact with a SQL database through simple, conversational commands.
- **Chat with Websites**: Enable the chatbot to interact with website contents.

To explore sample usage of each chatbot, please navigate to the corresponding chatbot section.
"""
)
