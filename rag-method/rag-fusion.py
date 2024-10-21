# Import libraries
import os
from operator import itemgetter
from typing import Dict, List, Optional, Sequence, Tuple

# import weaviate
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatCohere
from langchain_community.vectorstores import Weaviate
from langchain_core.documents import Document
from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import (
    ConfigurableField,
    Runnable,
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
    RunnableSequence,
    chain,
)

# from ingest import get_retriever, get_intent_retriever

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

from langchain_fireworks import ChatFireworks
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langsmith import Client

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

import re
import string
import json

import requests

# from search_fqa import searchSimilarity, encode
import pandas as pd

# import torch


from getpass import getpass
import os

# from langchain.llms import PromptLayerOpenAI

# PROMPTLAYER_API_KEY = getpass()
# os.environ["PROMPTLAYER_API_KEY"] = PROMPTLAYER_API_KEY

# load openai api key from file .env
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI

# LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

from langchain.prompts import ChatPromptTemplate

""" 
--
Nếu câu trả lời có link ảnh (type: image) hoặc link đường dẫn, hãy sử dụng dấu gạch chân (_) theo cú pháp [ảnh](url) để thể hiện link ảnh hoặc [đường dẫn](url) link đường dẫn.
--
 """

# Prompt
template = """Từ câu hỏi, thông tin tổ chức và văn bản sau đây, hãy trả lời câu hỏi dựa trên \nQuestion: {question}\nContext: {context}\n
Đưa ra câu trả lời liên quan nhất đến thông tin câu hỏi và context 
---\nOutput:"""

prompt = ChatPromptTemplate.from_template(template)

# get index
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

embed_model = OpenAIEmbeddings()
vectorstore = FAISS.load_local(
    "./Chatbot_17102024",
    OpenAIEmbeddings(),
    allow_dangerous_deserialization=True,
)

retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 15,
        # "threshold": 0.5
    }
)

template_vietnamese_fusion = """Bạn là một tư vấn viên chuyên nghiệp và là người giải quyết vấn đề, được giao nhiệm vụ trả lời bất kỳ câu hỏi nào \
về các thông tin về các standarzation.
Bạn có thể tạo ra nhiều truy vấn tìm kiếm dựa trên một truy vấn đầu vào duy nhất. \n
Tạo ra nhiều truy vấn tìm kiếm liên quan đến: {question} \n
Lưu ý đầu ra trả về các truy vấn tiếng Anh nhé
Đầu ra (3 truy vấn tiếng Anh):"""

prompt_rag_fusion = ChatPromptTemplate.from_template(template_vietnamese_fusion)

generate_queries = (
    prompt_rag_fusion
    | ChatOpenAI(temperature=0)
    | StrOutputParser()
    | (lambda x: x.split("\n"))
)

translate_prompt_template = """Bạn là 1 người giỏi tiếng anh và các ngôn ngữ khác bao gồm cả tiếng Việt. Với đầu vào sau đây: \n{question}Nếu câu đầu vào là tiếng Việt, hãy dịch trả về là tiếng Anh.
Nếu câu đầu vào là tiếng Anh, hãy trả về là tiếng Anh.
Đầu ra (Tiếng Anh):"""

translate_prompt = ChatPromptTemplate.from_template(translate_prompt_template)

translate_query = translate_prompt | ChatOpenAI(temperature=0) | StrOutputParser()

from langchain.load import dumps, loads


def reciprocal_rank_fusion(results: list[list], k=60):
    """Reciprocal_rank_fusion that takes multiple lists of ranked documents
    and an optional parameter k used in the RRF formula"""

    # Initialize a dictionary to hold fused scores for each unique document
    fused_scores = {}

    # Iterate through each list of ranked documents
    for docs in results:
        # Iterate through each document in the list, with its rank (position in the list)
        # print(docs)
        for rank, doc in enumerate(docs):
            # print(rank, doc)
            # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
            doc_str = dumps(doc)
            # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Retrieve the current score of the document, if any
            previous_score = fused_scores[doc_str]
            # Update the score of the document using the RRF formula: 1 / (rank + k).
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort the documents based on their fused scores in descending order to get the final reranked results
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # Return the reranked results as a list of tuples, each containing the document and its fused score
    return reranked_results


def get_results(question: str):
    docs = retrieval_chain_rag_fusion.invoke({"question": question})
    docs1 = retriever.get_relevant_documents(question)
    docs.append(docs1)
    docs = reciprocal_rank_fusion(docs)
    return docs1


retrieval_chain_rag_fusion = generate_queries | retriever.map()


class ChatRequest(BaseModel):
    question: str
    chat_history: Optional[List[Dict[str, str]]]


def _format_chat_history(chat_history: List[Dict[str, str]]) -> List:
    converted_chat_history = []
    for message in chat_history:
        if message.get("human") is not None:
            converted_chat_history.append(HumanMessage(content=message["human"]))
        if message.get("ai") is not None:
            converted_chat_history.append(AIMessage(content=message["ai"]))
    return converted_chat_history


_inputs = RunnableParallel(
    {
        "question": lambda x: x["question"],
        "organization": itemgetter("organization"),
        "chat_history": lambda x: _format_chat_history(x["chat_history"]),
        "context": RunnableLambda(itemgetter("question")) | get_results,
    }
).with_types(input_type=ChatRequest)

final_rag_chain = _inputs | prompt | llm | StrOutputParser()
