import os
import utils
import streamlit as st
from streaming import StreamHandler

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_text_splitters import RecursiveCharacterTextSplitter


st.set_page_config(page_title="ChatPDF", page_icon="📄")
st.header("Chat with your documents (RAG Fusion)")
st.write(
    "Has access to custom documents and can respond to user queries by referring to the content within those documents"
)
st.write(
    "[![view source code ](https://img.shields.io/badge/view_source_code-gray?logo=github)](https://github.com/shashankdeshpande/langchain-chatbot/blob/master/pages/4_%F0%9F%93%84_chat_with_your_documents.py)"
)

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    HumanMessagePromptTemplate,
)
from operator import itemgetter
from typing import Dict, List, Optional, Sequence, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
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


class CustomDocChatbot:

    def __init__(self):
        utils.sync_st_session()
        self.llm = utils.configure_llm()
        self.embedding_model = utils.configure_embedding_model()

    def save_file(self, file):
        folder = "tmp"
        if not os.path.exists(folder):
            os.makedirs(folder)

        file_path = f"./{folder}/{file.name}"
        with open(file_path, "wb") as f:
            f.write(file.getvalue())
        return file_path

    @st.spinner("Analyzing documents..")
    def setup_qa_chain(self, uploaded_files):
        # Load documents
        docs = []
        for file in uploaded_files:
            file_path = self.save_file(file)
            loader = PyPDFLoader(file_path)
            docs.extend(loader.load())

        # Split documents and store in vector db
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)
        vectordb = DocArrayInMemorySearch.from_documents(splits, self.embedding_model)

        # Define retriever
        retriever = vectordb.as_retriever(
            search_type="mmr", search_kwargs={"k": 2, "fetch_k": 4}
        )

        # Setup memory for contextual conversation
        memory = ConversationBufferMemory(
            memory_key="chat_history", output_key="answer", return_messages=True
        )

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
                    converted_chat_history.append(
                        HumanMessage(content=message["human"])
                    )
                if message.get("ai") is not None:
                    converted_chat_history.append(AIMessage(content=message["ai"]))
            return converted_chat_history

        _inputs = RunnableParallel(
            {
                "question": lambda x: x["question"],
                # "chat_history": lambda x: _format_chat_history(x["chat_history"]),
                "context": RunnableLambda(itemgetter("question")) | get_results,
            }
        ).with_types(input_type=ChatRequest)
        # Prompt
        template = """Từ câu hỏi, thông tin tổ chức và văn bản sau đây, hãy trả lời câu hỏi dựa trên \nQuestion: {question}\nContext: {context}\n
        Đưa ra câu trả lời liên quan nhất đến thông tin câu hỏi và context 
        ---\nOutput:"""

        prompt = ChatPromptTemplate.from_template(template)
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        final_rag_chain = _inputs | prompt | llm | StrOutputParser()
        return final_rag_chain

    @utils.enable_chat_history
    def main(self):

        # User Inputs
        uploaded_files = st.sidebar.file_uploader(
            label="Upload PDF files", type=["pdf"], accept_multiple_files=True
        )
        if not uploaded_files:
            st.error("Please upload PDF documents to continue!")
            st.stop()

        user_query = st.chat_input(placeholder="Ask me anything!")

        if uploaded_files and user_query:
            qa_chain = self.setup_qa_chain(uploaded_files)

            utils.display_msg(user_query, "user")

            with st.chat_message("assistant"):
                st_cb = StreamHandler(st.empty())
                result = qa_chain.invoke(
                    {"question": user_query}, {"callbacks": [st_cb]}
                )
                # response = result["answer"]
                response = result
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )
                utils.print_qa(CustomDocChatbot, user_query, response)

                # to show references
                for idx, doc in enumerate(result["source_documents"], 1):
                    filename = os.path.basename(doc.metadata["source"])
                    page_num = doc.metadata["page"]
                    ref_title = (
                        f":blue[Reference {idx}: *{filename} - page.{page_num}*]"
                    )
                    with st.popover(ref_title):
                        st.caption(doc.page_content)


if __name__ == "__main__":
    obj = CustomDocChatbot()
    obj.main()
