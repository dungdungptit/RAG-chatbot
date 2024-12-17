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
st.header("Chat with your documents (CRAG)")
st.write(
    "Has access to custom documents and can respond to user queries by referring to the content within those documents"
)
st.write(
    "[![view source code ](https://img.shields.io/badge/view_source_code-gray?logo=github)](https://github.com/shashankdeshpande/langchain-chatbot/blob/master/pages/4_%F0%9F%93%84_chat_with_your_documents.py)"
)

### Retrieval Grader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
# Data model
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )
# LLM with function call
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeDocuments)
# Prompt
system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader

### Generate
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
# Prompt
prompt = hub.pull("rlm/rag-prompt")
# LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
# Chain
rag_chain = prompt | llm | StrOutputParser()

### Question Re-writer
# LLM
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

# Prompt
system = """You a question re-writer that converts an input question to a better version that is optimized \n 
     for web search. Look at the input and try to reason about the underlying semantic intent / meaning."""
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Here is the initial question: \n\n {question} \n Formulate an improved question.",
        ),
    ]
)

question_rewriter = re_write_prompt | llm | StrOutputParser()

### Web Search Tool

from langchain_community.tools.tavily_search import TavilySearchResults

web_search_tool = TavilySearchResults(k=3)


### Now let's create our graph that will use CRAG
### Define Graph State
from typing import List
from typing_extensions import TypedDict

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """

    question: str
    generation: str
    web_search: str
    documents: List[str]






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

        # # Setup LLM and QA chain
        # qa_chain = ConversationalRetrievalChain.from_llm(
        #     llm=self.llm,
        #     retriever=retriever,
        #     memory=memory,
        #     return_source_documents=True,
        #     verbose=False,
        # )
        from langchain.schema import Document

        def retrieve(state):
            """
            Retrieve documents

            Args:
                state (dict): The current graph state

            Returns:
                state (dict): New key added to state, documents, that contains retrieved documents
            """
            print("---RETRIEVE---")
            question = state["question"]

            # Retrieval
            documents = retriever.get_relevant_documents(question)
            return {"documents": documents, "question": question}


        def generate(state):
            """
            Generate answer

            Args:
                state (dict): The current graph state

            Returns:
                state (dict): New key added to state, generation, that contains LLM generation
            """
            print("---GENERATE---")
            question = state["question"]
            documents = state["documents"]

            # RAG generation
            generation = rag_chain.invoke({"context": documents, "question": question})
            return {"documents": documents, "question": question, "generation": generation}


        def grade_documents(state):
            """
            Determines whether the retrieved documents are relevant to the question.

            Args:
                state (dict): The current graph state

            Returns:
                state (dict): Updates documents key with only filtered relevant documents
            """

            print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
            question = state["question"]
            documents = state["documents"]

            # Score each doc
            filtered_docs = []
            web_search = "No"
            for d in documents:
                score = retrieval_grader.invoke(
                    {"question": question, "document": d.page_content}
                )
                grade = score.binary_score
                if grade == "yes":
                    print("---GRADE: DOCUMENT RELEVANT---")
                    filtered_docs.append(d)
                else:
                    print("---GRADE: DOCUMENT NOT RELEVANT---")
                    web_search = "Yes"
                    continue
            return {"documents": filtered_docs, "question": question, "web_search": web_search}


        def transform_query(state):
            """
            Transform the query to produce a better question.

            Args:
                state (dict): The current graph state

            Returns:
                state (dict): Updates question key with a re-phrased question
            """

            print("---TRANSFORM QUERY---")
            question = state["question"]
            documents = state["documents"]

            # Re-write question
            better_question = question_rewriter.invoke({"question": question})
            return {"documents": documents, "question": better_question}


        def web_search(state):
            """
            Web search based on the re-phrased question.

            Args:
                state (dict): The current graph state

            Returns:
                state (dict): Updates documents key with appended web results
            """

            print("---WEB SEARCH---")
            question = state["question"]
            documents = state["documents"]

            # Web search
            docs = web_search_tool.invoke({"query": question})
            web_results = "\n".join([d["content"] for d in docs])
            web_results = Document(page_content=web_results)
            documents.append(web_results)

            return {"documents": documents, "question": question}


        ### Edges


        def decide_to_generate(state):
            """
            Determines whether to generate an answer, or re-generate a question.

            Args:
                state (dict): The current graph state

            Returns:
                str: Binary decision for next node to call
            """

            print("---ASSESS GRADED DOCUMENTS---")
            state["question"]
            web_search = state["web_search"]
            state["documents"]

            if web_search == "Yes":
                # All documents have been filtered check_relevance
                # We will re-generate a new query
                print(
                    "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
                )
                return "transform_query"
            else:
                # We have relevant documents, so generate answer
                print("---DECISION: GENERATE---")
                return "generate"


        ### Compile Graph
        ### The just follows the flow we outlined in the figure above.
        from langgraph.graph import END, StateGraph, START

        workflow = StateGraph(GraphState)

        # Define the nodes
        workflow.add_node("retrieve", retrieve)  # retrieve
        workflow.add_node("grade_documents", grade_documents)  # grade documents
        workflow.add_node("generate", generate)  # generatae
        workflow.add_node("transform_query", transform_query)  # transform_query
        workflow.add_node("web_search_node", web_search)  # web search

        # Build graph
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            decide_to_generate,
            {
                "transform_query": "transform_query",
                "generate": "generate",
            },
        )
        workflow.add_edge("transform_query", "web_search_node")
        workflow.add_edge("web_search_node", "generate")
        workflow.add_edge("generate", END)

        # Compile
        app = workflow.compile()
        return app

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
            app = self.setup_qa_chain(uploaded_files)

            utils.display_msg(user_query, "user")

            with st.chat_message("assistant"):
                st_cb = StreamHandler(st.empty())
                # result = qa_chain.invoke(
                #     {"question": user_query}, {"callbacks": [st_cb]}
                # )
                # from pprint import pprint

                # Run
                inputs = {"question": user_query}
                response = ""
                for output in app.stream(inputs):
                    for key, value in output.items():
                        # Node
                        # pprint(f"Node '{key}':")
                        st.write(f"Node '{key}':")
                        response += f"\n\nNode '{key}':" 
                        # Optional: print full state at each node
                        # pprint(value, indent=2, width=80, depth=None)
                        # st.write(str(value))
                        # response += str(value)
                    # pprint("\n---\n")

                # Final generation
                # pprint(value["generation"])
                # response = result["answer"]
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )
                
                st.write(value["generation"])
                
                utils.print_qa(CustomDocChatbot, user_query, response)

                # # to show references
                # for idx, doc in enumerate(result["source_documents"], 1):
                #     filename = os.path.basename(doc.metadata["source"])
                #     page_num = doc.metadata["page"]
                #     ref_title = (
                #         f":blue[Reference {idx}: *{filename} - page.{page_num}*]"
                #     )
                #     with st.popover(ref_title):
                #         st.caption(doc.page_content)


if __name__ == "__main__":
    obj = CustomDocChatbot()
    obj.main()
