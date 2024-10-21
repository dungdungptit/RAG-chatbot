import os
import utils
import streamlit as st
from streaming import StreamHandler

from langchain.memory import ConversationBufferMemory
from langchain.chains import StuffDocumentsChain, LLMChain, ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS


st.set_page_config(page_title="ChatPDF", page_icon="📄")
st.header("Chat with your documents (RankRAG)")
st.write(
    "Has access to custom documents and can respond to user queries by referring to the content within those documents"
)
st.write(
    "[![view source code ](https://img.shields.io/badge/view_source_code-gray?logo=github)](https://github.com/shashankdeshpande/langchain-chatbot/blob/master/pages/4_%F0%9F%93%84_chat_with_your_documents.py)"
)

# Helper function for printing docs


def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [
                f"Document {i+1}:\n\n{d.page_content}\nMetadata: {d.metadata}"
                for i, d in enumerate(docs)
            ]
        )
    )


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
        # vectordb = DocArrayInMemorySearch.from_documents(splits, self.embedding_model)

        # # Define retriever
        # retriever = vectordb.as_retriever(
        #     search_type="mmr", search_kwargs={"k": 2, "fetch_k": 4}
        # )
        retriever = FAISS.from_documents(splits, self.embedding_model).as_retriever(search_kwargs={"k": 20})

        # Setup memory for contextual conversation
        memory = ConversationBufferMemory(
            memory_key="chat_history", output_key="answer", return_messages=True
        )
        template = (
            "Combine the chat history and follow up question into "
            "a standalone question. Chat History: {chat_history}"
            "Follow up question: {question}"
        )
        prompt = PromptTemplate.from_template(template)
        
        # Setup LLM and QA chain
        # qa_chain = ConversationalRetrievalChain.from_llm(
        #     llm=self.llm,
        #     retriever=retriever,
        #     memory=memory,
        #     return_source_documents=True,
        #     verbose=False,
        # )
        
        from langchain.retrievers import ContextualCompressionRetriever
        from langchain.retrievers.document_compressors import FlashrankRerank
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(temperature=0)

        compressor = FlashrankRerank()
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=retriever
        )
        
        from langchain.chains import RetrievalQA

        chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=compression_retriever,
            # memory=memory,
            return_source_documents=True,
            verbose=False,)
        return chain

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
                    user_query
                )
                response = result["result"]
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )
                st.write(response)
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
