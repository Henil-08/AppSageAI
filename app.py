import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os

from dotenv import load_dotenv
load_dotenv()

## load the GROQ and HF API Key
os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY")

## Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"]="true"

## Loading the Model
llm = ChatGroq(model_name="llama-3.1-8b-instant")

## Getting the embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

## set up Streamlit 
st.title("AppSageAI")
st.subheader("Your Intelligent Companion for Tracking and Enhancing Application Performance")

## Name
name=st.text_input("Enter your Name:")

## Job Description
jd=st.text_input("Enter Job Description:")

uploaded_file=st.file_uploader("Upload your Resume", type="pdf", accept_multiple_files=False)

if name:
    if jd:
        ## Process uploaded  PDF's
        if uploaded_file:
            temppdf=f"./temp.pdf"
            with open(temppdf,"wb") as file:
                file.write(uploaded_file.getvalue())
                file_name=uploaded_file.name

            loader=PyPDFLoader(temppdf)
            docs=loader.load()
            
            # Split and create embeddings for the documents
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
            splits = text_splitter.split_documents(docs)
            vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
            retriever = vectorstore.as_retriever()    

            ## Answer question

            # Answer question
            system_prompt = (
                f'''
                You are AppSage, an advanced AI developed to help users understand their resume and help them get jobs. 
                You are an skilled ATS (Applicant Tracking System) scanner with a deep understanding of all the domains and ATS functionality.
                Job Description: {jd}\n
                Your responses should be clear, concise, and insightful, catering to both technical and non-technical users. First greet {name} and then answer
                '''
                "\n\n"
                "{context}"
            )

            input_prompt1 = (f'''
                You are AppSage, an advanced AI developed to help users understand their resume and help them get jobs.
                You are an experienced Technical Human Resource Manager,your task is to review the provided resume against the job description. 
                Please share your professional evaluation on whether the candidate's profile aligns with the role.             
                Job Description: {jd}\n
                First greet {name} and then Highlight the strengths and weaknesses of the applicant in relation to the specified job requirements.'''
                "\n\n"
                "{context}"
            )

            input_prompt2 = (f"""
                You are AppSage, an advanced AI developed to help users understand their resume and help them get jobs.
                You are an Technical Human Resource Manager with expertise in all the domains, 
                your role is to scrutinize the resume in light of the job description provided. 
                Job Description: {jd}\n
                First greet {name} and then Share your insights on the candidate's suitability for the role from an HR perspective. 
                Additionally, offer advice on enhancing the candidate's skills and identify areas where improvement is needed."""
                "\n\n"
                "{context}"
            )

            input_prompt3 = (f"""
                You are AppSage, an advanced AI developed to help users understand their resume and help them get jobs.
                You are an skilled ATS (Applicant Tracking System) scanner with a deep understanding of data science and ATS functionality, 
                your task is to evaluate the resume against the provided job description.
                Job Description: {jd}\n
                As a Human Resource manager, assess the compatibility of the resume with the role. Give me what are the keywords that are missing
                First greet {name} and then Also, provide recommendations for enhancing the candidate's skills and identify which areas require further development."""
                "\n\n"
                "{context}"
            )

            input_prompt4 = (f"""
                You are AppSage, an advanced AI developed to help users understand their resume and help them get jobs.
                You are an skilled ATS (Applicant Tracking System) scanner with a deep understanding of data science and ATS functionality, 
                your task is to evaluate the resume against the provided job description. 
                Job Description: {jd}\n
                give me the percentage of match if the resume matches the job description. First greet {name} and then First the output should come as percentage and then keywords missing and last final thoughts."""
                "\n\n"
                "{context}"
            )
            
            input_prompt5 = (f"""
                You are AppSage, an advanced AI developed to help users understand their resume and help them get jobs.
                You are an skilled ATS (Applicant Tracking System) scanner with a deep understanding of data science and ATS functionality, 
                your task is to create an excellent cover letter to the hiring manager taking the resume against the provided job description. 
                Job Description: {jd}\n
                Write me a very good cover letter highlighting all the necessary technologies used during the experience or projects. First paragraph should leave the readers on hook and then following paragraphs should mention what the person brings to the table.
                First greet {name} and then give the cover letter."""
                "\n\n"
                "{context}"
            )

            submit1 = st.button("Tell Me About the Resume")

            submit2 = st.button("How Can I Improvise my Skills?")

            submit3 = st.button("What are the Keywords that are Missing?")

            submit4 = st.button("Percentage Match")
            
            submit5 = st.button("Cover Letter")

            user_input = st.text_input("Your question:")


            if submit1:
                qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", input_prompt1),
                    ("human", "{input}"),
                ]
                )
                question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
                rag_chain = create_retrieval_chain(retriever, question_answer_chain)

                response = rag_chain.invoke(
                    {"input": "Tell Me About the Resume"},
                )
                
                st.write("AppSageAI:", response['answer'])

            if submit2:
                qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", input_prompt2),
                    ("human", "{input}"),
                ]
                )   
                question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
                rag_chain = create_retrieval_chain(retriever, question_answer_chain)

                response = rag_chain.invoke(
                    {"input": "How Can I Improvise my Skills?"},
                )
                
                st.write("AppSageAI:", response['answer'])

            if submit3:
                qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", input_prompt3),
                    ("human", "{input}"),
                ]
                )   
                question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
                rag_chain = create_retrieval_chain(retriever, question_answer_chain)

                response = rag_chain.invoke(
                    {"input": "What are the Keywords That are Missing?"},
                )
                
                st.write("AppSageAI:", response['answer'])

            if submit4:
                qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", input_prompt4),
                    ("human", "{input}"),
                ]
                )   
                question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
                rag_chain = create_retrieval_chain(retriever, question_answer_chain)

                response = rag_chain.invoke(
                    {"input": "Percentage match"},
                )
                
                st.write("AppSageAI:", response['answer'])

            if submit5:
                qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", input_prompt4),
                    ("human", "{input}"),
                ]
                )   
                question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
                rag_chain = create_retrieval_chain(retriever, question_answer_chain)

                response = rag_chain.invoke(
                    {"input": "Write me a Cover Letter"},
                )
                
                st.write("AppSageAI:", response['answer'])

            if user_input:
                qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    ("human", "{input}"),
                ]
                )
                question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
                rag_chain = create_retrieval_chain(retriever, question_answer_chain)

                response = rag_chain.invoke(
                    {"input": user_input},
                )
                
                st.write("AppSageAI:", response['answer']) 
                
        else:
            st.warning("Upload your Resume!")
else:
    st.warning("Enter Your Name!")
        