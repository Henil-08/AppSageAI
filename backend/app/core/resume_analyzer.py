"""Core resume analysis logic using RAG."""

import io
import time
from typing import Tuple, Dict, Any, Optional
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.documents import Document

from app.config.settings import settings
from app.logger import logger
from app.db.models import AnalysisType


class ResumeAnalyzer:
    """Handles resume analysis using RAG and LLM."""
    
    def __init__(self):
        """Initialize the analyzer with models and embeddings."""
        logger.info("Initializing Resume Analyzer...")
        
        # Initialize LLM
        self.llm = ChatGroq(
            model_name=settings.model_name,
            temperature=0.7,
            max_tokens=4000,
            groq_api_key=settings.get_groq_api_key()
        )
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Text splitter for documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        logger.info("Resume Analyzer initialized successfully")
    
    async def analyze(
        self,
        resume_content: bytes,
        job_description: str,
        analysis_type: AnalysisType,
        prompt_template: str,
        user_name: str = "Candidate",
        custom_query: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Analyze resume against job description.
        
        Args:
            resume_content: Resume file content (bytes)
            job_description: Job description text
            analysis_type: Type of analysis to perform
            prompt_template: Prompt template to use
            user_name: User's name or job title
            custom_query: Custom query for analysis
        
        Returns:
            Tuple of (analysis_result, metadata)
        """
        start_time = time.time()
        
        try:
            # Process resume PDF
            documents = self._process_pdf(resume_content)
            
            if not documents:
                raise ValueError("Could not extract text from resume")
            
            # Create vector store
            vectorstore = self._create_vectorstore(documents)
            
            # Create retriever with optimized settings
            retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 5,
                    "fetch_k": 10,
                    "lambda_mult": 0.5
                }
            )
            
            # Format prompt
            formatted_prompt = prompt_template.format(
                user_name=user_name,
                job_description=job_description,
                context="{context}",
                user_question=custom_query or ""
            )
            
            # Create QA chain
            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", formatted_prompt),
                ("human", "{input}")
            ])
            
            question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)
            
            # Determine the query based on analysis type
            query = self._get_analysis_query(analysis_type, custom_query)
            
            # Execute analysis
            response = rag_chain.invoke({"input": query})
            
            # Extract result
            answer = response.get('answer', 'Unable to generate analysis')
            
            # Calculate metadata
            end_time = time.time()
            response_time_ms = int((end_time - start_time) * 1000)
            
            # Estimate tokens (rough approximation)
            tokens_used = self._estimate_tokens(
                job_description + formatted_prompt + query + answer
            )
            
            metadata = {
                'response_time_ms': response_time_ms,
                'tokens_used': tokens_used,
                'model': settings.model_name,
                'chunks_retrieved': len(response.get('context', [])),
                'analysis_type': analysis_type.value
            }
            
            logger.info(f"Analysis completed: {analysis_type.value} in {response_time_ms}ms")
            
            return answer, metadata
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            raise
    
    def _process_pdf(self, pdf_content: bytes) -> list[Document]:
        """Process PDF content and extract documents."""
        try:
            # Create temporary file in memory
            pdf_file = io.BytesIO(pdf_content)
            
            # Save to temporary file (PyPDFLoader needs a file path)
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                tmp_file.write(pdf_content)
                tmp_path = tmp_file.name
            
            try:
                # Load PDF
                loader = PyPDFLoader(tmp_path)
                documents = loader.load()
                
                # Clean and split documents
                cleaned_docs = []
                for doc in documents:
                    # Clean text
                    text = doc.page_content
                    text = ' '.join(text.split())  # Remove excessive whitespace
                    
                    if text.strip():  # Only add non-empty documents
                        cleaned_doc = Document(
                            page_content=text,
                            metadata=doc.metadata
                        )
                        cleaned_docs.append(cleaned_doc)
                
                return cleaned_docs
                
            finally:
                # Clean up temporary file
                import os
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                    
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            raise ValueError(f"Could not process PDF: {str(e)}")
    
    def _create_vectorstore(self, documents: list[Document]) -> FAISS:
        """Create FAISS vector store from documents."""
        try:
            # Split documents
            splits = self.text_splitter.split_documents(documents)
            
            if not splits:
                raise ValueError("No content to vectorize")
            
            # Create vector store
            vectorstore = FAISS.from_documents(
                documents=splits,
                embedding=self.embeddings
            )
            
            return vectorstore
            
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            raise
    
    def _get_analysis_query(self, analysis_type: AnalysisType, custom_query: Optional[str]) -> str:
        """Get the appropriate query for the analysis type."""
        queries = {
            AnalysisType.RESUME_REVIEW: "Provide a comprehensive review of this resume against the job description",
            AnalysisType.SKILL_IMPROVEMENT: "What skills should I improve and how?",
            AnalysisType.KEYWORD_ANALYSIS: "What keywords are missing from my resume?",
            AnalysisType.PERCENTAGE_MATCH: "Calculate the match percentage with detailed breakdown",
            AnalysisType.COVER_LETTER: "Generate a compelling cover letter for this position",
            AnalysisType.CUSTOM_QUERY: custom_query or "Analyze my resume"
        }
        
        return queries.get(analysis_type, "Analyze my resume")
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        # Rough estimation: 1 token ≈ 4 characters
        return len(text) // 4