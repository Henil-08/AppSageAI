"""Resume processing and analysis with RAG."""

import os
import tempfile
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from logger import logger

class ResumeProcessor:
    """Handles resume processing, vectorization, and RAG operations."""
    
    def __init__(self, 
                 model_name: str = "llama-3.3-70b-versatile",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """
        Initialize the resume processor.
        
        Args:
            model_name: Name of the LLM model
            embedding_model: Name of the embedding model
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize LLM
        self.llm = ChatGroq(
            model_name=model_name,
            temperature=0.7,
            max_tokens=4000
        )
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Store for current session
        self.current_vectorstore = None
        self.current_retriever = None
        self.resume_hash = None
    
    def process_resume(self, file_content: bytes, filename: str) -> Tuple[bool, str, Optional[str]]:
        """
        Process uploaded resume file.
        
        Args:
            file_content: Raw file content
            filename: Name of the file
        
        Returns:
            Tuple of (success, message, resume_hash)
        """
        temp_path = None
        try:
            # Create temporary file with secure handling
            with tempfile.NamedTemporaryFile(
                mode='wb',
                suffix='.pdf',
                delete=False,
                dir=tempfile.gettempdir()
            ) as temp_file:
                temp_file.write(file_content)
                temp_path = temp_file.name
            
            # Calculate hash for the resume
            self.resume_hash = hashlib.sha256(file_content).hexdigest()[:16]
            
            # Load and process PDF
            loader = PyPDFLoader(temp_path)
            documents = loader.load()
            
            if not documents:
                return False, "Could not extract text from PDF", None
            
            # Extract and clean text
            resume_text = self._clean_text(documents)
            
            # Split documents
            splits = self.text_splitter.split_documents(documents)
            
            if not splits:
                return False, "Could not process document content", None
            
            # Create vector store
            self.current_vectorstore = FAISS.from_documents(
                documents=splits,
                embedding=self.embeddings
            )
            
            # Create retriever with optimized settings
            self.current_retriever = self.current_vectorstore.as_retriever(
                search_type="mmr",  # Maximum Marginal Relevance
                search_kwargs={
                    "k": 5,
                    "fetch_k": 10,
                    "lambda_mult": 0.5
                }
            )
            
            # Extract key information for memory
            key_info = self._extract_key_information(resume_text)
            
            logger.info(f"Successfully processed resume: {filename}")
            return True, "Resume processed successfully", self.resume_hash
            
        except Exception as e:
            logger.error(f"Error processing resume: {e}")
            return False, f"Error processing resume: {str(e)}", None
            
        finally:
            # Clean up temporary file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception as e:
                    logger.warning(f"Could not remove temp file: {e}")
    
    def _clean_text(self, documents: List[Document]) -> str:
        """Clean and concatenate document text."""
        text_parts = []
        for doc in documents:
            # Clean the text
            text = doc.page_content
            # Remove excessive whitespace
            text = ' '.join(text.split())
            text_parts.append(text)
        
        return '\n'.join(text_parts)
    
    def _extract_key_information(self, text: str) -> Dict[str, Any]:
        """Extract key information from resume for memory storage."""
        # This is a simplified extraction - enhance as needed
        key_info = {
            'skills': [],
            'experience_years': None,
            'education': [],
            'certifications': [],
            'languages': []
        }
        
        # Extract skills (simplified pattern matching)
        skill_keywords = ['python', 'java', 'javascript', 'react', 'node', 'sql', 
                         'aws', 'docker', 'kubernetes', 'machine learning', 'ai']
        
        text_lower = text.lower()
        for skill in skill_keywords:
            if skill in text_lower:
                key_info['skills'].append(skill)
        
        return key_info
    
    def create_rag_chain(self, prompt_template: str, config: Optional[Dict[str, Any]] = None):
        """
        Create a RAG chain for question answering.
        
        Args:
            prompt_template: The prompt template to use
            config: Optional configuration with callbacks
        
        Returns:
            The RAG chain
        """
        if not self.current_retriever:
            raise ValueError("No resume has been processed yet")
        
        # Create prompt
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_template),
            ("human", "{input}")
        ])
        
        # Create chains
        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        
        # Add config if provided (includes Langfuse callbacks)
        if config:
            rag_chain = create_retrieval_chain(
                self.current_retriever,
                question_answer_chain
            )
            # Wrap with config
            return lambda x: rag_chain.invoke(x, config=config)
        else:
            return create_retrieval_chain(
                self.current_retriever,
                question_answer_chain
            )
    
    def analyze_resume(self, 
                      prompt_template: str,
                      query: str,
                      user_name: str,
                      job_description: str,
                      config: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Analyze resume with given prompt and query.
        
        Args:
            prompt_template: The prompt template
            query: The query to answer
            user_name: User's name
            job_description: Job description
            config: Optional configuration with callbacks
        
        Returns:
            Tuple of (answer, metadata)
        """
        try:
            # Format prompt with user context
            formatted_prompt = prompt_template.format(
                user_name=user_name,
                job_description=job_description,
                context="{context}",
                user_question=query
            )
            
            # Create and run chain
            rag_chain = self.create_rag_chain(formatted_prompt, config)
            
            # Execute query
            import time
            start_time = time.time()
            
            response = rag_chain({"input": query})
            
            end_time = time.time()
            response_time_ms = int((end_time - start_time) * 1000)
            
            # Extract answer
            answer = response.get('answer', 'Unable to generate response')
            
            # Estimate tokens (rough approximation)
            tokens_used = len(answer.split()) * 1.3 + len(query.split()) * 1.3
            
            metadata = {
                'response_time_ms': response_time_ms,
                'tokens_used': int(tokens_used),
                'model': self.model_name,
                'retriever_docs': len(response.get('context', [])),
                'resume_hash': self.resume_hash
            }
            
            return answer, metadata
            
        except Exception as e:
            logger.error(f"Error in resume analysis: {e}")
            return f"Error analyzing resume: {str(e)}", {
                'error': str(e),
                'response_time_ms': 0,
                'tokens_used': 0
            }
    
    def get_relevant_context(self, query: str, k: int = 3) -> List[str]:
        """
        Get relevant context chunks for a query.
        
        Args:
            query: The query
            k: Number of chunks to retrieve
        
        Returns:
            List of relevant text chunks
        """
        if not self.current_retriever:
            return []
        
        try:
            docs = self.current_retriever.get_relevant_documents(query)[:k]
            return [doc.page_content for doc in docs]
        except Exception as e:
            logger.error(f"Error getting context: {e}")
            return []