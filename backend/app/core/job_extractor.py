"""Job details extraction using Groq LLM."""

from typing import Dict, Any
import json
from langchain_groq import ChatGroq
from google.oauth2 import service_account
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from app.config.settings import settings
from app.logger import logger


async def extract_job_details(text: str) -> Dict[str, Any]:
    """
    Extract job details from text using Groq LLM.
    
    Args:
        text: Raw text that might be a job description
        
    Returns:
        Dictionary with extracted job details
    """
    try:
        credentials = service_account.Credentials.from_service_account_file(
            settings.firebase_service_account_path,
            scopes=['https://www.googleapis.com/auth/generative-language']
        )

        # Initialize LLM
        llm = ChatGoogleGenerativeAI(
            model=settings.job_model_name,
            temperature=0.1,
            credentials=credentials,
        )
        
        # Create extraction prompt
        prompt = ChatPromptTemplate.from_template("""
        Analyze the following text and extract job details if it appears to be a job listing.
        If it's not a job listing, still try to understand the context. 
        
        Text:
        {text}
        
        Provide your response in JSON format with these fields:
        - is_job_listing: boolean (true if this is a job posting)
        - job_title: string (the position title, or best guess)
        - company: string (company name if mentioned)
        - location: string (job location if mentioned)
        - salary: string (salary range if mentioned)
        - job_type: string (full-time, part-time, contract, etc.)
        - job_description: string (cleaned/formatted job description)
        
        DON'T SUMMARIZE job_description or LEAVE OUT ANY WORD. Keep all the contents from the job description. Just format it cleanly. Not a single word should be missed!
                                                  
        If any field cannot be determined, use an empty string.
        
        RESPOND ONLY WITH VALID JSON. No additional text.
        """)
        
        # Create chain
        chain = prompt | llm
        
        # Execute extraction
        response = await chain.ainvoke({"text": text})
        
        # Parse response
        try:
            # Extract JSON from response
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Clean response (remove markdown if present)
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
            
            result = json.loads(response_text.strip())
            
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM response as JSON, using defaults")
            result = {
                "is_job_listing": len(text) > 200 and any(
                    keyword in text.lower() 
                    for keyword in ['requirements', 'qualifications', 'responsibilities', 'experience']
                ),
                "job_title": "",
                "company": "",
                "location": "",
                "salary": "",
                "job_type": "",
                "job_description": text
            }
        
        return result
        
    except Exception as e:
        logger.error(f"Error in job extraction: {e}")
        # Return defaults on error
        return {
            "is_job_listing": False,
            "job_title": "",
            "company": "",
            "location": "",
            "salary": "",
            "job_type": "",
            "job_description": text
        }