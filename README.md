# AppSageAI
### [Website](https://appsage.streamlit.app/)
AppSageAI is an intelligent application designed to assist job seekers in enhancing their resumes to better match job descriptions. Leveraging advanced AI technology, the application provides detailed feedback, suggestions for skills improvement, and a percentage match score, helping users refine their resumes to increase their chances of landing their desired jobs.

## Objectives
To provide an intuitive tool for job seekers to match their resumes with job descriptions.
To leverage advanced AI technology for analyzing and providing feedback on resumes.
To offer a user-friendly interface that simplifies the resume review process.

## Features
- *Resume Upload*: Users can upload their resume in PDF format.
- *Job Description Input*: A text input field allows users to paste the job description they are targeting.
- *AI-Powered Analysis*: Utilizing advanced AI, the application provides a detailed analysis of the resume in context with the job description.
- *Feedback on Different Aspects*:
    - Resume Review: General feedback on the resume.
    - Skills Improvement: Suggestions for skills enhancement.
    - Keywords Analysis: Identification of missing keywords in the resume.
    - Match Percentage: A percentage score indicating how well the resume matches the job description.
    - Cover Letter Generation: Create a customized cover letter based on the resume and job description.

## Technologies Used
- *Streamlit*: For creating the web application interface.
- *Langchain*: For handling the AI-based analysis and feedback.
- *FAISS*: For vector storage and retrieval.
- *HuggingFace Embeddings*: For embedding generation.
- *Python*: The primary programming language used for backend development.
- *PyPDFLoader*: For handling PDF file conversions and document loading.
- *Environment Variables*: Managed with dotenv for API keys and configuration.
- *Llama 3.1 Model*: The latest Llama 3.1 model powers the AI analysis and feedback, ensuring cutting-edge performance and accuracy.

## Challenges Faced
- *Integration with AI Models*: Ensuring seamless communication between the Streamlit interface and the AI models.
- *PDF Handling*: Efficiently converting PDF content to a format suitable for analysis by the AI model.
- *User Experience Optimization*: Creating an intuitive and responsive UI.

## Future Enhancements
- *Support for Multiple Pages*: Extend the functionality to handle multi-page resumes.
- *Customizable Feedback Categories*: Allow users to choose specific areas for feedback.
- *Interactive Resume Editing*: Integrate a feature to edit the resume directly based on the AI's suggestions.
- *Enhanced Error Handling*: Improve the system's robustness in handling various file formats and user inputs.

## Conclusion
AppSageAI stands as a significant tool in bridging the gap between job seekers and their ideal job roles. By harnessing the power of AI, it provides valuable insights and recommendations, making it a pivotal step in enhancing the job application process. The inclusion of the latest Llama 3.1 model ensures that users benefit from state-of-the-art AI capabilities.
