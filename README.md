# AppSageAI - Privacy-First AI Resume Analyzer

Transform your job search with intelligent resume analysis while keeping your data completely private.

## 🎯 What is AppSageAI?

AppSageAI is a privacy-focused job application companion that uses advanced AI to analyze your resume against job descriptions. Unlike other tools, we prioritize your data privacy with enterprise-grade encryption, ensuring your personal information remains secure.

## 🔐 Privacy-First Architecture

### Your Data, Your Control
- **AES-256 Encryption**: All resumes, chats, and personal data encrypted at rest
- **Anonymized Analytics**: No personally identifiable information in logs
- **Secure Infrastructure**: Hosted on Google Cloud with enterprise security
- **GDPR Compliant**: Full data portability and right to deletion

## ✨ Key Features

### 📊 Intelligent Analysis
- **Job Match Analysis**: Comprehensive review against requirements
- **ATS Optimization**: Keyword analysis for applicant tracking systems
- **Match Scoring**: Percentage-based compatibility assessment
- **Skill Gap Analysis**: Personalized improvement roadmap
- **Cover Letter Generation**: Tailored letters for each application

### 💼 Application Management
- **Multi-Resume Support**: Upload different versions for different roles
- **Job Tracker**: Track application status (Applied, Interviewing, Offered)
- **Chat History**: Maintain context across multiple applications
- **Custom Prompts**: Personalize AI responses to your industry

### 🚀 Modern User Experience
- **Real-time Streaming**: Instant AI responses as they generate
- **Smart Resume Tagging**: Reference specific resumes with @mentions
- **Markdown Support**: Rich text formatting in responses
- **Mobile Responsive**: Full functionality on any device

## Project Architecture
![AppSageAI Achitecture](media/appsageai_architecture.jpg)

## Technology Stack

### Frontend
- **Framework**: Next.js 14 with TypeScript
- **Authentication**: Firebase Auth (Google OAuth)
- **Styling**: Tailwind CSS with custom Claude-inspired design
- **State Management**: React Context API
- **Real-time**: Server-Sent Events (SSE) for streaming

### Backend
- **Framework**: FastAPI (Python 3.12)
- **AI Models**: 
  - Google Gemini 2.5 Flash (Analysis)
  - HuggingFace Embeddings (RAG)
- **Database**: Google Firestore
- **Security**: AES-256 Fernet Encryption
- **Vector Store**: FAISS for semantic search

### Infrastructure
- **Hosting**: Google Cloud Run (Serverless)
- **CI/CD**: GitHub Actions
- **Container**: Docker with Artifact Registry
- **Monitoring**: Cloud Run health checks

## 📁 Project Structure

```
appsageai/
├── frontend/               # Next.js application
│   ├── src/app/           # App router pages
│   ├── src/components/    # React components
│   ├── src/contexts/      # Context providers
│   └── src/lib/          # Utilities & Firebase
│
├── backend/               # FastAPI application
│   ├── app/api/          # API endpoints
│   ├── app/auth/         # Authentication
│   ├── app/core/         # Business logic & AI
│   ├── app/services/     # Encryption & utilities
│   ├── app/db/           # Pydantic Models
│   └── app/config/        # Config File
│
└── .github/workflows/     # CI/CD pipelines
```

## 🚀 Quick Start

### Prerequisites
- Node.js 18+
- Python 3.12+
- Google Cloud Account
- Firebase Project

### Local Development

#### Clone the repository
```bash
git clone https://github.com/yourusername/appsageai.git
cd appsageai
```

#### Backend Setup

```bash
cd backend

# Install dependencies
uv init
uv install -r pyproject.toml

cp .env.example .env
# Edit .env with your API keys

# Run server
uvicorn app.main:app --reload --port 8000
```

Read more here - [Backend.md](https://github.com/henil-08/AppSageAI/backend/README.md)

#### Frontend Setup

```bash
cd frontend

npm install
cp .env.local.example .env.local
# Add your Firebase config to .env.local

npm run dev
```

Read more here - [Frontend.md](https://github.com/henil-08/AppSageAI/frontend/README.md)

Then visit http://localhost:3000 to see the application in your local env.

### 🔒 Security Features

- End-to-End Encryption: All sensitive data encrypted before storage
- Zero-Knowledge: Even developers cannot access your encrypted data
- Secure Authentication: Firebase Auth with OAuth 2.0
- Rate Limiting: Protection against abuse
- CORS Protection: Restricted API access
- Input Sanitization: XSS and injection prevention

### Usage Limits (100% Free Forever)

- Unlimited resume uploads
- Unlimited chat sessions
- Rate Limit: 100 analyses per minute per user
- All features included
- No credit card required

### Contributing
We welcome contributions! Please see our Contributing Guide for details.

### License
This project is licensed under the MIT License - see the LICENSE file for details.

### 🙏 Acknowledgments
- Powered by Google Gemini AI
- Built with privacy as the foundation
- Inspired by the need for secure job application tools

### Contact
For questions or support, please open an issue on GitHub.