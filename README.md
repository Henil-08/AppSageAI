# AppSageAI - Privacy-First Resume Analysis Platform

## 🔐 Zero-Knowledge Architecture
This application implements client-side encryption to ensure your resume data remains private.

## 📁 Monorepo Structure
```
appsageai/
├── backend/          # FastAPI backend with Firebase Auth
├── frontend/         # Next.js frontend with E2E encryption  
├── shared/           # Shared types and crypto utilities
├── deployment/       # Docker, K8s, Terraform configs
└── config/          # Application configuration
```

## 🚀 Quick Start
See individual README files in backend/ and frontend/ directories.

## 🔒 Privacy Guarantee
- All sensitive data is encrypted client-side
- We cannot read your resumes or chat history
- Even with database access, your data remains private
EOF