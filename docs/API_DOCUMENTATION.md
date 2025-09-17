# AppSageAI API Documentation

## 🔑 Authentication

All API endpoints (except health check) require Firebase authentication token.

### Headers Required
```http
Authorization: Bearer <firebase-id-token>
Content-Type: application/json
```

### Getting Firebase Token (Frontend)
```javascript
const user = await firebase.auth().currentUser;
const token = await user.getIdToken();
```

---

## 📋 API Endpoints

### Base URL
```
Development: http://localhost:8000/api/v1
Production: https://api.appsageai.com/api/v1
```

---

## 🔐 Authentication Endpoints

### POST `/auth/verify`
Verify Firebase token and create/get user profile.

**Request:**
```json
// No body required, token in header
```

**Response:**
```json
{
  "uid": "user123",
  "email": "user@example.com",
  "name": "John Doe",
  "picture": "https://...",
  "created_at": "2024-01-15T10:30:00Z",
  "plan": "free",
  "usage": {
    "analyses_count": 10,
    "tokens_used": 50000,
    "last_analysis": "2024-01-15T10:30:00Z"
  }
}
```

### GET `/auth/me`
Get current user profile.

**Response:**
```json
{
  "uid": "user123",
  "email": "user@example.com",
  "name": "John Doe",
  "plan": "free",
  "usage": {...}
}
```

---

## 📄 Resume Management

### POST `/resume/upload`
Upload resume (one-time, reused across all chats).

**Request:**
```http
Content-Type: multipart/form-data

file: <pdf-file>
```

**Response:**
```json
{
  "message": "Resume uploaded successfully",
  "resume_id": "res_abc123",
  "filename": "john_doe_resume.pdf",
  "size_mb": 1.5,
  "is_duplicate": false
}
```

**Error Response (400):**
```json
{
  "detail": "File size exceeds 10MB limit"
}
```

### GET `/resume/list`
List all uploaded resumes.

**Response:**
```json
{
  "resumes": [
    {
      "resume_id": "res_abc123",
      "filename": "john_doe_resume.pdf",
      "uploaded_at": "2024-01-15T10:30:00Z",
      "analysis_count": 5,
      "is_active": true
    }
  ],
  "count": 1,
  "active_resume_id": "res_abc123"
}
```

### POST `/resume/{resume_id}/set-active`
Set a resume as active for analysis.

**Response:**
```json
{
  "message": "Active resume updated",
  "resume_id": "res_abc123"
}
```

### DELETE `/resume/{resume_id}`
Delete a resume.

**Response:**
```json
{
  "message": "Resume deleted successfully"
}
```

---

## 💬 Chat Session Management

### POST `/chat/create`
Create a new chat session for a job application.

**Request:**
```json
{
  "job_description": "We are looking for a Senior Full Stack Developer...",
  "job_title": "Senior Full Stack Developer",  // Optional
  "company": "TechCorp Inc."  // Optional
}
```

**Response:**
```json
{
  "session_id": "chat_xyz789",
  "message": "Chat session created",
  "job_title": "Senior Full Stack Developer",
  "company": "TechCorp Inc.",
  "resume_id": "res_abc123"
}
```

### GET `/chat/list`
List all chat sessions.

**Query Parameters:**
- `limit` (int, default: 20): Number of results
- `offset` (int, default: 0): Pagination offset

**Response:**
```json
{
  "chats": [
    {
      "session_id": "chat_xyz789",
      "created_at": "2024-01-15T10:30:00Z",
      "updated_at": "2024-01-15T11:00:00Z",
      "job_title": "Senior Full Stack Developer",
      "company": "TechCorp Inc.",
      "message_count": 5,
      "preview": "We are looking for a Senior..."
    }
  ],
  "total": 10,
  "limit": 20,
  "offset": 0,
  "has_more": false
}
```

### GET `/chat/{session_id}`
Get chat session with all messages.

**Response:**
```json
{
  "session_id": "chat_xyz789",
  "created_at": "2024-01-15T10:30:00Z",
  "job_title": "Senior Full Stack Developer",
  "company": "TechCorp Inc.",
  "job_description": "Full job description...",
  "resume_id": "res_abc123",
  "messages": [
    {
      "message_id": "msg_001",
      "role": "user",
      "encrypted_content": "encrypted_base64_string",
      "timestamp": "2024-01-15T10:35:00Z",
      "metadata": {}
    },
    {
      "message_id": "msg_002",
      "role": "assistant",
      "encrypted_content": "encrypted_base64_string",
      "timestamp": "2024-01-15T10:36:00Z",
      "metadata": {
        "analysis_type": "resume_review",
        "tokens_used": 1500
      }
    }
  ],
  "message_count": 2
}
```

### POST `/chat/{session_id}/message`
Add a message to chat session.

**Request:**
```json
{
  "role": "user",
  "encrypted_content": "encrypted_base64_string",
  "metadata": {
    "client_timestamp": "2024-01-15T10:35:00Z"
  }
}
```

**Response:**
```json
{
  "message_id": "msg_003",
  "timestamp": "2024-01-15T10:35:00Z"
}
```

### DELETE `/chat/{session_id}`
Delete a chat session.

**Response:**
```json
{
  "message": "Chat session deleted successfully"
}
```

---

## 🎯 Analysis Endpoints

### POST `/analysis/analyze/{session_id}`
Perform resume analysis.

**Request:**
```json
{
  "analysis_type": "resume_review",  // See types below
  "custom_query": null  // Required only for "custom_query" type
}
```

**Analysis Types:**
- `resume_review` - Comprehensive resume review
- `skill_improvement` - Skill gap analysis and roadmap
- `keyword_analysis` - ATS keyword optimization
- `percentage_match` - Match percentage calculation
- `cover_letter` - Generate cover letter
- `custom_query` - Custom question

**Response:**
```json
{
  "analysis_id": "analysis_001",
  "encrypted_response": "encrypted_analysis_result",
  "metadata": {
    "tokens_used": 1500,
    "response_time_ms": 2340,
    "model": "llama-3.3-70b-versatile",
    "analysis_type": "resume_review"
  },
  "timestamp": "2024-01-15T10:40:00Z"
}
```

### GET `/analysis/quick-actions/{session_id}`
Get available quick actions for chat.

**Response:**
```json
{
  "session_id": "chat_xyz789",
  "actions": [
    {
      "id": "job_match",
      "type": "resume_review",
      "label": "📊 Job Match Analysis",
      "description": "Comprehensive review against job requirements",
      "icon": "📊"
    },
    {
      "id": "ats_scan",
      "type": "keyword_analysis",
      "label": "🎯 ATS Scan",
      "description": "Check ATS compatibility and keywords",
      "icon": "🎯"
    }
    // ... more actions
  ]
}
```

### POST `/analysis/feedback/{analysis_id}`
Submit feedback for an analysis.

**Request:**
```json
{
  "feedback_type": "thumbs_up",  // or "thumbs_down"
  "comment": "Very helpful analysis!"  // Optional
}
```

**Response:**
```json
{
  "message": "Thank you for your feedback!"
}
```

---

## 👤 User Management

### GET `/user/stats`
Get user statistics.

**Response:**
```json
{
  "total_analyses": 25,
  "total_tokens_used": 150000,
  "total_resumes": 3,
  "total_chats": 10,
  "plan": "free",
  "joined_date": "2024-01-01T00:00:00Z"
}
```

### GET `/user/usage`
Get usage limits and current usage.

**Response:**
```json
{
  "plan": "free",
  "limits": {
    "max_analyses_per_day": 10,
    "max_tokens_per_month": 100000,
    "max_file_size_mb": 10,
    "max_stored_resumes": 5,
    "max_stored_chats": 50
  },
  "usage": {
    "analyses_today": 3,
    "tokens_this_month": 45000,
    "stored_resumes": 2,
    "stored_chats": 8
  },
  "remaining": {
    "analyses_today": 7,
    "tokens_this_month": 55000
  }
}
```

### DELETE `/user/account`
Delete user account (irreversible).

**Query Parameters:**
- `confirm` (bool): Must be `true` to confirm

**Response:**
```json
{
  "message": "Account deleted successfully"
}
```

---

## 🏥 Health & Status

### GET `/health`
Health check endpoint (no auth required).

**Response:**
```json
{
  "status": "healthy",
  "app": "AppSageAI",
  "version": "2.0.0",
  "environment": "development"
}
```

---

## 🔴 Error Responses

### 400 Bad Request
```json
{
  "detail": "Invalid request parameters",
  "request_id": "req_123"
}
```

### 401 Unauthorized
```json
{
  "detail": "Invalid authentication token",
  "request_id": "req_123"
}
```

### 403 Forbidden
```json
{
  "detail": "Email verification required",
  "request_id": "req_123"
}
```

### 404 Not Found
```json
{
  "detail": "Resource not found",
  "request_id": "req_123"
}
```

### 429 Too Many Requests
```json
{
  "detail": "Rate limit exceeded. Please try again later."
}
```

### 500 Internal Server Error
```json
{
  "detail": "Internal server error",
  "request_id": "req_123"
}
```

---

## 🔄 Workflow Examples

### Complete User Journey

1. **Authentication**
```bash
# Get Firebase token from frontend
# Include in all requests as Bearer token
```

2. **Upload Resume (One Time)**
```bash
POST /api/v1/resume/upload
Content-Type: multipart/form-data
Authorization: Bearer <token>

file: resume.pdf
```

3. **Create Chat Session**
```bash
POST /api/v1/chat/create
{
  "job_description": "Full stack developer role...",
  "job_title": "Full Stack Developer",
  "company": "TechCorp"
}
```

4. **Perform Analysis**
```bash
POST /api/v1/analysis/analyze/{session_id}
{
  "analysis_type": "resume_review"
}
```

5. **Continue Conversation**
```bash
POST /api/v1/chat/{session_id}/message
{
  "role": "user",
  "encrypted_content": "What skills should I focus on?",
  "metadata": {}
}

POST /api/v1/analysis/analyze/{session_id}
{
  "analysis_type": "skill_improvement"
}
```

6. **Submit Feedback**
```bash
POST /api/v1/analysis/feedback/{analysis_id}
{
  "feedback_type": "thumbs_up",
  "comment": "Very helpful!"
}
```

---

## 🔒 Encryption Notes

### Client-Side Encryption
The API expects certain fields to be encrypted client-side:
- Resume content
- Chat messages
- Analysis results

### Encryption Format
```javascript
// Frontend encryption example
const encrypted = await encryptWithUserKey(plaintext);
// Send as base64 string
```

### Why Encryption?
- Zero-knowledge architecture
- User privacy protection
- Compliance with data regulations

---

## 📊 Rate Limits

| Plan | Requests/Minute | Daily Analyses | Monthly Tokens |
|------|----------------|----------------|----------------|
| Free | 60 | 10 | 100,000 |
| Pro | 300 | 100 | 1,000,000 |

---

## 🧪 Testing with cURL

### Test Health Check
```bash
curl http://localhost:8000/health
```

### Test with Authentication
```bash
# Set your Firebase token
TOKEN="your-firebase-id-token"

# Test auth endpoint
curl -X POST http://localhost:8000/api/v1/auth/verify \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json"
```

### Upload Resume
```bash
curl -X POST http://localhost:8000/api/v1/resume/upload \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@/path/to/resume.pdf"
```

---

## 📝 Notes

- All timestamps are in UTC ISO format
- File uploads limited to 10MB
- Encrypted fields should be base64 encoded
- Request IDs are included in error responses for debugging
- Analytics are collected without PII

---

For more examples and Postman collection, see the `tests/postman/` directory.