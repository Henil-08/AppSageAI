# API Documentation for AppSageAI

## Base URL
- **Development:** `http://localhost:8000`
- **Production:** `https://appsageai-backend-964026407675.us-central1.run.app`

## Authentication
All API endpoints (except health check) require Firebase authentication token in the header:
```
Authorization: Bearer <firebase-id-token>
```

---

## Health Check

### `/health`
**Method:** `GET`  
**Description:** Health check endpoint to verify API is running  
**Authentication:** Not required

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

## Authentication Endpoints

### `/api/v1/auth/verify`
**Method:** `POST`  
**Description:** Verify Firebase token and get/create user profile  
**Authentication:** Required

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

### `/api/v1/auth/me`
**Method:** `GET`  
**Description:** Get current authenticated user's profile  
**Authentication:** Required

**Response:** Same as `/verify` endpoint

### `/api/v1/auth/logout`
**Method:** `POST`  
**Description:** Logout endpoint (client-side token invalidation)  
**Authentication:** Required

**Response:**
```json
{
  "message": "Logged out successfully"
}
```

### `/api/v1/auth/check`
**Method:** `GET`  
**Description:** Quick auth status check without fetching full profile  
**Authentication:** Required

**Response:**
```json
{
  "authenticated": true,
  "uid": "user123",
  "email": "user@example.com",
  "email_verified": true
}
```

---

## Resume Management

### `/api/v1/resume/upload`
**Method:** `POST`  
**Description:** Upload a resume PDF file (encrypted server-side)  
**Authentication:** Required  
**Content-Type:** `multipart/form-data`

**Form Data:**
| Parameter | Type | Required | Description |
| --- | --- | --- | --- |
| file | File | Yes | PDF file to upload (max 10MB) |
| target_role | string | No | Target role for this resume version |

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

### `/api/v1/resume/list`
**Method:** `GET`  
**Description:** List all uploaded resumes for current user  
**Authentication:** Required

**Response:**
```json
{
  "resumes": [
    {
      "resume_id": "res_abc123",
      "filename": "john_doe_resume.pdf",
      "uploaded_at": "2024-01-15T10:30:00Z",
      "analysis_count": 5,
      "target_role": "Full Stack Developer",
      "is_active": true
    }
  ],
  "count": 1,
  "active_resume_id": "res_abc123"
}
```

### `/api/v1/resume/{resume_id}/set-active`
**Method:** `POST`  
**Description:** Set a resume as active for analysis  
**Authentication:** Required

**Path Parameters:**
| Parameter | Type | Description |
| --- | --- | --- |
| resume_id | string | Resume ID to set as active |

**Response:**
```json
{
  "message": "Active resume updated",
  "resume_id": "res_abc123"
}
```

### `/api/v1/resume/{resume_id}/download`
**Method:** `GET`  
**Description:** Download a resume file  
**Authentication:** Required

**Path Parameters:**
| Parameter | Type | Description |
| --- | --- | --- |
| resume_id | string | Resume ID to download |

**Response:** Binary PDF file with appropriate headers

### `/api/v1/resume/{resume_id}/target-role`
**Method:** `PATCH`  
**Description:** Update the target role for a resume  
**Authentication:** Required

**Path Parameters:**
| Parameter | Type | Description |
| --- | --- | --- |
| resume_id | string | Resume ID to update |

**Request Body:**
```json
{
  "target_role": "Senior Full Stack Developer"
}
```

**Response:**
```json
{
  "message": "Target role updated",
  "target_role": "Senior Full Stack Developer"
}
```

### `/api/v1/resume/{resume_id}`
**Method:** `DELETE`  
**Description:** Delete a resume  
**Authentication:** Required

**Path Parameters:**
| Parameter | Type | Description |
| --- | --- | --- |
| resume_id | string | Resume ID to delete |

**Response:**
```json
{
  "message": "Resume deleted successfully"
}
```

---

## Chat Management

### `/api/v1/chat/create`
**Method:** `POST`  
**Description:** Create a new chat session for job analysis  
**Authentication:** Required

**Query Parameters:**
| Parameter | Type | Required | Description |
| --- | --- | --- | --- |
| job_description | string | No | Job description text |
| job_title | string | No | Job title |
| company | string | No | Company name |
| initial_message | string | No | Initial message for the chat |

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

### `/api/v1/chat/list`
**Method:** `GET`  
**Description:** List all chat sessions for the user  
**Authentication:** Required

**Query Parameters:**
| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| limit | integer | 20 | Number of results to return |
| offset | integer | 0 | Pagination offset |

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
      "preview": "We are looking for a Senior...",
      "tracker_status": "applied",
      "applied_date": "2024-01-16",
      "tracker_notes": "Submitted via LinkedIn"
    }
  ],
  "total": 10,
  "limit": 20,
  "offset": 0,
  "has_more": false
}
```

### `/api/v1/chat/{session_id}`
**Method:** `GET`  
**Description:** Get a specific chat session with all messages  
**Authentication:** Required

**Path Parameters:**
| Parameter | Type | Description |
| --- | --- | --- |
| session_id | string | Chat session ID |

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
      "encrypted_content": "What are my chances?",
      "timestamp": "2024-01-15T10:35:00Z",
      "metadata": {}
    },
    {
      "message_id": "msg_002",
      "role": "assistant",
      "encrypted_content": "Based on my analysis...",
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

### `/api/v1/chat/{session_id}/message`
**Method:** `POST`  
**Description:** Add a message to the chat session  
**Authentication:** Required

**Path Parameters:**
| Parameter | Type | Description |
| --- | --- | --- |
| session_id | string | Chat session ID |

**Request Body:**
```json
{
  "role": "user",
  "encrypted_content": "What skills should I improve?",
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

### `/api/v1/chat/{session_id}/update`
**Method:** `PATCH`  
**Description:** Update chat session details  
**Authentication:** Required

**Path Parameters:**
| Parameter | Type | Description |
| --- | --- | --- |
| session_id | string | Chat session ID |

**Request Body:**
```json
{
  "job_title": "Updated Job Title",
  "company": "Updated Company",
  "job_description": "Updated job description..."
}
```

**Response:**
```json
{
  "message": "Chat details updated successfully",
  "session_id": "chat_xyz789",
  "job_title": "Updated Job Title",
  "company": "Updated Company"
}
```

### `/api/v1/chat/{session_id}/tracker`
**Method:** `PATCH`  
**Description:** Update job tracker status for a chat  
**Authentication:** Required

**Path Parameters:**
| Parameter | Type | Description |
| --- | --- | --- |
| session_id | string | Chat session ID |

**Query Parameters:**
| Parameter | Type | Required | Description |
| --- | --- | --- | --- |
| tracker_status | string | Yes | Status: interested, applied, interviewing, offered, rejected |
| applied_date | string | No | Date applied (ISO format) |
| notes | string | No | Tracker notes |

**Response:**
```json
{
  "message": "Tracker status updated",
  "status": "applied"
}
```

### `/api/v1/chat/{session_id}`
**Method:** `DELETE`  
**Description:** Delete a chat session and all its messages  
**Authentication:** Required

**Path Parameters:**
| Parameter | Type | Description |
| --- | --- | --- |
| session_id | string | Chat session ID |

**Response:**
```json
{
  "message": "Chat session deleted successfully"
}
```

---

## Analysis Endpoints

### `/api/v1/analysis/analyze/{session_id}`
**Method:** `POST`  
**Description:** Perform AI analysis on resume against job description  
**Authentication:** Required

**Path Parameters:**
| Parameter | Type | Description |
| --- | --- | --- |
| session_id | string | Chat session ID |

**Request Body:**
```json
{
  "analysis_type": "resume_review",
  "custom_query": null,
  "resume_id": null
}
```

**Analysis Types:**
- `resume_review` - Comprehensive resume review
- `skill_improvement` - Skill gap analysis and roadmap
- `keyword_analysis` - ATS keyword optimization
- `percentage_match` - Match percentage calculation
- `cover_letter` - Generate cover letter
- `custom_query` - Custom question (requires custom_query field)

**Response:**
```json
{
  "analysis_id": "analysis_001",
  "encrypted_response": "Analysis result...",
  "metadata": {
    "tokens_used": 1500,
    "response_time_ms": 2340,
    "model": "llama-3.3-70b-versatile",
    "analysis_type": "resume_review"
  },
  "timestamp": "2024-01-15T10:40:00Z"
}
```

### `/api/v1/analysis/analyze-stream/{session_id}`
**Method:** `POST`  
**Description:** Perform analysis with Server-Sent Events streaming  
**Authentication:** Required  
**Response Type:** `text/event-stream`

**Path Parameters:** Same as non-streaming endpoint  
**Request Body:** Same as non-streaming endpoint

**Response Stream:**
```
data: {"type": "token", "content": "Based"}
data: {"type": "token", "content": " on"}
data: {"type": "token", "content": " my"}
data: {"type": "metadata", "metadata": {"tokens_used": 150}}
data: {"type": "done", "analysis_id": "analysis_001"}
```

### `/api/v1/analysis/extract-job-details`
**Method:** `POST`  
**Description:** Extract job details from text using AI  
**Authentication:** Required

**Request Body:**
```json
{
  "text": "We are looking for a Senior Full Stack Developer..."
}
```

**Response:**
```json
{
  "job_title": "Senior Full Stack Developer",
  "company": "TechCorp Inc.",
  "job_description": "Formatted job description...",
  "location": "Remote",
  "salary": "$120,000 - $180,000",
  "job_type": "Full-time",
  "is_job_listing": true
}
```

### `/api/v1/analysis/quick-actions/{session_id}`
**Method:** `GET`  
**Description:** Get available quick actions for a chat session  
**Authentication:** Required

**Path Parameters:**
| Parameter | Type | Description |
| --- | --- | --- |
| session_id | string | Chat session ID |

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
  ]
}
```

### `/api/v1/analysis/feedback/{message_id}`
**Method:** `POST`  
**Description:** Submit or toggle feedback for a message  
**Authentication:** Required

**Path Parameters:**
| Parameter | Type | Description |
| --- | --- | --- |
| message_id | string | Message ID to provide feedback for |

**Request Body:**
```json
{
  "feedback_type": "thumbs_up",
  "comment": "Very helpful analysis!"
}
```

**Response:**
```json
{
  "message": "Thank you for your feedback!",
  "feedback_type": "thumbs_up"
}
```

### `/api/v1/analysis/feedback/bulk`
**Method:** `GET`  
**Description:** Get feedback status for multiple messages  
**Authentication:** Required

**Query Parameters:**
| Parameter | Type | Description |
| --- | --- | --- |
| message_ids | string | Comma-separated list of message IDs |

**Response:**
```json
{
  "feedback": {
    "msg_001": "thumbs_up",
    "msg_002": null,
    "msg_003": "thumbs_down"
  }
}
```

---

## Custom Prompts

### `/api/v1/prompts/defaults`
**Method:** `GET`  
**Description:** Get all default prompt templates  
**Authentication:** Required

**Response:**
```json
{
  "prompts": {
    "resume_review": "Default resume review prompt...",
    "skill_improvement": "Default skill improvement prompt...",
    "keyword_analysis": "Default ATS analysis prompt...",
    "percentage_match": "Default match calculation prompt...",
    "cover_letter": "Default cover letter prompt..."
  },
  "message": "Default prompts retrieved successfully"
}
```

### `/api/v1/prompts/custom`
**Method:** `GET`  
**Description:** Get user's custom prompt templates  
**Authentication:** Required

**Response:**
```json
{
  "prompts": {
    "resume_review": "Custom resume review prompt...",
    "skill_improvement": "Custom skill improvement prompt..."
  },
  "has_custom": true,
  "message": "Custom prompts retrieved successfully"
}
```

### `/api/v1/prompts/update`
**Method:** `POST`  
**Description:** Update a single custom prompt template  
**Authentication:** Required

**Request Body:**
```json
{
  "analysis_type": "resume_review",
  "prompt_template": "Your custom prompt with {context} placeholder..."
}
```

**Note:** Prompt must include `{context}` placeholder for RAG to work

**Response:**
```json
{
  "message": "Prompt updated successfully",
  "analysis_type": "resume_review"
}
```

### `/api/v1/prompts/update-all`
**Method:** `POST`  
**Description:** Update all custom prompts at once  
**Authentication:** Required

**Request Body:**
```json
{
  "prompts": {
    "resume_review": "Custom prompt 1 with {context}...",
    "skill_improvement": "Custom prompt 2 with {context}..."
  }
}
```

**Response:**
```json
{
  "message": "All prompts updated successfully",
  "updated_count": 2
}
```

### `/api/v1/prompts/reset/{analysis_type}`
**Method:** `POST`  
**Description:** Reset a specific prompt to default  
**Authentication:** Required

**Path Parameters:**
| Parameter | Type | Description |
| --- | --- | --- |
| analysis_type | string | Type of analysis prompt to reset |

**Response:**
```json
{
  "message": "Prompt reset to default",
  "analysis_type": "resume_review",
  "default_prompt": "Default prompt content..."
}
```

### `/api/v1/prompts/reset-all`
**Method:** `POST`  
**Description:** Reset all prompts to default  
**Authentication:** Required

**Response:**
```json
{
  "message": "All prompts reset to default successfully",
  "deleted_count": 3
}
```

### `/api/v1/prompts/validate/{analysis_type}`
**Method:** `GET`  
**Description:** Validate if a custom prompt exists and is properly formatted  
**Authentication:** Required

**Path Parameters:**
| Parameter | Type | Description |
| --- | --- | --- |
| analysis_type | string | Type of analysis prompt to validate |

**Response:**
```json
{
  "valid": true,
  "uses_default": false,
  "has_context": true,
  "message": "Custom prompt is valid"
}
```

---

## User Management

### `/api/v1/user/stats`
**Method:** `GET`  
**Description:** Get user statistics and usage  
**Authentication:** Required

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

### `/api/v1/user/usage`
**Method:** `GET`  
**Description:** Get usage limits and current usage  
**Authentication:** Required

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

### `/api/v1/user/account`
**Method:** `DELETE`  
**Description:** Delete user account and all associated data (irreversible)  
**Authentication:** Required

**Query Parameters:**
| Parameter | Type | Required | Description |
| --- | --- | --- |
| confirm | boolean | Yes | Must be `true` to confirm deletion |

**Response:**
```json
{
  "message": "Account deleted successfully"
}
```

---

## Error Responses

All endpoints may return the following error responses:

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

## Rate Limits

| Plan | Requests/Minute | Daily Analyses | Monthly Tokens |
|------|-----------------|----------------|----------------|
| Free | 60 | 10 | 100,000 |

---

## Security Notes

- All sensitive data is encrypted server-side using AES-256 encryption
- Authentication is handled via Firebase Auth with JWT tokens
- CORS is configured to allow only whitelisted origins
- All API calls must use HTTPS in production
- Rate limiting is enforced to prevent abuse

---

## Testing with cURL

### Health Check
```bash
curl http://localhost:8000/health
```

### Authenticated Request Example
```bash
# Get Firebase token from browser console or auth flow
TOKEN="your-firebase-id-token"

# Test authentication
curl -X POST http://localhost:8000/api/v1/auth/verify \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json"

# Upload Resume
curl -X POST http://localhost:8000/api/v1/resume/upload \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@resume.pdf" \
  -F "target_role=Full Stack Developer"

# Create Chat Session
curl -X POST "http://localhost:8000/api/v1/chat/create?job_title=Developer&company=TechCorp" \
  -H "Authorization: Bearer $TOKEN"

# Perform Analysis
curl -X POST http://localhost:8000/api/v1/analysis/analyze/chat_xyz789 \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"analysis_type": "resume_review"}'
```