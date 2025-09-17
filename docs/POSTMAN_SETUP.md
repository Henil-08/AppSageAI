# Postman Setup & Testing Guide

## 🔑 Getting Firebase Token for API Testing

### What is Firebase Token?
The Firebase token (ID token) is a JWT that proves a user's identity. Since our API uses Firebase Authentication, you need this token to make authenticated requests.

## Method 1: Using Firebase Auth Emulator (Development)

### Step 1: Install Firebase Tools
```bash
npm install -g firebase-tools
```

### Step 2: Set up Firebase Emulator
```bash
# In project root
firebase init emulators
# Select Authentication emulator
firebase emulators:start
```

### Step 3: Get Test Token
Visit: http://localhost:9099/emulator/v1/projects/YOUR_PROJECT/accounts

## Method 2: Using Browser Console (Quick Testing)

### Step 1: Create a Simple Test Page
Create `test-auth.html`:
```html
<!DOCTYPE html>
<html>
<head>
    <title>Firebase Token Generator</title>
    <script src="https://www.gstatic.com/firebasejs/10.7.1/firebase-app-compat.js"></script>
    <script src="https://www.gstatic.com/firebasejs/10.7.1/firebase-auth-compat.js"></script>
</head>
<body>
    <h1>Firebase Token Generator for Testing</h1>
    <button onclick="signIn()">Sign In with Google</button>
    <button onclick="getToken()">Get Token</button>
    <button onclick="signOut()">Sign Out</button>
    
    <div id="result">
        <h3>Your Token:</h3>
        <textarea id="token" rows="10" cols="80"></textarea>
        <button onclick="copyToken()">Copy Token</button>
    </div>

    <script>
        // Your Firebase config (from Firebase Console)
        const firebaseConfig = {
            apiKey: "YOUR_API_KEY",
            authDomain: "appsageai-472321.firebaseapp.com",
            projectId: "appsageai-472321",
            storageBucket: "appsageai-472321.appspot.com",
            messagingSenderId: "YOUR_SENDER_ID",
            appId: "YOUR_APP_ID"
        };

        // Initialize Firebase
        firebase.initializeApp(firebaseConfig);
        const auth = firebase.auth();
        
        function signIn() {
            const provider = new firebase.auth.GoogleAuthProvider();
            auth.signInWithPopup(provider)
                .then((result) => {
                    console.log("Signed in:", result.user.email);
                    alert("Signed in successfully! Now click 'Get Token'");
                })
                .catch((error) => {
                    console.error("Error signing in:", error);
                });
        }
        
        async function getToken() {
            const user = auth.currentUser;
            if (user) {
                const token = await user.getIdToken();
                document.getElementById('token').value = token;
                console.log("Token generated!");
                alert("Token generated! Copy it to use in Postman");
            } else {
                alert("Please sign in first!");
            }
        }
        
        function copyToken() {
            const tokenField = document.getElementById('token');
            tokenField.select();
            document.execCommand('copy');
            alert("Token copied to clipboard!");
        }
        
        function signOut() {
            auth.signOut().then(() => {
                alert("Signed out!");
                document.getElementById('token').value = "";
            });
        }
    </script>
</body>
</html>
```

### Step 2: Use the Test Page
1. Replace Firebase config values with your actual values
2. Open the HTML file in a browser
3. Click "Sign In with Google"
4. Click "Get Token"
5. Copy the token

## Method 3: Using cURL (Command Line)

### Step 1: Get Google OAuth Token
```bash
# This is complex, easier to use the HTML method above
```

## Method 4: Using Python Script

Create `get_firebase_token.py`:
```python
import firebase_admin
from firebase_admin import auth, credentials
import google.auth
from google.auth.transport.requests import Request
from google.oauth2 import id_token

# Initialize Firebase Admin
cred = credentials.Certificate('backend/service-account.json')
firebase_admin.initialize_app(cred)

# Create custom token
uid = 'test-user-123'
custom_token = auth.create_custom_token(uid)
print(f"Custom Token: {custom_token.decode()}")

# Note: You'll need to exchange this custom token for an ID token
# using Firebase client SDK
```

## 📮 Setting Up Postman

### Step 1: Import Collection
1. Open Postman
2. Click **Import** → **Upload Files**
3. Select `backend/tests/postman_collection.json`

### Step 2: Create Environment
1. Click **Environments** (left sidebar)
2. Click **+** to create new environment
3. Name it: `AppSageAI Development`
4. Add variables:

| Variable | Type | Initial Value | Current Value |
|----------|------|---------------|---------------|
| `base_url` | default | `http://localhost:8000` | `http://localhost:8000` |
| `api_version` | default | `v1` | `v1` |
| `firebase_token` | secret | | `<paste-your-token-here>` |
| `session_id` | default | | |
| `resume_id` | default | | |
| `analysis_id` | default | | |

### Step 3: Set Firebase Token
1. Get token using one of the methods above
2. In Postman environment, paste token in `firebase_token` Current Value
3. Save the environment
4. Select the environment from the dropdown (top-right)

### Step 4: Test Authentication
1. Run **Health Check** (no auth required) - Should return 200
2. Run **Verify Token** - Should return your user profile

## 🧪 Testing Workflow in Postman

### Complete User Journey Test

#### 1. Setup & Authentication
```
✅ Health Check
✅ Verify Token
✅ Get Current User
```

#### 2. Resume Upload (One Time)
```
✅ Upload Resume
   - Select a PDF file in Body → form-data → file
✅ List Resumes
✅ Set Active Resume
```

#### 3. Create Chat Session
```
✅ Create Chat Session
   - Provide job_description in body
   - Note: session_id is saved automatically
```

#### 4. Perform Analyses
```
✅ Resume Review Analysis
✅ ATS Keyword Analysis  
✅ Match Percentage
✅ Skill Improvement Roadmap
✅ Generate Cover Letter
```

#### 5. Chat Interaction
```
✅ Add Message to Chat
✅ Get Chat Session
✅ List Chat Sessions
```

#### 6. Feedback & Stats
```
✅ Submit Feedback
✅ Get User Stats
✅ Get Usage Limits
```

## 🔄 Automated Testing

### Run Collection
1. Click **Collections** → **AppSageAI API**
2. Click **Run** (three dots menu)
3. Select environment
4. Configure:
   - Iterations: 1
   - Delay: 500ms
   - Save responses: Yes
5. Click **Run AppSageAI API**

### Test Results
- Green ✅ = Test passed
- Red ❌ = Test failed
- View response details for debugging

## 🐛 Common Issues

### Issue: 401 Unauthorized
**Solution**: Token expired. Generate a new token.

### Issue: 403 Forbidden  
**Solution**: Token is valid but missing required claims. Check Firebase Auth settings.

### Issue: 400 Bad Request
**Solution**: Check request body format. Ensure JSON is valid.

### Issue: Network Error
**Solution**: Ensure backend server is running on port 8000.

## 📝 Test Data Files

Create `test-data/` folder with:

### sample-resume.pdf
Use any PDF resume for testing

### job-descriptions.json
```json
{
  "software_engineer": {
    "title": "Senior Software Engineer",
    "company": "TechCorp",
    "description": "We are looking for a Senior Software Engineer..."
  },
  "data_scientist": {
    "title": "Data Scientist",
    "company": "DataCo",
    "description": "Join our data science team..."
  }
}
```

## 🔒 Security Notes

- **Never commit** tokens to version control
- Tokens expire after 1 hour
- Use environment variables for sensitive data
- Create separate environments for dev/staging/production

## 📊 Performance Testing

### Load Testing with Newman
```bash
# Install Newman
npm install -g newman

# Run collection
newman run AppSageAI_API.postman_collection.json \
  -e AppSageAI_Development.postman_environment.json \
  -n 10 \
  --delay-request 1000
```

### Monitor API Performance
- Response times should be < 2s for analyses
- Health check should be < 100ms
- Upload should handle 10MB files

## 📚 Additional Resources

- [Postman Documentation](https://learning.postman.com/docs/)
- [Firebase Auth REST API](https://firebase.google.com/docs/reference/rest/auth)
- [Newman CLI](https://www.npmjs.com/package/newman)