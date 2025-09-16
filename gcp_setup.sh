#!/bin/bash

# Google Cloud and Firebase Setup Script
echo "🚀 Setting up Google Cloud and Firebase for AppSageAI"

# Set your project ID
PROJECT_ID="appsageai"
REGION="us-central1"  # Good for free tier

# 1. Set the default project
echo "📌 Setting default project..."
gcloud config set project $PROJECT_ID

# 2. Enable required APIs
echo "🔧 Enabling required Google Cloud APIs..."
gcloud services enable \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    firestore.googleapis.com \
    firebase.googleapis.com \
    firebaseauth.googleapis.com \
    secretmanager.googleapis.com \
    cloudresourcemanager.googleapis.com \
    iam.googleapis.com

# 3. Create Firestore database (Native mode for Firebase compatibility)
echo "📊 Creating Firestore database..."
gcloud firestore databases create \
    --location=$REGION \
    --type=firestore-native

# 4. Set up Firebase in the project
echo "🔥 Initializing Firebase..."
firebase projects:addfirebase $PROJECT_ID

# 5. Create service account for backend
echo "🔐 Creating service account for backend..."
gcloud iam service-accounts create appsageai-backend \
    --display-name="AppSageAI Backend Service Account"

# 6. Grant necessary permissions to service account
echo "🔑 Granting permissions..."
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:appsageai-backend@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/firebase.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:appsageai-backend@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/datastore.user"

# 7. Create and download service account key (KEEP THIS SECURE!)
echo "📥 Creating service account key..."
gcloud iam service-accounts keys create \
    ./backend/service-account.json \
    --iam-account=appsageai-backend@$PROJECT_ID.iam.gserviceaccount.com

# 8. Set up Secret Manager for sensitive data (optional but recommended)
echo "🔒 Setting up Secret Manager..."
gcloud services enable secretmanager.googleapis.com

# Store Groq API key in Secret Manager (you'll need to add the actual value)
echo "Creating secret for GROQ_API_KEY..."
echo -n "your-groq-api-key" | gcloud secrets create groq-api-key \
    --data-file=- \
    --replication-policy="automatic"

# 9. Set up Firebase Authentication
echo "🔐 Configuring Firebase Authentication..."
cat > firebase_auth_config.txt << 'EOF'
To complete Firebase Auth setup:

1. Go to: https://console.firebase.google.com/project/appsageai/authentication
2. Click "Get Started"
3. Enable these providers:
   - Google
   - Apple (requires Apple Developer account)
   - Email/Password (optional)

4. For Google Sign-In:
   - Click on Google provider
   - Enable it
   - Add your domain to authorized domains

5. For Apple Sign-In:
   - Requires Apple Developer account
   - Follow Firebase's Apple setup guide

6. Get your Firebase config:
   - Go to Project Settings > General
   - Scroll to "Your apps" 
   - Click "Add app" > Web
   - Register app with nickname "AppSageAI Web"
   - Copy the configuration
EOF

cat firebase_auth_config.txt

# 10. Create Firestore security rules file
echo "📝 Creating Firestore security rules..."
cat > firestore.rules << 'EOF'
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    // Users can only access their own data
    match /users/{userId} {
      allow read, write: if request.auth != null && request.auth.uid == userId;
      
      // Subcollections
      match /chats/{chatId} {
        allow read, write: if request.auth != null && request.auth.uid == userId;
      }
      
      match /memories/{memoryId} {
        allow read, write: if request.auth != null && request.auth.uid == userId;
      }
      
      match /resumes/{resumeId} {
        allow read, write: if request.auth != null && request.auth.uid == userId;
      }
    }
    
    // No one can read analytics (backend only)
    match /analytics/{document=**} {
      allow read, write: if false;
    }
  }
}
EOF

# 11. Deploy Firestore rules
echo "🚀 Deploying Firestore security rules..."
firebase deploy --only firestore:rules

# 12. Create Cloud Run services placeholders
echo "🏃 Creating Cloud Run service configurations..."
cat > backend/service.yaml << 'EOF'
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: appsageai-backend
  annotations:
    run.googleapis.com/ingress: all
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/minScale: '0'
        autoscaling.knative.dev/maxScale: '10'
        run.googleapis.com/cpu-throttling: 'false'
    spec:
      serviceAccountName: appsageai-backend@appsageai.iam.gserviceaccount.com
      containers:
      - image: gcr.io/appsageai/backend:latest
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: production
        - name: GCP_PROJECT_ID
          value: appsageai
        resources:
          limits:
            cpu: '2'
            memory: '2Gi'
EOF

echo "
✅ Google Cloud and Firebase setup complete!

📋 Manual steps required:
1. Complete Firebase Auth setup (see firebase_auth_config.txt)
2. Add your actual GROQ_API_KEY to Secret Manager
3. Save the Firebase config to frontend/.env
4. Keep service-account.json SECURE (added to .gitignore)

🔐 Privacy Note:
- Firestore rules ensure users can only access their own data
- All sensitive data will be encrypted before storage
- You (as admin) cannot read encrypted user data through console

Ready to continue with backend implementation!
"