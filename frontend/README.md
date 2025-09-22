# AppSageAI Frontend

Next.js 14 frontend for AppSageAI - Privacy-first AI Resume Analysis Platform.

## Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/appsageai.git
cd appsageai/frontend

# Install dependencies
npm install

# Setup environment
cp .env.local.example .env.local
# Edit .env.local with your Firebase configuration

# Run development server
npm run dev

# Open browser
open http://localhost:3000
```

## Project Structure

```
frontend/
├── src/
│   ├── app/                    # Next.js App Router
│   │   ├── (auth)/            # Authentication pages
│   │   ├── dashboard/         # Main application
│   │   │   ├── page.tsx       # Dashboard home
│   │   │   ├── chat/          # Chat interface
│   │   │   ├── resume/        # Resume management
│   │   │   ├── jobs/          # Job tracker
│   │   │   └── settings/      # Custom prompts
│   │   ├── layout.tsx         # Root layout
│   │   ├── page.tsx           # Landing page
│   │   └── globals.css        # Global styles
│   ├── components/            # React components
│   │   ├── ConfirmationModal.tsx
│   │   └── ResumeSelector.tsx
│   ├── contexts/              # Context providers
│   │   ├── AuthContext.tsx    # Firebase auth
│   │   └── SidebarContext.tsx # UI state
│   ├── lib/                   # Utilities
│   │   └── firebase.ts        # Firebase config
│   └── types/                 # TypeScript types
├── public/                    # Static assets
│   ├── appsageai-icon.png
│   ├── shield-privacy.png
│   └── ...
├── package.json
├── tsconfig.json
├── tailwind.config.js
├── next.config.js
└── .env.local                 # Environment variables (git-ignored)
```

## Development Commands

### Running the Application

```bash
# Development server with hot reload
npm run dev

# Production build
npm run build

# Start production server
npm start

# Analyze bundle size
npm run analyze
```

### Code Quality

```bash
# Type checking
npm run type-check

# Linting
npm run lint
npm run lint:fix

# Format with Prettier
npm run format

# Run all checks
npm run check-all
```

## Environment Configuration

Create `.env.local` with the following variables:

```env
# Firebase Configuration (Public keys - safe to expose)
NEXT_PUBLIC_FIREBASE_API_KEY=your-firebase-api-key
NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN=your-project.firebaseapp.com
NEXT_PUBLIC_FIREBASE_PROJECT_ID=your-project-id
NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET=your-project.firebasestorage.app
NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID=your-sender-id
NEXT_PUBLIC_FIREBASE_APP_ID=your-app-id

# API Configuration
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_ENVIRONMENT=development

# Optional: Analytics
NEXT_PUBLIC_GA_MEASUREMENT_ID=G-XXXXXXXXXX
```

## UI Components & Features

### Core Features

- **Authentication**: Google OAuth via Firebase
- **Chat Interface**: Real-time streaming responses
- **Resume Management**: Multi-resume support with tagging
- **Quick Actions**: 5 AI analysis types
- **Job Tracker**: Application status management
- **Custom Prompts**: Personalize AI responses
- **Responsive Design**: Mobile-first approach

### Design System

We use a Claude-inspired design system with Tailwind CSS:

```javascript
// Color palette
claude: {
  background: '#FAFAF8',
  surface: '#FFFFFF',
  border: '#E5E5E0',
  text: {
    primary: '#2D2D2D',
    secondary: '#706F6C',
    muted: '#A8A29E',
  },
  accent: {
    orange: '#EA5A0C',
    'orange-light': '#FFF4ED',
    'orange-hover': '#DC4A00',
  },
}
```

## Testing

```bash
# Run unit tests
npm test

# Run tests in watch mode
npm run test:watch

# Run E2E tests with Playwright
npm run test:e2e

# Generate test coverage
npm run test:coverage
```

## Building for Production

```bash
# Create production build
npm run build

# Check for build errors
npm run build:check

# Analyze bundle size
npm run build:analyze

# Start production server locally
npm start
```

### Production Optimizations

- **Image Optimization**: Next.js Image component with lazy loading
- **Code Splitting**: Automatic route-based splitting
- **Font Optimization**: Next/font with preloading
- **Static Generation**: Pre-rendered marketing pages
- **API Route Caching**: SWR for data fetching

## Performance Targets

- **Lighthouse Score**: 95+ on all metrics
- **First Contentful Paint**: < 1.2s
- **Time to Interactive**: < 3.5s
- **Bundle Size**: < 200KB initial JS

## Key User Flows

### 1. Authentication Flow
```
Landing Page → Google Sign In → Dashboard
```

### 2. Resume Analysis Flow
```
Upload Resume → Create Chat → Paste Job Description → Select Analysis → View Results
```

### 3. Job Application Tracking
```
Chat Session → Update Status → Track Applications → Export Data
```

## Security Features

- **Firebase Authentication**: Secure OAuth flow
- **Content Security Policy**: XSS protection
- **HTTPS Only**: Enforced in production
- **Secure Cookies**: HttpOnly, SameSite
- **Input Sanitization**: React default escaping

## Debugging

```bash
# Enable debug mode
DEBUG=* npm run dev

# Check Next.js build output
npm run build -- --debug

# Analyze webpack bundle
ANALYZE=true npm run build

# Clear Next.js cache
rm -rf .next
```

### Browser DevTools

1. **React DevTools**: Component inspection
2. **Redux DevTools**: State management (if using)
3. **Network Tab**: API call monitoring
4. **Console**: Error tracking

## Responsive Design

### Breakpoints
```css
sm: 640px   /* Mobile landscape */
md: 768px   /* Tablet */
lg: 1024px  /* Desktop */
xl: 1280px  /* Wide desktop */
2xl: 1536px /* Ultra-wide */
```

### Mobile-First Approach
```jsx
className="text-sm md:text-base lg:text-lg"
className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3"
```

## API Integration

### Fetch Configuration
```typescript
// Base configuration for API calls
const API_BASE = process.env.NEXT_PUBLIC_API_URL;

const fetchConfig = {
  headers: {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json',
  },
};
```

### Key API Endpoints Used
- `POST /api/v1/auth/verify` - Verify Firebase token
- `GET /api/v1/resume/list` - List resumes
- `POST /api/v1/chat/create` - Create chat session
- `POST /api/v1/analysis/analyze-stream/{id}` - Stream analysis

## Package Scripts

```json
{
  "dev": "next dev",
  "build": "next build",
  "start": "next start",
  "lint": "next lint",
  "type-check": "tsc --noEmit",
  "format": "prettier --write .",
  "test": "jest",
  "test:watch": "jest --watch",
  "test:coverage": "jest --coverage"
}
```

## Troubleshooting

### Common Issues

**Module not found errors**
```bash
# Clear node_modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

**Firebase Auth issues**
```bash
# Check Firebase config in .env.local
# Ensure domain is whitelisted in Firebase Console
```

**API connection errors**
```bash
# Verify backend is running
curl http://localhost:8000/health

# Check CORS settings in backend
```

**Build failures**
```bash
# Clear Next.js cache
rm -rf .next
npm run build
```

## 🔗 Related Documentation

- [Complete Setup Guide](../docs/SETUP.md)
- [Backend README](../backend/README.md)
- [API Documentation](../docs/API_DOCUMENTATION.md)
- [Main README](../README.md)

## 📧 Support

For issues or questions:
- Create an issue on GitHub
- Check existing issues for solutions
- Review the setup guide

---

**Built with Next.js 14, TypeScript, Tailwind CSS, and Firebase**