'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '../contexts/AuthContext';
import { Sparkles, ArrowRight, Shield, Zap, Brain } from 'lucide-react';

export default function HomePage() {
  const { user, loading, signIn } = useAuth();
  const router = useRouter();
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    if (user && !loading) {
      router.push('/dashboard');
    }
  }, [user, loading, router]);

  useEffect(() => {
    // Trigger animations on mount
    setIsVisible(true);
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-gray-50 to-orange-50">
        <div className="flex flex-col items-center space-y-4">
          <div className="w-12 h-12 border-4 border-orange-500 border-t-transparent rounded-full animate-spin"></div>
          <p className="text-gray-600">Loading...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-orange-50 flex flex-col">
      {/* Navigation Bar */}
      <nav className="w-full bg-white/80 backdrop-blur-md border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            {/* Logo */}
            <div className="flex items-center space-x-2">
              <Sparkles className="w-7 h-7 text-orange-500" />
              <span className="text-xl font-semibold text-gray-900">AppSageAI</span>
            </div>
            
            {/* Get Started Button */}
            <button
              onClick={signIn}
              className="px-4 py-2 bg-orange-500 text-white font-medium rounded-lg hover:bg-orange-600 transition-all duration-200 transform hover:scale-105 flex items-center space-x-2"
            >
              <span>Get Started</span>
              <ArrowRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      </nav>

      {/* Main Content - Centered */}
      <main className="flex-1 flex items-center justify-center px-4 py-16">
        <div className="max-w-7xl mx-auto w-full">
          {/* Hero Section */}
          <div className="text-center">
            {/* Badge with animation */}
            <div 
              className={`inline-flex items-center space-x-2 bg-orange-100 text-orange-600 px-3 py-1 rounded-full text-sm font-medium mb-6 transition-all duration-700 transform ${
                isVisible ? 'translate-y-0 opacity-100' : '-translate-y-4 opacity-0'
              }`}
            >
              <Sparkles className="w-4 h-4" />
              <span>AI-Powered Resume Analysis</span>
            </div>

            {/* Main Headline with animation */}
            <h1 
              className={`text-5xl md:text-6xl font-bold text-gray-900 mb-6 transition-all duration-700 delay-100 transform ${
                isVisible ? 'translate-y-0 opacity-100' : 'translate-y-4 opacity-0'
              }`}
            >
              Land Your Dream Job with
              <span className="bg-gradient-to-r from-orange-500 to-orange-600 bg-clip-text text-transparent"> Intelligence</span>
            </h1>

            {/* Subheadline with animation */}
            <p 
              className={`text-xl text-gray-600 max-w-3xl mx-auto mb-8 transition-all duration-700 delay-200 transform ${
                isVisible ? 'translate-y-0 opacity-100' : 'translate-y-4 opacity-0'
              }`}
            >
              Upload your resume once, analyze it against any job description, and get
              personalized insights to maximize your chances of success.
            </p>

            {/* Powered by Meta Llama Badge */}
            <div 
              className={`inline-flex items-center space-x-3 bg-white rounded-full px-4 py-2 shadow-md mb-8 transition-all duration-700 delay-300 transform ${
                isVisible ? 'scale-100 opacity-100' : 'scale-95 opacity-0'
              }`}
            >
              {/* Meta Logo SVG */}
              <svg className="w-6 h-6" viewBox="0 0 24 24" fill="none">
                <path d="M12 0C5.373 0 0 5.373 0 12s5.373 12 12 12 12-5.373 12-12S18.627 0 12 0zm5.894 8.221l-1.97-1.97a4.37 4.37 0 00-3.067-1.267 4.37 4.37 0 00-3.067 1.267c-.27.27-.49.576-.656.906a3.578 3.578 0 00-.656-.906 4.37 4.37 0 00-3.067-1.267 4.37 4.37 0 00-3.067 1.267l-1.97 1.97a.75.75 0 000 1.06l7.814 7.814a.75.75 0 001.06 0l7.814-7.814a.75.75 0 000-1.06z" fill="#0866FF"/>
              </svg>
              <span className="text-sm font-medium text-gray-700">
                Powered by <span className="font-semibold">Meta Llama 3.3 70B</span>
              </span>
              <span className="text-xs bg-blue-100 text-blue-700 px-2 py-0.5 rounded-full font-medium">
                State-of-the-art
              </span>
            </div>

            {/* Action Buttons with animation */}
            <div 
              className={`flex flex-col sm:flex-row gap-4 justify-center transition-all duration-700 delay-400 transform ${
                isVisible ? 'translate-y-0 opacity-100' : 'translate-y-4 opacity-0'
              }`}
            >
              {/* Google Sign In Button */}
              <button
                onClick={signIn}
                className="px-8 py-3 bg-orange-500 text-white font-medium rounded-lg hover:bg-orange-600 transition-all duration-200 transform hover:scale-105 flex items-center justify-center space-x-3 shadow-lg animate-pulse-slow"
              >
                {/* Google Icon */}
                <svg className="w-5 h-5" viewBox="0 0 24 24">
                  <path fill="currentColor" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                  <path fill="currentColor" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                  <path fill="currentColor" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                  <path fill="currentColor" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
                </svg>
                <span>Sign in with Google</span>
              </button>
              
              {/* Watch Demo Button */}
              <button className="px-8 py-3 bg-white text-gray-700 font-medium rounded-lg border border-gray-300 hover:bg-gray-50 hover:shadow-md transition-all duration-200 transform hover:scale-105">
                Watch Demo
              </button>
            </div>
          </div>

          {/* Features Grid with staggered animations */}
          <div className="grid md:grid-cols-3 gap-8 mt-20">
            {/* Secure & Private Card */}
            <div 
              className={`bg-white rounded-xl p-6 shadow-md hover:shadow-xl transition-all duration-500 delay-500 transform hover:-translate-y-1 ${
                isVisible ? 'translate-y-0 opacity-100' : 'translate-y-8 opacity-0'
              }`}
            >
              <Shield className="w-10 h-10 text-orange-500 mb-4" />
              <h3 className="text-lg font-semibold text-gray-900 mb-2">Secure & Private</h3>
              <p className="text-gray-600">
                Your data is encrypted and stored securely. We prioritize your privacy.
              </p>
            </div>
            
            {/* Instant Analysis Card */}
            <div 
              className={`bg-white rounded-xl p-6 shadow-md hover:shadow-xl transition-all duration-500 delay-600 transform hover:-translate-y-1 ${
                isVisible ? 'translate-y-0 opacity-100' : 'translate-y-8 opacity-0'
              }`}
            >
              <Zap className="w-10 h-10 text-orange-500 mb-4" />
              <h3 className="text-lg font-semibold text-gray-900 mb-2">Instant Analysis</h3>
              <p className="text-gray-600">
                Get comprehensive feedback in seconds using cutting-edge AI.
              </p>
            </div>
            
            {/* AI-Powered Card */}
            <div 
              className={`bg-white rounded-xl p-6 shadow-md hover:shadow-xl transition-all duration-500 delay-700 transform hover:-translate-y-1 ${
                isVisible ? 'translate-y-0 opacity-100' : 'translate-y-8 opacity-0'
              }`}
            >
              <Brain className="w-10 h-10 text-orange-500 mb-4" />
              <h3 className="text-lg font-semibold text-gray-900 mb-2">Advanced AI Models</h3>
              <p className="text-gray-600">
                Powered by Meta's Llama 3.3 70B for accurate, actionable insights.
              </p>
            </div>
          </div>
        </div>
      </main>

      {/* Add custom animation styles */}
      <style jsx global>{`
        @keyframes pulse-slow {
          0%, 100% {
            opacity: 1;
          }
          50% {
            opacity: 0.9;
          }
        }
        
        .animate-pulse-slow {
          animation: pulse-slow 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
        }
      `}</style>
    </div>
  );
}