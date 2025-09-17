'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '../contexts/AuthContext';
import { Sparkles, ArrowRight, Shield, Zap, Brain } from 'lucide-react';

export default function HomePage() {
  const { user, loading, signIn } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (user && !loading) {
      router.push('/dashboard');
    }
  }, [user, loading, router]);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="flex flex-col items-center space-y-4">
          <div className="w-12 h-12 border-4 border-orange-500 border-t-transparent rounded-full animate-spin"></div>
          <p className="text-gray-600">Loading...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-orange-50">
      {/* Navigation Bar */}
      <nav className="sticky top-0 w-full bg-white/80 backdrop-blur-md border-b border-gray-200 z-50">
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
              className="px-4 py-2 bg-orange-500 text-white font-medium rounded-lg hover:bg-orange-600 transition-colors duration-200 flex items-center space-x-2"
            >
              <span>Get Started</span>
              <ArrowRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="pt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
          {/* Hero Section */}
          <div className="text-center mb-20">
            {/* Badge */}
            <div className="inline-flex items-center space-x-2 bg-orange-100 text-orange-600 px-3 py-1 rounded-full text-sm font-medium mb-6">
              <Sparkles className="w-4 h-4" />
              <span>AI-Powered Resume Analysis</span>
            </div>

            {/* Main Headline */}
            <h1 className="text-5xl md:text-6xl font-bold text-gray-900 mb-6">
              Land Your Dream Job with
              <span className="bg-gradient-to-r from-orange-500 to-orange-600 bg-clip-text text-transparent"> Intelligence</span>
            </h1>

            {/* Subheadline */}
            <p className="text-xl text-gray-600 max-w-3xl mx-auto mb-8">
              Upload your resume once, analyze it against any job description, and get
              personalized insights to maximize your chances of success.
            </p>

            {/* Action Buttons */}
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              {/* Google Sign In Button */}
              <button
                onClick={signIn}
                className="px-8 py-3 bg-orange-500 text-white font-medium rounded-lg hover:bg-orange-600 transition-all duration-200 transform hover:scale-105 flex items-center justify-center space-x-3 shadow-lg"
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
              <button className="px-8 py-3 bg-white text-gray-700 font-medium rounded-lg border border-gray-300 hover:bg-gray-50 transition-all duration-200">
                Watch Demo
              </button>
            </div>
          </div>

          {/* Features Grid */}
          <div className="grid md:grid-cols-3 gap-8 mb-20">
            {/* Privacy First Card */}
            <div className="bg-white rounded-xl p-6 shadow-md hover:shadow-xl transition-shadow duration-300 border border-gray-100">
              <Shield className="w-10 h-10 text-orange-500 mb-4" />
              <h3 className="text-lg font-semibold text-gray-900 mb-2">Privacy First</h3>
              <p className="text-gray-600">
                Your data is encrypted client-side. We never see your resume content.
              </p>
            </div>
            
            {/* Instant Analysis Card */}
            <div className="bg-white rounded-xl p-6 shadow-md hover:shadow-xl transition-shadow duration-300 border border-gray-100">
              <Zap className="w-10 h-10 text-orange-500 mb-4" />
              <h3 className="text-lg font-semibold text-gray-900 mb-2">Instant Analysis</h3>
              <p className="text-gray-600">
                Get comprehensive feedback in seconds, not hours.
              </p>
            </div>
            
            {/* AI-Powered Card */}
            <div className="bg-white rounded-xl p-6 shadow-md hover:shadow-xl transition-shadow duration-300 border border-gray-100">
              <Brain className="w-10 h-10 text-orange-500 mb-4" />
              <h3 className="text-lg font-semibold text-gray-900 mb-2">AI-Powered Insights</h3>
              <p className="text-gray-600">
                Powered by advanced AI to give you actionable recommendations.
              </p>
            </div>
          </div>

          {/* Stats Section */}
          <div className="bg-white rounded-2xl p-8 shadow-lg border border-gray-100">
            <div className="flex justify-around">
              <div className="text-center">
                <div className="text-3xl font-bold text-gray-900">10k+</div>
                <div className="text-gray-600 mt-1">Resumes Analyzed</div>
              </div>
              <div className="text-center border-l border-gray-200 pl-8">
                <div className="text-3xl font-bold text-gray-900">95%</div>
                <div className="text-gray-600 mt-1">Success Rate</div>
              </div>
              <div className="text-center border-l border-gray-200 pl-8">
                <div className="text-3xl font-bold text-gray-900">2 sec</div>
                <div className="text-gray-600 mt-1">Avg. Response</div>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}