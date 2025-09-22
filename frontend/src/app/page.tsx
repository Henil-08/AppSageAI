'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '../contexts/AuthContext';
import { Sparkles, Lock } from 'lucide-react';

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
    <div className="h-screen overflow-hidden bg-gradient-to-br from-gray-50 to-orange-50">
      <main className="h-full flex flex-col justify-center items-center px-16 py-[clamp(2rem,5vh,6rem)] space-y-4">
        <div className="max-w-7xl mx-auto">

          {/* Hero Section */}
          <div className="flex flex-col text-center items-center justify-center">
            <div 
              className={`shadow-md inline-flex items-center space-x-2 bg-orange-100 text-orange-600 px-3 py-1 rounded-full text-m font-medium mb-[clamp(0.5rem,2vh,2rem)] transition-all duration-700 delay-100 transform ${
                isVisible ? 'translate-y-0 opacity-100' : '-translate-y-4 opacity-0'
              }`}
            >
              <Sparkles className="w-5 h-5" />
              <span>AI-Powered Resume Analysis</span>
            </div>
            
            <div className="flex items-center space-x-7 mb-[clamp(1rem,3vh,3rem)]">
              <img
                src="/appsageai-icon.png"
                alt="AppSageAI Logo"
                className={`w-[110px] h-[110px] flex-shrink-0`}
              />
              <div className="flex-1 justify-start space-y-3"> 
                <h1 className="text-7xl font-bold text-gray-900">AppSageAI</h1>
                <h2 className={`text-xl md:text-xl font-bold text-gray-900 transition-all duration-700 delay-100 transform ${
                  isVisible ? 'translate-y-0 opacity-100' : 'translate-y-4 opacity-0'
                }`}
                >
                  Land Your Dream Job with...
                  <span className="bg-gradient-to-r from-orange-500 to-orange-600 bg-clip-text text-transparent"> Intelligence</span>
                </h2>
              </div> 
            </div>

            {/* Subheadline with animation */}
            <p 
              className={`text-xl text-gray-600 max-w-3xl mx-auto mb-[clamp(0.5rem,2.5vh,2rem)] transition-all duration-700 delay-200 transform ${
                isVisible ? 'translate-y-0 opacity-100' : 'translate-y-4 opacity-0'
              }`}
            >
              Upload your resume once, analyze it against any job description, and get
              personalized insights to maximize your chances of success.
            </p>
              
            {/* Powered by Gemini Badge */}
            <div className="relative mb-[clamp(0.5rem,2.5vh,2rem)]">
            <div className={`glowing-pill absolute inset-0 z-0 inline-flex items-center space-x-3 bg-white rounded-full px-5 py-3 shadow-md transition-all duration-500 delay-300 transform ${
                isVisible ? 'scale-100 opacity-100' : 'scale-50 opacity-0'
              }`}>
                {/* Gemini Logo SVG */}
                <Sparkles className="w-5 h-5 flex-shrink-0 text-transparent" />
                <span className="text-sm font-medium text-transparent">
                  Powered by <span className="font-semibold">Llama 3.3 70B</span>
                </span>
            </div>

            <div 
              className={`absolute inset-0 z-10 inline-flex items-center space-x-3 bg-white rounded-full px-4 py-2 shadow-md transition-all duration-500 transform ${
                isVisible ? 'scale-100 opacity-100' : 'scale-105 opacity-0'
              }`}
            >
              {/* Gemini Logo SVG */}
              <img
                src="/gemini.png"
                alt="Gemini Logo"
                className={`w-5 h-5 flex-shrink-0`}
              />
              <span className="text-sm font-medium text-gray-700">
                Powered by <span className="font-semibold">Google Gemini</span>
              </span>
            </div>
            
            </div>

            {/* Action Buttons with animation */}
            <div 
              className={`flex flex-col sm:flex-row gap-[clamp(0.5rem,1.5vh,1.5rem)] justify-center transition-all duration-700 delay-400 transform ${
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
              <button className="px-8 py-3 bg-white text-gray-700 font-medium rounded-lg border border-gray-300 hover:bg-gray-50 hover:scale-105 shadow-lg transition-all duration-200 transform hover:scale-105">
                Watch Demo
              </button>
            </div>
          </div>

          {/* Features Grid with staggered animations */}
          <div className="grid md:grid-cols-3 gap-8 mt-[clamp(2rem,5vh,6rem)]">
            {/* Secure & Private Card */}
            <div 
              className={`flex flex-col justify-between bg-white rounded-xl p-6 shadow-md hover:shadow-xl transition-all duration-500 transform hover:-translate-y-1 ${
                isVisible ? 'translate-y-0 opacity-100' : 'translate-y-8 opacity-0'
              }`}
            >
              <img
                src="/shield-privacy.png"
                alt="Privacy Shield"
                className={`w-10 h-10 flex-shrink-0 mb-4`}
              />
              <div className="flex flex-col gap-2 mt-auto">
                <h3 className="text-lg font-bold text-green-700">Your Data, Your Control</h3>
                <p className="text-grey-900">
                  Your data is stored securely with server-side encryption, so that even I can’t access your information. Your privacy is my top priority.
                </p>
                <div className="flex flex-wrap gap-3">
                  <span className="inline-flex justify-center gap-1 items-center px-2 py-1 bg-green-100 text-green-700 text-xs rounded-full">
                      <Lock className="w-3 h-3" />
                      AES-256 
                  </span>
                  <span className="inline-flex items-center px-2 py-1 bg-green-100 text-green-700 text-xs rounded-full">
                      Rotating Keys
                  </span>
                  <span className="inline-flex items-center px-2 py-1 bg-green-100 text-green-700 text-xs rounded-full">
                      No PII on Logs
                  </span>
                </div>
              </div>
            </div>
            
            {/* Speed and Simplicity */}
            <div 
              className={`flex flex-col justify-between bg-white rounded-xl p-6 shadow-md hover:shadow-xl transition-all duration-500 transform hover:-translate-y-1 ${
                isVisible ? 'translate-y-0 opacity-100' : 'translate-y-8 opacity-0'
              }`}
            >
              <img
                src="/lightning-fast.png"
                alt="Unlimited Free"
                className={`w-10 h-10 flex-shrink-0 mb-4`}
              />
              <div className="flex flex-col gap-2 mt-auto">
                <h3 className="text-lg font-bold text-[#FDB441]">Real Results, Real Fast</h3>
                <p className="text-grey-900">
                  Fast, lightweight, and easy to use. Six powerful tools in one to focus on your job hunt, not on the clunky software.
                </p>
                <div className="flex flex-wrap gap-3">
                  <span className="inline-flex items-center px-2 py-1 bg-[#FCF0DC] text-[#FDB441] text-xs rounded-full">
                      ATS 
                  </span>
                  <span className="inline-flex items-center px-2 py-1 bg-[#FCF0DC] text-[#FDB441] text-xs rounded-full">
                      Job Match
                  </span>
                  <span className="inline-flex items-center px-2 py-1 bg-[#FCF0DC] text-[#FDB441] text-xs rounded-full">
                      Cover Letter
                  </span>
                  <span className="inline-flex items-center px-2 py-1 bg-[#FCF0DC] text-[#FDB441] text-xs rounded-full">
                      Much More
                  </span>
                </div>
              </div>
            </div>
            {/* No Hidden Fees */}
            <div 
              className={`flex flex-col justify-between bg-white rounded-xl p-6 shadow-md hover:shadow-xl transition-all duration-500 transform hover:-translate-y-1 ${
                isVisible ? 'translate-y-0 opacity-100' : 'translate-y-8 opacity-0'
              }`}
            >
              <img
                src="/unlimited-free.png"
                alt="Unlimited Free"
                className={`w-10 h-10 flex-shrink-0 mb-4`}
              />
              <div className="flex flex-col gap-2 mt-auto">
                <h3 className="text-lg font-bold text-[#FA4360]">No Hidden Costs, Ever</h3>
                <p className="text-grey-900">
                  Full access to all features - Unlimited resume uploads, chats, and job tracking. I built this for the community, Go Wild!
                </p>
                <div className="flex flex-wrap gap-3">
                  <span className="inline-flex items-center px-2 py-1 bg-[#FAE6E9] text-[#FA4360] text-xs rounded-full">
                      No Trials 
                  </span>
                  <span className="inline-flex items-center px-2 py-1 bg-[#FAE6E9] text-[#FA4360] text-xs rounded-full">
                      No Credit Card
                  </span>
                  <span className="inline-flex items-center px-2 py-1 bg-[#FAE6E9] text-[#FA4360] text-xs rounded-full">
                      No Paywall
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* Add custom animation styles */}
      <style jsx global>{`
        @property --rotate {
          syntax: "<angle>";
          initial-value: 0deg;
          inherits: false;
        }

        .glowing-pill {
          position: relative;
        }
        
        .glowing-pill::before {
          content: "";
          width: calc(100% + 4px);
          height: calc(100% + 4px);
          position: absolute;
          top: -2px;
          left: -2px;
          z-index: -1;
          border-radius: inherit;
          background-image: linear-gradient(
            var(--rotate),
            #5ddcff, #3c67e3 43%, #4e00c2
          );
          animation: spin 3s linear infinite;
        }

        .glowing-pill::after {
          content: "";
          position: absolute;
          width: 100%;
          height: 100%;
          top: 0;
          left: 0;
          z-index: -1;
          border-radius: inherit;
          filter: blur(1rem);
          background-image: linear-gradient(
            var(--rotate),
            #5ddcff, #3c67e3 43%, #4e00c2
          );
          animation: spin 3s linear infinite;
        }

        @keyframes spin {
          0% {
            --rotate: 0deg;
          }
          25% {
            --rotate: 90deg;
          }
          50% {
            --rotate: 180deg;
          }
          75% {
            --rotate: 270deg;
          }
          100% {
            --rotate: 360deg;
          }
        }
      `}</style>
    </div>
  );
}