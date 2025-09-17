"use client";

import Link from 'next/link';
import { Home, ArrowLeft } from 'lucide-react';

export default function NotFound() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-claude-background to-orange-50 flex items-center justify-center px-4">
      <div className="max-w-md w-full text-center">
        {/* 404 Number */}
        <div className="relative">
          <h1 className="text-[150px] font-bold text-claude-accent-orange opacity-20">
            404
          </h1>
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="bg-white rounded-2xl shadow-medium p-8">
              <h2 className="text-2xl font-semibold text-claude-text-primary mb-2">
                Page Not Found
              </h2>
              <p className="text-claude-text-secondary mb-6">
                Sorry, the page you're looking for doesn't exist or has been moved.
              </p>
              
              <div className="flex flex-col sm:flex-row gap-3 justify-center">
                <Link
                  href="/"
                  className="inline-flex items-center justify-center space-x-2 px-4 py-2 bg-claude-accent-orange text-white font-medium rounded-lg hover:bg-claude-accent-orange-hover transition-colors"
                >
                  <Home className="w-4 h-4" />
                  <span>Go Home</span>
                </Link>
                
                <button
                  onClick={() => window.history.back()}
                  className="inline-flex items-center justify-center space-x-2 px-4 py-2 bg-white text-claude-text-primary font-medium rounded-lg border border-claude-border hover:bg-claude-background transition-colors"
                >
                  <ArrowLeft className="w-4 h-4" />
                  <span>Go Back</span>
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}