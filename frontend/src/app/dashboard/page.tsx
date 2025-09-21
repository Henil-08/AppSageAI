'use client';

import { useState, useEffect } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import { useSidebar } from '../../contexts/SidebarContext';
import Link from 'next/link';
import { 
  FileText,
  Plus,
  Clock,
  TrendingUp,
  Heart,
  Github,
  Linkedin,
  GraduationCap,
  Lock,
  MessageSquareDot,
  LucideAppWindow
} from 'lucide-react';

export default function DashboardHomePage() {
  const { user, getToken } = useAuth();
  const { sidebarOpen, setSidebarOpen } = useSidebar();
  const [greeting, setGreeting] = useState('');
  const [timeOfDay, setTimeOfDay] = useState('');
  const [isVisible, setIsVisible] = useState(false);
  const [stats, setStats] = useState({
    activeChats: 0,
    applications: 0,
    thisWeek: 0
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Set greeting based on time
    const hour = new Date().getHours();
    if (hour >= 6 && hour < 12) {
      setTimeOfDay('morning');
      setGreeting('Good Morning');
    } else if (hour >= 12 && hour < 18) {
      setTimeOfDay('afternoon');
      setGreeting('Good Afternoon');
    } else {
      setTimeOfDay('evening');
      setGreeting('Good Evening');
    }

    // Trigger animations
    setTimeout(() => setIsVisible(true), 100);
    
    // Fetch stats
    fetchStats();
  }, []);

  const fetchStats = async () => {
    try {
      const token = await getToken();
      
      // Fetch chats
      const chatsResponse = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/api/v1/chat/list?limit=100`,
        {
          headers: { 'Authorization': `Bearer ${token}` }
        }
      );
      
      if (chatsResponse.ok) {
        const chatsData = await chatsResponse.json();
        const activeChats = chatsData.chats.filter((chat: any) => 
          chat.message_count > 0
        ).length;
        
        // Count applications (chats with tracker status other than 'not_applicable')
        const applications = chatsData.chats.filter((chat: any) => 
          chat.tracker_status && chat.tracker_status !== 'not_applicable'
        ).length;
        
        // Count this week's activity
        const oneWeekAgo = new Date();
        oneWeekAgo.setDate(oneWeekAgo.getDate() - 7);
        const thisWeek = chatsData.chats.filter((chat: any) => 
          new Date(chat.updated_at) > oneWeekAgo
        ).length;
        
        setStats({
          activeChats,
          applications,
          thisWeek
        });
      }
    } catch (error) {
      console.error('Error fetching stats:', error);
    } finally {
      setLoading(false);
    }
  };

  const statCards = [
    { icon: MessageSquareDot, label: 'Active Chats', value: stats.activeChats, color: 'text-blue-500' },
    { icon: TrendingUp, label: 'Applications', value: stats.applications, color: 'text-yellow-500' },
    { icon: Clock, label: 'This Week', value: stats.thisWeek, color: 'text-pink-500' }
  ];

  const socialLinks = [
    { name: 'Website', icon: LucideAppWindow, href: 'https://henilgajjar.framer.ai/', color: 'hover:text-claude-accent-orange' },
    { name: 'GitHub', icon: Github, href: 'https://github.com/Henil-08', color: 'hover:text-gray-800' },
    { name: 'LinkedIn', icon: Linkedin, href: 'https://linkedin.com/in/henilgajjar', color: 'hover:text-blue-600' },
    { name: 'Google Scholar', icon: GraduationCap, href: 'https://scholar.google.com/citations?user=RdSGiWYAAAAJ&hl=en', color: 'hover:text-green-600' }
  ];

  const userName = user?.displayName?.split(' ')[0] || 'there';

  return (
    <div className={`flex flex-col h-screen bg-claude-background`}>
      <div className={`h-screen flex flex-col ${sidebarOpen ? 'pl-10 pr-10': 'pl-20 pr-20'} duration-500 transition-all ease-in-out`}>
        {/* Main content area */}
        <div className="flex-1 flex flex-col justify-center px-20 py-4">
            <div className="max-w-7xl mx-auto">

              {/* Greeting Section */}
              <div className={`text-center transition-all duration-700 transform ${
                  isVisible ? 'translate-y-0 opacity-100' : '-translate-y-2 opacity-0'
              }`}>
                  <h1 className="text-6xl md:text-6xl font-bold text-claude-text-primary mb-2">
                  Hi, {userName}!
                  <span className="inline-block ml-2 animate-wave">👋</span>
                  </h1>
                  <p className="text-l md:text-l text-claude-text-secondary">
                  {greeting}! Ready to land your dream job?
                  </p>
              </div>
            </div>
        </div>

        <div className={`${sidebarOpen ? 'pr-10 pl-10': 'px-20'} transition-all duration-500 ease-in-out`}>
          <div className="w-full space-y-2 pl-10 pr-10">
            {/* What is AppSageAI */}
            <div className={`bg-white rounded-2xl p-6 shadow-md border border-claude-border transition-all duration-500 ease-in-out delay-100 ${
                  isVisible ? 'translate-y-0 opacity-100' : '-translate-y-2 opacity-0'
              }`}>
                <div className="flex items-start justify-center space-x-3 transition-all duration-500 ease-in-out">
                    <div className="p-2 w-12 h-12 bg-claude-accent-orange-light rounded-lg">
                        <img
                            src="/appsageai-icon.png"
                            alt="AppSageAI Logo"
                            className={`w-full h-full flex-shrink-0`}
                        />
                    </div>
                    <div className="flex-1">
                        <h2 className="text-lg font-semibold text-claude-text-primary mb-1 flex items-center">
                        Welcome to AppSageAI
                        </h2>
                        <p className="text-sm text-claude-text-secondary leading-relaxed">
                        Your AI-powered job application assistant. I help you analyze job descriptions, 
                        optimize your resume for ATS systems, track applications, and provide personalized 
                        insights. Let's make your job search smarter and more successful!
                        </p>
                        <div className="mt-2 flex flex-wrap gap-3">
                        <Link
                            href="/dashboard/chat/new"
                            className="inline-flex h-10 items-center space-x-2 px-4 py-2 bg-claude-accent-orange text-white rounded-lg hover:bg-claude-accent-orange-hover transition-colors"
                        >
                            <Plus className="w-4 h-4" />
                            <span>Start New Chat</span>
                        </Link>
                        <Link
                            href="/dashboard/resume"
                            className="inline-flex h-10 items-center space-x-2 px-4 py-2 bg-white text-claude-text-primary border border-claude-border rounded-lg hover:bg-claude-background transition-colors"
                        >
                            <FileText className="w-4 h-4" />
                            <span>Upload Resume</span>
                        </Link>
                        </div>
                    </div>
                </div>
            </div>

            {/* Privacy Notice */}
            <div className={`bg-gradient-to-r from-green-50 to-emerald-50 rounded-2xl p-6 border border-green-200 shadow-md transition-all duration-500 ease-in-out delay-200 ${
                  isVisible ? 'translate-y-0 opacity-100' : '-translate-y-2 opacity-0'
              }
            `}>
                <div className="flex items-start space-x-3 transition-all duration-500 ease-in-out">
                    <div className="p-2 w-12 h-12 bg-green-100 rounded-lg">
                    <img
                        src="/shield-privacy.png"
                        alt="Privacy Shield"
                        className={`w-full h-full flex-shrink-0`}
                    />
                    </div>
                    <div className="flex-1">
                        <h2 className="text-lg font-semibold text-green-900 mb-1 flex items-center">
                        Privacy-First Architecture
                        </h2>
                        <p className="text-sm text-green-800 leading-relaxed">
                        Your data is protected with server-side AES-256 encryption using dynamically generated keys. 
                        Each user has a unique encryption key that automatically rotates, ensuring even I (the developer) 
                        cannot access your personal information.
                        </p>
                        <div className="mt-2 flex flex-wrap gap-3">
                        <span className="inline-flex justify-center gap-1 items-center px-2 py-1 bg-green-100 text-green-700 text-xs rounded-full">
                            <Lock className="w-3 h-3" />
                            <span className='font-semibold'> AES-256 </span> 
                            Encryption
                        </span>
                        <span className="inline-flex items-center px-2 py-1 bg-green-100 text-green-700 text-xs rounded-full">
                            Rotating Keys
                        </span>
                        <span className="inline-flex items-center px-2 py-1 bg-green-100 text-green-700 text-xs rounded-full">
                            No PII in Logs
                        </span>
                        </div>
                    </div>
                </div>
            </div>

            {/* Quick Stats */}
            <div className={` grid grid-cols-3 gap-4 transition-all duration-700 delay-300 ${
                  isVisible ? 'translate-y-0 opacity-100' : '-translate-y-2 opacity-0'
              }`}>
                {statCards.map((stat, index) => {
                const Icon = stat.icon;
                return (
                    <div 
                    key={index}
                    className="bg-white rounded-xl p-4 border border-claude-border shadow-md transition-all transform"
                    >
                    <div className="flex items-center justify-between mb-2">
                      <Icon className={`w-5 h-5 ${stat.color}`} />

                      {/* This container holds both the loading placeholder and the final number */}
                      <div className="relative h-8 w-10 flex items-center justify-end">
                          {/* 1. The background pulse animation (visible only when loading) */}
                          <div className={`
                              absolute top-0 right-0 h-full w-full bg-claude-background rounded-md
                              transition-opacity duration-500 ease-in-out
                              ${loading ? 'opacity-100' : 'opacity-0'}
                          `}>
                          </div>

                          {/* 2. The actual number that fades in when loading is complete */}
                          <span className={`
                              text-2xl font-bold text-claude-text-primary
                              transition-opacity duration-500 ease-in-out
                              ${loading ? 'opacity-0' : 'opacity-100'}
                          `}>
                              {stat.value}
                          </span>
                      </div>
                    </div>
                    <p className="text-sm text-claude-text-secondary">{stat.label}</p>
                    </div>
                );
                })}
            </div>
          </div>
        </div>  
      </div>
      
      {/* Footer */}
      <div className={`bg-claude-background mt-2 py-7 border-t border-claude-border transition-all duration-500 delay-400 ${
                isVisible ? 'translate-y-0 opacity-100' : '-translate-y-2 opacity-0'
            }`}>
          <div className="h-8 flex items-center justify-center">
          <div className="max-w-7xl mx-auto text-center">
              <div className="flex items-center justify-center space-x-2 text-claude-text-secondary mb-4">
              <span>Made with</span>
              <Heart className="w-4 h-4 text-red-500 animate-pulse fill-current" />
              <span>by Henil</span>
              </div>
              <div className="flex items-center justify-center space-x-6">
              {socialLinks.map((link, index) => {
                  const Icon = link.icon;
                  return (
                  <a 
                      key={index}
                      href={link.href}
                      target="_blank" 
                      rel="noopener noreferrer"
                      className={`flex items-center space-x-2 text-claude-text-secondary transition-colors ${link.color}`}
                  >
                      <Icon className="w-5 h-5"/>
                      <span className="text-sm font-medium">{link.name}</span>
                  </a>
                  );
              })}
              </div>
          </div>
          </div>
      </div>

      {/* Wave animation */}
      <style jsx global>{`
          @keyframes wave {
          0%, 100% { transform: rotate(0deg); }
          10%, 30%, 50%, 70% { transform: rotate(-10deg); }
          20%, 40%, 60% { transform: rotate(10deg); }
          80% { transform: rotate(8deg); }
          90% { transform: rotate(-8deg); }
          }
          
          .animate-wave {
          animation: wave 10s ease-in-out infinite;
          transform-origin: 70% 70%;
          display: inline-block;
          }
      `}</style>
    </div>
  );
}