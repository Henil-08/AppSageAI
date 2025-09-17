'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '../../contexts/AuthContext';
import { 
  Sparkles, 
  LogOut, 
  User, 
  FileText, 
  MessageSquare,
  Settings,
  ChevronLeft,
  Menu,
  Briefcase
} from 'lucide-react';
import Image from 'next/image';

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const { user, loading, signOut } = useAuth();
  const router = useRouter();
  const [sidebarOpen, setSidebarOpen] = useState(true);

  useEffect(() => {
    if (!loading && !user) {
      router.push('/');
    }
  }, [user, loading, router]);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-claude-background">
        <div className="flex flex-col items-center space-y-4">
          <div className="w-12 h-12 border-4 border-claude-accent-orange border-t-transparent rounded-full animate-spin"></div>
          <p className="text-claude-text-secondary">Loading your workspace...</p>
        </div>
      </div>
    );
  }

  if (!user) return null;

  return (
    <div className="min-h-screen bg-claude-background flex">
      {/* Sidebar */}
      <aside className={`${
        sidebarOpen ? 'w-64' : 'w-16'
      } bg-white border-r border-claude-border transition-all duration-300 flex flex-col`}>
        
        {/* Logo Section */}
        <div className="h-16 flex items-center justify-between px-4 border-b border-claude-border">
          {sidebarOpen && (
            <div className="flex items-center space-x-2">
              <Sparkles className="w-6 h-6 text-claude-accent-orange" />
              <span className="font-semibold text-lg">AppSageAI</span>
            </div>
          )}
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="p-1.5 hover:bg-claude-background rounded-lg transition-colors"
          >
            {sidebarOpen ? (
              <ChevronLeft className="w-5 h-5 text-claude-text-secondary" />
            ) : (
              <Menu className="w-5 h-5 text-claude-text-secondary" />
            )}
          </button>
        </div>

        {/* Navigation */}
        <nav className="flex-1 p-4">
          {/* New Chat Button */}
          <a
            href="/dashboard/new-chat"
            className="w-full flex items-center justify-center space-x-2 mb-4 px-3 py-2 bg-claude-accent-orange text-white font-medium rounded-lg hover:bg-claude-accent-orange-hover transition-colors"
          >
            <Sparkles className="w-4 h-4" />
            {sidebarOpen && <span>New Chat</span>}
          </a>
          
          <ul className="space-y-2">
            <li>
              <a
                href="/dashboard/jobs"
                className="flex items-center space-x-3 px-3 py-2 rounded-lg hover:bg-claude-accent-orange-light text-claude-text-primary hover:text-claude-accent-orange transition-colors"
              >
                <Briefcase className="w-5 h-5 flex-shrink-0" />
                {sidebarOpen && <span>Job Tracker</span>}
              </a>
            </li>
            <li>
              <a
                href="/dashboard"
                className="flex items-center space-x-3 px-3 py-2 rounded-lg hover:bg-claude-accent-orange-light text-claude-text-primary hover:text-claude-accent-orange transition-colors"
              >
                <FileText className="w-5 h-5 flex-shrink-0" />
                {sidebarOpen && <span>Resume</span>}
              </a>
            </li>
            <li>
              <a
                href="/dashboard/chats"
                className="flex items-center space-x-3 px-3 py-2 rounded-lg hover:bg-claude-accent-orange-light text-claude-text-primary hover:text-claude-accent-orange transition-colors"
              >
                <MessageSquare className="w-5 h-5 flex-shrink-0" />
                {sidebarOpen && <span>Chats</span>}
              </a>
            </li>
            <li>
              <a
                href="/dashboard/settings"
                className="flex items-center space-x-3 px-3 py-2 rounded-lg hover:bg-claude-accent-orange-light text-claude-text-primary hover:text-claude-accent-orange transition-colors"
              >
                <Settings className="w-5 h-5 flex-shrink-0" />
                {sidebarOpen && <span>Settings</span>}
              </a>
            </li>
          </ul>
        </nav>

        {/* User Section */}
        <div className="p-4 border-t border-claude-border">
          <div className={`flex items-center ${sidebarOpen ? 'space-x-3' : 'justify-center'}`}>
            <div className="relative">
              {user.photoURL ? (
                <img
                  src={user.photoURL}
                  alt={user.displayName || 'User'}
                  className="w-10 h-10 rounded-full"
                />
              ) : (
                <div className="w-10 h-10 rounded-full bg-claude-accent-orange-light flex items-center justify-center">
                  <User className="w-5 h-5 text-claude-accent-orange" />
                </div>
              )}
              <div className="absolute bottom-0 right-0 w-3 h-3 bg-green-500 rounded-full border-2 border-white"></div>
            </div>
            
            {sidebarOpen && (
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-claude-text-primary truncate">
                  {user.displayName || 'User'}
                </p>
                <p className="text-xs text-claude-text-muted truncate">
                  {user.email}
                </p>
              </div>
            )}
          </div>
          
          {sidebarOpen && (
            <button
              onClick={signOut}
              className="mt-4 w-full flex items-center justify-center space-x-2 px-3 py-2 bg-claude-background hover:bg-gray-100 rounded-lg transition-colors"
            >
              <LogOut className="w-4 h-4 text-claude-text-secondary" />
              <span className="text-sm text-claude-text-secondary">Sign out</span>
            </button>
          )}
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 overflow-auto">
        {children}
      </main>
    </div>
  );
}