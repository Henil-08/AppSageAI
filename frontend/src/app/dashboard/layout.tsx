'use client';

import { useEffect, useState } from 'react';
import { useRouter, usePathname } from 'next/navigation';
import { useAuth } from '../../contexts/AuthContext';
import { 
  Sparkles, 
  LogOut, 
  User, 
  FileText, 
  MessageSquare,
  Settings,
  ChevronLeft,
  ChevronDown,
  Menu,
  Briefcase,
  Plus,
  Trash2
} from 'lucide-react';
import toast from 'react-hot-toast';

interface ChatSession {
  session_id: string;
  job_title: string;
  company: string;
  updated_at: string;
}

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const { user, loading, signOut, getToken } = useAuth();
  const router = useRouter();
  const pathname = usePathname();
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [chatsExpanded, setChatsExpanded] = useState(true);
  const [recentChats, setRecentChats] = useState<ChatSession[]>([]);
  const [loadingChats, setLoadingChats] = useState(false);

  // Fetch recent chats
  const fetchRecentChats = async () => {
    if (!user) return;
    
    setLoadingChats(true);
    try {
      const token = await getToken();
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/api/v1/chat/list?limit=10&offset=0`,
        {
          headers: {
            'Authorization': `Bearer ${token}`,
          },
        }
      );

      if (response.ok) {
        const data = await response.json();
        setRecentChats(data.chats);
      }
    } catch (error) {
      console.error('Error fetching chats:', error);
    } finally {
      setLoadingChats(false);
    }
  };

  // Delete chat
  const deleteChat = async (sessionId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    e.preventDefault();
    
    const confirmed = window.confirm('Are you sure you want to delete this chat?');
    if (!confirmed) return;

    try {
      const token = await getToken();
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/api/v1/chat/${sessionId}`,
        {
          method: 'DELETE',
          headers: {
            'Authorization': `Bearer ${token}`,
          },
        }
      );

      if (response.ok) {
        toast.success('Chat deleted');
        fetchRecentChats(); // Refresh the list
        
        // If we're on the deleted chat page, redirect
        if (pathname === `/dashboard/${sessionId}`) {
          router.push('/dashboard/new');
        }
      } else {
        toast.error('Failed to delete chat');
      }
    } catch (error) {
      console.error('Error deleting chat:', error);
      toast.error('Failed to delete chat');
    }
  };

  useEffect(() => {
    if (!loading && !user) {
      router.push('/');
    }
  }, [user, loading, router]);

  useEffect(() => {
    if (user) {
      fetchRecentChats();
    }
  }, [user]);

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
        <nav className="flex-1 p-4 overflow-y-auto">
          {/* New Chat Button */}
          <a
            href="/dashboard/chat/new"
            className="w-full flex items-center justify-center space-x-2 mb-4 px-3 py-2 bg-claude-accent-orange text-white font-medium rounded-lg hover:bg-claude-accent-orange-hover transition-colors"
          >
            <Plus className="w-4 h-4" />
            {sidebarOpen && <span>New Chat</span>}
          </a>
          
          {/* Collapsible Chats Section */}
          {sidebarOpen && (
            <div className="mb-4">
              <button
                onClick={() => setChatsExpanded(!chatsExpanded)}
                className="w-full flex items-center justify-between px-3 py-2 text-sm font-medium text-claude-text-secondary hover:bg-claude-background rounded-lg transition-colors"
              >
                <div className="flex items-center space-x-2">
                  <MessageSquare className="w-4 h-4" />
                  <span>Recent Chats</span>
                </div>
                <ChevronDown className={`w-4 h-4 transition-transform ${
                  chatsExpanded ? 'rotate-0' : '-rotate-90'
                }`} />
              </button>
              
              {chatsExpanded && (
                <div className="mt-2 space-y-1">
                  {loadingChats ? (
                    <div className="px-3 py-2 text-xs text-claude-text-muted">
                      Loading...
                    </div>
                  ) : recentChats.length === 0 ? (
                    <div className="px-3 py-2 text-xs text-claude-text-muted">
                      No chats yet
                    </div>
                  ) : (
                    recentChats.map((chat) => (
                      <div
                        key={chat.session_id}
                        className={`group flex items-center justify-between px-3 py-1.5 rounded-lg hover:bg-claude-background transition-colors cursor-pointer ${
                          pathname === `/dashboard/chat/${chat.session_id}` ? 'bg-claude-accent-orange-light' : ''
                        }`}
                      >
                        <a
                          href={`/dashboard/chat/${chat.session_id}`}
                          className="flex-1 min-w-0"
                        >
                          <div className="text-sm text-claude-text-primary truncate">
                            {chat.job_title || 'Untitled'}
                          </div>
                          <div className="text-xs text-claude-text-muted truncate">
                            {chat.company || 'No company'}
                          </div>
                        </a>
                        <button
                          onClick={(e) => deleteChat(chat.session_id, e)}
                          className="opacity-0 group-hover:opacity-100 p-1 hover:bg-red-50 rounded transition-all"
                        >
                          <Trash2 className="w-3 h-3 text-red-500" />
                        </button>
                      </div>
                    ))
                  )}
                  
                  {recentChats.length > 0 && (
                    <a
                      href="/dashboard/chat"
                      className="block px-3 py-2 text-xs text-claude-accent-orange hover:underline"
                    >
                      View all chats →
                    </a>
                  )}
                </div>
              )}
            </div>
          )}
          
          <ul className="space-y-2">
            <li>
              <a
                href="/dashboard/chat"
                className={`flex items-center space-x-3 px-3 py-2 rounded-lg transition-colors ${
                  pathname === '/dashboard/chat'
                    ? 'bg-claude-accent-orange-light text-claude-accent-orange'
                    : 'hover:bg-claude-accent-orange-light text-claude-text-primary hover:text-claude-accent-orange'
                }`}
              >
                <MessageSquare className="w-5 h-5 flex-shrink-0" />
                {sidebarOpen && <span>All Chats</span>}
              </a>
            </li>
            <li>
              <a
                href="/dashboard/jobs"
                className={`flex items-center space-x-3 px-3 py-2 rounded-lg transition-colors ${
                  pathname === '/dashboard/jobs'
                    ? 'bg-claude-accent-orange-light text-claude-accent-orange'
                    : 'hover:bg-claude-accent-orange-light text-claude-text-primary hover:text-claude-accent-orange'
                }`}
              >
                <Briefcase className="w-5 h-5 flex-shrink-0" />
                {sidebarOpen && <span>Job Tracker</span>}
              </a>
            </li>
            <li>
              <a
                href="/dashboard"
                className={`flex items-center space-x-3 px-3 py-2 rounded-lg transition-colors ${
                  pathname === '/dashboard'
                    ? 'bg-claude-accent-orange-light text-claude-accent-orange'
                    : 'hover:bg-claude-accent-orange-light text-claude-text-primary hover:text-claude-accent-orange'
                }`}
              >
                <FileText className="w-5 h-5 flex-shrink-0" />
                {sidebarOpen && <span>Resume</span>}
              </a>
            </li>
            <li>
              <a
                href="/dashboard/settings"
                className={`flex items-center space-x-3 px-3 py-2 rounded-lg transition-colors ${
                  pathname === '/dashboard/settings'
                    ? 'bg-claude-accent-orange-light text-claude-accent-orange'
                    : 'hover:bg-claude-accent-orange-light text-claude-text-primary hover:text-claude-accent-orange'
                }`}
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