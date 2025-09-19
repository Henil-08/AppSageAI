'use client';
import { Trash2 } from 'lucide-react';

import { useEffect, useState } from 'react';
import { useRouter, usePathname } from 'next/navigation';
import Link from 'next/link';
import { useAuth } from '../../contexts/AuthContext';
import {
  Sparkles,
  LogOut,
  User,
  FileText,
  MessageSquare,
  ChevronLeft,
  ChevronDown,
  Menu,
  Briefcase,
  Plus,
  Shield
} from 'lucide-react';
import toast from 'react-hot-toast';
import ConfirmationModal from '../../components/ConfirmationModal';

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
  const [deleteModal, setDeleteModal] = useState<{
    isOpen: boolean;
    chatId: string | null;
    chatTitle: string;
  }>({ isOpen: false, chatId: null, chatTitle: '' });

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
  const deleteChat = async (sessionId: string) => {
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
        if (pathname === `/dashboard/chat/${sessionId}`) {
          router.push('/dashboard/chat/new');
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

  // Listen for refresh events
  useEffect(() => {
    const handleRefresh = () => {
      fetchRecentChats();
    };

    window.addEventListener('refreshSidebarChats', handleRefresh);
    return () => {
      window.removeEventListener('refreshSidebarChats', handleRefresh);
    };
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
    <div className="min-h-screen bg-claude-background">
      {/* Fixed Sidebar */}
      <aside
        className={`${sidebarOpen ? 'w-64' : 'w-20'
          } bg-white border-r border-claude-border transition-all duration-300 flex flex-col h-screen overflow-hidden fixed left-0 top-0 z-40`}
      >
        {/* Logo Section */}
        <div className="h-[65px] flex items-center justify-between px-4 border-b border-claude-border flex-shrink-0">
          {sidebarOpen && (
            <div className="flex items-center space-x-2 transition-opacity duration-300">
              <Sparkles className="w-6 h-6 text-claude-accent-orange" />
              <span className="font-semibold text-lg">AppSageAI</span>
            </div>
          )}
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="flex items-center justify-center w-12 h-12 rounded-lg hover:bg-claude-background transition-colors"
          >
            {sidebarOpen ? (
              <ChevronLeft className="w-5 h-5 text-claude-text-secondary" />
            ) : (
              <Menu className="w-5 h-5 text-claude-text-secondary" />
            )}
          </button>
        </div>

        {/* Navigation */}
        <nav className="flex-1 h-full p-4 overflow-y-auto">
          {/* New Chat */}
          <Link
            href="/dashboard/chat/new"
            className="w-full flex mb-4 items-center h-10 justify-center md:justify-center px-3 py-2 rounded-lg bg-claude-accent-orange text-white hover:bg-claude-accent-orange-hover transition-all duration-300"
          >
            <Plus className="w-5 h-5 flex-shrink-0" />
            <span
              className={`overflow-hidden whitespace-nowrap transition-all duration-300 ease-in-out
                ${sidebarOpen ? 'ml-2 opacity-100 translate-x-0' : 'ml-0 opacity-0 -translate-x-2'}`}
            >
              New Chat
            </span>
          </Link>

          {/* Collapsible Chats */}
          {sidebarOpen && (
            <div className={`transition-all duration-300 ease-in-out mb-2 overflow-hidden ${sidebarOpen ? (chatsExpanded ? 'max-h-[400px]' : 'max-h-10') : 'max-h-0'}`}>
              <button
                onClick={() => setChatsExpanded(!chatsExpanded)}
                className={`w-full h-10 flex items-center justify-center md:justify-between px-3 rounded-lg ${chatsExpanded ? 'bg-claude-accent-orange-light' : 'bg-claude-light'} transition-colors`}
              >
                {/* Icon + Text */}
                <div className={`flex items-center ${chatsExpanded ? 'text-claude-accent-orange' : 'text-claude-primary'}`}>
                  <MessageSquare
                    className={`w-5 h-5 flex-shrink-0 transition-opacity duration-300 ease-in-out
                      ${sidebarOpen ? 'opacity-100' : 'opacity-0'}`}
                  />
                  <span
                    className={`overflow-hidden whitespace-nowrap transition-all duration-300 ease-in-out
                      ${sidebarOpen ? 'ml-3 opacity-100 translate-x-0' : 'ml-0 opacity-0 -translate-x-2'}`}
                  >
                    Recent Chats
                  </span>
                </div>

                {/* Chevron */}
                {sidebarOpen && (
                  <ChevronDown
                    className={`w-4 h-4 transition-transform duration-300 ${chatsExpanded ? 'rotate-0' : '-rotate-90'}`}
                  />
                )}
              </button>

              {/* Expanded Chat List */}
              <div
                className={`mt-2 space-y-1 max-h-64 overflow-y-auto transition-all duration-300 ease-in-out
                  ${chatsExpanded && sidebarOpen ? 'opacity-100' : 'opacity-0 pointer-events-none'}`}
              >
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
                      className={`group flex items-center justify-between px-3 py-1.5 rounded-lg hover:bg-claude-accent-orange-light transition-colors ${pathname === `/dashboard/chat/${chat.session_id}` ? 'bg-claude-accent-orange-light' : ''
                        }`}
                    >
                      <Link
                        href={`/dashboard/chat/${chat.session_id}`}
                        className="flex-1 min-w-0"
                      >
                        <div className="text-sm text-claude-text-primary truncate">
                          {chat.job_title || 'Untitled'}
                        </div>
                        <div className="text-xs text-claude-text-muted truncate">
                          {chat.company || 'No company'}
                        </div>
                      </Link>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setDeleteModal({
                            isOpen: true,
                            chatId: chat.session_id,
                            chatTitle: chat.job_title || 'Untitled'
                          });
                        }}
                        className="opacity-0 group-hover:opacity-100 p-1 hover:bg-red-50 rounded transition-all"
                      >
                        <Trash2 className="w-3 h-3 text-red-500" />
                      </button>
                    </div>
                  ))
                )}

                {recentChats.length > 0 && (
                  <Link
                    href="/dashboard/chat"
                    className="block px-3 py-2 text-xs text-claude-accent-orange hover:underline"
                  >
                    View all chats →
                  </Link>
                )}
              </div>
            </div>
          )}

          {/* Main Nav Items */}
          <ul className="space-y-2">
            {[
              { href: '/dashboard', icon: Sparkles, label: 'Dashboard' },
              { href: '/dashboard/chat', icon: MessageSquare, label: 'All Chats' },
              { href: '/dashboard/jobs', icon: Briefcase, label: 'Job Tracker' },
              { href: '/dashboard/resume', icon: FileText, label: 'Resume' },
            ].map((item) => (
              <li key={item.href}>
                <Link
                  href={item.href}
                  className={`flex items-center justify-center md:justify-start space-x-3 px-3 py-2 rounded-lg transition-all duration-300 ${pathname === item.href
                    ? 'bg-claude-accent-orange-light text-claude-accent-orange'
                    : 'hover:bg-claude-accent-orange-light text-claude-text-primary hover:text-claude-accent-orange'
                    }`}
                >
                  <item.icon className="w-5 h-5 flex-shrink-0" />
                  {sidebarOpen && (
                    <span className="transition-all duration-300">{item.label}</span>
                  )}
                </Link>
              </li>
            ))}
          </ul>
        </nav>
        
        {/* Meta Llama Card */}
        <div className="px-4">
          <div className="justify-center space-x-2 mb-4 relative bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg p-3 border border-blue-200 shadow-sm flex items-center transition-all duration-300">
            <img
              src="/meta-logo.png"
              alt="Meta Logo"
              className={`w-5 h-5 flex-shrink-0 ${sidebarOpen ? '' : '-mr-2'}`}
            />
            <span
              className={`overflow-hidden text-sm font-medium text-blue-800 whitespace-nowrap transition-all duration-300 ease-in-out
                ${sidebarOpen ? 'ml-2 opacity-100 translate-x-0' : 'ml-0 opacity-0 -translate-x-2'}`}
            >
              Powered by Llama 3.3 
            </span>
          </div>
        </div>

        {/* Privacy First Card */}
        <div className="px-4">
          <div className="justify-center space-x-2 mb-4 relative bg-gradient-to-r from-green-50 to-emerald-50 rounded-lg p-3 border border-green-200 shadow-sm flex items-center transition-all duration-300">
            <img
              src="/shield-privacy.png"
              alt="Privacy Shield"
              className={`w-5 h-5 flex-shrink-0 ${sidebarOpen ? '' : '-mr-2'}`}
            />
            <span
              className={`overflow-hidden text-sm font-medium text-green-800 whitespace-nowrap transition-all duration-300 ease-in-out
                ${sidebarOpen ? 'ml-2 opacity-100 translate-x-0' : 'ml-0 opacity-0 -translate-x-2'}`}
            >
              Privacy First Application
            </span>
          </div>
        </div>

        {/* User Section */}
        <div className="p-4 border-t border-claude-border flex-shrink-0">
          <div className={`flex items-center ${sidebarOpen ? 'space-x-3' : 'justify-center'}`}>
            <div className="relative">
              {user.photoURL ? (
                <img
                  src={user.photoURL}
                  alt={user.displayName || 'User'}
                  className="w-12 h-12 rounded-full"
                />
              ) : (
                <div className="w-12 h-12 rounded-full bg-claude-accent-orange-light flex items-center justify-center">
                  <User className="w-6 h-6 text-claude-accent-orange" />
                </div>
              )}
              <div className="absolute bottom-0 right-0 w-3 h-3 bg-green-500 rounded-full border-2 border-white"></div>
            </div>

            {sidebarOpen && (
              <div className="flex-1 min-w-0 transition-all duration-300">
                <p className="text-sm font-medium text-claude-text-primary truncate">
                  {user.displayName || 'User'}
                </p>
                <p className="text-xs text-claude-text-muted truncate">
                  {user.email}
                </p>
              </div>
            )}
          </div>

          {/* Sign out button */}
          <button
            onClick={signOut}
            className="mt-4 w-full flex items-center h-10 justify-center md:justify-center px-3 py-2 rounded-lg bg-claude-accent-orange hover:bg-claude-accent-orange-hover text-white transition-all duration-300"
          >
            <LogOut className="w-5 h-5 flex-shrink-0" />
            <span
              className={`overflow-hidden whitespace-nowrap transition-all duration-300 ease-in-out
                ${sidebarOpen ? 'ml-2 opacity-100 translate-x-0' : 'ml-0 opacity-0 -translate-x-2'}`}
            >
              Sign out
            </span>
          </button>
        </div>
      </aside>

      {/* Main Content with margin for fixed sidebar */}
      <main className={`transition-all duration-300 ${sidebarOpen ? 'ml-64' : 'ml-20'}`}>
        <div className="min-h-screen">
          {children}
        </div>
      </main>

      {/* Delete Confirmation Modal */}
      <ConfirmationModal
        isOpen={deleteModal.isOpen}
        onClose={() => setDeleteModal({ isOpen: false, chatId: null, chatTitle: '' })}
        onConfirm={() => {
          if (deleteModal.chatId) {
            deleteChat(deleteModal.chatId);
            setDeleteModal({ isOpen: false, chatId: null, chatTitle: '' });
          }
        }}
        title="Delete Chat"
        message={`Are you sure you want to delete "${deleteModal.chatTitle}"? This action cannot be undone.`}
        confirmText="Delete"
        cancelText="Cancel"
        type="danger"
      />
    </div>
  );
}