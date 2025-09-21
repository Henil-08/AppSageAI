'use client';
import { MessagesSquare, Trash2, Waypoints } from 'lucide-react';

import { useEffect, useState } from 'react';
import { useRouter, usePathname } from 'next/navigation';
import Link from 'next/link';
import { useAuth } from '../../contexts/AuthContext';
import { SidebarProvider, useSidebar } from '../../contexts/SidebarContext';
import {
  Sparkles,
  LogOut,
  User,
  FileText,
  MessageSquare,
  ChevronDown,
  Menu,
  Plus,
} from 'lucide-react';
import toast from 'react-hot-toast';
import ConfirmationModal from '../../components/ConfirmationModal';

interface ChatSession {
  session_id: string;
  job_title: string;
  company: string;
  updated_at: string;
}

function DashboardUI({ children }: { children: React.ReactNode }) {
  const { sidebarOpen, setSidebarOpen } = useSidebar();
  const { user, loading, signOut, getToken } = useAuth();
  const router = useRouter();
  const pathname = usePathname();
  // const [sidebarOpen, setSidebarOpen] = useState(true);
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
          } bg-white border-r border-claude-border transition-all duration-500 flex flex-col h-screen overflow-hidden fixed left-0 top-0 z-40`}
      >
        <div className="flex h-[65px] flex-shrink-0 items-center border-b border-claude-border px-4">
          {/* Logo Section */}
          
          {/* 1. Sidebar Toggle Button (Fixed Size) */}
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className={`flex h-10 w-12 flex-shrink-0 items-center justify-center rounded-lg transition-colors transition-shadow duration-500 ease-in-out
              ${sidebarOpen ? 'shadow-md bg-claude-accent-orange-light text-claude-accent-orange' : 'bg-claude-light text-claude-text-secondary'}
      `}
          >
            <Menu className="h-5 w-5 " />
          </button>

          {/* 2. Centering Container */}
          <div className="flex flex-1 items-center justify-center overflow-hidden">
            {/* 3. Animated Logo + Title Wrapper */}
            <div
              className={`flex items-center whitespace-nowrap transition-all duration-500 ease-in-out ${
                sidebarOpen
                  ? "translate-x-0 opacity-100"
                  : "-translate-x-2 opacity-0"
              }`}
            >
              <img
                src="/appsageai-icon.png"
                alt="AppSageAI Logo"
                className="h-6 w-6 flex-shrink-0"
              />
              <span className="ml-3 text-lg font-semibold">AppSageAI</span>
            </div>
          </div>
          
        </div>
        
        {/* New Chat */}
        <div className="px-4 mt-4">
        <Link
          href="/dashboard/chat/new"
          className="w-full p-4 flex mb-4 items-center shadow-lg h-10 justify-center md:justify-center px-3 py-2 rounded-lg bg-claude-accent-orange text-white hover:bg-claude-accent-orange-hover transition-all duration-500"
        >
          <Plus className="w-5 h-5 flex-shrink-0" />
          <span
            className={`overflow-hidden whitespace-nowrap transition-all duration-500 ease-in-out
              ${sidebarOpen ? 'ml-2 opacity-100 translate-x-0' : 'ml-0 opacity-0 -translate-x-2'}`}
          >
            New Chat
          </span>
        </Link>
        </div>

        {/* Navigation */}
        <nav className="flex-1 h-full pl-4 pr-4 overflow-y-auto">
          {/* Collapsible Section (Recent Chats + Main Nav Items wrapper) */}
          <div className="transition-all duration-500 ease-in-out">
            {/* Collapsible Chats */}
            <div
              className={`transition-all duration-500 ease-in-out overflow-hidden
                ${sidebarOpen 
                  ? (chatsExpanded 
                    ? 'mb-2 max-h-[250px] opacity-100' 
                    : 'mb-2 max-h-10 opacity-100') 
                  : 'mb-0 max-h-0 opacity-0 pointer-events-none'}`}
            >
              <button
                onClick={() => setChatsExpanded(!chatsExpanded)}
                className={`w-full h-10 flex items-center justify-center md:justify-between px-3 rounded-lg
                  ${chatsExpanded ? 'shadow-md bg-claude-accent-orange-light' : 'bg-claude-light'} transition-shadow transition-colors duration-500`}
              >
                <div className={`flex items-center ${chatsExpanded ? 'text-claude-accent-orange' : 'text-claude-primary'}`}>
                  <MessageSquare
                    className={`w-5 h-5 flex-shrink-0 transition-opacity duration-500 ease-in-out
                      ${sidebarOpen ? 'opacity-100' : 'opacity-0'}`}
                  />
                  <span
                    className={`overflow-hidden whitespace-nowrap transition-all duration-500 ease-in-out
                      ${sidebarOpen ? 'ml-3 opacity-100 translate-x-0' : 'ml-0 opacity-0 -translate-x-2'}`}
                  >
                    Recent Chats
                  </span>
                </div>
                {sidebarOpen && (
                  <ChevronDown
                    className={`w-4 h-4 transition-transform duration-500 ${chatsExpanded ? 'rotate-0' : '-rotate-90'}`}
                  />
                )}
              </button>

              {/* Expanded Chat List */}
              <div
                className={`mt-2 space-y-1 overflow-hidden transition-all duration-500 ease-in-out
                  ${chatsExpanded && sidebarOpen ? 'opacity-100' : 'opacity-0 pointer-events-none'}
                  `}
                style={{ 
                  maxHeight: 
                    chatsExpanded && sidebarOpen 
                      ? `${Math.min(recentChats.length * 56 + 46, 150)}px` 
                      : '0px',
                }}
              >
                <div className="space-y-1">
                  {recentChats.map((chat) => (
                    <div
                      key={chat.session_id}
                      className={`
                        group flex items-center justify-between px-3 py-1.5 rounded-lg
                        hover:bg-claude-accent-orange-light
                        transition-all duration-500 ease-in-out overflow-hidden
                        ${pathname === `/dashboard/chat/${chat.session_id}` ? 'bg-claude-accent-orange-light' : ''}
                      `}
                    >
                      <Link href={`/dashboard/chat/${chat.session_id}`} className="flex-1 min-w-0">
                        <div className="text-sm text-claude-text-primary truncate">{chat.job_title || 'Untitled'}</div>
                        <div className="text-xs text-claude-text-muted truncate">{chat.company || 'No company'}</div>
                      </Link>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setDeleteModal({
                            isOpen: true,
                            chatId: chat.session_id,
                            chatTitle: chat.job_title || 'Untitled',
                          });
                        }}
                        className="opacity-0 group-hover:opacity-100 p-1 hover:bg-red-50 rounded transition-all duration-500"
                      >
                        <Trash2 className="w-3 h-3 text-red-500" />
                      </button>
                    </div>
                  ))}

                  {/* Empty state */}
                  {!loadingChats && recentChats.length === 0 && (
                    <div className="px-3 py-2 text-xs text-claude-text-muted transition-all duration-500 ease-in-out opacity-100">
                      No chats yet
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Main Nav Items (slides up/down smoothly because Recent Chats block shrinks above it) */}
            <ul className="space-y-2 transition-all duration-500 ease-in-out">
              {[
                { href: '/dashboard', icon: Sparkles, label: 'Dashboard' },
                { href: '/dashboard/chat', icon: MessagesSquare, label: 'All Chats' },
                { href: '/dashboard/jobs', icon: Waypoints, label: 'Job Tracker' },
                { href: '/dashboard/resume', icon: FileText, label: 'Resume' },
              ].map((item) => (
                <li key={item.href}>
                  <Link
                    href={item.href}
                    className={`flex items-center justify-center md:justify-start px-3 py-2 rounded-lg transition-all duration-500
                      ${pathname === item.href
                        ? 'shadow-lg bg-claude-accent-orange-light text-claude-accent-orange'
                        : 'hover:bg-claude-accent-orange-light text-claude-text-primary hover:text-claude-accent-orange'}`}
                  >
                    <item.icon className="w-5 h-5 flex-shrink-0" />
                    <span
                      className={`overflow-hidden whitespace-nowrap transition-all duration-500 ease-in-out
                        ${sidebarOpen ? 'ml-2 opacity-100 translate-x-0' : 'ml-0 opacity-0 -translate-x-2'}`}
                    >
                      {item.label}
                    </span>
                  </Link>
                </li>
              ))}
            </ul>
          </div>
        </nav>
        
        {/* Meta Llama Card */}
        {/* <div className="px-4">
          <div className="justify-center space-x-2 mb-4 relative bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg p-3 border border-blue-200 shadow-sm flex items-center transition-all duration-500">
            <img
              src="/meta-logo.png"
              alt="Meta Logo"
              className={`w-5 h-5 flex-shrink-0 duration-500 ease-in-out ${sidebarOpen ? 'mr-0 translate-x-0' : '-mr-2 translate-x-0'}`}
            />
            <span
              className={`overflow-hidden text-sm font-medium text-blue-800 whitespace-nowrap transition-all duration-500 ease-in-out
                ${sidebarOpen ? 'ml-2 opacity-100 translate-x-0' : 'ml-0 opacity-0 -translate-x-2'}`}
            >
              Powered by Llama 3.3 
            </span>
          </div>
        </div> */}

        {/* Privacy First Card */}
        <div className="px-4">
          <div className="justify-center space-x-2 mb-2 relative bg-gradient-to-r from-green-50 to-emerald-50 rounded-lg p-3 border border-green-200 shadow-sm flex items-center transition-all duration-500">
            <img
              src="/shield-privacy.png"
              alt="Privacy Shield"
              className={`w-5 h-5 flex-shrink-0 transition-all duration-500 ease-in-out ${sidebarOpen ? 'mr-0 translate-x-0' : '-mr-2 translate-x-0'}`}
            />
            <span
              className={`overflow-hidden text-sm font-medium text-green-800 whitespace-nowrap transition-all duration-500 ease-in-out
                ${sidebarOpen ? 'ml-2 opacity-100 translate-x-0' : 'ml-0 opacity-0 -translate-x-2'}`}
            >
              Privacy First Architecture
            </span>
          </div>
        </div>

        {/* User Section */}
        <div className="p-4 border-t border-claude-border flex-shrink-0">
          <div className={`flex items-center ${sidebarOpen ? 'space-x-3' : ''}`}>
            <div className="relative flex-shrink-0">
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

            <div className={`flex-1 min-w-0 transition-all duration-500 ease-in-out ${sidebarOpen ? 'opacity-100 translate-x-0' : 'opacity-0 -translate-x-2'}`}>
              <p className="text-sm font-medium text-claude-text-primary truncate">
                {user.displayName || 'User'}
              </p>
              <p className="text-xs text-claude-text-muted truncate">
                {user.email}
              </p>
            </div>
          </div>

          {/* Sign out button */}
          <button
            onClick={signOut}
            className="mt-4 w-full flex items-center h-10 shadow-lg justify-center md:justify-center px-3 py-2 rounded-lg bg-claude-accent-orange hover:bg-claude-accent-orange-hover text-white transition-all duration-500"
          >
            <LogOut className="w-5 h-5 flex-shrink-0" />
            <span
              className={`overflow-hidden whitespace-nowrap transition-all duration-500 ease-in-out
                ${sidebarOpen ? 'ml-2 opacity-100 translate-x-0' : 'ml-0 opacity-0 -translate-x-2'}`}
            >
              Sign out
            </span>
          </button>
        </div>
      </aside>

      {/* Main Content with margin for fixed sidebar */}
      <main className={`transition-all duration-500 ${sidebarOpen ? 'ml-64' : 'ml-20'}`}>
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
        message={`Are you sure you want to delete "${deleteModal.chatTitle}"?`}
        confirmText="Delete"
        cancelText="Cancel"
        type="danger"
      />
    </div>
  );
}

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  return (
    <SidebarProvider>
      <DashboardUI>{children}</DashboardUI>
    </SidebarProvider>
  );
}