'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '../../../contexts/AuthContext';
import Link from 'next/link';
import { 
  Plus,
  MessageSquare,
  Calendar,
  ChevronRight,
  Search,
  Trash2,
} from 'lucide-react';
import toast from 'react-hot-toast';
import ConfirmationModal from '../../../components/ConfirmationModal';

interface ChatSession {
  session_id: string;
  created_at: string;
  updated_at: string;
  job_title: string;
  company: string;
  message_count: number;
  preview: string;
}

export default function ChatsPage() {
  const { getToken } = useAuth();
  const router = useRouter();
  const [chats, setChats] = useState<ChatSession[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [showNewChatModal, setShowNewChatModal] = useState(false);
  const [deleteModal, setDeleteModal] = useState<{
    isOpen: boolean;
    chatId: string | null;
    chatTitle: string;
  }>({ isOpen: false, chatId: null, chatTitle: '' });

  // Fetch chat sessions
  const fetchChats = async () => {
    try {
      const token = await getToken();
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/api/v1/chat/list?limit=20&offset=0`,
        {
          headers: {
            'Authorization': `Bearer ${token}`,
          },
        }
      );

      if (response.ok) {
        const data = await response.json();
        setChats(data.chats);
      }
    } catch (error) {
      console.error('Error fetching chats:', error);
      toast.error('Failed to load chats');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchChats();
  }, []);

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
        fetchChats(); // Refresh the list
      } else {
        toast.error('Failed to delete chat');
      }
    } catch (error) {
      console.error('Error deleting chat:', error);
      toast.error('Failed to delete chat');
    }
  };

  // Filter chats based on search
  const filteredChats = chats.filter(chat => 
    chat.job_title.toLowerCase().includes(searchQuery.toLowerCase()) ||
    chat.company.toLowerCase().includes(searchQuery.toLowerCase())
  );

  // Format date
  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const days = Math.floor(diff / (1000 * 60 * 60 * 24));
    
    if (days === 0) return 'Today';
    if (days === 1) return 'Yesterday';
    if (days < 7) return `${days} days ago`;
    return date.toLocaleDateString();
  };

  return (
    <div className="p-8">
      {/* Header */}
      <div className="flex justify-between items-start mb-5">
        <div>
          <h1 className="text-3xl font-semibold text-claude-text-primary mb-2">
            Your Chats
          </h1>
          <p className="text-claude-text-secondary">
            Manage your job application conversations and analyses
          </p>
        </div>
        
        <Link
          href='/dashboard/chat/new/'
          className="flex items-center space-x-2 px-4 py-2 bg-claude-accent-orange text-white font-medium rounded-lg hover:bg-claude-accent-orange-hover transition-colors"
        >
          <Plus className="w-5 h-5" />
          <span>New Chat</span>
        </Link>
      </div>

      {/* Privacy Info */}
      <div className="mt-4 items-center bg-gradient-to-r from-green-50 to-emerald-50 border border-green-200 rounded-xl p-4 flex items-start space-x-3 shadow-soft">
        <img
          src="/shield-privacy.png"
          alt="Privacy Shield"
          className={`w-5 h-5 flex-shrink-0`}
        />
        <p className="text-sm text-claude-text-secondary">
          All your Chats are <span className="font-medium text-green-700">stored securely</span> on the server with 
          <span className="font-medium text-green-700"> AES-256 encryption</span>. 
          Only visible to you.
        </p>
      </div>

      {/* Search and Filter Bar */}
      <div className="mt-6 flex items-center space-x-4 mb-6">
        <div className="flex-1 relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-claude-text-muted" />
          <input
            type="text"
            placeholder="Search by job title or company..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-10 pr-4 py-2 bg-white border border-claude-border rounded-lg focus:outline-none focus:ring-2 focus:ring-claude-accent-orange/20 focus:border-claude-accent-orange"
          />
        </div>
      </div>

      {/* Chats Grid */}
      {loading ? (
        <div className="flex items-center justify-center py-12">
          <div className="w-8 h-8 border-3 border-claude-accent-orange border-t-transparent rounded-full animate-spin"></div>
        </div>
      ) : filteredChats.length === 0 ? (
        <div className="bg-white rounded-xl border border-claude-border p-12 text-center">
          <MessageSquare className="w-12 h-12 text-claude-text-muted mx-auto mb-4" />
          <p className="text-lg font-medium text-claude-text-primary mb-2">
            {searchQuery ? 'No chats found' : 'No chats yet'}
          </p>
          <p className="text-sm text-claude-text-secondary mb-6">
            {searchQuery 
              ? 'Try a different search term' 
              : 'Start your first chat to analyze a job opportunity'}
          </p>
          {!searchQuery && (
            <button
              onClick={() => setShowNewChatModal(true)}
              className="inline-flex items-center space-x-2 px-4 py-2 bg-claude-accent-orange text-white font-medium rounded-lg hover:bg-claude-accent-orange-hover transition-colors"
            >
              <Plus className="w-5 h-5" />
              <span>Create First Chat</span>
            </button>
          )}
        </div>
      ) : (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {filteredChats.map((chat) => (
            <div
              key={chat.session_id}
              onClick={() => router.push(`/dashboard/chat/${chat.session_id}`)}
              className="bg-white rounded-xl border border-claude-border p-5 hover:shadow-medium hover:border-claude-accent-orange/30 transition-all cursor-pointer group"
            >
              <div className="flex justify-between mb-4 items-start">
                <div className="flex items-center space-x-3 flex-1 min-w-0">
                  <div className="w-10 h-10 bg-claude-accent-orange-light rounded-lg flex items-center justify-center flex-shrink-0">
                    <img
                      src="/application.png"
                      alt="Application"
                      className={`w-5 h-5 flex-shrink-0`}
                    />
                  </div>
                  <div className="flex-1 min-w-0">
                    <h3 className="font-medium text-claude-text-primary line-clamp-1">
                      {chat.job_title || 'Untitled Position'}
                    </h3>
                    <p className="text-sm text-claude-text-secondary line-clamp-1">
                      {chat.company || 'Unknown Company'}
                    </p>
                  </div>
                </div>
                
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    setDeleteModal({
                      isOpen: true,
                      chatId: chat.session_id,
                      chatTitle: chat.job_title || 'Untitled',
                    });
                  }}
                  className="w-7 h-7 flex items-center justify-center opacity-0 group-hover:opacity-100 p-1 hover:bg-red-50 rounded transition-all flex-shrink-0 ml-1"
                >
                  <Trash2 className="w-4 h-4 text-red-500" />
                </button>
              </div>

              <div className="flex-1 mb-4">
              <p className="text-sm text-claude-text-secondary line-clamp-3 h-[60px] mb-4">
                {chat.preview}
              </p>
              </div>

              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-4 text-xs text-claude-text-muted">
                  <span className="flex items-center">
                    <Calendar className="w-3 h-3 mr-1" />
                    {formatDate(chat.updated_at)}
                  </span>
                  <span className="flex items-center">
                    <MessageSquare className="w-3 h-3 mr-1" />
                    {chat.message_count} messages
                  </span>
                </div>
                
                <ChevronRight className="w-4 h-4 text-claude-text-muted group-hover:text-claude-accent-orange transition-colors" />
              </div>
            </div>
          ))}
        </div>
      )}

      {/* New Chat Modal */}
      {showNewChatModal && (
        <NewChatModal 
          onClose={() => setShowNewChatModal(false)}
          onSuccess={() => {
            setShowNewChatModal(false);
            fetchChats();
          }}
        />
      )}

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

// New Chat Modal Component
function NewChatModal({ onClose, onSuccess }: { onClose: () => void; onSuccess: () => void }) {
  const { getToken } = useAuth();
  const router = useRouter();
  const [jobTitle, setJobTitle] = useState('');
  const [company, setCompany] = useState('');
  const [jobDescription, setJobDescription] = useState('');
  const [creating, setCreating] = useState(false);

  const handleCreate = async () => {
    if (!jobDescription.trim()) {
      toast.error('Please provide a job description');
      return;
    }

    setCreating(true);
    try {
      const token = await getToken();
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/api/v1/chat/create`,
        {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            job_title: jobTitle || 'Untitled Position',
            company: company || 'Unknown Company',
            job_description: jobDescription,
          }),
        }
      );

      if (response.ok) {
        const data = await response.json();
        toast.success('Chat created successfully');
        router.push(`/dashboard/chat/${data.session_id}`);
        onSuccess();
      } else {
        const error = await response.json();
        // Handle different error formats
        const errorMessage = typeof error.detail === 'string' 
          ? error.detail 
          : Array.isArray(error.detail) 
            ? error.detail[0]?.msg || 'Failed to create chat'
            : 'Failed to create chat';
        toast.error(errorMessage);
      }
    } catch (error) {
      console.error('Error creating chat:', error);
      toast.error('Failed to create chat');
    } finally {
      setCreating(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50">
      <div className="bg-white rounded-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">
        <div className="p-6 border-b border-claude-border">
          <h2 className="text-xl font-semibold text-claude-text-primary">
            New Job Application Chat
          </h2>
          <p className="text-sm text-claude-text-secondary mt-1">
            Start analyzing a new job opportunity
          </p>
        </div>

        <div className="p-6 space-y-4">
          <div>
            <label className="block text-sm font-medium text-claude-text-primary mb-2">
              Job Title
            </label>
            <input
              type="text"
              placeholder="e.g., Senior Software Engineer"
              value={jobTitle}
              onChange={(e) => setJobTitle(e.target.value)}
              className="w-full px-3 py-2 bg-white border border-claude-border rounded-lg focus:outline-none focus:ring-2 focus:ring-claude-accent-orange/20 focus:border-claude-accent-orange"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-claude-text-primary mb-2">
              Company
            </label>
            <input
              type="text"
              placeholder="e.g., TechCorp Inc."
              value={company}
              onChange={(e) => setCompany(e.target.value)}
              className="w-full px-3 py-2 bg-white border border-claude-border rounded-lg focus:outline-none focus:ring-2 focus:ring-claude-accent-orange/20 focus:border-claude-accent-orange"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-claude-text-primary mb-2">
              Job Description <span className="text-red-500">*</span>
            </label>
            <textarea
              placeholder="Paste the full job description here..."
              value={jobDescription}
              onChange={(e) => setJobDescription(e.target.value)}
              rows={8}
              className="w-full px-3 py-2 bg-white border border-claude-border rounded-lg focus:outline-none focus:ring-2 focus:ring-claude-accent-orange/20 focus:border-claude-accent-orange resize-none"
            />
            <p className="text-xs text-claude-text-muted mt-1">
              The more detailed the job description, the better the analysis
            </p>
          </div>
        </div>

        <div className="p-6 border-t border-claude-border flex justify-end space-x-3">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-white border border-claude-border rounded-lg hover:bg-claude-background transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleCreate}
            disabled={creating || !jobDescription.trim()}
            className="px-4 py-2 bg-claude-accent-orange text-white font-medium rounded-lg hover:bg-claude-accent-orange-hover transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {creating ? 'Creating...' : 'Create Chat'}
          </button>
        </div>
      </div>
    </div>
    
  );
}