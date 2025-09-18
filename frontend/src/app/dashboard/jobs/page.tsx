'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '../../../contexts/AuthContext';
import { 
  Briefcase,
  Calendar,
  MapPin,
  CheckCircle,
  Clock,
  XCircle,
  MessageSquare,
  Filter,
  AlertCircle,
  TrendingUp,
  Edit2,
  Save,
  X
} from 'lucide-react';
import toast from 'react-hot-toast';

interface ChatWithTracker {
  session_id: string;
  job_title: string;
  company: string;
  created_at: string;
  updated_at: string;
  message_count: number;
  tracker_status?: string;
  applied_date?: string;
  tracker_notes?: string;
  has_job_description: boolean;
}

const STATUS_OPTIONS = [
  { value: 'not_applicable', label: 'N/A', color: 'bg-gray-100 text-gray-700' },
  { value: 'interested', label: 'Interested', color: 'bg-blue-100 text-blue-700' },
  { value: 'applied', label: 'Applied', color: 'bg-green-100 text-green-700' },
  { value: 'interviewing', label: 'Interviewing', color: 'bg-yellow-100 text-yellow-700' },
  { value: 'offered', label: 'Offered', color: 'bg-purple-100 text-purple-700' },
  { value: 'rejected', label: 'Rejected', color: 'bg-red-100 text-red-700' },
];

const STATUS_ICONS = {
  not_applicable: AlertCircle,
  interested: Clock,
  applied: CheckCircle,
  interviewing: MessageSquare,
  offered: TrendingUp,
  rejected: XCircle,
};

export default function JobTrackerPage() {
  const { getToken } = useAuth();
  const router = useRouter();
  const [chats, setChats] = useState<ChatWithTracker[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<string>('all');
  const [editingChat, setEditingChat] = useState<string | null>(null);
  const [editData, setEditData] = useState<{
    status: string;
    applied_date: string;
    notes: string;
  }>({ status: '', applied_date: '', notes: '' });

  useEffect(() => {
    fetchChatsWithTracker();
  }, []);

  const fetchChatsWithTracker = async () => {
    try {
      const token = await getToken();
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/api/v1/chat/list?limit=100`,
        {
          headers: {
            'Authorization': `Bearer ${token}`,
          },
        }
      );

      if (response.ok) {
        const data = await response.json();
        // Process chats to determine if they have job descriptions
        const processedChats = data.chats.map((chat: any) => ({
          ...chat,
          tracker_status: chat.tracker_status || 'not_applicable',
          has_job_description: chat.job_title !== 'General Consultation' && 
                               chat.company !== 'Career Development'
        }));
        setChats(processedChats);
      }
    } catch (error) {
      console.error('Error fetching chats:', error);
      toast.error('Failed to load job applications');
    } finally {
      setLoading(false);
    }
  };

  const updateTrackerStatus = async (
    sessionId: string, 
    status: string,
    appliedDate?: string,
    notes?: string
  ) => {
    try {
      const token = await getToken();
      const params = new URLSearchParams({ tracker_status: status });
      if (appliedDate) params.append('applied_date', appliedDate);
      if (notes) params.append('notes', notes);

      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/api/v1/chat/${sessionId}/tracker?${params}`,
        {
          method: 'PATCH',
          headers: {
            'Authorization': `Bearer ${token}`,
          },
        }
      );

      if (response.ok) {
        toast.success('Status updated');
        fetchChatsWithTracker();
        setEditingChat(null);
      } else {
        toast.error('Failed to update status');
      }
    } catch (error) {
      console.error('Error updating tracker:', error);
      toast.error('Failed to update status');
    }
  };

  const startEditing = (chat: ChatWithTracker) => {
    setEditingChat(chat.session_id);
    setEditData({
      status: chat.tracker_status || 'not_applicable',
      applied_date: chat.applied_date || '',
      notes: chat.tracker_notes || ''
    });
  };

  const saveEdit = () => {
    if (editingChat) {
      updateTrackerStatus(
        editingChat,
        editData.status,
        editData.applied_date,
        editData.notes
      );
    }
  };

  const cancelEdit = () => {
    setEditingChat(null);
    setEditData({ status: '', applied_date: '', notes: '' });
  };

  const filteredChats = filter === 'all' 
    ? chats 
    : chats.filter(chat => chat.tracker_status === filter);

  const stats = {
    total: chats.length,
    jobs: chats.filter(c => c.has_job_description).length,
    applied: chats.filter(c => c.tracker_status === 'applied').length,
    interviewing: chats.filter(c => c.tracker_status === 'interviewing').length,
    offered: chats.filter(c => c.tracker_status === 'offered').length,
  };

  if (loading) {
    return (
      <div className="p-8 flex items-center justify-center min-h-[60vh]">
        <div className="w-8 h-8 border-3 border-claude-accent-orange border-t-transparent rounded-full animate-spin"></div>
      </div>
    );
  }

  return (
    <div className="p-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-semibold text-claude-text-primary mb-2">
          Application Tracker
        </h1>
        <p className="text-claude-text-secondary">
          Track the status of all your job application chats
        </p>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-8">
        <div className="bg-white rounded-xl border border-claude-border p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-claude-text-secondary text-sm">Total Chats</span>
            <MessageSquare className="w-4 h-4 text-claude-text-muted" />
          </div>
          <div className="text-2xl font-semibold text-claude-text-primary">{stats.total}</div>
        </div>
        
        <div className="bg-white rounded-xl border border-claude-border p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-claude-text-secondary text-sm">Job Apps</span>
            <Briefcase className="w-4 h-4 text-claude-accent-orange" />
          </div>
          <div className="text-2xl font-semibold text-claude-text-primary">{stats.jobs}</div>
        </div>
        
        <div className="bg-white rounded-xl border border-claude-border p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-claude-text-secondary text-sm">Applied</span>
            <CheckCircle className="w-4 h-4 text-green-500" />
          </div>
          <div className="text-2xl font-semibold text-claude-text-primary">{stats.applied}</div>
        </div>
        
        <div className="bg-white rounded-xl border border-claude-border p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-claude-text-secondary text-sm">Interviewing</span>
            <MessageSquare className="w-4 h-4 text-yellow-500" />
          </div>
          <div className="text-2xl font-semibold text-claude-text-primary">{stats.interviewing}</div>
        </div>
        
        <div className="bg-white rounded-xl border border-claude-border p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-claude-text-secondary text-sm">Offers</span>
            <TrendingUp className="w-4 h-4 text-purple-500" />
          </div>
          <div className="text-2xl font-semibold text-claude-text-primary">{stats.offered}</div>
        </div>
      </div>

      {/* Filter Tabs */}
      <div className="flex items-center space-x-2 mb-6 overflow-x-auto">
        <button
          onClick={() => setFilter('all')}
          className={`px-4 py-2 rounded-lg font-medium transition-colors whitespace-nowrap ${
            filter === 'all' 
              ? 'bg-claude-accent-orange text-white' 
              : 'bg-white text-claude-text-secondary hover:bg-claude-background'
          }`}
        >
          All ({chats.length})
        </button>
        {STATUS_OPTIONS.map(status => {
          const count = chats.filter(c => c.tracker_status === status.value).length;
          return (
            <button
              key={status.value}
              onClick={() => setFilter(status.value)}
              className={`px-4 py-2 rounded-lg font-medium transition-colors whitespace-nowrap ${
                filter === status.value 
                  ? 'bg-claude-accent-orange text-white' 
                  : 'bg-white text-claude-text-secondary hover:bg-claude-background'
              }`}
            >
              {status.label} ({count})
            </button>
          );
        })}
      </div>

      {/* Applications List */}
      <div className="space-y-4">
        {filteredChats.length === 0 ? (
          <div className="bg-white rounded-xl border border-claude-border p-12 text-center">
            <Briefcase className="w-12 h-12 text-claude-text-muted mx-auto mb-4" />
            <p className="text-lg font-medium text-claude-text-primary mb-2">
              {filter === 'all' ? 'No chats yet' : `No ${filter.replace('_', ' ')} applications`}
            </p>
            <p className="text-sm text-claude-text-secondary">
              Start a new chat to track your job applications
            </p>
          </div>
        ) : (
          filteredChats.map(chat => {
            const StatusIcon = STATUS_ICONS[chat.tracker_status as keyof typeof STATUS_ICONS] || AlertCircle;
            const statusOption = STATUS_OPTIONS.find(s => s.value === chat.tracker_status);
            const isEditing = editingChat === chat.session_id;
            
            return (
              <div
                key={chat.session_id}
                className="bg-white rounded-xl border border-claude-border p-6 hover:shadow-soft transition-all"
              >
                {isEditing ? (
                  // Edit Mode
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <h3 className="text-lg font-medium text-claude-text-primary">
                        {chat.job_title}
                      </h3>
                      <div className="flex items-center space-x-2">
                        <button
                          onClick={saveEdit}
                          className="p-2 text-green-600 hover:bg-green-50 rounded-lg"
                        >
                          <Save className="w-4 h-4" />
                        </button>
                        <button
                          onClick={cancelEdit}
                          className="p-2 text-red-600 hover:bg-red-50 rounded-lg"
                        >
                          <X className="w-4 h-4" />
                        </button>
                      </div>
                    </div>
                    
                    <div className="grid grid-cols-3 gap-4">
                      <div>
                        <label className="block text-xs text-claude-text-secondary mb-1">
                          Status
                        </label>
                        <select
                          value={editData.status}
                          onChange={(e) => setEditData({...editData, status: e.target.value})}
                          className="w-full px-3 py-2 bg-white border border-claude-border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-claude-accent-orange/20"
                        >
                          {STATUS_OPTIONS.map(opt => (
                            <option key={opt.value} value={opt.value}>
                              {opt.label}
                            </option>
                          ))}
                        </select>
                      </div>
                      
                      <div>
                        <label className="block text-xs text-claude-text-secondary mb-1">
                          Applied Date
                        </label>
                        <input
                          type="date"
                          value={editData.applied_date}
                          onChange={(e) => setEditData({...editData, applied_date: e.target.value})}
                          className="w-full px-3 py-2 bg-white border border-claude-border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-claude-accent-orange/20"
                        />
                      </div>
                      
                      <div>
                        <label className="block text-xs text-claude-text-secondary mb-1">
                          Notes
                        </label>
                        <input
                          type="text"
                          placeholder="Add notes..."
                          value={editData.notes}
                          onChange={(e) => setEditData({...editData, notes: e.target.value})}
                          className="w-full px-3 py-2 bg-white border border-claude-border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-claude-accent-orange/20"
                        />
                      </div>
                    </div>
                  </div>
                ) : (
                  // View Mode
                  <div className="flex items-start justify-between">
                    <div 
                      className="flex-1 cursor-pointer"
                      onClick={() => router.push(`/dashboard/chat/${chat.session_id}`)}
                    >
                      <div className="flex items-center space-x-3 mb-2">
                        <h3 className="text-lg font-medium text-claude-text-primary hover:text-claude-accent-orange transition-colors">
                          {chat.job_title}
                        </h3>
                        {statusOption && (
                          <span className={`px-2 py-1 rounded-full text-xs font-medium ${statusOption.color}`}>
                            <StatusIcon className="w-3 h-3 inline mr-1" />
                            {statusOption.label}
                          </span>
                        )}
                        {!chat.has_job_description && (
                          <span className="px-2 py-1 bg-gray-100 text-gray-600 rounded-full text-xs">
                            Resume Enhancement
                          </span>
                        )}
                      </div>
                      
                      <div className="flex items-center space-x-4 text-sm text-claude-text-secondary mb-2">
                        <span className="flex items-center">
                          <Briefcase className="w-3 h-3 mr-1" />
                          {chat.company}
                        </span>
                        <span className="flex items-center">
                          <MessageSquare className="w-3 h-3 mr-1" />
                          {chat.message_count} messages
                        </span>
                      </div>
                      
                      <div className="flex items-center space-x-4 text-xs text-claude-text-muted">
                        <span className="flex items-center">
                          <Calendar className="w-3 h-3 mr-1" />
                          Created: {new Date(chat.created_at).toLocaleDateString()}
                        </span>
                        {chat.applied_date && (
                          <span className="flex items-center">
                            <CheckCircle className="w-3 h-3 mr-1" />
                            Applied: {new Date(chat.applied_date).toLocaleDateString()}
                          </span>
                        )}
                      </div>
                      
                      {chat.tracker_notes && (
                        <div className="mt-2 text-sm text-claude-text-secondary italic">
                          Note: {chat.tracker_notes}
                        </div>
                      )}
                    </div>
                    
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        startEditing(chat);
                      }}
                      className="p-2 hover:bg-claude-background rounded-lg transition-colors ml-4"
                    >
                      <Edit2 className="w-4 h-4 text-claude-text-secondary" />
                    </button>
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}