'use client';

import { useState, useEffect, useRef } from 'react';
import { useParams, useRouter } from 'next/navigation';
import { useAuth } from '../../../../contexts/AuthContext';
import { 
  Send,
  ArrowLeft,
  Briefcase,
  Sparkles,
  FileText,
  Target,
  TrendingUp,
  PenTool,
  Percent,
  MessageSquare,
  ThumbsUp,
  ThumbsDown,
  RefreshCw
} from 'lucide-react';
import toast from 'react-hot-toast';

interface Message {
  message_id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  metadata?: {
    analysis_type?: string;
    tokens_used?: number;
  };
}

interface ChatSession {
  session_id: string;
  job_title: string;
  company: string;
  job_description: string;
  messages: Message[];
}

interface QuickAction {
  id: string;
  type: string;
  label: string;
  description: string;
  icon: React.ComponentType<any>;
  color: string;
}

const QUICK_ACTIONS: QuickAction[] = [
  {
    id: 'job_match',
    type: 'resume_review',
    label: 'Job Match Analysis',
    description: 'Comprehensive review',
    icon: FileText,
    color: 'bg-blue-500'
  },
  {
    id: 'ats_scan',
    type: 'keyword_analysis',
    label: 'ATS Scan',
    description: 'Keyword optimization',
    icon: Target,
    color: 'bg-green-500'
  },
  {
    id: 'match_percentage',
    type: 'percentage_match',
    label: 'Match Score',
    description: 'Compatibility percentage',
    icon: Percent,
    color: 'bg-purple-500'
  },
  {
    id: 'improve_skills',
    type: 'skill_improvement',
    label: 'Skill Roadmap',
    description: 'Improvement plan',
    icon: TrendingUp,
    color: 'bg-yellow-500'
  },
  {
    id: 'cover_letter',
    type: 'cover_letter',
    label: 'Cover Letter',
    description: 'Generate letter',
    icon: PenTool,
    color: 'bg-pink-500'
  },
];

export default function ChatPage() {
  const params = useParams();
  const router = useRouter();
  const { getToken } = useAuth();
  const messagesEndRef = useRef<HTMLDivElement>(null);
  
  const sessionId = params.id as string;
  
  const [chatSession, setChatSession] = useState<ChatSession | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [loading, setLoading] = useState(true);
  const [sending, setSending] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [currentAnalysisType, setCurrentAnalysisType] = useState<string | null>(null);

  // Fetch chat session
  const fetchChatSession = async () => {
    try {
      const token = await getToken();
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/api/v1/chat/${sessionId}`,
        {
          headers: {
            'Authorization': `Bearer ${token}`,
          },
        }
      );

      if (response.ok) {
        const data = await response.json();
        setChatSession(data);
        // Decrypt messages here if using encryption
        const decryptedMessages = data.messages.map((msg: any) => ({
          ...msg,
          content: msg.encrypted_content // In production, decrypt this
        }));
        setMessages(decryptedMessages);
      }
    } catch (error) {
      console.error('Error fetching chat:', error);
      toast.error('Failed to load chat');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchChatSession();
  }, [sessionId]);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  // Run analysis
  const runAnalysis = async (analysisType: string) => {
    setAnalyzing(true);
    setCurrentAnalysisType(analysisType);

    try {
      const token = await getToken();
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/api/v1/analysis/analyze/${sessionId}`,
        {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            analysis_type: analysisType,
          }),
        }
      );

      if (response.ok) {
        const data = await response.json();
        
        // Add the analysis result as a message
        const newMessage: Message = {
          message_id: data.analysis_id,
          role: 'assistant',
          content: data.encrypted_response, // In production, decrypt this
          timestamp: data.timestamp,
          metadata: data.metadata,
        };
        
        setMessages(prev => [...prev, newMessage]);
        toast.success('Analysis complete!');
      } else {
        const error = await response.json();
        toast.error(error.detail || 'Analysis failed');
      }
    } catch (error) {
      console.error('Error running analysis:', error);
      toast.error('Failed to run analysis');
    } finally {
      setAnalyzing(false);
      setCurrentAnalysisType(null);
    }
  };

  // Send message
  const sendMessage = async () => {
    if (!inputMessage.trim() || sending) return;

    const userMessage: Message = {
      message_id: `temp_${Date.now()}`,
      role: 'user',
      content: inputMessage,
      timestamp: new Date().toISOString(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setSending(true);

    try {
      const token = await getToken();
      
      // Add message to chat
      await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/api/v1/chat/${sessionId}/message`,
        {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            role: 'user',
            encrypted_content: inputMessage, // In production, encrypt this
            metadata: {},
          }),
        }
      );

      // Run custom analysis
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/api/v1/analysis/analyze/${sessionId}`,
        {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            analysis_type: 'custom_query',
            custom_query: inputMessage,
          }),
        }
      );

      if (response.ok) {
        const data = await response.json();
        
        const assistantMessage: Message = {
          message_id: data.analysis_id,
          role: 'assistant',
          content: data.encrypted_response,
          timestamp: data.timestamp,
          metadata: data.metadata,
        };
        
        setMessages(prev => [...prev, assistantMessage]);
      }
    } catch (error) {
      console.error('Error sending message:', error);
      toast.error('Failed to send message');
    } finally {
      setSending(false);
    }
  };

  // Submit feedback
  const submitFeedback = async (messageId: string, feedbackType: 'thumbs_up' | 'thumbs_down') => {
    try {
      const token = await getToken();
      await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/api/v1/analysis/feedback/${messageId}`,
        {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            feedback_type: feedbackType,
          }),
        }
      );
      
      toast.success('Thanks for your feedback!');
    } catch (error) {
      console.error('Error submitting feedback:', error);
    }
  };

  if (loading) {
    return (
      <div className="h-screen flex items-center justify-center">
        <div className="w-8 h-8 border-3 border-claude-accent-orange border-t-transparent rounded-full animate-spin"></div>
      </div>
    );
  }

  return (
    <div className="h-screen flex flex-col">
      {/* Header */}
      <div className="bg-white border-b border-claude-border px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <button
              onClick={() => router.push('/dashboard/chats')}
              className="p-2 hover:bg-claude-background rounded-lg transition-colors"
            >
              <ArrowLeft className="w-5 h-5 text-claude-text-secondary" />
            </button>
            
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-claude-accent-orange-light rounded-lg flex items-center justify-center">
                <Briefcase className="w-5 h-5 text-claude-accent-orange" />
              </div>
              <div>
                <h2 className="font-medium text-claude-text-primary">
                  {chatSession?.job_title || 'Loading...'}
                </h2>
                <p className="text-sm text-claude-text-secondary">
                  {chatSession?.company || 'Unknown Company'}
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto bg-claude-background p-6">
        <div className="max-w-4xl mx-auto space-y-6">
          {messages.length === 0 && (
            <div className="text-center py-12">
              <Sparkles className="w-12 h-12 text-claude-accent-orange mx-auto mb-4" />
              <h3 className="text-lg font-medium text-claude-text-primary mb-2">
                Ready to Analyze
              </h3>
              <p className="text-claude-text-secondary mb-6">
                Choose an analysis type below or ask me anything about this role
              </p>
            </div>
          )}

          {messages.map((message) => (
            <div
              key={message.message_id}
              className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div className={`max-w-[80%] ${message.role === 'user' ? 'order-2' : ''}`}>
                <div className={`rounded-2xl px-4 py-3 ${
                  message.role === 'user'
                    ? 'bg-claude-accent-orange text-white'
                    : 'bg-white border border-claude-border'
                }`}>
                  <p className="whitespace-pre-wrap">{message.content}</p>
                  
                  {message.metadata?.analysis_type && (
                    <div className={`mt-2 pt-2 border-t ${
                      message.role === 'user' 
                        ? 'border-white/20' 
                        : 'border-claude-border'
                    }`}>
                      <span className={`text-xs ${
                        message.role === 'user'
                          ? 'text-white/80'
                          : 'text-claude-text-muted'
                      }`}>
                        {message.metadata.analysis_type.replace('_', ' ')} • 
                        {message.metadata.tokens_used} tokens
                      </span>
                    </div>
                  )}
                </div>
                
                {message.role === 'assistant' && (
                  <div className="flex items-center space-x-2 mt-2 px-2">
                    <button
                      onClick={() => submitFeedback(message.message_id, 'thumbs_up')}
                      className="p-1 hover:bg-claude-background rounded transition-colors"
                    >
                      <ThumbsUp className="w-4 h-4 text-claude-text-muted hover:text-green-500" />
                    </button>
                    <button
                      onClick={() => submitFeedback(message.message_id, 'thumbs_down')}
                      className="p-1 hover:bg-claude-background rounded transition-colors"
                    >
                      <ThumbsDown className="w-4 h-4 text-claude-text-muted hover:text-red-500" />
                    </button>
                  </div>
                )}
              </div>
            </div>
          ))}

          {analyzing && (
            <div className="flex justify-start">
              <div className="bg-white border border-claude-border rounded-2xl px-4 py-3">
                <div className="flex items-center space-x-2">
                  <RefreshCw className="w-4 h-4 text-claude-accent-orange animate-spin" />
                  <span className="text-claude-text-secondary">
                    Running {currentAnalysisType?.replace('_', ' ')} analysis...
                  </span>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Quick Actions */}
      <div className="bg-white border-t border-claude-border px-6 py-3">
        <div className="max-w-4xl mx-auto">
          <div className="flex items-center space-x-2 overflow-x-auto pb-2">
            {QUICK_ACTIONS.map((action) => {
              const Icon = action.icon;
              return (
                <button
                  key={action.id}
                  onClick={() => runAnalysis(action.type)}
                  disabled={analyzing}
                  className="flex items-center space-x-2 px-3 py-1.5 bg-claude-background hover:bg-claude-accent-orange-light rounded-lg transition-colors whitespace-nowrap group disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <Icon className="w-4 h-4 text-claude-text-secondary group-hover:text-claude-accent-orange" />
                  <span className="text-sm text-claude-text-secondary group-hover:text-claude-accent-orange">
                    {action.label}
                  </span>
                </button>
              );
            })}
          </div>
        </div>
      </div>

      {/* Input Area */}
      <div className="bg-white border-t border-claude-border px-6 py-4">
        <div className="max-w-4xl mx-auto">
          <div className="flex items-end space-x-3">
            <textarea
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  sendMessage();
                }
              }}
              placeholder="Ask anything about this role..."
              rows={1}
              className="flex-1 px-4 py-2 bg-claude-background border border-claude-border rounded-lg focus:outline-none focus:ring-2 focus:ring-claude-accent-orange/20 focus:border-claude-accent-orange resize-none"
            />
            
            <button
              onClick={sendMessage}
              disabled={!inputMessage.trim() || sending}
              className="p-2 bg-claude-accent-orange text-white rounded-lg hover:bg-claude-accent-orange-hover transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Send className="w-5 h-5" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}