'use client';

import { useState, useEffect, useRef } from 'react';
import { useParams, useRouter, useSearchParams } from 'next/navigation';
import { useAuth } from '../../../../contexts/AuthContext';
import ResumeSelector from '../../../../components/ResumeSelector';
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
  RefreshCw,
  Settings,
  Edit2,
  Check,
  X
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
    resume_used?: string;
  };
}

interface ChatSession {
  session_id: string;
  job_title: string;
  company: string;
  job_description: string;
  messages: Message[];
}

interface Resume {
  resume_id: string;
  filename: string;
  is_active: boolean;
  target_role?: string;
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
  const searchParams = useSearchParams();
  const { getToken } = useAuth();
  const messagesEndRef = useRef<HTMLDivElement>(null);
  
  const sessionId = params.id as string;
  const isNewChat = sessionId === 'new';
  
  const [chatSession, setChatSession] = useState<ChatSession | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [loading, setLoading] = useState(!isNewChat);
  const [sending, setSending] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [currentAnalysisType, setCurrentAnalysisType] = useState<string | null>(null);
  const [showResumeSelector, setShowResumeSelector] = useState(false);
  const [resumeSelectorPosition, setResumeSelectorPosition] = useState({ top: 0, left: 0 });
  const [selectedResume, setSelectedResume] = useState<Resume | null>(null);
  const [selectedResumeIndex, setSelectedResumeIndex] = useState(0);
  const [resumes, setResumes] = useState<Resume[]>([]);
  const [showSettings, setShowSettings] = useState(false);
  const [editingTitle, setEditingTitle] = useState(false);
  const [editedTitle, setEditedTitle] = useState('');
  const [editedCompany, setEditedCompany] = useState('');
  const [detectingJob, setDetectingJob] = useState(false);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const [cursorPosition, setCursorPosition] = useState(0);

  // Parse message content to render inline resume tags with better highlighting
  const renderMessageContent = (content: string, isUser: boolean = false) => {
    // Find @resume_name patterns and highlight them
    const parts = content.split(/(@[\w.-]+)/g);
    
    return parts.map((part, index) => {
      if (part.startsWith('@')) {
        return (
          <span
            key={index}
            className={`inline-flex items-center px-2 py-0.5 mx-1 rounded-md text-sm font-medium ${
              isUser 
                ? 'bg-white/20 text-white border border-white/30' 
                : 'bg-claude-accent-orange/10 text-claude-accent-orange border border-claude-accent-orange/20'
            }`}
          >
            <FileText className="w-3 h-3 mr-1" />
            {part.substring(1)}
          </span>
        );
      }
      return part;
    });
  };

  // Initialize for new chat
  useEffect(() => {
    if (isNewChat) {
      setChatSession({
        session_id: 'new',
        job_title: 'New Chat',
        company: '',
        job_description: '',
        messages: []
      });
      setLoading(false);
    } else {
      fetchChatSession();
    }
  }, [sessionId]);

  // Fetch existing chat session
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
        setEditedTitle(data.job_title);
        setEditedCompany(data.company);
        const decryptedMessages = data.messages.map((msg: any) => ({
          ...msg,
          content: msg.encrypted_content
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
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  // Fetch resumes when showing selector
  const fetchResumes = async () => {
    try {
      const token = await getToken();
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/v1/resume/list`, {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      if (response.ok) {
        const data = await response.json();
        setResumes(data.resumes);
      }
    } catch (error) {
      console.error('Error fetching resumes:', error);
    }
  };

  // Auto-detect job and create/update chat
  const detectAndProcessInput = async (text: string, analysisType?: string) => {
    setDetectingJob(true);
    
    try {
      const token = await getToken();
      let detectedTitle = chatSession?.job_title || "General Consultation";
      let detectedCompany = chatSession?.company || "Career Development";
      let isJobListing = false;
      
      // Try to detect job details if text is substantial
      if (text.length > 100) {
        try {
          const extractResponse = await fetch(
            `${process.env.NEXT_PUBLIC_API_URL}/api/v1/analysis/extract-job-details`,
            {
              method: 'POST',
              headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json',
              },
              body: JSON.stringify({ text }),
            }
          );

          if (extractResponse.ok) {
            const details = await extractResponse.json();
            if (details.is_job_listing) {
              detectedTitle = details.job_title || "Untitled Position";
              detectedCompany = details.company || "Unknown Company";
              isJobListing = true;
              
              // Animate title change with smooth transition
              setChatSession(prev => prev ? {
                ...prev,
                job_title: detectedTitle,
                company: detectedCompany,
                job_description: text
              } : null);
              setEditedTitle(detectedTitle);
              setEditedCompany(detectedCompany);
              
              // Add a subtle animation by updating with a small delay
              setTimeout(() => {
                setDetectingJob(false);
              }, 300);
            }
          }
        } catch (error) {
          console.log('Detection failed, continuing...');
        }
      }
      
      // If new chat, create it first
      if (isNewChat) {
        const createResponse = await fetch(
          `${process.env.NEXT_PUBLIC_API_URL}/api/v1/chat/create`,
          {
            method: 'POST',
            headers: {
              'Authorization': `Bearer ${token}`,
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              job_title: detectedTitle,
              company: detectedCompany,
              job_description: text || "General consultation",
            }),
          }
        );

        if (createResponse.ok) {
          const data = await createResponse.json();
          // Navigate to the new chat
          window.history.replaceState({}, '', `/dashboard/${data.session_id}`);
          
          // Save to job tracker if it's a job
          if (isJobListing) {
            localStorage.setItem(`job_${data.session_id}`, JSON.stringify({
              title: detectedTitle,
              company: detectedCompany,
              added_date: new Date().toISOString(),
              status: 'interested',
              chatSessionId: data.session_id
            }));
          }
          
          return data.session_id;
        }
      }
      
      return sessionId;
      
    } catch (error) {
      console.error('Error processing input:', error);
      return sessionId;
    } finally {
      setDetectingJob(false);
    }
  };

  // Run analysis
  const runAnalysis = async (analysisType: string) => {
    // If new chat with no input, prompt user
    if (isNewChat && !inputMessage.trim()) {
      toast.error('Please enter a job description or question first');
      inputRef.current?.focus();
      return;
    }
    
    setAnalyzing(true);
    setCurrentAnalysisType(analysisType);

    try {
      // If new chat or first message, detect job and create chat
      let actualSessionId = sessionId;
      if (isNewChat || messages.length === 0) {
        actualSessionId = await detectAndProcessInput(inputMessage, analysisType);
      }
      
      const token = await getToken();
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/api/v1/analysis/analyze/${actualSessionId}`,
        {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            analysis_type: analysisType,
            resume_id: selectedResume?.resume_id,
            custom_query: analysisType === 'custom_query' ? inputMessage : undefined
          }),
        }
      );

      if (response.ok) {
        const data = await response.json();
        
        // Add user message if it's from input
        if (inputMessage.trim() && analysisType === 'custom_query') {
          const userMessage: Message = {
            message_id: `user_${Date.now()}`,
            role: 'user',
            content: inputMessage,
            timestamp: new Date().toISOString(),
          };
          setMessages(prev => [...prev, userMessage]);
          setInputMessage('');
        }
        
        const newMessage: Message = {
          message_id: data.analysis_id,
          role: 'assistant',
          content: data.encrypted_response,
          timestamp: data.timestamp,
          metadata: {
            ...data.metadata,
            resume_used: selectedResume?.filename
          },
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
    
    // Process as custom query
    await runAnalysis('custom_query');
  };

  // Update chat details
  const updateChatDetails = async () => {
    try {
      const token = await getToken();
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/api/v1/chat/${sessionId}/update`,
        {
          method: 'PATCH',
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            job_title: editedTitle,
            company: editedCompany,
          }),
        }
      );

      if (response.ok) {
        setChatSession(prev => prev ? {
          ...prev,
          job_title: editedTitle,
          company: editedCompany
        } : null);
        setEditingTitle(false);
        toast.success('Chat details updated');
      }
    } catch (error) {
      console.error('Error updating chat:', error);
      toast.error('Failed to update chat details');
    }
  };

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

  // Handle keyboard navigation for resume selector
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (showResumeSelector) {
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        setSelectedResumeIndex(prev => Math.min(prev + 1, resumes.length - 1));
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        setSelectedResumeIndex(prev => Math.max(prev - 1, 0));
      } else if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        if (resumes[selectedResumeIndex]) {
          // Select the resume
          const resume = resumes[selectedResumeIndex];
          const beforeAt = inputMessage.substring(0, cursorPosition - 1);
          const afterCursor = inputMessage.substring(cursorPosition);
          const newText = `${beforeAt}@${resume.filename.replace('.pdf', '')} ${afterCursor}`;
          setInputMessage(newText);
          setSelectedResume(resume);
          setShowResumeSelector(false);
        }
      } else if (e.key === 'Escape') {
        setShowResumeSelector(false);
      }
    } else {
      // Submit on Cmd/Ctrl + Enter
      if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        sendMessage();
      }
    }
  };

  // Adjust textarea height dynamically
  const adjustTextareaHeight = () => {
    const textarea = inputRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      const scrollHeight = textarea.scrollHeight;
      const maxHeight = 200; // Max height in pixels
      textarea.style.height = `${Math.min(scrollHeight, maxHeight)}px`;
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
              onClick={() => router.push('/dashboard/chat')}
              className="p-2 hover:bg-claude-background rounded-lg transition-colors"
            >
              <ArrowLeft className="w-5 h-5 text-claude-text-secondary" />
            </button>
            
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-claude-accent-orange-light rounded-lg flex items-center justify-center">
                <Briefcase className="w-5 h-5 text-claude-accent-orange" />
              </div>
              <div>
                {editingTitle ? (
                  <div className="flex items-center space-x-2">
                    <input
                      type="text"
                      value={editedTitle}
                      onChange={(e) => setEditedTitle(e.target.value)}
                      className="px-2 py-1 border border-claude-border rounded focus:outline-none focus:ring-1 focus:ring-claude-accent-orange"
                      autoFocus
                    />
                    <input
                      type="text"
                      value={editedCompany}
                      onChange={(e) => setEditedCompany(e.target.value)}
                      placeholder="Company"
                      className="px-2 py-1 border border-claude-border rounded focus:outline-none focus:ring-1 focus:ring-claude-accent-orange text-sm"
                    />
                    <button
                      onClick={updateChatDetails}
                      className="p-1 text-green-600 hover:bg-green-50 rounded"
                    >
                      <Check className="w-4 h-4" />
                    </button>
                    <button
                      onClick={() => {
                        setEditingTitle(false);
                        setEditedTitle(chatSession?.job_title || '');
                        setEditedCompany(chatSession?.company || '');
                      }}
                      className="p-1 text-red-600 hover:bg-red-50 rounded"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  </div>
                ) : (
                  <>
                    <h2 className="font-medium text-claude-text-primary flex items-center transition-all duration-500 ease-in-out">
                      {detectingJob && (
                        <RefreshCw className="w-4 h-4 mr-2 text-claude-accent-orange animate-spin" />
                      )}
                      <span className="transition-all duration-500">
                        {chatSession?.job_title || 'New Chat'}
                      </span>
                      {!isNewChat && (
                        <button
                          onClick={() => setEditingTitle(true)}
                          className="ml-2 p-1 hover:bg-claude-background rounded opacity-0 hover:opacity-100 transition-opacity"
                        >
                          <Edit2 className="w-3 h-3 text-claude-text-muted" />
                        </button>
                      )}
                    </h2>
                    <p className="text-sm text-claude-text-secondary transition-all duration-500">
                      {chatSession?.company || 'Start typing to begin'}
                    </p>
                  </>
                )}
              </div>
            </div>
          </div>
          
          {!isNewChat && (
            <button
              onClick={() => setShowSettings(!showSettings)}
              className="p-2 hover:bg-claude-background rounded-lg transition-colors"
            >
              <Settings className="w-5 h-5 text-claude-text-secondary" />
            </button>
          )}
        </div>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto bg-claude-background p-6">
        <div className="max-w-4xl mx-auto">
          {messages.length === 0 ? (
            <div className="flex items-center justify-center min-h-[60vh]">
              <div className="text-center max-w-2xl">
                <Sparkles className="w-12 h-12 text-claude-accent-orange mx-auto mb-4" />
                <h3 className="text-2xl font-medium text-claude-text-primary mb-2">
                  How can I help you today?
                </h3>
                <p className="text-claude-text-secondary mb-8">
                  Paste a job listing, ask a question about your resume, or choose a quick action
                </p>
                
                {/* Show quick actions only when user starts typing */}
                {inputMessage.length > 0 && (
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-3 animate-fadeIn">
                    {QUICK_ACTIONS.map((action) => {
                      const Icon = action.icon;
                      return (
                        <button
                          key={action.id}
                          onClick={() => runAnalysis(action.type)}
                          disabled={analyzing}
                          className="bg-white border border-claude-border rounded-xl p-3 hover:shadow-soft hover:border-claude-accent-orange/30 transition-all group disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                          <div className={`w-10 h-10 ${action.color} rounded-lg flex items-center justify-center mx-auto mb-2 group-hover:scale-110 transition-transform`}>
                            <Icon className="w-5 h-5 text-white" />
                          </div>
                          <h4 className="font-medium text-claude-text-primary text-sm">
                            {action.label}
                          </h4>
                          <p className="text-xs text-claude-text-muted mt-1">
                            {action.description}
                          </p>
                        </button>
                      );
                    })}
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="space-y-6">
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
                      <div className="whitespace-pre-wrap">
                        {renderMessageContent(message.content, message.role === 'user')}
                      </div>
                      
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

              {(analyzing || detectingJob) && (
                <div className="flex justify-start">
                  <div className="bg-white border border-claude-border rounded-2xl px-4 py-3">
                    <div className="flex items-center space-x-2">
                      <RefreshCw className="w-4 h-4 text-claude-accent-orange animate-spin" />
                      <span className="text-claude-text-secondary">
                        {detectingJob ? 'Detecting job details...' : `Running ${currentAnalysisType?.replace('_', ' ')} analysis...`}
                      </span>
                    </div>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>
          )}
        </div>
      </div>

      {/* Quick Actions Bar (only for existing chats with messages) */}
      {messages.length > 0 && !isNewChat && (
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
      )}

      {/* Input Area */}
      <div className="bg-white border-t border-claude-border px-6 py-4">
        <div className="max-w-4xl mx-auto">
          <div className="flex items-end space-x-3 relative">
            <div className="flex-1 relative">
              <textarea
                ref={inputRef}
                value={inputMessage}
                onChange={(e) => {
                  setInputMessage(e.target.value);
                  setCursorPosition(e.target.selectionStart);
                  adjustTextareaHeight();
                  
                  // Check for @ symbol at cursor position
                  const text = e.target.value;
                  const cursorPos = e.target.selectionStart;
                  const textBeforeCursor = text.substring(0, cursorPos);
                  const lastAtIndex = textBeforeCursor.lastIndexOf('@');
                  
                  // Only show selector if @ is at the end or followed by incomplete text
                  if (lastAtIndex !== -1 && lastAtIndex === cursorPos - 1) {
                    fetchResumes();
                    setShowResumeSelector(true);
                    setSelectedResumeIndex(0);
                  }
                }}
                onKeyDown={handleKeyDown}
                placeholder={
                  isNewChat 
                    ? "Paste a job listing or ask about your resume... (Cmd/Ctrl + Enter to send)"
                    : "Ask a follow-up question... (Cmd/Ctrl + Enter to send)"
                }
                className="w-full px-4 py-2 bg-claude-background border border-claude-border rounded-lg focus:outline-none focus:ring-2 focus:ring-claude-accent-orange/20 focus:border-claude-accent-orange resize-none overflow-y-auto"
                style={{ minHeight: '40px', maxHeight: '200px' }}
              />
              
              {/* Resume Selector - positioned above input */}
              {showResumeSelector && (
                <div 
                  className="absolute bottom-full mb-2 left-0 bg-white rounded-lg shadow-lg border border-claude-border z-50 w-72 max-h-64 overflow-y-auto"
                >
                  <div className="p-2">
                    <div className="text-xs font-medium text-claude-text-secondary px-2 py-1">
                      Select Resume (↑↓ to navigate, Enter to select)
                    </div>
                    {resumes.length === 0 ? (
                      <div className="px-2 py-3 text-sm text-claude-text-secondary">
                        No resumes uploaded
                      </div>
                    ) : (
                      resumes.map((resume, index) => (
                        <button
                          key={resume.resume_id}
                          onClick={() => {
                            const beforeAt = inputMessage.substring(0, cursorPosition - 1);
                            const afterCursor = inputMessage.substring(cursorPosition);
                            const newText = `${beforeAt}@${resume.filename.replace('.pdf', '')} ${afterCursor}`;
                            setInputMessage(newText);
                            setSelectedResume(resume);
                            setShowResumeSelector(false);
                            inputRef.current?.focus();
                          }}
                          className={`w-full flex items-center space-x-2 px-2 py-2 rounded text-left transition-colors ${
                            index === selectedResumeIndex 
                              ? 'bg-claude-accent-orange-light' 
                              : 'hover:bg-claude-background'
                          }`}
                        >
                          <FileText className="w-4 h-4 text-claude-text-secondary flex-shrink-0" />
                          <div className="flex-1 min-w-0">
                            <div className="text-sm text-claude-text-primary truncate">
                              {resume.filename}
                            </div>
                            {resume.target_role && (
                              <div className="text-xs text-claude-text-muted truncate">
                                Target: {resume.target_role}
                              </div>
                            )}
                          </div>
                          {resume.is_active && (
                            <span className="px-1.5 py-0.5 bg-claude-accent-orange text-white text-xs rounded">
                              Active
                            </span>
                          )}
                        </button>
                      ))
                    )}
                  </div>
                </div>
              )}
            </div>
            
            <button
              onClick={sendMessage}
              disabled={!inputMessage.trim() || sending || analyzing}
              className="p-2 bg-claude-accent-orange text-white rounded-lg hover:bg-claude-accent-orange-hover transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Send className="w-5 h-5" />
            </button>
          </div>
          
          <div className="mt-2 text-xs text-claude-text-muted text-center">
            Press @ to select resume • Cmd/Ctrl + Enter to send
          </div>
        </div>
      </div>
    </div>
  );
}