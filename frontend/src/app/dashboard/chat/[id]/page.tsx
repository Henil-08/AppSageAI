'use client';

import { useState, useEffect, useRef } from 'react';
import { useParams, useRouter } from 'next/navigation';
import { useAuth } from '../../../../contexts/AuthContext';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
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
  Copy,
  CheckCheck
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
  const [selectedResume, setSelectedResume] = useState<Resume | null>(null);
  const [selectedResumeIndex, setSelectedResumeIndex] = useState(0);
  const [resumes, setResumes] = useState<Resume[]>([]);
  const [showSettings, setShowSettings] = useState(false);
  const [editedTitle, setEditedTitle] = useState('');
  const [editedCompany, setEditedCompany] = useState('');
  const [editedJobDescription, setEditedJobDescription] = useState('');
  const [detectingJob, setDetectingJob] = useState(false);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const [cursorPosition, setCursorPosition] = useState(0);
  const [actualSessionId, setActualSessionId] = useState<string>(sessionId);
  const [copiedMessageId, setCopiedMessageId] = useState<string | null>(null);
  const [selectedTags, setSelectedTags] = useState<Set<string>>(new Set());

  // Enhanced render function for inline resume highlighting in textarea
  const renderInputWithHighlights = () => {
    const parts = inputMessage.split(/(@[\w\s.-]+)/g);
    
    return (
      <div 
        className="absolute inset-0 px-4 py-2 pointer-events-none whitespace-pre-wrap break-words overflow-hidden"
        style={{ 
          wordBreak: 'break-word',
          overflowWrap: 'break-word',
          fontSize: '14px',
          lineHeight: '20px',
          fontFamily: 'inherit'
        }}
      >
        {parts.map((part, index) => {
          if (part.startsWith('@') && selectedTags.has(part)) {
            return (
              <span
                key={index}
                className="inline-block px-1.5 py-0.5 rounded-md"
                style={{ 
                  backgroundColor: '#FFE8D9',   // light orange background
                  color: '#EA5A0C',             // strong orange text
                  // borderBottom: '2px solid #EA5A0C',
                  fontWeight: '500'
                }}
              >
                {part}
              </span>
            );
          }
          return <span key={index} style={{ color: 'transparent' }}>{part}</span>;
        })}
      </div>
    );
  };

  // Parse message content for display with markdown support
  const renderMessageContent = (content: string, isUser: boolean = false) => {
    // For user messages, only highlight actual selected resume tags
    if (isUser) {
      const parts = content.split(/(@[\w\s.-]+)/g);
      
      return parts.map((part, index) => {
        if (part.startsWith('@') && selectedTags.has(part)) {
          return (
            <span
              key={index}
              style={{ 
                color: '#EA5A0C',   // orange
                fontWeight: 600     // bold
              }}
            >
              {part}
            </span>
          );
        }
        return <span key={index}>{part}</span>;
      });
    }
    
    // For assistant messages, render markdown
    return (
      <div className="prose prose-sm max-w-none">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={{
            h1: ({ children }) => <h1 className="text-xl font-bold mb-3 text-claude-text-primary">{children}</h1>,
            h2: ({ children }) => <h2 className="text-lg font-semibold mb-2 text-claude-text-primary">{children}</h2>,
            h3: ({ children }) => <h3 className="text-base font-semibold mb-2 text-claude-text-primary">{children}</h3>,
            p: ({ children }) => <p className="mb-3 text-claude-text-primary leading-relaxed">{children}</p>,
            ul: ({ children }) => <ul className="list-disc list-inside mb-3 space-y-1">{children}</ul>,
            ol: ({ children }) => <ol className="list-decimal list-inside mb-3 space-y-1">{children}</ol>,
            li: ({ children }) => <li className="text-claude-text-primary">{children}</li>,
            strong: ({ children }) => <strong className="font-semibold text-claude-text-primary">{children}</strong>,
            em: ({ children }) => <em className="italic">{children}</em>,
            code: (props: any) => {
              const { inline, children } = props;
              return inline ? (
                <code className="px-1.5 py-0.5 bg-claude-background text-claude-accent-orange text-sm rounded">
                  {children}
                </code>
              ) : (
                <code className="block p-3 bg-claude-background text-sm rounded-lg overflow-x-auto">
                  {children}
                </code>
              );
            },
            pre: ({ children }) => <pre className="mb-3">{children}</pre>,
            blockquote: ({ children }) => (
              <blockquote className="border-l-4 border-claude-accent-orange-light pl-4 py-1 mb-3 italic text-claude-text-secondary">
                {children}
              </blockquote>
            ),
            hr: () => <hr className="my-4 border-claude-border" />,
            a: ({ href, children }) => (
              <a href={href} target="_blank" rel="noopener noreferrer" className="text-claude-accent-orange hover:underline">
                {children}
              </a>
            ),
            table: ({ children }) => (
              <div className="overflow-x-auto mb-3">
                <table className="min-w-full border-collapse border border-claude-border">
                  {children}
                </table>
              </div>
            ),
            th: ({ children }) => (
              <th className="border border-claude-border px-3 py-2 bg-claude-background font-semibold text-left">
                {children}
              </th>
            ),
            td: ({ children }) => (
              <td className="border border-claude-border px-3 py-2">
                {children}
              </td>
            ),
          }}
        >
          {content}
        </ReactMarkdown>
      </div>
    );
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
      setActualSessionId('new');
    } else {
      setActualSessionId(sessionId);
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
        setEditedJobDescription(data.job_description);
        
        const decryptedMessages = data.messages.map((msg: any) => ({
          message_id: msg.message_id,
          role: msg.role,
          content: msg.encrypted_content,
          timestamp: msg.timestamp,
          metadata: msg.metadata
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

  // Fetch resumes
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
        if (!selectedResume && data.resumes.length > 0) {
          const activeResume = data.resumes.find((r: Resume) => r.is_active);
          if (activeResume) {
            setSelectedResume(activeResume);
          }
        }
      }
    } catch (error) {
      console.error('Error fetching resumes:', error);
    }
  };

  useEffect(() => {
    fetchResumes();
  }, []);

  // Create chat and detect job - Store FULL JD
  const createChatSession = async (text: string) => {
    try {
      const token = await getToken();
      let detectedTitle = "General Consultation";
      let detectedCompany = "Career Development";
      let fullJobDescription = text; // Store the FULL text
      
      // Try to detect job details (but keep the full JD)
      if (text.length > 100) {
        try {
          setDetectingJob(true);
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
              // Keep the FULL original text as job description, not the summary
              fullJobDescription = text;
            }
          }
        } catch (error) {
          console.log('Detection failed, continuing...');
        } finally {
          setDetectingJob(false);
        }
      }
      
      // Create the chat with FULL job description
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
            job_description: fullJobDescription, // Send FULL text
          }),
        }
      );

      if (createResponse.ok) {
        const data = await createResponse.json();
        const newSessionId = data.session_id;
        
        // Update state with detected title/company but FULL job description
        setActualSessionId(newSessionId);
        setChatSession({
          session_id: newSessionId,
          job_title: detectedTitle,
          company: detectedCompany,
          job_description: fullJobDescription, // Store FULL text
          messages: []
        });
        setEditedTitle(detectedTitle);
        setEditedCompany(detectedCompany);
        setEditedJobDescription(fullJobDescription); // Store FULL text
        
        // Update URL
        window.history.replaceState({}, '', `/dashboard/chat/${newSessionId}`);
        
        // Refresh sidebar chats after creation
        setTimeout(() => {
          window.dispatchEvent(new CustomEvent('refreshSidebarChats'));
        }, 1000);
        
        return newSessionId;
      }
      
      throw new Error('Failed to create chat');
    } catch (error) {
      console.error('Error creating chat:', error);
      setDetectingJob(false);
      throw error;
    }
  };

  // Extract resume from message
  const extractResumeFromMessage = (message: string) => {
    // Only extract if it's a properly selected tag
    const tags = Array.from(selectedTags);
    for (const tag of tags) {
      if (message.includes(tag)) {
        const filename = tag.substring(1).trim();
        const matchedResume = resumes.find(r => 
          r.filename === filename || 
          r.filename === `${filename}.pdf` ||
          r.filename.replace('.pdf', '') === filename
        );
        if (matchedResume) return matchedResume;
      }
    }
    return null;
  };

  // Run analysis
  const runAnalysis = async (analysisType: string, customQuery?: string) => {
    // For quick actions, use the input as job description if it's a new chat
    let jobDescriptionText = customQuery || inputMessage;
    
    // Check if we need to create a chat first
    if (actualSessionId === 'new') {
      if (!jobDescriptionText.trim()) {
        toast.error('Please enter a job description or question first');
        inputRef.current?.focus();
        return;
      }
      
      try {
        setDetectingJob(true);
        const newId = await createChatSession(jobDescriptionText);
        setDetectingJob(false);
        
        // For quick actions, add a user message showing which action was clicked
        if (analysisType !== 'custom_query') {
          const actionLabel = QUICK_ACTIONS.find(a => a.type === analysisType)?.label || analysisType;
          const userMessage: Message = {
            message_id: `user_${Date.now()}`,
            role: 'user',
            content: actionLabel,
            timestamp: new Date().toISOString(),
            metadata: {
              analysis_type: analysisType
            }
          };
          setMessages(prev => [...prev, userMessage]);
          setInputMessage(''); // Clear the input
        }
        
        // Continue with analysis using new session
        await performAnalysis(newId, analysisType, analysisType === 'custom_query' ? jobDescriptionText : undefined);
      } catch (error) {
        setDetectingJob(false);
        toast.error('Failed to create chat session');
        return;
      }
    } else {
      // Existing chat
      await performAnalysis(actualSessionId, analysisType, customQuery);
    }
  };

  // Perform the actual analysis
  const performAnalysis = async (sessionId: string, analysisType: string, customQuery?: string) => {
    setAnalyzing(true);
    setCurrentAnalysisType(analysisType);

    try {
      // Determine which resume to use
      let resumeToUse = selectedResume;
      if (customQuery) {
        const mentionedResume = extractResumeFromMessage(customQuery);
        if (mentionedResume) {
          resumeToUse = mentionedResume;
        }
      }
      
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
            resume_id: resumeToUse?.resume_id,
            custom_query: customQuery
          }),
        }
      );

      if (response.ok) {
        const data = await response.json();
        
        // Add user message if custom query
        if (customQuery && analysisType === 'custom_query') {
          const userMessage: Message = {
            message_id: `user_${Date.now()}`,
            role: 'user',
            content: customQuery,
            timestamp: new Date().toISOString(),
            metadata: {
              resume_used: resumeToUse?.filename
            }
          };
          setMessages(prev => [...prev, userMessage]);
          setInputMessage('');
          setSelectedTags(new Set()); // Clear tags after sending
        }
        
        // Add assistant response
        const newMessage: Message = {
          message_id: data.analysis_id,
          role: 'assistant',
          content: data.encrypted_response,
          timestamp: data.timestamp,
          metadata: {
            ...data.metadata,
            resume_used: resumeToUse?.filename
          },
        };
        
        setMessages(prev => [...prev, newMessage]);
        toast.success('Analysis complete!');
      } else {
        const error = await response.json();
        console.error('Analysis error:', error);
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
    
    setSending(true);
    try {
      await runAnalysis('custom_query', inputMessage);
    } finally {
      setSending(false);
    }
  };

  // Settings Modal
  const SettingsModal = () => {
    const [localTitle, setLocalTitle] = useState(editedTitle);
    const [localCompany, setLocalCompany] = useState(editedCompany);
    const [localJobDescription, setLocalJobDescription] = useState(editedJobDescription);
    const [saving, setSaving] = useState(false);

    const handleSave = async () => {
      if (actualSessionId === 'new') {
        toast.error('Please send a message first to create the chat');
        return;
      }
      
      setSaving(true);
      try {
        const token = await getToken();
        const response = await fetch(
          `${process.env.NEXT_PUBLIC_API_URL}/api/v1/chat/${actualSessionId}/update`,
          {
            method: 'PATCH',
            headers: {
              'Authorization': `Bearer ${token}`,
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              job_title: localTitle || 'General Consultation',
              company: localCompany || 'Career Development',
              job_description: localJobDescription || '',
            }),
          }
        );

        if (response.ok) {
          // Update local state
          setChatSession(prev => prev ? {
            ...prev,
            job_title: localTitle,
            company: localCompany,
            job_description: localJobDescription
          } : null);
          setEditedTitle(localTitle);
          setEditedCompany(localCompany);
          setEditedJobDescription(localJobDescription);
          
          // Force refresh sidebar chats
          window.dispatchEvent(new CustomEvent('refreshSidebarChats'));
          
          toast.success('Chat details updated');
          setShowSettings(false);
        } else {
          const error = await response.json();
          console.error('Update error:', error);
          toast.error(error.detail || 'Failed to update details');
        }
      } catch (error) {
        console.error('Error updating chat:', error);
        toast.error('Failed to update details');
      } finally {
        setSaving(false);
      }
    };

    return (
      <div className="fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50">
        <div className="bg-white rounded-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">
          <div className="p-6 border-b border-claude-border">
            <h2 className="text-xl font-semibold text-claude-text-primary">
              Chat Settings
            </h2>
            <p className="text-sm text-claude-text-secondary mt-1">
              Update job details for this chat
            </p>
          </div>

          <div className="p-6 space-y-4">
            <div>
              <label className="block text-sm font-medium text-claude-text-primary mb-2">
                Job Title
              </label>
              <input
                type="text"
                value={localTitle}
                onChange={(e) => setLocalTitle(e.target.value)}
                className="w-full px-3 py-2 bg-white border border-claude-border rounded-lg focus:outline-none focus:ring-2 focus:ring-claude-accent-orange/20 focus:border-claude-accent-orange"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-claude-text-primary mb-2">
                Company
              </label>
              <input
                type="text"
                value={localCompany}
                onChange={(e) => setLocalCompany(e.target.value)}
                className="w-full px-3 py-2 bg-white border border-claude-border rounded-lg focus:outline-none focus:ring-2 focus:ring-claude-accent-orange/20 focus:border-claude-accent-orange"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-claude-text-primary mb-2">
                Job Description
              </label>
              <textarea
                value={localJobDescription}
                onChange={(e) => setLocalJobDescription(e.target.value)}
                rows={8}
                className="w-full px-3 py-2 bg-white border border-claude-border rounded-lg focus:outline-none focus:ring-2 focus:ring-claude-accent-orange/20 focus:border-claude-accent-orange resize-none"
              />
            </div>
          </div>

          <div className="p-6 border-t border-claude-border flex justify-end space-x-3">
            <button
              onClick={() => setShowSettings(false)}
              className="px-4 py-2 bg-white border border-claude-border rounded-lg hover:bg-claude-background transition-colors"
            >
              Cancel
            </button>
            <button
              onClick={handleSave}
              disabled={saving}
              className="px-4 py-2 bg-claude-accent-orange text-white font-medium rounded-lg hover:bg-claude-accent-orange-hover transition-colors disabled:opacity-50"
            >
              {saving ? 'Saving...' : 'Save Changes'}
            </button>
          </div>
        </div>
      </div>
    );
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

  const copyMessage = (content: string, messageId: string) => {
    const plainText = content.replace(/[#*`_~\[\]()]/g, '');
    navigator.clipboard.writeText(plainText);
    setCopiedMessageId(messageId);
    setTimeout(() => setCopiedMessageId(null), 2000);
    toast.success('Copied to clipboard');
  };

  // Handle input changes with smart tag management
  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const newValue = e.target.value;
    const cursorPos = e.target.selectionStart;
    
    setInputMessage(newValue);
    setCursorPosition(cursorPos);
    adjustTextareaHeight();
    
    // Check for @ symbol
    const textBeforeCursor = newValue.substring(0, cursorPos);
    const lastAtIndex = textBeforeCursor.lastIndexOf('@');
    
    if (lastAtIndex !== -1 && lastAtIndex === cursorPos - 1) {
      setShowResumeSelector(true);
      setSelectedResumeIndex(0);
    } else {
      setShowResumeSelector(false);
    }
  };

  // Handle keyboard navigation
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
          const resume = resumes[selectedResumeIndex];
          const beforeAt = inputMessage.substring(0, cursorPosition - 1);
          const afterCursor = inputMessage.substring(cursorPosition);
          const tagText = `@${resume.filename.replace('.pdf', '')}`;
          const newText = `${beforeAt}${tagText} ${afterCursor}`;
          setInputMessage(newText);
          setSelectedResume(resume);
          setSelectedTags(prev => new Set(prev).add(tagText));
          setShowResumeSelector(false);
          // Set cursor position after the tag
          setTimeout(() => {
            if (inputRef.current) {
              const newPos = beforeAt.length + tagText.length + 1;
              inputRef.current.setSelectionRange(newPos, newPos);
              inputRef.current.focus();
            }
          }, 0);
        }
      } else if (e.key === 'Escape') {
        setShowResumeSelector(false);
      }
    } else {
      // Handle backspace on tags
      if (e.key === 'Backspace') {
        const cursorPos = e.currentTarget.selectionStart;
        
        // Check if we're right after a tag
        for (const tag of selectedTags) {
          const tagEnd = inputMessage.indexOf(tag) + tag.length;
          if (tagEnd === cursorPos || tagEnd + 1 === cursorPos) {
            e.preventDefault();
            const tagStart = inputMessage.indexOf(tag);
            const beforeTag = inputMessage.substring(0, tagStart);
            const afterTag = inputMessage.substring(tagEnd);
            const newText = beforeTag + afterTag;
            setInputMessage(newText);
            setSelectedTags(prev => {
              const newTags = new Set(prev);
              newTags.delete(tag);
              return newTags;
            });
            e.currentTarget.setSelectionRange(tagStart, tagStart);
            return;
          }
        }
      }
      
      // Submit on Cmd/Ctrl + Enter
      if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        sendMessage();
      }
    }
  };

  const adjustTextareaHeight = () => {
    const textarea = inputRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      const scrollHeight = textarea.scrollHeight;
      const maxHeight = 200;
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
    <div className="h-screen flex flex-col overflow-hidden">
      {/* Header */}
      <div className="h-[65px] bg-white border-b border-claude-border px-6 flex items-center flex-shrink-0">
        <div className="flex items-center justify-between w-full">
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
                <h2 className="font-medium text-claude-text-primary flex items-center">
                  {detectingJob && (
                    <RefreshCw className="w-4 h-4 mr-2 text-claude-accent-orange animate-spin" />
                  )}
                  <span>{chatSession?.job_title || 'New Chat'}</span>
                </h2>
                <p className="text-sm text-claude-text-secondary">
                  {chatSession?.company || 'Start typing to begin'}
                </p>
              </div>
            </div>
          </div>
          
          <button
            onClick={() => setShowSettings(true)}
            className="p-2 hover:bg-claude-background rounded-lg transition-colors"
            title="Edit chat details"
          >
            <Settings className="w-5 h-5 text-claude-text-secondary" />
          </button>
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
                
                {/* Show quick actions when user starts typing more than 100 chars */}
                {inputMessage.length > 100 && (
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
                      <div className={message.role === 'user' ? '' : 'message-content'}>
                        {renderMessageContent(message.content, message.role === 'user')}
                      </div>
                      
                      {message.metadata && (message.metadata.analysis_type || message.metadata.resume_used) && (
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
                            {message.metadata.resume_used && `Using: ${message.metadata.resume_used}`}
                            {message.metadata.analysis_type && message.metadata.resume_used && ' • '}
                            {message.metadata.analysis_type && message.metadata.analysis_type.replace('_', ' ')}
                            {message.metadata.tokens_used && ` • ${message.metadata.tokens_used} tokens`}
                          </span>
                        </div>
                      )}
                    </div>
                    
                    {message.role === 'assistant' && (
                      <div className="flex items-center space-x-2 mt-2 px-2">
                        <button
                          onClick={() => copyMessage(message.content, message.message_id)}
                          className="p-1 hover:bg-claude-background rounded transition-colors"
                          title="Copy message"
                        >
                          {copiedMessageId === message.message_id ? (
                            <CheckCheck className="w-4 h-4 text-green-500" />
                          ) : (
                            <Copy className="w-4 h-4 text-claude-text-muted hover:text-claude-text-secondary" />
                          )}
                        </button>
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

      {/* Quick Actions Bar */}
      {messages.length > 0 && actualSessionId !== 'new' && (
        <div className="bg-white border-t border-claude-border px-6 py-3 flex-shrink-0">
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
      <div className="bg-white border-t border-claude-border px-6 py-4 flex-shrink-0">
        <div className="max-w-4xl mx-auto">
          <div className="flex items-end space-x-3 relative">
            <div className="flex-1 relative">
              {/* Highlight overlay */}
              {inputMessage && renderInputWithHighlights()}
              
              <textarea
                ref={inputRef}
                value={inputMessage}
                onChange={handleInputChange}
                onKeyDown={handleKeyDown}
                placeholder={
                  isNewChat 
                    ? "Paste a job listing or ask about your resume... (Cmd/Ctrl + Enter to send)"
                    : "Ask a follow-up question... (Cmd/Ctrl + Enter to send)"
                }
                className="w-full px-4 py-2 bg-transparent border border-claude-border rounded-lg focus:outline-none focus:ring-2 focus:ring-claude-accent-orange/20 focus:border-claude-accent-orange resize-none overflow-y-auto relative z-10"
                style={{ minHeight: '40px', maxHeight: '200px' }}
              />
              
              {/* Resume Selector */}
              {showResumeSelector && (
                <div className="absolute bottom-full mb-2 left-0 bg-white rounded-lg shadow-lg border border-claude-border z-50 w-72 max-h-64 overflow-y-auto">
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
                            const tagText = `@${resume.filename.replace('.pdf', '')}`;
                            const newText = `${beforeAt}${tagText} ${afterCursor}`;
                            setInputMessage(newText);
                            setSelectedResume(resume);
                            setSelectedTags(prev => new Set(prev).add(tagText));
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

      {/* Settings Modal */}
      {showSettings && <SettingsModal />}
      
      {/* Add fadeIn animation to global styles */}
      <style jsx global>{`
        @keyframes fadeIn {
          from {
            opacity: 0;
            transform: translateY(10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        
        .animate-fadeIn {
          animation: fadeIn 0.3s ease-out;
        }
      `}</style>
    </div>
  );
}