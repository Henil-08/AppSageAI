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
  FileText,
  Target,
  TrendingUp,
  PenTool,
  Percent,
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
  const isInitialLoad = useRef(true);
  
  const sessionId = params.id as string;
  const isNewChat = sessionId === 'new';
  
  const [messageFeedback, setMessageFeedback] = useState<Record<string, 'thumbs_up' | 'thumbs_down' | null>>({});
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
  const backdropRef = useRef<HTMLDivElement>(null);
  const [cursorPosition, setCursorPosition] = useState(0);
  const [actualSessionId, setActualSessionId] = useState<string>(sessionId);
  const [copiedMessageId, setCopiedMessageId] = useState<string | null>(null);
  const [selectedTags, setSelectedTags] = useState<Set<string>>(new Set());

  useEffect(() => {
    if (messages.length > 0) {
      fetchMessageFeedback();
    }
  }, [messages]);

  useEffect(() => {
    const handlePopState = () => {
      // Force update of any pathname-dependent UI
      // This will update the sidebar highlighting
      window.dispatchEvent(new CustomEvent('refreshSidebarChats'));
    };
    
    window.addEventListener('popstate', handlePopState);
    return () => window.removeEventListener('popstate', handlePopState);
  }, []);

  // Fetch feedback status for all messages
  const fetchMessageFeedback = async () => {
    try {
      const assistantMessages = messages.filter(m => m.role === 'assistant');
      if (assistantMessages.length === 0) return;
      
      const messageIds = assistantMessages.map(m => m.message_id).join(',');
      const token = await getToken();
      
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/api/v1/analysis/feedback/bulk?message_ids=${messageIds}`,
        {
          headers: {
            'Authorization': `Bearer ${token}`,
          },
        }
      );
      
      if (response.ok) {
        const data = await response.json();
        setMessageFeedback(data.feedback);
      }
    } catch (error) {
      console.error('Error fetching feedback status:', error);
    }
  };

  const saveMessageToBackend = async (sessionId: string, message: Omit<Message, 'message_id' | 'timestamp'>) => {
    try {
      const token = await getToken();
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/api/v1/chat/${sessionId}/message`,
        {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            role: message.role,
            encrypted_content: message.content, // Backend handles encryption now
            metadata: message.metadata || {}
          }),
        }
      );
      if (!response.ok) {
        throw new Error('Failed to save message');
      }
    } catch (error) {
      console.error('Error saving message:', error);
      toast.error('Could not save your message. Please try again.');
    }
  };

  // Parse message content with bold orange for tags
  const parseMessageWithTags = (text: string) => {
    if (selectedTags.size === 0) {
      return <span className="text-claude-text-primary">{text}</span>;
    }

    // Create a regex from the selected tags to find all matches
    const tagsRegex = new RegExp(`(${Array.from(selectedTags).join('|')})`, 'g');
    const parts = text.split(tagsRegex);

    return parts.map((part, index) => {
      if (selectedTags.has(part)) {
        return (
          <span key={index} className="font-semibold text-claude-accent-orange">
            {part}
          </span>
        );
      }
      return <span key={index} className="text-claude-text-primary">{part}</span>;
    });
  };

  // Parse message content for display with markdown support
  const renderMessageContent = (content: string, isUser: boolean = false) => {
    // For user messages, only highlight actual selected resume tags
    if (isUser) {
      const parts = content.split(/(@[\w\s.-]+)/g);
      
      return parts.map((part, index) => {
        if (part.startsWith('@') && selectedTags.has(part)) {
          const filename = part.substring(1).trim();
          return (
            <span
              key={index}
              className="inline-flex items-center px-1.5 py-0.5 mx-0.5 rounded text-xs font-medium bg-white/20 text-white border border-white/30"
            >
              <FileText className="w-3 h-3 mr-1" />
              {filename}
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

  useEffect(() => {
    const initializeChat = async () => {
      setLoading(true);
      if (isNewChat) {
        // Reset all state for a new chat session
        setChatSession({
          session_id: 'new',
          job_title: 'New Chat',
          company: '',
          job_description: '',
          messages: []
        });
        setMessages([]);
        setInputMessage('');
        setEditedTitle('');
        setEditedCompany('');
        setEditedJobDescription('');
        setSelectedTags(new Set());
        setActualSessionId('new');
        setMessageFeedback({});
        setLoading(false);
      } else {
        // Fetch existing chat data
        setActualSessionId(sessionId);
        await fetchChatSession();
      }
    };

    initializeChat();
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

        const sortedMessages = data.messages.sort((a: Message, b: Message) => 
          new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
        );

        const decryptedMessages = sortedMessages.map((msg: any) => ({
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

  // Resets the 'initial load' flag whenever you switch to a new chat
  useEffect(() => {
    isInitialLoad.current = true;
  }, [sessionId]);

  // Scrolls to the bottom, using 'auto' for the initial load and 'smooth' for new messages
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({
        behavior: isInitialLoad.current ? 'auto' : 'smooth',
      });
      // After the first render/scroll, set the flag to false
      isInitialLoad.current = false;
    }
  }, [messages, sessionId]); // Reruns when messages or the chat itself changes

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

  // Update chat in backend after creation
  const updateChatInBackend = async (sessionId: string, title: string, company: string, jobDesc: string) => {
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
            job_title: title,
            company: company,
            job_description: jobDesc,
          }),
        }
      );

      if (response.ok) {
        // Refresh sidebar
        window.dispatchEvent(new CustomEvent('refreshSidebarChats'));
      }
    } catch (error) {
      console.error('Error updating chat:', error);
    }
  };

  // Create chat and detect job - Store FULL JD and UPDATE backend
  const createChatSession = async (text: string) => {
    try {
      const token = await getToken();
      let detectedTitle = "General Consultation";
      let detectedCompany = "Career Development";
      let fullJobDescription = ``;
      
      // Create the chat first with default values
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
            job_description: fullJobDescription,
          }),
        }
      );

      if (!createResponse.ok) {
        throw new Error('Failed to create chat');
      }

      const data = await createResponse.json();
      const newSessionId = data.session_id;
      
      // Update state immediately
      setActualSessionId(newSessionId);
      setChatSession({
        session_id: newSessionId,
        job_title: detectedTitle,
        company: detectedCompany,
        job_description: fullJobDescription,
        messages: []
      });
      setEditedTitle(detectedTitle);
      setEditedCompany(detectedCompany);
      setEditedJobDescription(fullJobDescription);
      
      // Try to detect job details if text is long enough
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

            if (details.is_job_listing && (details.job_title || details.company || details.job_description)) {
              detectedTitle = details.job_title || detectedTitle;
              detectedCompany = details.company || detectedCompany;
              fullJobDescription = details.job_description || text;
              
              // Update local state with detected values
              setChatSession(prev => prev ? {
                ...prev,
                job_title: detectedTitle,
                company: detectedCompany,
                job_description: fullJobDescription
              } : null);
              setEditedTitle(detectedTitle);
              setEditedCompany(detectedCompany);
              setEditedJobDescription(fullJobDescription);
              
              // Update backend with detected title and company
              await updateChatInBackend(newSessionId, detectedTitle, detectedCompany, fullJobDescription);
            }
          }
        } catch (error) {
          console.log('Detection failed, continuing with defaults');
        } finally {
          setDetectingJob(false);
        }
      }
      
      // Refresh sidebar
      window.dispatchEvent(new CustomEvent('refreshSidebarChats'));
      
      return newSessionId;
    } catch (error) {
      console.error('Error creating chat:', error);
      setDetectingJob(false);
      throw error;
    }
  };

  const handleScroll = () => {
    if (inputRef.current && backdropRef.current) {
      backdropRef.current.scrollTop = inputRef.current.scrollTop;
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
    const isQuickAction = analysisType !== 'custom_query';

    const messageContent = isQuickAction
      ? QUICK_ACTIONS.find(a => a.type === analysisType)?.label || analysisType
      : customQuery;

    const analysisQuery = customQuery || inputMessage || '';

    if (!messageContent || !messageContent.trim()) {
      toast.error('Please enter a message or select an action.');
      return;
    }

    // --- Session Creation ---
    let currentSessionId = actualSessionId;
    if (isNewChat) {
      const jobDescription = analysisQuery || '';
      if (!jobDescription.trim() && !isQuickAction) {
        toast.error('Please paste a job description to start a new chat.');
        return;
      }
      try {
        currentSessionId = await createChatSession(jobDescription);
        await new Promise(resolve => setTimeout(resolve, 50));
      } catch (error) {
        toast.error('Failed to create chat session');
        return;
      }
    }
    
    // --- User Message Handling ---
    // 1. Create the user message object
    const userMessage: Message = {
      message_id: `user_${Date.now()}`,
      role: 'user',
      content: messageContent,
      timestamp: new Date().toISOString(),
      metadata: {
        analysis_type: isQuickAction ? analysisType : undefined,
        resume_used: extractResumeFromMessage(messageContent)?.filename,
      },
    };

    // 2. Optimistically update UI with the user message BEFORE the API call
    setMessages(prev => [...prev, userMessage]);
    
    // 3. Save the user message to the database (fire and forget)
    saveMessageToBackend(currentSessionId, userMessage);

    // 4. Clear the input for the user
    setInputMessage('');
    setSelectedTags(new Set());

    // --- Assistant Message Handling ---
    // This will now only add the assistant's message, preventing order mix-ups
    // await performAnalysis(currentSessionId, analysisType, customQuery);
    await performStreamingAnalysis(currentSessionId, analysisType, analysisQuery);
  };

  // Function for streaming analysis
  const performStreamingAnalysis = async (
    sessionId: string, 
    analysisType: string, 
    customQuery?: string
  ) => {
    setAnalyzing(true);
    setCurrentAnalysisType(analysisType);
    
    const wasNewChat = actualSessionId === 'new';
    let fullResponse = '';
    let streamingMessageId = `streaming_${Date.now()}`;
    let messageAddedToUI = false;

    try {
      let resumeToUse = selectedResume;
      if (customQuery) {
        const mentionedResume = extractResumeFromMessage(customQuery);
        if (mentionedResume) {
          resumeToUse = mentionedResume;
        }
      }
      
      const token = await getToken();
      
      // Start streaming
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/api/v1/analysis/analyze-stream/${sessionId}`,
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

      if (!response.ok) {
        throw new Error('Failed to start streaming');
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      if (!reader) {
        throw new Error('No response body');
      }

      while (true) {
        const { done, value } = await reader.read();
        
        if (done) break;
        
        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              
              if (data.type === 'token') {
                // Add the message to the UI only on the first token
                if (!messageAddedToUI) {
                  const streamingMessage: Message = {
                    message_id: streamingMessageId,
                    role: 'assistant',
                    content: '',
                    timestamp: new Date().toISOString(),
                    metadata: {
                      analysis_type: analysisType,
                      resume_used: resumeToUse?.filename
                    }
                  };
                  setMessages(prev => [...prev, streamingMessage]);
                  messageAddedToUI = true;
                }
                
                fullResponse += data.content;
                
                // Update the streaming message
                setMessages(prev => 
                  prev.map(msg => 
                    msg.message_id === streamingMessageId
                      ? { ...msg, content: fullResponse }
                      : msg
                  )
                );
              } else if (data.type === 'metadata') {
                // Update message with metadata
                setMessages(prev => 
                  prev.map(msg => 
                    msg.message_id === streamingMessageId
                      ? { ...msg, metadata: { ...msg.metadata, ...data.metadata } }
                      : msg
                  )
                );
              } else if (data.type === 'done') {
                // Replace streaming message with final message
                setMessages(prev => 
                  prev.map(msg => 
                    msg.message_id === streamingMessageId
                      ? { ...msg, message_id: data.analysis_id }
                      : msg
                  )
                );
                
                if (wasNewChat && sessionId !== 'new') {
                  window.history.replaceState({}, '', `/dashboard/chat/${sessionId}`);
                }
              } else if (data.type === 'error') {
                throw new Error(data.error);
              }
            } catch (e) {
              console.error('Error parsing SSE data:', e);
            }
          }
        }
      }

    } catch (error) {
      console.error('Error in streaming analysis:', error);
      toast.error('Failed to complete analysis');
      
      // Remove the streaming message on error
      setMessages(prev => prev.filter(msg => msg.message_id !== streamingMessageId));
    } finally {
      setAnalyzing(false);
      setCurrentAnalysisType(null);
    }
  };

  // Perform the actual analysis (invoke method)
  const performAnalysis = async (sessionId: string, analysisType: string, customQuery?: string) => {
    setAnalyzing(true);
    setCurrentAnalysisType(analysisType);

    const wasNewChat = actualSessionId === 'new';

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
        
        // Add assistant response to the state
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

        if (wasNewChat && sessionId !== 'new') {
          // Use replaceState to update URL without triggering a re-render
          window.history.replaceState({}, '', `/dashboard/chat/${sessionId}`);
        }

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
      const currentFeedback = messageFeedback[messageId];
      
      const response = await fetch(
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
      
      if (response.ok) {
        const data = await response.json();
        
        // Update local state
        setMessageFeedback(prev => ({
          ...prev,
          [messageId]: data.feedback_type
        }));
        
        // Show appropriate toast
        if (data.feedback_type === null) {
          toast.success('Feedback removed');
        } else if (currentFeedback && currentFeedback !== data.feedback_type) {
          toast.success('Feedback updated');
        } else {
          toast.success('Thanks for your feedback!');
        }
      }
    } catch (error) {
      console.error('Error submitting feedback:', error);
      toast.error('Failed to submit feedback');
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

  const toTitleCase = (str: string | undefined) => {
    if (!str) return '';
    return str
      .replace('_', ' ')  // 1. Replace all underscores with spaces
      .split(' ')         // 2. Split the string into an array of words
      .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase()) // 3. Capitalize each word
      .join(' ');         // 4. Join them back together
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
      } else if (e.key === 'Enter' || e.key === 'Tab') {
        e.preventDefault();
        if (resumes[selectedResumeIndex]) {
          const resume = resumes[selectedResumeIndex];
          const beforeAt = inputMessage.substring(0, cursorPosition - 1);
          const afterCursor = inputMessage.substring(cursorPosition);
          const tagText = `@${resume.filename.replace('.pdf', '')}`;

          // --- FIX IS HERE: Removed the extra space after ${tagText} ---
          const newText = `${beforeAt}${tagText}${afterCursor}`;

          setInputMessage(newText);
          setSelectedResume(resume);
          setSelectedTags(prev => new Set(prev).add(tagText));
          setShowResumeSelector(false);

          // Set cursor position right after the tag
          setTimeout(() => {
            if (inputRef.current) {
              const newPos = beforeAt.length + tagText.length;
              inputRef.current.focus();
              inputRef.current.setSelectionRange(newPos, newPos);
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
        if (cursorPos === 0) return;

        let tagToDelete: string | null = null;
        for (const tag of selectedTags) {
          const tagEnd = inputMessage.indexOf(tag) + tag.length;
          if (tagEnd === cursorPos && inputMessage.substring(tagEnd-1, tagEnd) !== ' ') {
            tagToDelete = tag;
            break;
          }
        }

        if (tagToDelete) {
          e.preventDefault();
          const tagStart = inputMessage.indexOf(tagToDelete);
          const newText = inputMessage.substring(0, tagStart) + inputMessage.substring(cursorPos);
          setInputMessage(newText);

          setSelectedTags(prev => {
            const newTags = new Set(prev);
            newTags.delete(tagToDelete!);
            return newTags;
          });

          setTimeout(() => {
            if (inputRef.current) {
              inputRef.current.focus();
              inputRef.current.setSelectionRange(tagStart, tagStart);
            }
          }, 0);
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
      // If the textarea is empty, remove the inline height to reset to CSS default
      if (!textarea.value) {
        textarea.style.height = '';
        return;
      }

      // Otherwise, calculate the new height
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
                <img
                  src="/application.png"
                  alt="Application"
                  className={`w-5 h-5 flex-shrink-0`}
                />
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
          
          {/* Right side: Privacy + Settings */}
          <div className="flex items-center space-x-3">
            {/* Privacy Shield */}
            <div className="bg-gradient-to-r from-green-50 to-emerald-50 rounded-lg p-2 border border-green-200 shadow-sm flex items-center">
              <img
                  src="/shield-privacy.png"
                  alt="Privacy Shield"
                  className={`w-5 h-5 flex-shrink-0`}
              />
            </div>

            {/* Settings */}
            <button
              onClick={() => setShowSettings(true)}
              className="p-2 hover:bg-claude-accent-orange-light text-claude-text-secondary hover:text-claude-accent-orange rounded-lg transition-colors"
              title="Edit chat details"
            >
              <Settings className="w-5 h-5" />
            </button>
          </div>
        </div>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto bg-claude-background px-6 pt-4">
        <div className="max-w-4xl mx-auto">
          {messages.length === 0 ? (
            <div className="flex items-center justify-center min-h-[60vh]">
              <div className="text-center max-w-2xl">
                <img
                  src="/appsageai-icon.png"
                  alt="AppSageAI Logo"
                  className={`w-12 h-12 mx-auto mb-2 flex-shrink-0`}
                />
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
            <div className="flex flex-col gap-2">
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
                      
                      {/* Only show metadata for assistant messages */}
                      {message.role !== 'user' && message.metadata && (message.metadata.analysis_type || message.metadata.resume_used || message.metadata.tokens_used) && (
                        <div className="mt-2 pt-2 border-t border-claude-border">
                          <span className="text-xs text-claude-text-muted">
                            {message.metadata.resume_used && `Using: ${message.metadata.resume_used}`}
                            {message.metadata.analysis_type && message.metadata.resume_used && ' • '}
                            {toTitleCase(message.metadata.analysis_type)}
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
                            <CheckCheck className="w-4 h-4 text-blue-500" />
                          ) : (
                            <Copy className="w-4 h-4 text-claude-text-muted hover:text-blue-500" />
                          )}
                        </button>
                        <button
                          onClick={() => submitFeedback(message.message_id, 'thumbs_up')}
                          className="p-1 hover:bg-claude-background rounded transition-colors"
                          title={messageFeedback[message.message_id] === 'thumbs_up' ? 'Remove feedback' : 'Good response'}
                        >
                          <ThumbsUp 
                            className={`w-4 h-4 transition-colors duration-500 ease-in-out ${
                              messageFeedback[message.message_id] === 'thumbs_up'
                                ? 'text-green-500 fill-green-500'
                                : 'text-claude-text-muted hover:text-green-500'
                            }`} 
                          />
                        </button>
                        <button
                          onClick={() => submitFeedback(message.message_id, 'thumbs_down')}
                          className="p-1 hover:bg-claude-background rounded transition-colors"
                          title={messageFeedback[message.message_id] === 'thumbs_down' ? 'Remove feedback' : 'Poor response'}
                        >
                          <ThumbsDown 
                            className={`w-4 h-4 transition-colors duration-500 ease-in-out ${
                              messageFeedback[message.message_id] === 'thumbs_down'
                                ? 'text-red-500 fill-red-500'
                                : 'text-claude-text-muted hover:text-red-500'
                            }`} 
                          />
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              ))}

              {(analyzing || detectingJob) && (
                <div className="px-3 mt-2 lex justify-start">
                  <div className="flex items-center space-x-3">
                      <img
                        src="/appsageai-icon.png"
                        alt="AppSageAI"
                        className="w-6 h-6 animate-glow"
                      />
                      <span className="text-claude-text-secondary">
                        {detectingJob 
                          ? 'Detecting job details...' 
                          : ""
                        }
                      </span>
                    </div>
                </div>
              )}

              {/* {(analyzing || detectingJob) && (
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
              )} */}

              <div ref={messagesEndRef} />
            </div>
          )}
        </div>
      </div>

      {/* Quick Actions Bar */}
      {messages.length > 0 && actualSessionId !== 'new' && (
        <div className="mt-2">
            {/* Warning text is now cleaner and has bottom padding */}
            <div className="pb-2 text-xs text-claude-text-muted text-center">
                AppSageAI can make mistakes, so always double-check
            </div>

            {/* Your Quick Actions Bar */}
            <div className="bg-white border-t border-claude-border px-6 flex items-center justify-center flex-shrink-0" style={{ height: '52px' }}>
                <div className="max-w-4xl w-full mx-auto">
                    <div className="flex items-center justify-center space-x-2">
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
        </div>
      )}

      {/* Input Area */}
      <div className="bg-white border-t border-claude-border px-6 py-3 flex-shrink-0">
        <div className="mt-2 max-w-4xl mx-auto">
          <div className="flex items-start space-x-5 relative">
            <div className="flex-1 relative">
              {/* Grid container to perfectly align the backdrop and textarea */}
              <div className="grid grid-cols-1 grid-rows-1">
                {/* Backdrop for syntax highlighting (in the back) */}
                <div
                  ref={backdropRef}
                  className="col-start-1 row-start-1 w-full min-h-[80px] max-h-[200px] px-4 py-3 box-border bg-claude-background rounded-lg whitespace-pre-wrap break-words text-base leading-relaxed pointer-events-none overflow-y-auto"
                >
                  {parseMessageWithTags(inputMessage)}
                  {inputMessage.endsWith('\n') ? '\u00A0' : ''}
                </div>

                {/* The actual textarea (in the front) */}
                <textarea
                  ref={inputRef}
                  value={inputMessage}
                  onChange={handleInputChange}
                  onKeyDown={handleKeyDown}
                  onScroll={handleScroll}
                  placeholder={
                    "Ask AppSageAI"
                  }
                  className="col-start-1 row-start-1 w-full min-h-[80px] max-h-[200px] px-4 py-3 box-border bg-transparent border border-claude-border rounded-lg focus:outline-none focus:ring-2 focus:ring-claude-accent-orange/20 focus:border-claude-accent-orange resize-none overflow-y-auto text-base leading-relaxed"
                  style={{
                    color: 'transparent',
                    caretColor: '#1F1F1F',
                  }}
                  spellCheck="false"
                />
              </div>

              {/* Resume Selector */}
              {showResumeSelector && (
                <div className="absolute bottom-full mb-2 left-0 bg-white rounded-lg shadow-lg border border-claude-border z-10 w-auto max-h-64 overflow-y-auto">
                  <div className="p-2">
                    <div className="text-xs font-medium text-claude-text-secondary px-2 py-1">
                      Select Resume (↑↓ to navigate, Enter/Tab to select)
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
                            const newText = `${beforeAt}${tagText}${afterCursor}`;
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

            {/* Send Button */}
            <button
              onClick={sendMessage}
              disabled={!inputMessage.trim() || sending || analyzing}
              className="w-40 self-end h-20 flex items-center justify-center space-x-2 px-4 bg-claude-accent-orange text-white rounded-lg hover:bg-claude-accent-orange-hover transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <span className="font-medium leading-none">Send</span>
              <Send className="w-5 h-5" />
            </button>
          </div>
            
          
          <div className="flex items-start space-x-5"> 
            <div className="flex-1 mt-2 text-xs justify-center text-claude-text-muted text-center">
              Press @ to select resume • Cmd/Ctrl + Enter to send
            </div>
            <div className="w-40"></div>
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
        
        /* Add smooth height transition for message containers */
        .message-container {
          transition: height 0.3s ease-in-out;
        }
        
        /* Glowing animation for the logo */
        @keyframes glow {
          0%, 100% {
            filter: drop-shadow(0 0 10px rgba(255, 117, 24, 0.8));
          }
          50% {
            filter: drop-shadow(0 0 0px rgba(255, 117, 24, 0.5));
          }
        }
        
        .animate-glow {
          animation: glow 2s ease-in-out infinite;
        }
      `}</style>
    </div>
  );
}