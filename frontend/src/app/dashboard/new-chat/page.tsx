'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '../../../contexts/AuthContext';
import { 
  Sparkles,
  Briefcase,
  FileText,
  MessageSquare,
  Zap,
  ArrowRight,
  Wand2,
  ClipboardPaste,
  Target
} from 'lucide-react';
import toast from 'react-hot-toast';

export default function NewChatPage() {
  const { getToken } = useAuth();
  const router = useRouter();
  const [inputText, setInputText] = useState('');
  const [chatType, setChatType] = useState<'job' | 'general' | 'auto'>('auto');
  const [creating, setCreating] = useState(false);

  // Quick templates
  const templates = [
    {
      icon: Briefcase,
      title: 'Job Application',
      description: 'Paste a job listing',
      action: () => setChatType('job'),
      color: 'bg-blue-500'
    },
    {
      icon: FileText,
      title: 'Resume Review',
      description: 'Get general feedback',
      action: () => {
        setChatType('general');
        setInputText('Please review my resume and suggest improvements for tech roles.');
      },
      color: 'bg-green-500'
    },
    {
      icon: Target,
      title: 'ATS Optimization',
      description: 'Improve ATS score',
      action: () => {
        setChatType('general');
        setInputText('Help me optimize my resume for ATS systems.');
      },
      color: 'bg-purple-500'
    },
    {
      icon: MessageSquare,
      title: 'Interview Prep',
      description: 'Practice questions',
      action: () => {
        setChatType('general');
        setInputText('Help me prepare for technical interviews.');
      },
      color: 'bg-orange-500'
    },
  ];

  // Smart detection of job listing
  const detectJobListing = (text: string): boolean => {
    const jobKeywords = [
      'requirements:', 'qualifications:', 'responsibilities:',
      'we are looking for', 'we are hiring', 'job description',
      'years of experience', 'bachelor', 'master',
      'salary', 'benefits', 'location:', 'remote', 'hybrid',
      'apply now', 'job type:', 'about us:', 'role:'
    ];
    
    const lowerText = text.toLowerCase();
    const keywordCount = jobKeywords.filter(keyword => lowerText.includes(keyword)).length;
    
    return keywordCount >= 3 || text.length > 500;
  };

  // Extract job details from text
  const extractJobDetails = (text: string) => {
    const lines = text.split('\n');
    let title = 'Untitled Position';
    let company = 'Unknown Company';
    
    // Try to extract title and company
    for (const line of lines.slice(0, 10)) {
      const lowerLine = line.toLowerCase();
      
      // Common patterns
      if (lowerLine.includes('position:') || lowerLine.includes('role:') || lowerLine.includes('title:')) {
        title = line.split(':')[1]?.trim() || title;
      }
      if (lowerLine.includes('company:') || lowerLine.includes('about') && lowerLine.includes('us')) {
        company = line.split(':')[1]?.trim() || company;
      }
      
      // First line might be the title
      if (lines.indexOf(line) === 0 && line.length > 5 && line.length < 100) {
        title = line.trim();
      }
    }
    
    return { title, company };
  };

  const handleCreate = async () => {
    if (!inputText.trim()) {
      toast.error('Please enter some text to start');
      return;
    }

    setCreating(true);
    
    try {
      const token = await getToken();
      
      // Auto-detect if it's a job listing
      const isJobListing = chatType === 'job' || 
        (chatType === 'auto' && detectJobListing(inputText));
      
      if (isJobListing) {
        // Extract job details
        const { title, company } = extractJobDetails(inputText);
        
        // Create job-specific chat
        const response = await fetch(
          `${process.env.NEXT_PUBLIC_API_URL}/api/v1/chat/create`,
          {
            method: 'POST',
            headers: {
              'Authorization': `Bearer ${token}`,
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              job_title: title,
              company: company,
              job_description: inputText,
            }),
          }
        );

        if (response.ok) {
          const data = await response.json();
          toast.success('Job chat created! Analyzing the role...');
          
          // Optionally save to job tracker
          localStorage.setItem(`job_${data.session_id}`, JSON.stringify({
            title,
            company,
            added_date: new Date().toISOString(),
            status: 'analyzing'
          }));
          
          router.push(`/dashboard/chat/${data.session_id}`);
        } else {
          throw new Error('Failed to create chat');
        }
      } else {
        // Create general chat
        const response = await fetch(
          `${process.env.NEXT_PUBLIC_API_URL}/api/v1/chat/create`,
          {
            method: 'POST',
            headers: {
              'Authorization': `Bearer ${token}`,
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              job_title: 'General Consultation',
              company: 'Career Development',
              job_description: inputText,
            }),
          }
        );

        if (response.ok) {
          const data = await response.json();
          toast.success('Chat created!');
          router.push(`/dashboard/chat/${data.session_id}`);
        } else {
          throw new Error('Failed to create chat');
        }
      }
    } catch (error) {
      console.error('Error:', error);
      toast.error('Failed to create chat');
    } finally {
      setCreating(false);
    }
  };

  return (
    <div className="p-8 max-w-5xl mx-auto">
      {/* Header */}
      <div className="text-center mb-8">
        <div className="inline-flex items-center justify-center w-12 h-12 bg-claude-accent-orange-light rounded-xl mb-4">
          <Sparkles className="w-6 h-6 text-claude-accent-orange" />
        </div>
        <h1 className="text-3xl font-semibold text-claude-text-primary mb-2">
          Start a New Conversation
        </h1>
        <p className="text-claude-text-secondary">
          Paste a job listing, ask questions, or get resume feedback
        </p>
      </div>

      {/* Main Input Area */}
      <div className="bg-white rounded-2xl border border-claude-border p-6 mb-6">
        <div className="flex items-center space-x-2 mb-4">
          <ClipboardPaste className="w-5 h-5 text-claude-text-secondary" />
          <h2 className="font-medium text-claude-text-primary">What would you like help with?</h2>
        </div>
        
        <textarea
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          placeholder="Paste a job listing, ask a question about your resume, or describe what you need help with..."
          className="w-full h-40 px-4 py-3 bg-claude-background border border-claude-border rounded-lg focus:outline-none focus:ring-2 focus:ring-claude-accent-orange/20 focus:border-claude-accent-orange resize-none"
        />
        
        {/* Smart Detection Indicator */}
        {inputText.length > 50 && (
          <div className="mt-3 flex items-center space-x-2">
            <Wand2 className="w-4 h-4 text-claude-accent-orange" />
            <span className="text-sm text-claude-text-secondary">
              {detectJobListing(inputText) 
                ? 'Detected: Job Listing - Will create a job-specific chat'
                : 'Detected: General Query - Will create a consultation chat'}
            </span>
          </div>
        )}
        
        <div className="mt-4 flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <label className="text-sm text-claude-text-secondary">Type:</label>
            <select
              value={chatType}
              onChange={(e) => setChatType(e.target.value as any)}
              className="px-3 py-1 bg-claude-background border border-claude-border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-claude-accent-orange/20"
            >
              <option value="auto">Auto-detect</option>
              <option value="job">Job Application</option>
              <option value="general">General Chat</option>
            </select>
          </div>
          
          <button
            onClick={handleCreate}
            disabled={!inputText.trim() || creating}
            className="flex items-center space-x-2 px-4 py-2 bg-claude-accent-orange text-white font-medium rounded-lg hover:bg-claude-accent-orange-hover transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <span>{creating ? 'Creating...' : 'Start Chat'}</span>
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Quick Start Templates */}
      <div>
        <h3 className="text-sm font-medium text-claude-text-secondary mb-3">QUICK START</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {templates.map((template, index) => {
            const Icon = template.icon;
            return (
              <button
                key={index}
                onClick={template.action}
                className="bg-white border border-claude-border rounded-xl p-4 hover:shadow-soft hover:border-claude-accent-orange/30 transition-all text-left group"
              >
                <div className={`w-10 h-10 ${template.color} rounded-lg flex items-center justify-center mb-3 group-hover:scale-110 transition-transform`}>
                  <Icon className="w-5 h-5 text-white" />
                </div>
                <h4 className="font-medium text-claude-text-primary text-sm mb-1">
                  {template.title}
                </h4>
                <p className="text-xs text-claude-text-secondary">
                  {template.description}
                </p>
              </button>
            );
          })}
        </div>
      </div>

      {/* Tips */}
      <div className="mt-8 bg-claude-accent-orange-light rounded-xl p-4">
        <div className="flex items-start space-x-3">
          <Zap className="w-5 h-5 text-claude-accent-orange flex-shrink-0 mt-0.5" />
          <div>
            <h3 className="font-medium text-claude-text-primary text-sm mb-1">
              Pro Tips
            </h3>
            <ul className="text-sm text-claude-text-secondary space-y-1">
              <li>• Just paste any job listing - we'll extract the details automatically</li>
              <li>• No job? Ask general questions about your resume or career</li>
              <li>• Each chat saves your conversation history for future reference</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}