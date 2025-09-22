'use client';

import { useState, useEffect } from 'react';
import { useAuth } from '../../../contexts/AuthContext';
import ConfirmationModal from '../../../components/ConfirmationModal';
import { 
  FileText,
  Target,
  Percent,
  TrendingUp,
  PenTool,
  Save,
  RotateCcw,
  ChevronDown,
  ChevronRight,
  Copy,
  Check
} from 'lucide-react';
import toast from 'react-hot-toast';

interface PromptTemplate {
  type: string;
  label: string;
  description: string;
  icon: React.ComponentType<any>;
  prompt: string;
}

const ANALYSIS_TYPES = [
  {
    type: 'resume_review',
    label: 'Job Match Analysis',
    description: 'Comprehensive review of resume against job requirements',
    icon: FileText,
  },
  {
    type: 'keyword_analysis',
    label: 'ATS Scan',
    description: 'Keyword optimization for Applicant Tracking Systems',
    icon: Target,
  },
  {
    type: 'percentage_match',
    label: 'Match Score',
    description: 'Calculate compatibility percentage with detailed breakdown',
    icon: Percent,
  },
  {
    type: 'skill_improvement',
    label: 'Skill Roadmap',
    description: 'Personalized skill improvement plan',
    icon: TrendingUp,
  },
  {
    type: 'cover_letter',
    label: 'Cover Letter',
    description: 'Generate tailored cover letters',
    icon: PenTool,
  },
];

export default function SettingsPage() {
  const { getToken } = useAuth();
  const [prompts, setPrompts] = useState<Record<string, string>>({});
  const [defaultPrompts, setDefaultPrompts] = useState<Record<string, string>>({});
  const [expandedPrompt, setExpandedPrompt] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [loading, setLoading] = useState(true);
  const [hasChanges, setHasChanges] = useState(false);
  const [copiedType, setCopiedType] = useState<string | null>(null);
  const [contentHeights, setContentHeights] = useState<Record<string, number>>({});

  const [resetModal, setResetModal] = useState({
    isOpen: false,
    type: 'all' as 'all' | 'single',
    promptType: '' // For single prompt reset
  });

  // Fetch custom prompts
  const fetchPrompts = async () => {
    try {
      const token = await getToken();
      
      // Fetch custom prompts
      const customResponse = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/api/v1/prompts/custom`,
        {
          headers: {
            'Authorization': `Bearer ${token}`,
          },
        }
      );

      // Fetch default prompts
      const defaultResponse = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/api/v1/prompts/defaults`,
        {
          headers: {
            'Authorization': `Bearer ${token}`,
          },
        }
      );

      if (customResponse.ok && defaultResponse.ok) {
        const customData = await customResponse.json();
        const defaultData = await defaultResponse.json();
        
        setPrompts(customData.prompts || {});
        setDefaultPrompts(defaultData.prompts || {});
      }
    } catch (error) {
      console.error('Error fetching prompts:', error);
      toast.error('Failed to load prompts');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPrompts();
  }, []);

  // Handle prompt change
  const handlePromptChange = (type: string, value: string) => {
    setPrompts(prev => ({
      ...prev,
      [type]: value
    }));
    setHasChanges(true);
  };

  // Save all prompts
  const saveAllPrompts = async () => {
    setSaving(true);
    try {
      const token = await getToken();
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/api/v1/prompts/update-all`,
        {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ prompts }),
        }
      );

      if (response.ok) {
        toast.success('All prompts saved successfully');
        setHasChanges(false);
      } else {
        toast.error('Failed to save prompts');
      }
    } catch (error) {
      console.error('Error saving prompts:', error);
      toast.error('Failed to save prompts');
    } finally {
      setSaving(false);
    }
  };

  // Reset single prompt to default
  const resetPrompt = async (type: string) => {
    try {
      const token = await getToken();
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/api/v1/prompts/reset/${type}`,
        {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${token}`,
          },
        }
      );

      if (response.ok) {
        const data = await response.json();
        setPrompts(prev => ({
          ...prev,
          [type]: data.default_prompt
        }));
        toast.success('Prompt reset to default');
        setHasChanges(true);
      }
    } catch (error) {
      console.error('Error resetting prompt:', error);
      toast.error('Failed to reset prompt');
    }
  };

  // Reset all prompts to default
  const resetAllPrompts = async () => {
  try {
    const token = await getToken();
    const response = await fetch(
      `${process.env.NEXT_PUBLIC_API_URL}/api/v1/prompts/reset-all`,
      {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      }
    );

    if (response.ok) {
      // After resetting, fetch the default prompts again to display them
      const defaultResponse = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/api/v1/prompts/defaults`,
        {
          headers: {
            'Authorization': `Bearer ${token}`,
          },
        }
      );

      if (defaultResponse.ok) {
        const defaultData = await defaultResponse.json();
        setPrompts(defaultData.prompts || {});
        setDefaultPrompts(defaultData.prompts || {});
      }

      toast.success('All prompts reset to default');
      setHasChanges(false);
      setResetModal({ isOpen: false, type: 'all', promptType: '' });
    }
  } catch (error) {
    console.error('Error resetting prompts:', error);
    toast.error('Failed to reset prompts');
  }
};


  // Copy prompt to clipboard
  const copyPrompt = (type: string) => {
    const prompt = prompts[type];
    if (prompt) {
      navigator.clipboard.writeText(prompt);
      setCopiedType(type);
      setTimeout(() => setCopiedType(null), 2000);
      toast.success('Prompt copied to clipboard');
    }
  };

  if (loading) {
    return (
      <div className="p-8">
        <div className="flex items-center justify-center py-12">
          <div className="w-8 h-8 border-3 border-claude-accent-orange border-t-transparent rounded-full animate-spin"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="p-8">
      {/* Header */}
      <div className="flex justify-between items-start mb-4">
        <div>
          <h1 className="text-3xl font-semibold text-claude-text-primary mb-2">
            Customize Prompts
          </h1>
          <p className="text-claude-text-secondary">
            Change your quick action prompts to get personalized results
          </p>
        </div>
        
        <div className="flex space-x-3">
          <button
            onClick={() => setResetModal({ isOpen: true, type: 'all', promptType: '' })}
            className="flex items-center space-x-2 px-4 py-2 bg-white border border-claude-border rounded-lg hover:bg-claude-background transition-colors"
           >
            <RotateCcw className="w-4 h-4" />
            <span>Reset All</span>
            </button>
          
          <button
            onClick={saveAllPrompts}
            disabled={!hasChanges || saving}
            className="flex items-center space-x-2 px-4 py-2 bg-claude-accent-orange text-white font-medium rounded-lg hover:bg-claude-accent-orange-hover transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Save className="w-4 h-4" />
            <span>{saving ? 'Saving...' : 'Save All Changes'}</span>
          </button>
        </div>
      </div>
      
      {/* Security Info */}
        <div className="items-center mt-4 bg-gradient-to-r from-green-50 to-emerald-50 border border-green-200 rounded-xl p-4 flex items-start space-x-3 shadow-soft mb-4">
          <img
              src="/shield-privacy.png"
              alt="Privacy Shield"
              className={`w-5 h-5 flex-shrink-0`}
          />
          <p className="text-sm text-claude-text-secondary">
            Your prompts are <span className="font-medium text-green-700">stored securely</span> on the server with 
            <span className="font-medium text-green-700"> AES-256 encryption</span>. 
            Only you can edit, update or reset them.
          </p>
        </div>

      {/* Info Box */}
      <div className="bg-yellow-50 border border-yellow-200 rounded-xl p-4 mb-4 shadow-soft">
        <div className="flex items-start space-x-3">
            <img
              src="/customize.png"
              alt="Privacy Shield"
              className={`w-5 h-5 flex-shrink-0`}
          />
          <div className="text-sm text-claude-text-secondary">
              These prompts control how AppSageAI responds when you use each quick action button. 
              Customize them to get responses tailored to your industry, experience level, or specific needs. 
              For Ex., make responses more technical for engineering roles or more creative for design positions.
          </div>
        </div>
      </div>

      {/* Prompt Templates */}
      <div className="space-y-4">
        {ANALYSIS_TYPES.map((analysis) => {
          const Icon = analysis.icon;
          const isExpanded = expandedPrompt === analysis.type;
          const currentPrompt = prompts[analysis.type] || '';
          const isModified = currentPrompt !== defaultPrompts[analysis.type];
          
          return (
            <div
              key={analysis.type}
              className="bg-white rounded-xl border border-claude-border overflow-hidden"
            >
              {/* Header */}
              <div
                onClick={() => setExpandedPrompt(isExpanded ? null : analysis.type)}
                className="p-5 cursor-pointer hover:bg-claude-background transition-colors"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-4">
                    <div className="w-10 h-10 bg-claude-accent-orange-light rounded-lg flex items-center justify-center">
                      <Icon className="w-5 h-5 text-claude-accent-orange" />
                    </div>
                    <div>
                      <h3 className="font-medium text-claude-text-primary flex items-center">
                        {analysis.label}
                        {isModified && (
                          <span className="ml-2 px-2 py-0.5 bg-yellow-100 text-yellow-700 text-xs rounded">
                            Modified
                          </span>
                        )}
                      </h3>
                      <p className="text-sm text-claude-text-secondary">
                        {analysis.description}
                      </p>
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-2">
                    {isExpanded ? (
                      <ChevronDown className="w-5 h-5 text-claude-text-muted" />
                    ) : (
                      <ChevronRight className="w-5 h-5 text-claude-text-muted" />
                    )}
                  </div>
                </div>
              </div>

              {/* Expanded Content */}
                <div 
                className={`overflow-hidden transition-all duration-500 ease-in-out ${isExpanded ? 'shadow-md' : ''}`}
                style={{
                    maxHeight: isExpanded ? `${contentHeights[analysis.type] || 1000}px` : '0px'
                }}
                >
                <div 
                    ref={(el) => {
                    if (el && !contentHeights[analysis.type]) {
                        setContentHeights(prev => ({
                        ...prev,
                        [analysis.type]: el.scrollHeight
                        }));
                    }
                    }}
                    className="border-t border-claude-border"
                >
                    <div className="p-5">
                        {/* Available Variables */}
                        <div className="mb-3 p-3 bg-claude-background rounded-lg">
                            <p className="text-xs font-medium text-claude-text-secondary mb-2">
                            Available Variables:
                            </p>
                            <div className="flex flex-wrap gap-2">
                            {['{context}', '{user_name}', '{job_description}', '{user_question}'].map((variable) => (
                                <code
                                key={variable}
                                className="px-2 py-1 bg-claude-accent-orange-light text-claude-accent-orange text-xs rounded"
                                >
                                {variable}
                                </code>
                            ))}
                        </div>
                            
                        </div>
                    {/* Rest of your existing content stays exactly the same */}
                    <div className="mb-3">
                        <label className="block text-sm font-medium text-claude-text-primary mb-2">
                        Prompt Template
                        </label>
                        <textarea
                        value={currentPrompt}
                        onChange={(e) => handlePromptChange(analysis.type, e.target.value)}
                        rows={12}
                        className="w-full px-3 bg-white border border-claude-border rounded-lg focus:outline-none focus:ring-2 focus:ring-claude-accent-orange/20 focus:border-claude-accent-orange font-mono text-sm"
                        placeholder="Enter your custom prompt template..."
                        />
                    </div>

                    {/* Actions */}
                    <div className="flex justify-between">
                        <div className="flex space-x-2">
                        <button
                            onClick={() => copyPrompt(analysis.type)}
                            className="flex items-center space-x-2 px-3 py-1.5 bg-white border border-claude-border rounded-lg hover:bg-claude-background transition-colors"
                        >
                            {copiedType === analysis.type ? (
                            <Check className="w-4 h-4 text-green-500" />
                            ) : (
                            <Copy className="w-4 h-4" />
                            )}
                            <span className="text-sm">Copy</span>
                        </button>
                        
                        <button
                            onClick={() => setResetModal({ 
                            isOpen: true, 
                            type: 'single', 
                            promptType: analysis.type 
                            })}
                            disabled={!isModified}
                            className="flex items-center space-x-2 px-3 py-1.5 bg-white border border-claude-border rounded-lg hover:bg-claude-background transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                            <RotateCcw className="w-4 h-4" />
                            <span className="text-sm">Reset to Default</span>
                        </button>
                        </div>

                        <div className="text-xs text-claude-text-muted">
                        {currentPrompt.length} characters
                        </div>
                    </div>
                    </div>
                </div>
                </div>
            </div>
          );
        })}
      </div>

      {/* Bottom Save Bar (sticky) */}
      {hasChanges && (
        <div className="fixed bottom-0 left-0 right-0 bg-white border-t border-claude-border p-4 shadow-lg">
          <div className="max-w-7xl mx-auto flex items-center justify-between">
            <p className="text-sm text-claude-text-secondary">
              You have unsaved changes
            </p>
            <div className="flex space-x-3">
              <button
                onClick={() => {
                  fetchPrompts();
                  setHasChanges(false);
                }}
                className="px-4 py-2 bg-white border border-claude-border rounded-lg hover:bg-claude-background transition-colors"
              >
                Discard
              </button>
              <button
                onClick={saveAllPrompts}
                disabled={saving}
                className="px-4 py-2 bg-claude-accent-orange text-white font-medium rounded-lg hover:bg-claude-accent-orange-hover transition-colors disabled:opacity-50"
              >
                {saving ? 'Saving...' : 'Save Changes'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Reset Confirmation Modal */}
<ConfirmationModal
  isOpen={resetModal.isOpen}
  onClose={() => setResetModal({ isOpen: false, type: 'all', promptType: '' })}
  onConfirm={() => {
    if (resetModal.type === 'all') {
      resetAllPrompts();
    } else if (resetModal.type === 'single' && resetModal.promptType) {
      resetPrompt(resetModal.promptType);
      setResetModal({ isOpen: false, type: 'all', promptType: '' });
    }
  }}
  title={resetModal.type === 'all' ? 'Reset All Prompts' : 'Reset Prompt to Default'}
  message={
    resetModal.type === 'all' 
      ? 'Are you sure you want to reset all prompts to their default templates?'
      : `Are you sure you want to reset the "${
          ANALYSIS_TYPES.find(a => a.type === resetModal.promptType)?.label || 'this'
        }" prompt to its default template?`
  }
  confirmText="Reset"
  cancelText="Cancel"
  type="danger"
/>
    </div>
  );
}