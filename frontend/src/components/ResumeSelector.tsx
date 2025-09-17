import { useState, useEffect } from 'react';
import { FileText, RefreshCw } from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';
import toast from 'react-hot-toast';

interface Resume {
  resume_id: string;
  filename: string;
  is_active: boolean;
  target_role?: string;
}

interface ResumeSelectorProps {
  onSelect: (resume: Resume) => void;
  onClose: () => void;
  position: { top: number; left: number };
}

export default function ResumeSelector({ onSelect, onClose, position }: ResumeSelectorProps) {
  const { getToken } = useAuth();
  const [resumes, setResumes] = useState<Resume[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchResumes();
  }, []);

  const fetchResumes = async () => {
    try {
      const token = await getToken();
      if (!token) {
        toast.error('Please sign in again');
        onClose();
        return;
      }
      
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/v1/resume/list`, {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      if (response.ok) {
        const data = await response.json();
        setResumes(data.resumes);
      } else {
        toast.error('Failed to load resumes');
      }
    } catch (error) {
      console.error('Error fetching resumes:', error);
      toast.error('Failed to load resumes');
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      {/* Backdrop */}
      <div className="fixed inset-0 z-40" onClick={onClose} />
      
      {/* Dropdown */}
      <div 
        className="absolute bg-white rounded-lg shadow-lg border border-claude-border p-2 z-50 min-w-[280px] max-h-[300px] overflow-y-auto"
        style={{ 
          top: `${position.top}px`, 
          left: `${position.left}px`,
          maxWidth: '90vw'
        }}
      >
        <div className="text-xs font-medium text-claude-text-secondary px-2 py-1 border-b border-claude-border mb-2">
          Select Resume to Use
        </div>
        
        {loading ? (
          <div className="px-2 py-4 flex items-center justify-center">
            <RefreshCw className="w-4 h-4 text-claude-accent-orange animate-spin" />
            <span className="ml-2 text-sm text-claude-text-secondary">Loading...</span>
          </div>
        ) : resumes.length === 0 ? (
          <div className="px-2 py-4 text-center">
            <p className="text-sm text-claude-text-secondary mb-2">No resumes uploaded</p>
            <a 
              href="/dashboard"
              className="text-xs text-claude-accent-orange hover:underline"
            >
              Upload a resume first
            </a>
          </div>
        ) : (
          resumes.map((resume) => (
            <button
              key={resume.resume_id}
              onClick={() => {
                onSelect(resume);
                onClose();
              }}
              className="w-full flex items-center space-x-2 px-2 py-2 hover:bg-claude-background rounded transition-colors text-left group"
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
                <span className="px-1.5 py-0.5 bg-claude-accent-orange-light text-claude-accent-orange text-xs rounded opacity-0 group-hover:opacity-100 transition-opacity">
                  Active
                </span>
              )}
            </button>
          ))
        )}
      </div>
    </>
  );
}