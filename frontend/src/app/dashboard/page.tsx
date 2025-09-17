'use client';

import { useState, useCallback, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import { 
  Upload, 
  FileText, 
  CheckCircle, 
  AlertCircle,
  Trash2,
  Eye,
  Download,
  Clock,
  RefreshCw
} from 'lucide-react';
import { useAuth } from '../../contexts/AuthContext';
import toast from 'react-hot-toast';

interface Resume {
  resume_id: string;
  filename: string;
  uploaded_at: string;
  analysis_count: number;
  is_active: boolean;
}

export default function DashboardPage() {
  const { user, getToken } = useAuth();
  const [resumes, setResumes] = useState<Resume[]>([]);
  const [uploading, setUploading] = useState(false);
  const [loading, setLoading] = useState(true);
  const [dragActive, setDragActive] = useState(false);

  // Fetch existing resumes
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
      toast.error('Failed to load resumes');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchResumes();
  }, []);

  // Handle file upload
  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (!file) return;

    if (file.type !== 'application/pdf') {
      toast.error('Please upload a PDF file');
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      toast.error('File size must be less than 10MB');
      return;
    }

    setUploading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const token = await getToken();
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/v1/resume/upload`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
        body: formData,
      });

      const data = await response.json();

      if (response.ok) {
        toast.success(data.message);
        fetchResumes(); // Refresh the list
      } else {
        toast.error(data.detail || 'Upload failed');
      }
    } catch (error) {
      console.error('Upload error:', error);
      toast.error('Failed to upload resume');
    } finally {
      setUploading(false);
      setDragActive(false);
    }
  }, [getToken]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
    },
    maxFiles: 1,
    maxSize: 10 * 1024 * 1024, // 10MB
  });

  // Set active resume
  const setActiveResume = async (resumeId: string) => {
    try {
      const token = await getToken();
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/api/v1/resume/${resumeId}/set-active`,
        {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${token}`,
          },
        }
      );

      if (response.ok) {
        toast.success('Resume activated');
        fetchResumes();
      }
    } catch (error) {
      console.error('Error setting active resume:', error);
      toast.error('Failed to activate resume');
    }
  };

  // Delete resume
  const deleteResume = async (resumeId: string) => {
    if (!confirm('Are you sure you want to delete this resume?')) return;

    try {
      const token = await getToken();
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/api/v1/resume/${resumeId}`,
        {
          method: 'DELETE',
          headers: {
            'Authorization': `Bearer ${token}`,
          },
        }
      );

      if (response.ok) {
        toast.success('Resume deleted');
        fetchResumes();
      }
    } catch (error) {
      console.error('Error deleting resume:', error);
      toast.error('Failed to delete resume');
    }
  };

  return (
    <div className="p-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-semibold text-claude-text-primary mb-2">
          Resume Management
        </h1>
        <p className="text-claude-text-secondary">
          Upload and manage your resume. You can upload multiple versions and choose which one to use.
        </p>
      </div>

      {/* Upload Section */}
      <div className="mb-8">
        <div
          {...getRootProps()}
          className={`
            border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-all
            ${isDragActive || dragActive 
              ? 'border-claude-accent-orange bg-claude-accent-orange-light' 
              : 'border-claude-border bg-white hover:border-claude-accent-orange hover:bg-claude-accent-orange-light/50'
            }
            ${uploading ? 'opacity-50 cursor-not-allowed' : ''}
          `}
        >
          <input {...getInputProps()} disabled={uploading} />
          
          <div className="flex flex-col items-center">
            {uploading ? (
              <>
                <RefreshCw className="w-12 h-12 text-claude-accent-orange mb-4 animate-spin" />
                <p className="text-lg font-medium text-claude-text-primary mb-2">
                  Uploading your resume...
                </p>
              </>
            ) : isDragActive ? (
              <>
                <Upload className="w-12 h-12 text-claude-accent-orange mb-4" />
                <p className="text-lg font-medium text-claude-text-primary mb-2">
                  Drop your resume here
                </p>
              </>
            ) : (
              <>
                <FileText className="w-12 h-12 text-claude-accent-orange mb-4" />
                <p className="text-lg font-medium text-claude-text-primary mb-2">
                  Drag & drop your resume here
                </p>
                <p className="text-sm text-claude-text-secondary mb-4">
                  or click to browse
                </p>
                <div className="flex items-center space-x-4 text-xs text-claude-text-muted">
                  <span className="flex items-center">
                    <CheckCircle className="w-3 h-3 mr-1" />
                    PDF only
                  </span>
                  <span className="flex items-center">
                    <CheckCircle className="w-3 h-3 mr-1" />
                    Max 10MB
                  </span>
                </div>
              </>
            )}
          </div>
        </div>
      </div>

      {/* Resumes List */}
      <div>
        <h2 className="text-xl font-semibold text-claude-text-primary mb-4">
          Your Resumes
        </h2>

        {loading ? (
          <div className="flex items-center justify-center py-12">
            <RefreshCw className="w-6 h-6 text-claude-accent-orange animate-spin" />
          </div>
        ) : resumes.length === 0 ? (
          <div className="bg-white rounded-xl border border-claude-border p-12 text-center">
            <FileText className="w-12 h-12 text-claude-text-muted mx-auto mb-4" />
            <p className="text-lg font-medium text-claude-text-primary mb-2">
              No resumes uploaded yet
            </p>
            <p className="text-sm text-claude-text-secondary">
              Upload your first resume to get started
            </p>
          </div>
        ) : (
          <div className="space-y-4">
            {resumes.map((resume) => (
              <div
                key={resume.resume_id}
                className={`
                  bg-white rounded-xl border p-6 transition-all
                  ${resume.is_active 
                    ? 'border-claude-accent-orange shadow-medium' 
                    : 'border-claude-border hover:shadow-soft'
                  }
                `}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-4">
                    <div className={`
                      w-12 h-12 rounded-lg flex items-center justify-center
                      ${resume.is_active 
                        ? 'bg-claude-accent-orange text-white' 
                        : 'bg-claude-background text-claude-text-secondary'
                      }
                    `}>
                      <FileText className="w-6 h-6" />
                    </div>
                    
                    <div>
                      <div className="flex items-center space-x-2">
                        <h3 className="font-medium text-claude-text-primary">
                          {resume.filename}
                        </h3>
                        {resume.is_active && (
                          <span className="px-2 py-0.5 bg-claude-accent-orange-light text-claude-accent-orange text-xs font-medium rounded-full">
                            Active
                          </span>
                        )}
                      </div>
                      <div className="flex items-center space-x-4 mt-1 text-sm text-claude-text-secondary">
                        <span className="flex items-center">
                          <Clock className="w-3 h-3 mr-1" />
                          {new Date(resume.uploaded_at).toLocaleDateString()}
                        </span>
                        <span>
                          {resume.analysis_count} analyses
                        </span>
                      </div>
                    </div>
                  </div>

                  <div className="flex items-center space-x-2">
                    {!resume.is_active && (
                      <button
                        onClick={() => setActiveResume(resume.resume_id)}
                        className="px-3 py-1.5 bg-claude-accent-orange text-white text-sm font-medium rounded-lg hover:bg-claude-accent-orange-hover transition-colors"
                      >
                        Set Active
                      </button>
                    )}
                    
                    <button
                      className="p-2 hover:bg-claude-background rounded-lg transition-colors"
                      title="View"
                    >
                      <Eye className="w-4 h-4 text-claude-text-secondary" />
                    </button>
                    
                    <button
                      className="p-2 hover:bg-claude-background rounded-lg transition-colors"
                      title="Download"
                    >
                      <Download className="w-4 h-4 text-claude-text-secondary" />
                    </button>
                    
                    <button
                      onClick={() => deleteResume(resume.resume_id)}
                      className="p-2 hover:bg-red-50 rounded-lg transition-colors"
                      title="Delete"
                    >
                      <Trash2 className="w-4 h-4 text-red-500" />
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Info Section */}
      <div className="mt-8 bg-claude-accent-orange-light rounded-xl p-6">
        <div className="flex items-start space-x-3">
          <AlertCircle className="w-5 h-5 text-claude-accent-orange flex-shrink-0 mt-0.5" />
          <div>
            <h3 className="font-medium text-claude-text-primary mb-1">
              Pro Tip
            </h3>
            <p className="text-sm text-claude-text-secondary">
              Upload your most recent resume first. You can upload multiple versions and switch between them
              for different job applications. Your resume is encrypted and stored securely.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}