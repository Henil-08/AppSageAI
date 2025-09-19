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
  RefreshCw,
  Edit2,
  Save,
  X,
  Shield
} from 'lucide-react';
import { useAuth } from '../../../contexts/AuthContext';
import toast from 'react-hot-toast';
import ConfirmationModal from '../../../components/ConfirmationModal';

interface Resume {
  resume_id: string;
  filename: string;
  uploaded_at: string;
  analysis_count: number;
  is_active: boolean;
  target_role?: string;
}

export default function DashboardPage() {
  const { user, getToken } = useAuth();
  const [resumes, setResumes] = useState<Resume[]>([]);
  const [uploading, setUploading] = useState(false);
  const [loading, setLoading] = useState(true);
  const [dragActive, setDragActive] = useState(false);
  const [showTargetRoleModal, setShowTargetRoleModal] = useState(false);
  const [pendingFile, setPendingFile] = useState<File | null>(null);
  const [targetRole, setTargetRole] = useState('');
  const [editingResumeId, setEditingResumeId] = useState<string | null>(null);
  const [editingTargetRole, setEditingTargetRole] = useState('');
  const [deleteModal, setDeleteModal] = useState<{
    isOpen: boolean;
    resumeId: string | null;
    filename: string;
  }>({ isOpen: false, resumeId: null, filename: '' });
  const [previewingResume, setPreviewingResume] = useState<string | null>(null);

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

  // Update target role for existing resume
  const updateTargetRole = async (resumeId: string) => {
    try {
        const token = await getToken();
        const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/api/v1/resume/${resumeId}/target-role`,
        {
            method: 'PATCH',
            headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json',
            },
            body: JSON.stringify({ target_role: editingTargetRole }),
        }
        );

        if (response.ok) {
        toast.success('Target role updated');
        
        // Update local state immediately
        setResumes(prevResumes => 
            prevResumes.map(resume => 
            resume.resume_id === resumeId 
                ? { ...resume, target_role: editingTargetRole }
                : resume
            )
        );
        
        setEditingResumeId(null);
        setEditingTargetRole('');
        
        // Then fetch to ensure consistency
        fetchResumes();
        } else {
        toast.error('Failed to update target role');
        }
    } catch (error) {
        console.error('Error updating target role:', error);
        toast.error('Failed to update target role');
    }
  };

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

    setPendingFile(file);
    setShowTargetRoleModal(true);
  }, []);

  // Upload with target role
  const uploadWithTargetRole = async () => {
    if (!pendingFile) return;

    setUploading(true);
    setShowTargetRoleModal(false);
    
    const formData = new FormData();
    formData.append('file', pendingFile);
    if (targetRole) {
      formData.append('target_role', targetRole);
    }

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
        fetchResumes();
      } else {
        toast.error(data.detail || 'Upload failed');
      }
    } catch (error) {
      console.error('Upload error:', error);
      toast.error('Failed to upload resume');
    } finally {
      setUploading(false);
      setDragActive(false);
      setPendingFile(null);
      setTargetRole('');
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
    },
    maxFiles: 1,
    maxSize: 10 * 1024 * 1024,
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
        setDeleteModal({ isOpen: false, resumeId: null, filename: '' });
      }
    } catch (error) {
      console.error('Error deleting resume:', error);
      toast.error('Failed to delete resume');
    }
  };

  // Preview resume
  const previewResume = async (resumeId: string) => {
    try {
      const token = await getToken();
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/api/v1/resume/${resumeId}/download`,
        {
          headers: {
            'Authorization': `Bearer ${token}`,
          },
        }
      );

      if (response.ok) {
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        setPreviewingResume(url);
        
        // Open in new tab
        window.open(url, '_blank');
        
        // Clean up
        setTimeout(() => {
          URL.revokeObjectURL(url);
          setPreviewingResume(null);
        }, 1000);
      } else {
        toast.error('Failed to preview resume');
      }
    } catch (error) {
      console.error('Error previewing resume:', error);
      toast.error('Unable to preview resume');
    }
  };

  // Download resume
  const downloadResume = async (resumeId: string, filename: string) => {
    try {
      const token = await getToken();
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/api/v1/resume/${resumeId}/download`,
        {
          headers: {
            'Authorization': `Bearer ${token}`,
          },
        }
      );

      if (response.ok) {
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        toast.success('Download started');
      } else {
        toast.error('Failed to download resume');
      }
    } catch (error) {
      console.error('Error downloading resume:', error);
      toast.error('Unable to download resume');
    }
  };

  return (
    <div className="p-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-semibold text-claude-text-primary mb-2">
          Resume Management
        </h1>
        <p className="text-claude-text-secondary mb-2">
          Upload and manage your resume. You can upload multiple versions and choose which one to use.
        </p>

        {/* Security Info */}
        <div className="items-center mt-4 bg-gradient-to-r from-green-50 to-emerald-50 border border-green-200 rounded-xl p-4 flex items-start space-x-3 shadow-soft">
          <img
              src="/shield-privacy.png"
              alt="Privacy Shield"
              className={`w-5 h-5 flex-shrink-0`}
          />
          <p className="text-sm text-claude-text-secondary">
            Your resumes are <span className="font-medium text-green-700">stored securely</span> on the server with 
            <span className="font-medium text-green-700"> AES-256 encryption</span>. 
            Only you can download or manage them.
          </p>
        </div>
      </div>

      {/* Upload Section */}
      <div className="-mt-1 mb-8">
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
      
      {/* Info Section */}
      <div className="mt-8 bg-claude-accent-orange-light rounded-xl p-6 mb-8">
        <div className="flex items-start space-x-3">
          <AlertCircle className="w-5 h-5 text-claude-accent-orange flex-shrink-0 mt-0.5" />
          <div>
            <h3 className="font-medium text-claude-text-primary mb-1">
              Pro Tip
            </h3>
            <p className="text-sm text-claude-text-secondary">
              Upload your most recent resume first. You can upload multiple versions and switch between them
              for different job applications.
            </p>
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
                      
                      {/* Target Role Display/Edit */}
                      <div className="mt-1">
                        {editingResumeId === resume.resume_id ? (
                          <div className="flex items-center space-x-2">
                            <input
                              type="text"
                              value={editingTargetRole}
                              onChange={(e) => setEditingTargetRole(e.target.value)}
                              placeholder="Enter target role"
                              className="px-2 py-1 text-sm border border-claude-border rounded focus:outline-none focus:ring-1 focus:ring-claude-accent-orange"
                              autoFocus
                            />
                            <button
                              onClick={() => updateTargetRole(resume.resume_id)}
                              className="p-1 text-green-600 hover:bg-green-50 rounded"
                            >
                              <Save className="w-4 h-4" />
                            </button>
                            <button
                              onClick={() => {
                                setEditingResumeId(null);
                                setEditingTargetRole('');
                              }}
                              className="p-1 text-red-600 hover:bg-red-50 rounded"
                            >
                              <X className="w-4 h-4" />
                            </button>
                          </div>
                        ) : (
                          <div className="flex items-center space-x-2">
                            <span className="text-sm text-claude-text-secondary">
                              Target: {resume.target_role || 'Not specified'}
                            </span>
                            <button
                              onClick={() => {
                                setEditingResumeId(resume.resume_id);
                                setEditingTargetRole(resume.target_role || '');
                              }}
                              className="p-1 text-claude-text-muted hover:text-claude-accent-orange"
                            >
                              <Edit2 className="w-3 h-3" />
                            </button>
                          </div>
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
                      onClick={() => previewResume(resume.resume_id)}
                      className="p-2 hover:bg-claude-background rounded-lg transition-colors"
                      title="Preview"
                    >
                      <Eye className="w-4 h-4 text-claude-text-secondary" />
                    </button>
                    
                    <button
                      onClick={() => downloadResume(resume.resume_id, resume.filename)}
                      className="p-2 hover:bg-claude-background rounded-lg transition-colors"
                      title="Download"
                    >
                      <Download className="w-4 h-4 text-claude-text-secondary" />
                    </button>
                    
                    <button
                      onClick={() => setDeleteModal({
                        isOpen: true,
                        resumeId: resume.resume_id,
                        filename: resume.filename
                      })}
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

      {/* Target Role Modal */}
      {showTargetRoleModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-2xl max-w-md w-full">
            <div className="p-6 border-b border-claude-border">
              <h2 className="text-xl font-semibold text-claude-text-primary">
                Add Target Role
              </h2>
              <p className="text-sm text-claude-text-secondary mt-1">
                What type of role are you targeting with this resume?
              </p>
            </div>
            
            <div className="p-6">
              <input
                type="text"
                placeholder="e.g., Senior Software Engineer, Product Manager"
                value={targetRole}
                onChange={(e) => setTargetRole(e.target.value)}
                className="w-full px-3 py-2 bg-white border border-claude-border rounded-lg focus:outline-none focus:ring-2 focus:ring-claude-accent-orange/20 focus:border-claude-accent-orange"
                autoFocus
              />
              <p className="text-xs text-claude-text-muted mt-2">
                This helps us provide better job matching and recommendations
              </p>
            </div>
            
            <div className="p-6 border-t border-claude-border flex justify-end space-x-3">
              <button
                onClick={() => {
                  setShowTargetRoleModal(false);
                  uploadWithTargetRole();
                }}
                className="px-4 py-2 bg-white border border-claude-border rounded-lg hover:bg-claude-background transition-colors"
              >
                Skip
              </button>
              <button
                onClick={uploadWithTargetRole}
                className="px-4 py-2 bg-claude-accent-orange text-white font-medium rounded-lg hover:bg-claude-accent-orange-hover transition-colors"
              >
                Upload Resume
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Delete Confirmation Modal */}
      <ConfirmationModal
        isOpen={deleteModal.isOpen}
        onClose={() => setDeleteModal({ isOpen: false, resumeId: null, filename: '' })}
        onConfirm={() => {
          if (deleteModal.resumeId) {
            deleteResume(deleteModal.resumeId);
          }
        }}
        title="Delete Resume"
        message={`Are you sure you want to delete "${deleteModal.filename}"? This action cannot be undone.`}
        confirmText="Delete Resume"
        cancelText="Cancel"
        type="danger"
      />
    </div>
  );
}