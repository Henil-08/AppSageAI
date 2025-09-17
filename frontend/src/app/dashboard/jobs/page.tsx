'use client';

import { useState, useEffect } from 'react';
import { 
  Briefcase,
  Calendar,
  MapPin,
  DollarSign,
  CheckCircle,
  Clock,
  XCircle,
  MessageSquare,
  ExternalLink,
  Filter
} from 'lucide-react';

interface JobApplication {
  id: string;
  title: string;
  company: string;
  location?: string;
  salary?: string;
  status: 'interested' | 'applied' | 'interviewing' | 'offered' | 'rejected';
  appliedDate?: string;
  deadline?: string;
  notes?: string;
  chatSessionId?: string;
  url?: string;
}

const STATUS_COLORS = {
  interested: 'bg-gray-100 text-gray-700',
  applied: 'bg-blue-100 text-blue-700',
  interviewing: 'bg-yellow-100 text-yellow-700',
  offered: 'bg-green-100 text-green-700',
  rejected: 'bg-red-100 text-red-700',
};

const STATUS_ICONS = {
  interested: Clock,
  applied: CheckCircle,
  interviewing: MessageSquare,
  offered: CheckCircle,
  rejected: XCircle,
};

export default function JobTrackerPage() {
  const [jobs, setJobs] = useState<JobApplication[]>([]);
  const [filter, setFilter] = useState<string>('all');

  useEffect(() => {
    // Load jobs from localStorage (in production, this would be from backend)
    const loadJobs = () => {
      const jobKeys = Object.keys(localStorage).filter(key => key.startsWith('job_'));
      const loadedJobs = jobKeys.map(key => {
        const data = JSON.parse(localStorage.getItem(key) || '{}');
        return {
          id: key.replace('job_', ''),
          ...data
        };
      });
      setJobs(loadedJobs);
    };
    
    loadJobs();
  }, []);

  const updateJobStatus = (jobId: string, newStatus: JobApplication['status']) => {
    const updatedJobs = jobs.map(job => 
      job.id === jobId ? { ...job, status: newStatus } : job
    );
    setJobs(updatedJobs);
    
    // Update localStorage
    const job = updatedJobs.find(j => j.id === jobId);
    if (job) {
      localStorage.setItem(`job_${jobId}`, JSON.stringify(job));
    }
  };

  const filteredJobs = filter === 'all' 
    ? jobs 
    : jobs.filter(job => job.status === filter);

  const stats = {
    total: jobs.length,
    applied: jobs.filter(j => j.status === 'applied').length,
    interviewing: jobs.filter(j => j.status === 'interviewing').length,
    offered: jobs.filter(j => j.status === 'offered').length,
  };

  return (
    <div className="p-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-semibold text-claude-text-primary mb-2">
          Job Application Tracker
        </h1>
        <p className="text-claude-text-secondary">
          Track your job applications and their progress
        </p>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-4 gap-4 mb-8">
        <div className="bg-white rounded-xl border border-claude-border p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-claude-text-secondary text-sm">Total Applications</span>
            <Briefcase className="w-4 h-4 text-claude-text-muted" />
          </div>
          <div className="text-2xl font-semibold text-claude-text-primary">{stats.total}</div>
        </div>
        
        <div className="bg-white rounded-xl border border-claude-border p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-claude-text-secondary text-sm">Applied</span>
            <CheckCircle className="w-4 h-4 text-blue-500" />
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
            <CheckCircle className="w-4 h-4 text-green-500" />
          </div>
          <div className="text-2xl font-semibold text-claude-text-primary">{stats.offered}</div>
        </div>
      </div>

      {/* Filter Tabs */}
      <div className="flex items-center space-x-2 mb-6">
        <button
          onClick={() => setFilter('all')}
          className={`px-4 py-2 rounded-lg font-medium transition-colors ${
            filter === 'all' 
              ? 'bg-claude-accent-orange text-white' 
              : 'bg-white text-claude-text-secondary hover:bg-claude-background'
          }`}
        >
          All
        </button>
        {Object.keys(STATUS_COLORS).map(status => (
          <button
            key={status}
            onClick={() => setFilter(status)}
            className={`px-4 py-2 rounded-lg font-medium transition-colors capitalize ${
              filter === status 
                ? 'bg-claude-accent-orange text-white' 
                : 'bg-white text-claude-text-secondary hover:bg-claude-background'
            }`}
          >
            {status}
          </button>
        ))}
      </div>

      {/* Jobs List */}
      <div className="space-y-4">
        {filteredJobs.length === 0 ? (
          <div className="bg-white rounded-xl border border-claude-border p-12 text-center">
            <Briefcase className="w-12 h-12 text-claude-text-muted mx-auto mb-4" />
            <p className="text-lg font-medium text-claude-text-primary mb-2">
              No applications tracked yet
            </p>
            <p className="text-sm text-claude-text-secondary">
              Start a chat with a job listing to track it here
            </p>
          </div>
        ) : (
          filteredJobs.map(job => {
            const StatusIcon = STATUS_ICONS[job.status];
            return (
              <div
                key={job.id}
                className="bg-white rounded-xl border border-claude-border p-6 hover:shadow-soft transition-all"
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center space-x-3 mb-2">
                      <h3 className="text-lg font-medium text-claude-text-primary">
                        {job.title}
                      </h3>
                      <span className={`px-2 py-1 rounded-full text-xs font-medium ${STATUS_COLORS[job.status]}`}>
                        {job.status}
                      </span>
                    </div>
                    
                    <div className="flex items-center space-x-4 text-sm text-claude-text-secondary mb-3">
                      <span className="flex items-center">
                        <Briefcase className="w-3 h-3 mr-1" />
                        {job.company}
                      </span>
                      {job.location && (
                        <span className="flex items-center">
                          <MapPin className="w-3 h-3 mr-1" />
                          {job.location}
                        </span>
                      )}
                      {job.salary && (
                        <span className="flex items-center">
                          <DollarSign className="w-3 h-3 mr-1" />
                          {job.salary}
                        </span>
                      )}
                    </div>
                    
                    <div className="flex items-center space-x-4 text-xs text-claude-text-muted">
                      {job.appliedDate && (
                        <span className="flex items-center">
                          <Calendar className="w-3 h-3 mr-1" />
                          Applied: {new Date(job.appliedDate).toLocaleDateString()}
                        </span>
                      )}
                      {job.deadline && (
                        <span className="flex items-center">
                          <Clock className="w-3 h-3 mr-1" />
                          Deadline: {new Date(job.deadline).toLocaleDateString()}
                        </span>
                      )}
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-2">
                    {/* Status Update Dropdown */}
                    <select
                      value={job.status}
                      onChange={(e) => updateJobStatus(job.id, e.target.value as any)}
                      className="px-3 py-1 bg-claude-background border border-claude-border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-claude-accent-orange/20"
                    >
                      <option value="interested">Interested</option>
                      <option value="applied">Applied</option>
                      <option value="interviewing">Interviewing</option>
                      <option value="offered">Offered</option>
                      <option value="rejected">Rejected</option>
                    </select>
                    
                    {job.chatSessionId && (
                      <a
                        href={`/dashboard/chat/${job.chatSessionId}`}
                        className="p-2 hover:bg-claude-background rounded-lg transition-colors"
                        title="Open Chat"
                      >
                        <MessageSquare className="w-4 h-4 text-claude-text-secondary" />
                      </a>
                    )}
                    
                    {job.url && (
                      <a
                        href={job.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="p-2 hover:bg-claude-background rounded-lg transition-colors"
                        title="View Job Posting"
                      >
                        <ExternalLink className="w-4 h-4 text-claude-text-secondary" />
                      </a>
                    )}
                  </div>
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}