import { AlertTriangle, Trash2, X } from 'lucide-react';

interface ConfirmationModalProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: () => void;
  title: string;
  message: string;
  confirmText?: string;
  cancelText?: string;
  type?: 'danger' | 'warning' | 'info';
}

export default function ConfirmationModal({
  isOpen,
  onClose,
  onConfirm,
  title,
  message,
  confirmText = 'Confirm',
  cancelText = 'Cancel',
  type = 'danger'
}: ConfirmationModalProps) {
  if (!isOpen) return null;

  const iconColors = {
    danger: 'text-claude-accent-orange bg-claude-accent-orange-light',
    warning: 'text-yellow-500 bg-yellow-50',
    info: 'text-claude-accent-orange bg-claude-accent-orange-light'
  };

  const buttonColors = {
    danger: 'bg-claude-accent-orange hover:bg-claude-accent-orange-hover',
    warning: 'bg-yellow-500 hover:bg-yellow-600',
    info: 'bg-claude-accent-orange hover:bg-claude-accent-orange-hover'
  };

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto">
      {/* Backdrop */}
      <div 
        className="fixed inset-0 bg-black/50 transition-opacity"
        onClick={onClose}
      />
      
      {/* Modal */}
      <div className="flex min-h-full items-center justify-center p-4">
        <div className="relative bg-white rounded-2xl shadow-lg max-w-md w-full transform transition-all">
          {/* Close button */}
          <button
            onClick={onClose}
            className="absolute top-4 right-4 p-1 rounded-lg hover:bg-claude-background transition-colors"
          >
            <X className="w-5 h-5 text-claude-text-muted" />
          </button>
          
          {/* Content */}
          <div className="p-6">
            {/* Icon */}
            <div className={`w-12 h-12 rounded-full ${iconColors[type]} flex items-center justify-center mb-4`}>
              {type === 'danger' ? (
                <Trash2 className="w-6 h-6" />
              ) : (
                <AlertTriangle className="w-6 h-6" />
              )}
            </div>
            
            {/* Title */}
            <h3 className="text-lg font-semibold text-claude-text-primary mb-2">
              {title}
            </h3>
            
            {/* Message */}
            <p className="text-claude-text-secondary text-sm">
              {message}
            </p>
          </div>
          
          {/* Actions */}
          <div className="px-6 pb-6 flex items-center justify-end space-x-3">
            <button
              onClick={onClose}
              className="px-4 py-2 bg-white border border-claude-border rounded-lg hover:bg-claude-background transition-colors text-claude-text-primary font-medium"
            >
              {cancelText}
            </button>
            <button
              onClick={() => {
                onConfirm();
                onClose();
              }}
              className={`px-4 py-2 text-white font-medium rounded-lg transition-colors ${buttonColors[type]}`}
            >
              {confirmText}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}