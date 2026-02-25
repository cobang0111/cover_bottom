'use client';

import { useEffect } from 'react';

interface ToastProps {
  message: string;
  type?: 'error' | 'warning' | 'success';
  onClose: () => void;
}

export default function Toast({ message, type = 'error', onClose }: ToastProps) {
  useEffect(() => {
    const timer = setTimeout(() => {
      onClose();
    }, 5000);

    return () => clearTimeout(timer);
  }, [onClose]);

  return (
    <div 
      className="fixed right-[-350px] top-[-150px]"
      style={{
        animation: 'slideIn 0.3s ease-out forwards',
      }}
    >
      <div
        className={`
          flex items-center gap-3 px-6 py-4 rounded-xl shadow-lg
          backdrop-blur-sm
          ${type === 'error' 
            ? 'bg-red-500/95 text-white' 
            : type === 'warning'
            ? 'bg-amber-500/95 text-white'
            : 'bg-emerald-500/95 text-white'}
        `}
        role="alert"
      >
        {/* Icon */}
        <div className="flex-shrink-0">
          {type === 'error' && (
            <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <circle cx="12" cy="12" r="10" strokeWidth="2"/>
              <path d="M12 8v5" strokeWidth="2" strokeLinecap="round"/>
              <circle cx="12" cy="16" r="1" fill="currentColor"/>
            </svg>
          )}
          {type === 'warning' && (
            <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <path d="M12 3L2 21h20L12 3z" strokeWidth="2"/>
              <path d="M12 9v5" strokeWidth="2" strokeLinecap="round"/>
              <circle cx="12" cy="17" r="1" fill="currentColor"/>
            </svg>
          )}
          {type === 'success' && (
            <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <circle cx="12" cy="12" r="10" strokeWidth="2"/>
              <path d="M8 12l3 3 5-5" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          )}
        </div>

        {/* Message */}
        <p className="text-sm font-medium">{message}</p>

        {/* Close Button */}
        <button
          onClick={onClose}
          className="ml-2 p-1 rounded-full transition-colors duration-200 close-button"
        >
          <span className="sr-only">Close</span>
          <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor">
            <path d="M18 6L6 18M6 6l12 12" strokeWidth="2" strokeLinecap="round"/>
          </svg>
        </button>
      </div>

      <style jsx>{`
        .close-button:hover {
          background-color: rgba(255, 255, 255, 0.1);
        }
        @keyframes slideIn {
          0% {
            transform: translate(-50%, -150%);
            opacity: 0;
          }
          100% {
            transform: translate(-50%, 0);
            opacity: 1;
          }
        }
      `}</style>
    </div>
  );
}