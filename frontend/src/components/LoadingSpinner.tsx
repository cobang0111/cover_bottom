'use client';
import { useEffect, useState } from 'react';
import { createPortal } from 'react-dom';

const LoadingSpinner = () => {
  const [scrollY, setScrollY] = useState(0);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    const handleScroll = () => setScrollY(window.scrollY);
    window.addEventListener('scroll', handleScroll);
    setScrollY(window.scrollY);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  if (!mounted) return null;

  return createPortal(
    <div
      style={{ top: scrollY, left: 0, width: '100vw', height: '100vh', position: 'absolute' }}
      className="bg-white/80 backdrop-blur-sm flex flex-col gap-4 justify-center items-center z-50"
    >
      <div className="relative w-16 h-16">
        <div className="absolute inset-0 border-4 border-[#a50034]/20 rounded-full" />
        <div className="absolute inset-0">
          <div
            className="w-full h-full border-4 border-[#a50034] border-l-transparent rounded-full animate-spin"
            style={{ animationDuration: '1s' }}
          />
        </div>
      </div>
    </div>,
    document.body
  );
};

export default LoadingSpinner;