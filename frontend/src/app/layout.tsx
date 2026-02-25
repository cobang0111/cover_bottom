import './globals.css';
import type { ReactNode } from 'react';

export const metadata = {
  title: 'Cover Bottom 분석',
  description: 'Next.js + FastAPI'
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="ko">
      <body>{children}</body>
    </html>
  );
}