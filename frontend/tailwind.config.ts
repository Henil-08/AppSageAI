import type { Config } from 'tailwindcss'

const config: Config = {
  darkMode: 'class',
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        // Claude-like color palette
        claude: {
          background: '#FDFCFA',
          surface: '#FFFFFF',
          border: '#E5E3DF',
          text: {
            primary: '#1F1F1F',
            secondary: '#6B6967',
            muted: '#A8A5A0',
          },
          accent: {
            orange: '#DC6027',
            'orange-light': '#FFF4ED',
            blue: '#3B82F6',
            'blue-light': '#EFF6FF',
          },
        },
        // Dark mode colors
        dark: {
          background: '#1C1917',
          surface: '#27241F',
          border: '#3F3A34',
          text: {
            primary: '#FDFCFA',
            secondary: '#A8A5A0',
            muted: '#78736B',
          },
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['SF Mono', 'Monaco', 'Consolas', 'monospace'],
      },
      animation: {
        'fade-in': 'fadeIn 0.3s ease-in-out',
        'slide-up': 'slideUp 0.3s ease-out',
        'slide-down': 'slideDown 0.3s ease-out',
        'shimmer': 'shimmer 2s linear infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { transform: 'translateY(10px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        slideDown: {
          '0%': { transform: 'translateY(-10px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        shimmer: {
          '0%': { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition: '200% 0' },
        },
      },
      boxShadow: {
        'claude': '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
        'claude-lg': '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
      },
    },
  },
  plugins: [],
}