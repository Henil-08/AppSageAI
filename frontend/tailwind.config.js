/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        claude: {
          background: '#FAFAF8',
          surface: '#FFFFFF',
          border: '#E5E5E0',
          text: {
            primary: '#2D2D2D',
            secondary: '#706F6C',
            muted: '#A8A29E',
          },
          accent: {
            orange: '#EA5A0C',
            'orange-light': '#FFF4ED',
            'orange-hover': '#DC4A00',
          },
        },
      },
      fontFamily: {
        sans: ['Inter', '-apple-system', 'system-ui', 'sans-serif'],
      },
      boxShadow: {
        'soft': '0 2px 8px rgba(0, 0, 0, 0.04)',
        'medium': '0 4px 12px rgba(0, 0, 0, 0.08)',
      },
    },
  },
  plugins: [],
}