/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: '#165DFF',
          light: '#4080FF',
          dark: '#0B5CDE',
          50: '#E8F1FF',
          100: '#D1E5FF',
          500: '#165DFF',
          600: '#0F4CC5',
          700: '#0B3B8C',
        },
        secondary: {
          DEFAULT: '#36CFC9',
          light: '#5FDEDD',
          dark: '#2BA8A3',
        },
        warning: {
          DEFAULT: '#FF7D00',
          light: '#FF9D32',
          dark: '#CC6600',
        },
        danger: {
          DEFAULT: '#F53F3F',
          light: '#F77066',
          dark: '#C42F2F',
        },
        gray: {
          50: '#F8FAFC',
          100: '#F1F5F9',
          200: '#E2E8F0',
          300: '#CBD5E1',
          400: '#94A3B8',
          500: '#64748B',
          600: '#475569',
          700: '#334155',
          800: '#1E293B',
          900: '#0F172A',
        }
      },
      fontFamily: {
        sans: ['Inter', 'PingFang SC', 'Microsoft YaHei', 'sans-serif'],
      },
      boxShadow: {
        'card': '0 4px 12px rgba(0, 0, 0, 0.05)',
        'card-hover': '0 8px 24px rgba(0, 0, 0, 0.1)',
        'primary': '0 4px 16px rgba(22, 93, 255, 0.3)',
      },
      borderRadius: {
        'card': '8px',
      },
      animation: {
        'fade-in': 'fadeIn 0.3s ease-out',
        'slide-up': 'slideUp 0.3s ease-out',
        'float': 'float 3s ease-in-out infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { opacity: '0', transform: 'translateY(10px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-10px)' },
        },
      },
    },
  },
  plugins: [],
}
