/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./templates/**/*.html",
    "./static/js/**/*.js",
  ],
  safelist: [
    "sm:grid-cols-2",  // dynamic :class in _modal_model_settings.html
    "bg-emerald-500", "text-white", "border-emerald-500",
    "bg-emerald-50", "text-emerald-700", "border-emerald-200", "hover:bg-emerald-100",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
      colors: {
        surface: {
          DEFAULT: 'var(--bg-primary)',
          alt: 'var(--bg-secondary)',
          muted: 'var(--bg-tertiary)',
        },
        fg: {
          DEFAULT: 'var(--text-primary)',
          secondary: 'var(--text-secondary)',
          tertiary: 'var(--text-tertiary)',
          muted: 'var(--text-muted)',
        },
        line: {
          DEFAULT: 'var(--border-faint)',
          strong: 'var(--border-normal)',
        },
        accent: {
          DEFAULT: 'var(--btn-primary)',
          hover: 'var(--btn-primary-hover)',
          fg: 'var(--btn-primary-text)',
        },
        danger: {
          DEFAULT: 'var(--text-danger)',
          bg: 'var(--bg-danger-hover)',
        },
        code: 'var(--code-bg)',
      },
      animation: {
        'fade-in-up': 'fadeInUp 0.5s ease-out forwards',
      },
      keyframes: {
        fadeInUp: {
          '0%': { opacity: '0', transform: 'translateY(10px)' },
          '100%': { opacity: '1', transform: 'none' },
        }
      }
    },
  },
  plugins: [],
}
