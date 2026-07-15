/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ["'Plus Jakarta Sans'", "sans-serif"],
        mono: ["'JetBrains Mono'", "monospace"],
      },
      colors: {
        darkBg: "#08080C",
        panelBg: "#111116",
        cardBg: "rgba(20, 20, 28, 0.6)",
        accentPurple: "#8B5CF6",
        accentIndigo: "#6366F1",
        accentCyan: "#06B6D4",
        borderPurple: "rgba(139, 92, 246, 0.2)",
        borderCyan: "rgba(6, 182, 212, 0.2)",
      },
      backdropBlur: {
        xs: '2px',
      },
      boxShadow: {
        glowPurple: "0 0 20px rgba(139, 92, 246, 0.15)",
        glowCyan: "0 0 20px rgba(6, 182, 212, 0.15)",
      }
    },
  },
  plugins: [],
}
