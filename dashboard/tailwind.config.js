/** @type {import('tailwindcss').Config} */
export default {
  darkMode: ["class"],
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        carbon: {
          900: "#05070B",
          800: "#0F1117",
          700: "#1B1F2B",
        },
        neon: {
          red: "#FF3358",
          orange: "#FF8038",
          teal: "#00FFC6",
        },
      },
      fontFamily: {
        sans: ["Space Grotesk", "Inter", "system-ui", "sans-serif"],
        display: ["Space Grotesk", "Inter", "system-ui", "sans-serif"],
      },
      backgroundImage: {
        "carbon-grid":
          "radial-gradient(circle at 1px 1px, rgba(255,255,255,0.05) 1px, transparent 0)",
        "hero-gradient":
          "linear-gradient(120deg, rgba(255,51,88,0.45), rgba(0,255,198,0.2))",
      },
      boxShadow: {
        neon: "0 10px 35px rgba(255,51,88,0.35)",
      },
    },
  },
  plugins: [],
};

