/** @type {import('tailwindcss').Config} */
export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        bg: "rgb(var(--color-bg) / <alpha-value>)",
        panel: "rgb(var(--color-panel) / <alpha-value>)",
        input: "rgb(var(--color-input) / <alpha-value>)",
        border: "rgb(var(--color-border) / <alpha-value>)",
        fg: "rgb(var(--color-fg) / <alpha-value>)",
        accent: "#27ae60",
        danger: "#c0392b",
      },
      fontFamily: {
        mono: [
          "Andale Mono",
          "AndaleMono",
          "ui-monospace",
          "SFMono-Regular",
          "Menlo",
          "monospace",
        ],
      },
    },
  },
  plugins: [],
};
