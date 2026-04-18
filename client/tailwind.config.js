/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        // Match the existing dark terminal aesthetic
        bg: "#212121",
        panel: "#202c33",
        input: "#2a3942",
        border: "#7f8c8d",
        fg: "#f5f6fa",
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
