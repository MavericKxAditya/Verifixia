import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => ({
  // GitHub Pages serves from /Verifixia/ – use '/' for other hosts (Netlify)
  base: process.env.GITHUB_ACTIONS ? "/Verifixia/" : "/",
  server: {
    host: "0.0.0.0",
    port: 8086,
    strictPort: true,
    hmr: {
      overlay: false,
    },
  },
  preview: {
    host: "0.0.0.0",
    port: 8086,
    strictPort: true,
  },
  build: {
    outDir: "dist",
    sourcemap: false,
  },
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
}));
