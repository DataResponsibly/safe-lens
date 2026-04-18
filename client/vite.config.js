import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  // In the default dev stack (dev/docker-compose.yml) requests to /generate
  // and /regenerate are handled by the nginx reverse proxy in front of Vite,
  // so no Vite-side proxy is needed. For running `npm run dev` directly
  // against a local backend (no nginx), set VITE_PROXY_TARGET.
  const env = { ...process.env, ...loadEnv(mode, process.cwd(), "") };
  const proxyTarget = env.VITE_PROXY_TARGET;
  const proxy = proxyTarget
    ? {
        "/generate": { target: proxyTarget, changeOrigin: true },
        "/regenerate": { target: proxyTarget, changeOrigin: true },
      }
    : undefined;

  return {
    plugins: [react()],
    server: {
      host: true,
      port: 5173,
      strictPort: true,
      watch: {
        // Polling is more reliable for bind-mounted volumes on Linux hosts.
        usePolling: true,
        interval: 300,
      },
      // HMR websocket needs to know the browser-facing host/port, which in
      // the dev stack is the nginx container published on localhost:8080.
      hmr: {
        clientPort: env.VITE_HMR_CLIENT_PORT
          ? Number(env.VITE_HMR_CLIENT_PORT)
          : undefined,
      },
      proxy,
    },
    build: {
      outDir: "dist",
      sourcemap: false,
    },
  };
});
