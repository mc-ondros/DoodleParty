import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const rootDir = fileURLToPath(new URL('.', import.meta.url))

export default defineConfig({
  root: 'public',
  publicDir: false,
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(rootDir, './src')
    }
  },
  build: {
    outDir: '../dist',
    emptyOutDir: true
  }
})
