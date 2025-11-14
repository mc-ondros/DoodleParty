/**
 * Application Entry Point
 *
 * Initializes React root and renders main App component.
 * Sets up StrictMode for development checks.
 * 
 * Related:
 * - src/App.tsx (main application component)
 * 
 * Imports:
 * - React (core library)
 * - ReactDOM (rendering)
 * - App (root component)
 */

import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './index.css';

const root = ReactDOM.createRoot(document.getElementById('root') as HTMLElement);
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
