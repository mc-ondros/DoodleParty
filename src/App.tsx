import React from 'react';

import AdminPanel from '@/components/AdminPanel';

/**
 * Main App Component
 *
 * Application entry point that renders the admin interface.
 * Following DoodleParty's single-page app architecture.
 *
 * Related:
 * - src/components/AdminPanel.tsx (admin interface)
 * - src/index.tsx (application entry)
 */

// Single-page admin interface; remove all other routes/pages.
export default function App() {
  return <AdminPanel />
}
