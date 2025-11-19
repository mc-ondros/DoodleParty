# Design System

DoodleParty emphasizes a clean, accessible interface suitable for rapid drawing and gameplay.

## UI Components

The interface is built with **React** and uses **Lucide React** for iconography.

### Key Views
1.  **Drawing Interface (Sender)**: The primary mobile-first view for players.
    *   No lobby/waiting room; players join and start drawing immediately.
    *   Features: Canvas, Color Palette, Brush Size, Ink Meter.
    *   Served at `/doodleparty`.
2.  **Main Display (Viewer)**: A read-only view optimized for large screens or projectors.
    *   Displays all incoming strokes in real-time.
    *   Served at `/` (index).
3.  **Admin Panel**: A protected dashboard for managing the session.
    *   Controls: Timer, Clear Canvas, Kick Users, Toggle Moderation.

## Visual Style

*   **Colors**: High contrast for drawing visibility.
*   **Layout**: Responsive design to support both Desktop and Mobile/Tablet drawing.
*   **Feedback**: Immediate visual feedback for strokes and game state changes (e.g., Timer countdown).
