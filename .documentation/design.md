# DoodleParty Visual Design System

**Purpose:** Visual design documentation for DoodleParty's event-focused collaborative drawing interface with modern glassmorphism aesthetic.

**Status: Production Ready** - Comprehensive design guide for consistent user experience across all deployment targets.

## Table of Contents

- [Design Philosophy](#design-philosophy)
  - [Core Principle: Event-Focused Design](#core-principle-event-focused-design)
- [Visual Foundation](#visual-foundation)
  - [Glass Panel Component](#glass-panel-component)
- [Component Design](#component-design)
  - [Buttons](#buttons)
  - [Cards](#cards)
- [Color System](#color-system)
- [Typography](#typography)
- [Iconography](#iconography)
- [Animation](#animation)
- [Layout](#layout)
  - [Canvas Layout](#canvas-layout)
  - [Leaderboard Layout](#leaderboard-layout)
  - [Responsive Scaling](#responsive-scaling)
- [Drawing Canvas](#drawing-canvas)
- [Implementation Guidelines](#implementation-guidelines)
  - [CSS Architecture](#css-architecture)
  - [Performance](#performance)
- [Accessibility](#accessibility)
- [Theme Architecture](#theme-architecture)
- [Cross-Platform Scaling](#cross-platform-scaling)
- [Common Patterns](#common-patterns)
  - [Game Mode Card Pattern](#game-mode-card-pattern)
  - [Button Pattern](#button-pattern)
  - [Leaderboard Row Pattern](#leaderboard-row-pattern)

## Design Philosophy

### Core Principle: Event-Focused Design

The interface uses glassmorphism - translucent panels with backdrop blur that float above a vibrant event background. Elements feel tactile and three-dimensional through careful use of transparency, blur, and elevation. Designed for large group participation with projector-friendly displays.

**Key Characteristics:**
- Vibrant event background (#1a1a2e) for energy
- Glass panels with 10px backdrop blur
- Semi-transparent white surfaces (10-15% opacity)
- Soft shadows for elevation
- Bold accent colors for game modes
- Mobile-first responsive design

## Visual Foundation

### Glass Panel Component

```css
.glass-panel {
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.2);
  box-shadow:
    0 8px 32px 0 rgba(0, 0, 0, 0.2),
    0 2px 8px 0 rgba(0, 0, 0, 0.1),
    inset 0 1px 0 0 rgba(255, 255, 255, 0.3);
  border-radius: 16px;
}
```

**Properties:**
- Transparency: 10% white for subtle glass effect
- Blur: 10px for atmospheric depth
- Borders: 20% white opacity for edge definition
- Shadow: Multi-layer for tactile elevation
- Border radius: 16px for modern feel

## Component Design

### Buttons

**Primary Button (Game Action):**
```css
.btn-primary {
  background: rgba(255, 255, 255, 0.15);
  backdrop-filter: blur(10px);
  color: #ffffff;
  border: 1px solid rgba(255, 255, 255, 0.3);
  padding: 12px 24px;
  border-radius: 12px;
  font-weight: 600;
  font-size: 16px;
  transition: all 0.3s ease;
  cursor: pointer;
}

.btn-primary:hover {
  transform: translateY(-2px);
  background: rgba(255, 255, 255, 0.25);
  box-shadow: 0 12px 24px rgba(0, 0, 0, 0.3);
}

.btn-primary:active {
  transform: translateY(0);
  background: rgba(255, 255, 255, 0.2);
}
```

**Secondary Button (Navigation):**
```css
.btn-secondary {
  background: rgba(255, 255, 255, 0.08);
  backdrop-filter: blur(10px);
  color: rgba(255, 255, 255, 0.8);
  border: 1px solid rgba(255, 255, 255, 0.15);
  padding: 10px 20px;
  border-radius: 10px;
  font-weight: 500;
  font-size: 14px;
  transition: all 0.2s ease;
}

.btn-secondary:hover {
  background: rgba(255, 255, 255, 0.15);
  color: #ffffff;
}
```

**Accent Button (Game Mode):**
```css
.btn-accent {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: #ffffff;
  border: none;
  padding: 12px 24px;
  border-radius: 12px;
  font-weight: 600;
  transition: all 0.3s ease;
}

.btn-accent:hover {
  transform: translateY(-2px);
  box-shadow: 0 12px 24px rgba(102, 126, 234, 0.4);
}
```

### Cards

Cards use glass panel aesthetic with glass-panel class.

**Game Mode Card:**
```jsx
<div className="glass-panel p-6 cursor-pointer hover:scale-105 transition-transform">
  <div className="flex items-start justify-between mb-4">
    <h3 className="text-xl font-bold text-white">Speed Sketch</h3>
    <Zap className="w-6 h-6 text-yellow-400" />
  </div>
  <p className="text-gray-200 text-sm mb-4">Draw for 30 seconds, vote on best</p>
  <div className="flex justify-between text-xs text-gray-300">
    <span>👥 2-50 players</span>
    <span>⏱️ 5 min</span>
  </div>
</div>
```

**Leaderboard Card:**
```jsx
<div className="glass-panel p-4 flex items-center justify-between hover:bg-white/15 transition">
  <div className="flex items-center gap-4">
    <span className="text-2xl font-bold text-yellow-400 w-8">{rank}</span>
    <div>
      <p className="font-semibold text-white">{playerName}</p>
      <p className="text-xs text-gray-300">{roundWins} wins</p>
    </div>
  </div>
  <p className="text-2xl font-bold text-white">{score}</p>
</div>
```

## Color System

**Primary Colors:**
- **Dark Background:** `#1a1a2e` - Main background
- **Glass Surface:** `rgba(255, 255, 255, 0.1)` - Card backgrounds
- **Glass Border:** `rgba(255, 255, 255, 0.2)` - Card borders

**Accent Colors:**
- **Purple Gradient:** `#667eea` → `#764ba2` - Primary actions
- **Cyan:** `#00d4ff` - Secondary highlights
- **Yellow:** `#ffd700` - Badges, achievements
- **Green:** `#4ade80` - Success, approved
- **Red:** `#f87171` - Alerts, rejected

**Text Colors:**
- **Primary:** `#ffffff` - Main text
- **Secondary:** `rgba(255, 255, 255, 0.7)` - Secondary text
- **Tertiary:** `rgba(255, 255, 255, 0.5)` - Disabled, hints

**Implementation:**
```css
:root {
  /* Glass surfaces */
  --glass-bg: rgba(255, 255, 255, 0.1);
  --glass-border: rgba(255, 255, 255, 0.2);
  --glass-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);

  /* Colors */
  --bg-dark: #1a1a2e;
  --accent-purple: #667eea;
  --accent-cyan: #00d4ff;
  --accent-yellow: #ffd700;
  --accent-green: #4ade80;
  --accent-red: #f87171;

  /* Typography */
  --text-primary: #ffffff;
  --text-secondary: rgba(255, 255, 255, 0.7);
  --text-tertiary: rgba(255, 255, 255, 0.5);
}
```

## Typography

**Font Stack:**
- System fonts: SF Pro, Segoe UI, Roboto
- Fallback: sans-serif
- Monospace: Fira Code (for technical content)

**Type Scale:**
- **Display:** 48px, 700 weight (game titles)
- **Heading 1:** 32px, 700 weight (section headers)
- **Heading 2:** 24px, 600 weight (card titles)
- **Body:** 16px, 400 weight (main content)
- **Small:** 14px, 400 weight (secondary content)
- **Label:** 12px, 600 weight (uppercase labels)

**Typography Examples:**
```jsx
// Game title
<h1 className="text-5xl font-bold text-white">Speed Sketch Challenge</h1>

// Card title
<h2 className="text-2xl font-semibold text-white">Leaderboard</h2>

// Body text
<p className="text-base text-gray-200">Draw your interpretation of the prompt</p>

// Label
<span className="text-xs font-semibold text-gray-300 uppercase tracking-wide">
  Round 1 of 5
</span>
```

## Iconography

**Icon Library: Lucide React**
- Exclusive icon library for all icons
- 24px base size (adjustable: w-5, w-6, w-8)
- 1.5px stroke weight
- Semi-transparent (opacity 0.7-0.9)
- Touch targets: 44px minimum

**Common Icons:**
- Drawing: `pen-tool`, `pencil`
- Eraser: `eraser`
- Undo/Redo: `undo-2`, `redo-2`
- Delete: `trash-2`
- Settings: `settings`
- Users: `users`
- Trophy: `trophy`
- Energy: `zap`
- Timer: `clock`
- Vote: `thumbs-up`

**Usage:**
```jsx
import { Users, PenTool, Trophy, Zap } from 'lucide-react';

// In components
<Users className="w-5 h-5 text-gray-300" />
<Trophy className="w-6 h-6 text-yellow-400" />
<Zap className="w-8 h-8 text-cyan-400" />
```

## Animation

**Timing:**
- Micro-interactions: 0.2s
- State changes: 0.3s
- Easing: ease for natural motion

**Interactive States:**

**Hover:**
- Lift element 2px vertically
- Increase opacity slightly
- Enhance shadow depth
- Duration: 0.2s

**Active:**
- Return to base position
- Brief opacity reduction
- Duration: 0.1s

**Focus:**
- Visible focus ring (2px)
- High contrast outline
- Maintains glass aesthetic

**Example:**
```css
.glass-panel {
  transition: all 0.3s ease;
}

.glass-panel:hover {
  transform: translateY(-2px);
  background: rgba(255, 255, 255, 0.15);
  box-shadow: 0 12px 24px rgba(0, 0, 0, 0.3);
}

.glass-panel:focus-within {
  outline: 2px solid #00d4ff;
  outline-offset: 2px;
}
```

## Layout

### Canvas Layout

**Grid System:**
- Mobile: Single column (full width canvas)
- Tablet: Two columns (canvas + sidebar)
- Desktop: Three columns (sidebar + canvas + leaderboard)
- Base spacing unit: 16px

**Canvas Container:**
```jsx
<div className="
  grid
  grid-cols-1        /* Mobile */
  md:grid-cols-2     /* Tablet */
  lg:grid-cols-3     /* Desktop */
  gap-4              /* 16px gaps */
  h-screen
">
  <Sidebar />
  <Canvas />
  <Leaderboard />
</div>
```

### Leaderboard Layout

**Card Spacing:**
- Between cards: 12px
- Within cards: 16px padding
- Max height: 600px with scroll
- Fixed width: 320px on desktop

**Responsive Behavior:**
- Mobile: Full width, horizontal scroll
- Tablet: Side panel, 280px width
- Desktop: Fixed panel, 320px width

### Responsive Scaling

**Breakpoints:**
- Mobile: <640px (single column, full-width components)
- Tablet: 640px-1024px (two columns, adjusted spacing)
- Desktop: >1024px (three columns, full layout)

**Scaling Principles:**
- Fluid typography: Scale font sizes with viewport width
- Flexible spacing: Use relative units (rem, %) not fixed pixels
- Touch-friendly: Minimum 44px touch targets on all devices
- Aspect ratio preservation: Maintain 16:9 canvas ratio across devices
- Safe area consideration: Account for notches and safe zones on mobile

## Drawing Canvas

**Canvas Container:**
- Background: `#1a1a2e`
- Border: 1px `rgba(255, 255, 255, 0.1)`
- Floating effect with shadow
- Aspect ratio: 16:9 (adjustable)

**Canvas Surface:**
- White background for drawing
- Smooth rendering
- Touch-optimized strokes
- Pressure sensitivity support

**Rationale:**
- User focus on drawing content
- Maximum clarity for collaboration
- Projector display optimization
- Reduces visual distractions

## Implementation Guidelines

### CSS Architecture

**Glass Panel Class:**
```css
.glass-panel {
  /* Applied to all cards, buttons, and containers */
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-radius: 16px;
}
```

**Component Structure:**
```jsx
<div className="glass-panel p-6">
  <h3 className="text-2xl font-semibold text-white mb-4">Title</h3>
  <p className="text-gray-200">Content goes here</p>
</div>
```

**Utility Classes:**
```css
/* Spacing */
.p-4 { padding: 16px; }
.p-6 { padding: 24px; }
.gap-4 { gap: 16px; }

/* Typography */
.text-white { color: #ffffff; }
.text-gray-200 { color: rgba(255, 255, 255, 0.7); }
.font-semibold { font-weight: 600; }

/* Effects */
.hover:scale-105 { transform: scale(1.05); }
.transition-all { transition: all 0.3s ease; }
```

### Performance

**Blur Optimization:**
- Limit backdrop-filter to 10px maximum
- Use -webkit-backdrop-filter for Safari compatibility
- GPU acceleration automatic on modern browsers
- Disable on low-end devices via media queries

**Low-End Device Optimization:**
- Detect low-end devices via media queries and feature detection
- Disable backdrop-filter on constrained devices (CPU-intensive)
- Use solid fallback colors instead of blur effects
- Reduce shadow complexity: 3 layers max (desktop), 1 layer (mobile)
- Avoid expensive properties: `left`, `top`, `width`, `height`, `filter`
- Respect `prefers-reduced-motion` user preference

**Animation Performance:**
- Use `transform` and `opacity` only (GPU-accelerated)
- Batch animations with `requestAnimationFrame`
- Disable animations on `prefers-reduced-motion`
- Avoid layout-triggering properties (position, size changes)

**Responsive Scaling:**
- Mobile-first approach: start with minimal effects, enhance for capable devices
- Use `@media` queries for device capability detection
- Test on actual low-end devices (mobile phones, older browsers)
- Monitor frame rate and memory usage during development

## Accessibility

**Contrast Ratios:**
- Text on glass: 4.5:1 minimum
- Text on background: 7:1 minimum
- Focus indicators: 3:1 minimum

**Focus Management:**
- Always visible focus rings (2px outline)
- Consistent focus order
- Keyboard navigation support
- Tab order follows visual flow

**Touch Targets:**
- Minimum 44px for all interactive elements
- Adequate spacing between targets (8px minimum)
- Large buttons for event environment

**Color Blindness:**
- Don't rely on color alone
- Use icons + color combinations
- Test with color blindness simulator

## Theme Architecture

**CSS Custom Properties:**
```css
:root {
  /* Glass surfaces */
  --glass-bg: rgba(255, 255, 255, 0.1);
  --glass-border: rgba(255, 255, 255, 0.2);
  --glass-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);

  /* Colors */
  --bg-dark: #1a1a2e;
  --accent-purple: #667eea;
  --accent-cyan: #00d4ff;
  --accent-yellow: #ffd700;
  --accent-green: #4ade80;
  --accent-red: #f87171;

  /* Typography */
  --text-primary: #ffffff;
  --text-secondary: rgba(255, 255, 255, 0.7);
  --text-tertiary: rgba(255, 255, 255, 0.5);
}
```

**Theme Application:**
```css
body {
  background: var(--bg-dark);
  color: var(--text-primary);
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}
```

## Cross-Platform Scaling

**Browser & Web App:**
- Use CSS media queries for responsive breakpoints
- Implement fluid typography with viewport-relative units
- Test on Chrome, Firefox, Safari, Edge (desktop and mobile)
- Verify touch interactions work on tablets and phones
- Monitor performance on low-end devices (older phones, low RAM)

**Native Mobile Apps:**
- Use platform-specific safe area insets (notches, home indicators)
- Adapt touch target sizes for finger input (minimum 44pt iOS, 48dp Android)
- Scale assets for device pixel ratios (1x, 2x, 3x)
- Optimize for landscape and portrait orientations
- Test on actual devices, not just emulators

**Shared Component Library:**
- Design components to scale from 320px (mobile) to 2560px (desktop)
- Use flexible layouts (flexbox, grid) instead of fixed dimensions
- Provide responsive variants for different screen sizes
- Document scaling behavior and breakpoints clearly
- Test component scaling across all target platforms

**Performance Across Platforms:**
- Web: Monitor CPU, memory, and frame rate on low-end devices
- Mobile: Optimize for battery life and data usage
- Projector displays: Ensure text is readable from distance
- Accessibility: Test with screen readers and keyboard navigation

## Common Patterns

### Game Mode Card Pattern

```jsx
<div className="glass-panel p-6 cursor-pointer hover:scale-105 transition-transform">
  <div className="flex items-start justify-between mb-4">
    <div>
      <h3 className="text-xl font-bold text-white">{mode.name}</h3>
      <p className="text-sm text-gray-300 mt-1">{mode.description}</p>
    </div>
    <div className="text-2xl">{mode.icon}</div>
  </div>
  <div className="flex justify-between text-xs text-gray-400 pt-4 border-t border-white/10">
    <span>👥 {mode.minPlayers}-{mode.maxPlayers}</span>
    <span>⏱️ {mode.duration}</span>
  </div>
</div>
```

### Button Pattern

```jsx
<button className="
  glass-panel
  px-6 py-3
  font-semibold
  text-white
  hover:translate-y-[-2px]
  hover:bg-white/15
  transition-all
  duration-300
">
  <PenTool className="w-4 h-4 inline mr-2" />
  Start Drawing
</button>
```

### Leaderboard Row Pattern

```jsx
<div className="glass-panel p-4 flex items-center justify-between hover:bg-white/15 transition">
  <div className="flex items-center gap-4 flex-1">
    <span className="text-2xl font-bold text-yellow-400 w-8 text-center">
      {rank}
    </span>
    <div className="flex-1">
      <p className="font-semibold text-white">{player.name}</p>
      <p className="text-xs text-gray-400">{player.wins} wins</p>
    </div>
  </div>
  <div className="text-right">
    <p className="text-2xl font-bold text-white">{player.score}</p>
    <p className="text-xs text-gray-400">+{player.lastRound}</p>
  </div>
</div>
```

## Conclusion

The DoodleParty visual design system creates a cohesive, event-focused interface through:

1. **Glassmorphism** - Translucent panels with backdrop blur
2. **Dark Backdrop** - High-energy dark background keeps focus on content
3. **Vibrant Accents** - Purple, cyan, and yellow for game modes and achievements
4. **Subtle Elevation** - Multi-layer shadows for depth
5. **Consistent Components** - Reusable glass panel pattern
6. **Lucide Icons** - Unified iconography throughout
7. **Responsive Design** - Mobile-first approach for all devices
8. **Accessibility** - High contrast and keyboard navigation

This design balances modern aesthetics with usability, creating an interface that's both beautiful and functional for large group events.

**Related Documentation:**
- [Architecture](architecture.md) - System design and components
- [API Reference](api.md) - WebSocket and REST API
- [Installation](installation.md) - Setup instructions
- [ML Pipeline](ml-pipeline.md) - Content moderation details
- [Development Roadmap](roadmap.md) - Planned features
- [Project Structure](structure.md) - File organization
- [README](../README.md) - Project overview

*Visual Design System for DoodleParty v1.0*
