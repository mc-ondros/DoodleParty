# DoodleParty Landing Page

## Overview
A modern, responsive landing page for DoodleParty showcasing the business concept and pricing options.

## Features Included

### Sections
1. **Hero Section** - Large banner with background image, gradient mask, and breadcrumb navigation
   - Live status indicator with pulsing green dot
   - Performance metrics (concurrent users, latency, inference time)
   - Background image fades to transparent at bottom
2. **Features Section** - Six key features with green-accented icon boxes:
   - 100+ Concurrent Users
   - Real-Time Collaboration  
   - AI-Powered Moderation
   - Multiple Game Modes
   - Edge ML Inference
   - Offline Capable
3. **Pricing Section** - Three-tier pricing model:
   - **Hobby**: Free (10 users, basic features)
   - **Raspberry Pi 4**: $499 one-time (100 users, offline) - FEATURED
   - **Cloud Pro**: $199/month (500 users, managed)
   - Enterprise tiers listed below: Pro ($399/mo), Enterprise ($799/mo)
4. **Use Cases Section** - Four target markets in clean card layout
5. **Call-to-Action Section** - Black background with green CTA button
6. **Footer** - Minimal footer with branding on black

### Technical Stack
- React 18.3
- TypeScript
- Tailwind CSS v4 with PostCSS
- Lucide React (icons)
- Vite (build tool)

## Running the Landing Page

### Development Mode
```bash
npm run dev
```
Then open http://localhost:5173 in your browser

### Production Build
```bash
npm run build
npm run preview
```

### Type Checking
```bash
npm run type-check
```

## File Structure
```
src/
├── index.tsx      # React entry point
├── App.tsx        # Main landing page component
└── index.css      # Global styles with Tailwind directives
```

## Pricing Details

### Hobby (Free)
- **Price**: Free
- **Capacity**: Up to 10 concurrent users
- **Features**: Classic Canvas mode, Basic AI moderation, Community support
- **Ideal For**: Individuals and small groups getting started

### Raspberry Pi 4 (Featured)
- **Price**: $499 one-time
- **Capacity**: 100 concurrent users
- **Features**: All game modes, Complete offline operation, No internet required, Full data privacy, One-time purchase
- **Ideal For**: Events, festivals, conferences with local network

### Cloud Pro
- **Price**: $199/month
- **Capacity**: Up to 500 concurrent users
- **Features**: Auto-scaling infrastructure, 99.9% uptime SLA, Full AI features & LLM, 24/7 support, Analytics dashboard
- **Ideal For**: Growing communities and managed SaaS

### Enterprise Tiers
- **Pro**: $399/month - Up to 1,000 users
- **Enterprise**: $799/month - Up to 2,000+ users
- **Custom**: Contact for larger deployments

## Design System
- **Primary Colors**: Green (#22C55E / green-500) and Black
- **Background**: Pure black (#000000) throughout
- **Accent Color**: Green-500 for CTAs and highlights
- **Typography**: Inter font family
- **Style**: Clean, modern aesthetic inspired by ExplorePage
- **Hero Image**: Colorful wall art from Unsplash (Seoul, South Korea)
- **Features**: 
  - Hero with background image and gradient mask effect
  - Live status indicators with pulse animations
  - Icon boxes with green-500/20 backgrounds
  - Zinc-700/800 borders with green-500 on hover
  - All sections use black background for maximum contrast
- **Responsive**: Mobile-first design (320px - 2560px)

## Next Steps
1. Connect call-to-action buttons to actual signup/demo flows
2. Add analytics tracking
3. Implement contact form
4. Add customer testimonials section
5. Create product screenshots/demo video
6. Set up email capture for newsletter

## Notes
- All pricing values are reference examples and can be adjusted
- Icons from Lucide React provide a consistent, modern look
- The page is fully responsive and optimized for all device sizes
- Tailwind CSS enables rapid design iterations
