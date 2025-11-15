import React from 'react';
import {
  Compass,
  Paintbrush,
  Users,
  Trophy,
  Search,
  Home,
  BookOpen,
  Bell,
  Store,
  Play,
  Clock,
  Settings,
  ChevronRight,
  CheckCircle2,
  HelpCircle,
  Star,
  Flame,
  Sparkles,
  Target,
  Swords,
  BookText,
} from 'lucide-react';

export const DoodlePartyLogo: React.FC<{ className?: string }> = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M4 20L8 16M12.5 4.5L19.5 11.5L11.5 19.5L4.5 12.5L12.5 4.5Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
    <path d="M16.5 8.5L19.5 11.5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
  </svg>
);

export const CompassIcon: React.FC<{ className?: string }> = ({ className }) => (
  <Compass className={className} />
);

export const BrushIcon: React.FC<{ className?: string }> = ({ className }) => (
  <Paintbrush className={className} />
);

export const UsersIcon: React.FC<{ className?: string }> = ({ className }) => (
  <Users className={className} />
);

export const TrophyIcon: React.FC<{ className?: string }> = ({ className }) => (
  <Trophy className={className} />
);

export const SearchIcon: React.FC<{ className?: string }> = ({ className }) => (
  <Search className={className} />
);

export const HomeIcon: React.FC<{ className?: string }> = ({ className }) => (
  <Home className={className} />
);

export const LibraryIcon: React.FC<{ className?: string }> = ({ className }) => (
  <BookOpen className={className} />
);

export const NotificationsIcon: React.FC<{ className?: string }> = ({ className }) => (
  <Bell className={className} />
);

export const StoreIcon: React.FC<{ className?: string }> = ({ className }) => (
  <Store className={className} />
);

export const PlayIcon: React.FC<{ className?: string }> = ({ className }) => (
  <Play className={className} />
);

export const ClockIcon: React.FC<{ className?: string }> = ({ className }) => (
  <Clock className={className} />
);

export const UsersIconNav: React.FC<{ className?: string }> = ({ className }) => (
  <Users className={className} />
);

export const SettingsIcon: React.FC<{ className?: string }> = ({ className }) => (
  <Settings className={className} />
);

export const ChevronRightIcon: React.FC<{ className?: string }> = ({ className }) => (
  <ChevronRight className={className} />
);

export const CheckCircleIcon: React.FC<{ className?: string }> = ({ className }) => (
  <CheckCircle2 className={className} />
);

export const HelpCircleIcon: React.FC<{ className?: string }> = ({ className }) => (
  <HelpCircle className={className} />
);

export const StarIcon: React.FC<{ className?: string }> = ({ className }) => (
  <Star className={className} />
);

export const FlameIcon: React.FC<{ className?: string }> = ({ className }) => (
  <Flame className={className} />
);

export const SparklesIcon: React.FC<{ className?: string }> = ({ className }) => (
  <Sparkles className={className} />
);

export const TargetIcon: React.FC<{ className?: string }> = ({ className }) => (
  <Target className={className} />
);

export const SwordsIcon: React.FC<{ className?: string }> = ({ className }) => (
  <Swords className={className} />
);

export const BookTextIcon: React.FC<{ className?: string }> = ({ className }) => (
  <BookText className={className} />
);
