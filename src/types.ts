
export interface User {
  name: string;
  avatarUrl: string;
}

export interface GameMode {
  name: string;
  imageUrl: string;
  comingSoon?: boolean;
}

export interface DiscussionPost {
  id: number;
  author: string;
  title: string;
  postedAgo: string;
  votes: number;
  comments: number;
  userAvatars: string[];
}

export interface Artwork {
  id: number;
  title: string;
  imageUrl: string;
  artist: User;
  theme: string;
  votes: number;
}

export type ModerationStatus = 'PENDING' | 'SAFE' | 'UNSAFE' | 'ERROR';

export interface DrawData {
  path: { x: number; y: number }[];
  color: string;
  strokeWeight: number;
  isFill?: boolean;
  userId?: string;
}
