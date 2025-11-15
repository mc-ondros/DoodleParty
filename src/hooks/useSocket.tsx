/**
 * WebSocket Connection Hook
 *
 * Manages WebSocket connections with automatic cleanup.
 * Provides functionality to emit and listen to events.
 *
 * Related:
 * - src/services/socket-service.tsx (WebSocket implementation)
 * - src/components/DrawingCanvas.tsx (real-time communication)
 *
 * Exports:
 * - useSocket (hook)
 */

import { useEffect, useRef, useCallback } from 'react';

export const useSocket = (url: string) => {
  const socketRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    socketRef.current = new WebSocket(url);

    return () => {
      if (socketRef.current) {
        socketRef.current.close();
      }
    };
  }, [url]);

  const emit = useCallback((event: string, data: any) => {
    if (socketRef.current?.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify({ event, data }));
    }
  }, []);

  const on = useCallback((event: string, callback: (data: any) => void) => {
    if (socketRef.current) {
      socketRef.current.onmessage = (e) => {
        const { event: eventName, data } = JSON.parse(e.data);
        if (eventName === event) {
          callback(data);
        }
      };
    }
  }, []);

  return { emit, on };
};
