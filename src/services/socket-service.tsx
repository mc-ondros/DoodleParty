/**
 * WebSocket Service
 *
 * Manages WebSocket connections for real-time communication.
 * Handles connection, disconnection, and event emission.
 *
 * Related:
 * - src/hooks/useSocket.tsx (React hook)
 * - src/App.tsx (application entry point)
 *
 * Exports:
 * - SocketService (class), socketService (instance)
 */

class SocketService {
  private socket: WebSocket | null = null;

  connect(url: string): WebSocket | null {
    this.socket = new WebSocket(url);
    return this.socket;
  }

  disconnect(): void {
    if (this.socket) {
      this.socket.close();
      this.socket = null;
    }
  }

  emit(event: string, data: unknown): void {
    if (this.socket?.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify({ event, data }));
    }
  }

  on(event: string, callback: (data: unknown) => void): void {
    if (this.socket) {
      this.socket.onmessage = (e: MessageEvent) => {
        const { event: eventName, data } = JSON.parse(e.data);
        if (eventName === event) {
          callback(data);
        }
      };
    }
  }

  off(_event: string): void {
    // TODO: Implement event listener removal
  }
}

export const socketService = new SocketService();
