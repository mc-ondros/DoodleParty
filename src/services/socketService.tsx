class SocketService {
  private socket: WebSocket | null = null;

  connect(url: string) {
    this.socket = new WebSocket(url);
    return this.socket;
  }

  disconnect() {
    if (this.socket) {
      this.socket.close();
      this.socket = null;
    }
  }

  emit(event: string, data: any) {
    if (this.socket?.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify({ event, data }));
    }
  }

  on(event: string, callback: (data: any) => void) {
    if (this.socket) {
      this.socket.onmessage = (e) => {
        const { event: eventName, data } = JSON.parse(e.data);
        if (eventName === event) {
          callback(data);
        }
      };
    }
  }

  off(event: string) {
    // Remove event listener
  }
}

export const socketService = new SocketService();
