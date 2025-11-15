class AnalyticsService {
  private events: Array<{
    name: string;
    timestamp: number;
    data?: Record<string, any>;
  }> = [];

  trackEvent(name: string, data?: Record<string, any>) {
    const event = {
      name,
      timestamp: Date.now(),
      data,
    };
    this.events.push(event);

    // Send to server if needed
    this.sendEvent(event);
  }

  private async sendEvent(event: any) {
    try {
      await fetch(`${process.env.VITE_API_URL}/analytics`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(event),
      });
    } catch (error) {
      console.error('Failed to send analytics event:', error);
    }
  }

  getEvents() {
    return this.events;
  }

  clearEvents() {
    this.events = [];
  }
}

export const analyticsService = new AnalyticsService();
