class ModerationService {
  private apiUrl: string;

  constructor(apiUrl: string = process.env.VITE_ML_URL || 'http://localhost:5001') {
    this.apiUrl = apiUrl;
  }

  async moderateDrawing(imageData: string): Promise<{
    status: 'safe' | 'warning' | 'blocked';
    confidence: number;
    reason?: string;
  }> {
    try {
      const response = await fetch(`${this.apiUrl}/moderate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: imageData }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      return result;
    } catch (error) {
      console.error('Moderation service error:', error);
      throw error;
    }
  }

  async classifyDrawing(imageData: string): Promise<{
    label: string;
    confidence: number;
  }> {
    try {
      const response = await fetch(`${this.apiUrl}/classify`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: imageData }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      return result;
    } catch (error) {
      console.error('Classification service error:', error);
      throw error;
    }
  }
}

export const moderationService = new ModerationService();
