class InkMeter {
    constructor(inkLevelElement) {
        this.inkLevelElement = inkLevelElement;
        this.currentInkLevel = 100; // Start with full ink
        this.updateInkLevelDisplay();
    }

    useInk(amount) {
        this.currentInkLevel = Math.max(0, this.currentInkLevel - amount);
        this.updateInkLevelDisplay();
    }

    refillInk() {
        this.currentInkLevel = 100; // Refill ink to full
        this.updateInkLevelDisplay();
    }

    updateInkLevelDisplay() {
        this.inkLevelElement.textContent = `Ink Level: ${this.currentInkLevel}%`;
        this.inkLevelElement.style.width = `${this.currentInkLevel}%`;
        this.inkLevelElement.style.backgroundColor = this.currentInkLevel > 20 ? 'green' : 'red';
    }
}

export default InkMeter;