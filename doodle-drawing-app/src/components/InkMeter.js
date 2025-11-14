class InkMeter {
    constructor(inkLevelElement) {
        this.inkLevelElement = inkLevelElement;
        this.currentInkLevel = 100; // Assuming ink level starts at 100%
        this.updateInkLevelDisplay();
    }

    updateInkLevel(amount) {
        this.currentInkLevel = Math.max(0, this.currentInkLevel - amount);
        this.updateInkLevelDisplay();
    }

    updateInkLevelDisplay() {
        this.inkLevelElement.textContent = `Ink Level: ${this.currentInkLevel}%`;
    }
}

export default InkMeter;