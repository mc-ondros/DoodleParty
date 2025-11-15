class GameModeSelector {
    constructor() {
        this.currentMode = 'freeDraw'; // Default mode
        this.modes = ['freeDraw', 'challenge', 'collaborative'];
        this.modeChangeCallback = null;
    }

    setMode(mode) {
        if (this.modes.includes(mode)) {
            this.currentMode = mode;
            this.notifyModeChange();
        }
    }

    getMode() {
        return this.currentMode;
    }

    onModeChange(callback) {
        this.modeChangeCallback = callback;
    }

    notifyModeChange() {
        if (this.modeChangeCallback) {
            this.modeChangeCallback(this.currentMode);
        }
    }

    render() {
        const modeSelectorContainer = document.createElement('div');
        modeSelectorContainer.className = 'game-mode-selector';

        this.modes.forEach(mode => {
            const button = document.createElement('button');
            button.innerText = mode.charAt(0).toUpperCase() + mode.slice(1);
            button.onclick = () => this.setMode(mode);
            modeSelectorContainer.appendChild(button);
        });

        return modeSelectorContainer;
    }
}

export default GameModeSelector;