class DrawerView {
    constructor({ onColorChange, onBrushSizeChange } = {}) {
        this.onColorChange = onColorChange;
        this.onBrushSizeChange = onBrushSizeChange;

        this.colorPicker = document.getElementById('colorPicker');
        this.colorPresetButtons = document.querySelectorAll('.color-dot');
        this.brushSizeSelector = document.getElementById('brushSize');
        this.brushSizeValue = document.getElementById('brushSizeValue');
        this.inkMeterText = document.getElementById('inkValue');
        this.inkMeterFill = document.getElementById('inkFill');
        this.timerDisplay = document.getElementById('timer');
        this.activeColorButton = document.querySelector('.color-dot.active');

        this.initialize();
    }

    initialize() {
        this.colorPresetButtons?.forEach((button) => {
            button.addEventListener('click', () => {
                const color = button.dataset.color;
                if (!color) return;
                this.setActiveColor(button);
                this.onColorChange?.(color);
            });
        });

        const initialButton =
            this.activeColorButton || this.colorPresetButtons?.[0];
        if (initialButton) {
            this.setActiveColor(initialButton);
            this.onColorChange?.(initialButton.dataset.color);
        }

        this.brushSizeSelector?.addEventListener('input', (event) => {
            this.updateBrushSize(event.target.value);
        });
    }

    updateBrushSize(size) {
        if (this.brushSizeValue) {
            this.brushSizeValue.textContent = `${size}px`;
        }
        this.onBrushSizeChange?.(Number(size));
    }

    updateInkLevel(percent) {
        if (this.inkMeterText) {
            this.inkMeterText.textContent = `${percent}%`;
        }
        if (this.inkMeterFill) {
            this.inkMeterFill.style.width = `${percent}%`;
        }
    }

    updateTimer(formattedTime) {
        if (this.timerDisplay) {
            this.timerDisplay.textContent = formattedTime;
        }
    }

    setActiveColor(button) {
        if (!button) return;
        this.activeColorButton?.classList.remove('active');
        button.classList.add('active');
        this.activeColorButton = button;
    }
}

export default DrawerView;