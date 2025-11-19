function initializeColorPicker(callback) {
    const colorPicker = document.createElement('input');
    colorPicker.type = 'color';
    colorPicker.value = '#000000'; // Default color

    colorPicker.addEventListener('input', (event) => {
        const selectedColor = event.target.value;
        callback(selectedColor);
    });

    document.body.appendChild(colorPicker);
}

export { initializeColorPicker };