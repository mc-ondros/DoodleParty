# Doodle Drawing Application

## Overview
The Doodle Drawing Application is a web-based drawing tool that allows users to create doodles with various features such as color picking, brush size selection, remaining ink display, and a timer. The application is built using HTML5, CSS, and JavaScript, following a modular structure for easy maintenance and scalability.

## Features
- **Drawing Canvas**: A responsive canvas area where users can draw.
- **Color Picker**: Users can select different colors for their brush.
- **Brush Size Selector**: Options to choose different brush sizes for drawing.
- **Ink Meter**: Displays the remaining ink level based on user activity.
- **Timer**: A countdown timer for the drawing session.
- **Moderation Shield**: Provides feedback on content moderation.

## Project Structure
```
doodle-drawing-app
├── index.html            # Main HTML document
├── src
│   ├── components
│   │   ├── DrawingCanvas.js  # Handles the drawing area
│   │   ├── DrawerView.js     # Manages the sidebar interface
│   │   ├── InkMeter.js       # Displays remaining ink level
│   │   ├── ModerationShield.js # Provides moderation feedback
│   │   └── GameModeSelector.js # Allows game mode selection
│   ├── scripts
│   │   ├── main.js           # Entry point for the application
│   │   ├── colorPicker.js     # Color picker functionality
│   │   ├── brushSize.js       # Brush size selection functionality
│   │   ├── inkMeter.js        # Updates ink meter display
│   │   └── timer.js           # Timer management
│   └── styles
│       └── main.css          # CSS styles for the application
└── README.md                # Project documentation
```

## Setup Instructions
1. Clone the repository or download the project files.
2. Open `index.html` in a web browser to run the application.
3. Ensure that JavaScript is enabled in your browser settings.

## Usage Guidelines
- Use the color picker to select your desired brush color.
- Adjust the brush size using the provided controls.
- Monitor the remaining ink level as you draw.
- Keep an eye on the timer to manage your drawing session effectively.
- Follow moderation guidelines while using the application.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any suggestions or improvements.

## License
This project is licensed under the MIT License. See the LICENSE file for details.