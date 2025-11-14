import React from 'react';
import { DrawingCanvas } from './components/DrawingCanvas';
import { DrawerView } from './components/DrawerView';
import { GameModeSelector } from './components/GameModeSelector';

function App() {
  return (
    <div className="app">
      <header className="app-header">
        <h1>DoodleParty</h1>
      </header>
      <main className="app-main">
        <DrawerView />
        <DrawingCanvas />
        <GameModeSelector onModeSelect={(mode) => console.log('Selected mode:', mode)} />
      </main>
    </div>
  );
}

export default App;
