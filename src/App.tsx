import './App.css';

const BOARD_SIZE = 32;

import { useState, createContext, useContext, useMemo } from 'react';
// Game state context
type Player = 'red' | 'blue';
type Dot = {
  player: Player;
};
type GameState = {
  dots: Map<[i: number, j: number], Player>;
};

const GameContext = createContext<GameState | undefined>(undefined);

function* dotsiterator(dots: Map<[i: number, j: number], Player>, width: number, height: number) {
  for (let i = 0; i < height; i++) {
    for (let j = 0; j < width; j++) {
      if (!dots.has([i, j])) {
        yield [i, j];
      }
    }
  }
}

function Grid() {
  const game = useContext(GameContext);
  const DOT_RADIUS = 2;
  const DOT_RADIUS_HOVER = 4;
  const DOT_SPACING = 24;
  const HIT_RADIUS = 10; // Larger than visible dot
  const [hovered, setHovered] = useState<{i: number, j: number} | null>(null);
  const empty_dots = useMemo(() => Array.from(dotsiterator(game!.dots, BOARD_SIZE, BOARD_SIZE)), [game]);
  return (
    <svg
      width={BOARD_SIZE * DOT_SPACING}
      height={BOARD_SIZE * DOT_SPACING}
      style={{ display: 'block' }}
    >
      {empty_dots.map(([i, j]) => {
          const isHovered = hovered && hovered.i === i && hovered.j === j;
          return (
            <g key={`${i}-${j}`}> 
              {/* Invisible hit area for mouse events */}
              <circle
                cx={j * DOT_SPACING + DOT_SPACING / 2}
                cy={i * DOT_SPACING + DOT_SPACING / 2}
                r={HIT_RADIUS}
                fill="transparent"
                style={{ pointerEvents: 'all' }}
                onMouseEnter={() => setHovered({i, j})}
                onMouseLeave={() => setHovered(null)}
              />
              {/* Visible dot */}
              <circle
                cx={j * DOT_SPACING + DOT_SPACING / 2}
                cy={i * DOT_SPACING + DOT_SPACING / 2}
                r={isHovered ? DOT_RADIUS_HOVER : DOT_RADIUS}
                fill={isHovered ? '#6b7280' : '#d1d5db'}
                style={{ transition: 'r 0.15s, fill 0.15s', pointerEvents: 'none' }}
              />
            </g>
          );
        })}
    </svg>
  );
}


function App() {
  return (
    <GameContext.Provider value={{ dots: new Map() }}>
      <div className="flex flex-col items-center justify-center min-h-screen">
        <h1 className="text-3xl font-bold mb-4">Крапки (Dots) Game</h1>
        <div className="overflow-auto max-h-[80vh] max-w-full border rounded bg-white p-4 shadow-lg">
          <Grid />
        </div>
      </div>
    </GameContext.Provider>
  );
}

export default App;
