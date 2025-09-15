import './App.css';

const BOARD_SIZE = 32;

import { useState, useEffect, createContext, useContext, useMemo } from 'react';
import { useHelloWorldWasm } from './WasmLogic';
// WASM loader hook for /public/hello-world.js


// Game state context
type Player = 'red' | 'blue';
type Dot = {
  player: Player;
};

type Polygon = {
  id: number;
  points: Array<[i: number, j: number]>;
  player: Player;
};

type GameState = {
  occupied_spaces: Set<[i: number, j: number]>;
  dots: Map<[i: number, j: number], Player>;
  polygons: Array<Polygon>;
};

const GameContext = createContext<GameState | undefined>(undefined);
  const DOT_SPACING = 24;

function* dotsiterator(occupied_spaces: Set<[i: number, j: number]>, width: number, height: number) {
  for (let i = 0; i < height; i++) {
    for (let j = 0; j < width; j++) {
      if (!occupied_spaces.has([i, j])) {
        yield [i, j];
      }
    }
  }
}

function PolygonSVG({polygon}: {polygon: Polygon}) {
  // Choose pattern and color by player
  const fillId = polygon.player === 'red' ? 'red-stripes' : 'blue-dots';
  const stroke = polygon.player === 'red' ? '#ef4444' : '#3b82f6';
  const r = 4; // smaller corner radius for better appearance
  // Convert grid points to midpoints
  const midpoints: [number, number][] = polygon.points.map(([i, j]) => [
    (j + 0.5) * DOT_SPACING,
    (i + 0.5) * DOT_SPACING
  ]);

  // Offset polygon outward by DOT_SPACING/2
  function offsetPolygon(points: [number, number][], offset: number): [number, number][] {
    const n = points.length;
    const result: [number, number][] = [];
    for (let i = 0; i < n; i++) {
      const [x0, y0] = points[(i - 1 + n) % n];
      const [x1, y1] = points[i];
      const [x2, y2] = points[(i + 1) % n];
      // Edge vectors
      const v1x = x1 - x0, v1y = y1 - y0;
      const v2x = x2 - x1, v2y = y2 - y1;
      // Outward normals (perpendicular, right-hand rule)
      const n1x = v1y, n1y = -v1x;
      const n2x = v2y, n2y = -v2x;
      // Normalize
      const len1 = Math.hypot(n1x, n1y);
      const len2 = Math.hypot(n2x, n2y);
      const n1nx = n1x / len1, n1ny = n1y / len1;
      const n2nx = n2x / len2, n2ny = n2y / len2;
      // Average normals
      let nx = n1nx + n2nx;
      let ny = n1ny + n2ny;
      const nlen = Math.hypot(nx, ny);
      if (nlen > 0) {
        nx /= nlen;
        ny /= nlen;
      }
      // Offset point
      result.push([x1 + nx * offset, y1 + ny * offset]);
    }
    return result;
  }

  // Generate SVG path with rounded corners
  function roundedPath(points: [number, number][], radius: number) {
    if (points.length < 3) return '';
    let d = '';
    for (let i = 0; i < points.length; i++) {
      const [x1, y1] = points[i];
      const [x2, y2] = points[(i + 1) % points.length];
      const [x0, y0] = points[(i - 1 + points.length) % points.length];
      // Compute direction vectors
      const v1x = x1 - x0, v1y = y1 - y0;
      const v2x = x2 - x1, v2y = y2 - y1;
      // Normalize
      const len1 = Math.hypot(v1x, v1y);
      const len2 = Math.hypot(v2x, v2y);
      const v1nx = v1x / len1, v1ny = v1y / len1;
      const v2nx = v2x / len2, v2ny = v2y / len2;
      // Start and end of the corner
      const startX = x1 - v1nx * radius;
      const startY = y1 - v1ny * radius;
      const endX = x1 + v2nx * radius;
      const endY = y1 + v2ny * radius;
      if (i === 0) {
        d += `M ${startX} ${startY} `;
      } else {
        d += `L ${startX} ${startY} `;
      }
      d += `Q ${x1} ${y1} ${endX} ${endY} `;
    }
    d += 'Z';
    return d;
  }

  const d = roundedPath(midpoints, r);
  const inflated = offsetPolygon(midpoints, DOT_SPACING / 3);
  const dInflated = roundedPath(inflated, r);

  return (
    <g>
      {/* Inflated boundary */}
      <path
        d={dInflated}
        fill="none"
        stroke={stroke}
        strokeWidth={4}
        opacity={0.7}
      />
      {/* Main polygon */}
      <path
        d={d}
        fill={`url(#${fillId})`}
        stroke={stroke}
        strokeWidth={2}
      />
    </g>
  );
}

function Grid() {
  const game = useContext(GameContext);
  const DOT_RADIUS = 2;
  const DOT_RADIUS_HOVER = 4;
  const HIT_RADIUS = 10; // Larger than visible dot
  const [hovered, setHovered] = useState<{i: number, j: number} | null>(null);
  const empty_dots = useMemo(() => Array.from(dotsiterator(game!.occupied_spaces, BOARD_SIZE, BOARD_SIZE)), [game]);
  return (
    <svg
      width={BOARD_SIZE * DOT_SPACING}
      height={BOARD_SIZE * DOT_SPACING}
      style={{ display: 'block' }}
    >
      <defs>
        {/* Red player: diagonal stripes */}
        <pattern id="red-stripes" patternUnits="userSpaceOnUse" width="8" height="8" patternTransform="rotate(45)">
          <rect width="8" height="8" fill="#fecaca" />
          <line x1="0" y1="0" x2="0" y2="8" stroke="#ef4444" strokeWidth="2" />
        </pattern>
        {/* Blue player: dots */}
        <pattern id="blue-dots" patternUnits="userSpaceOnUse" width="8" height="8">
          <rect width="8" height="8" fill="#dbeafe" />
          <circle cx="4" cy="4" r="2" fill="#3b82f6" />
        </pattern>
      </defs>
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
      {game!.polygons.map(polygon => (
        <PolygonSVG key={polygon.id} polygon={polygon} />
      ))}
    </svg>
  );
}


function App() {
  const wasmResult = useHelloWorldWasm();
  return (
    <GameContext.Provider value={{ occupied_spaces: new Set(), dots: new Map(), polygons: [
      {
      id: 1,
      points: [[10, 10], [11, 10], [12,10], [11, 11], [10, 11]],
      player: 'red'
    },
    {
      id: 2,
      points: [[22, 22], [23, 22], [23, 23], [22, 23]],
      player: 'blue'
    }
    
    ] }}>
      <div className="flex flex-col items-center justify-center min-h-screen">
        {/* WASM output at the top */}
        <div className="w-full text-center py-2 bg-gray-100 text-gray-700 text-lg font-mono">
          {wasmResult || 'Loading WASM...'}
        </div>
        <h1 className="text-3xl font-bold mb-4">Крапки (Dots) Game</h1>
        <div className="overflow-auto max-h-[80vh] max-w-full border rounded bg-white p-4 shadow-lg">
          <Grid />
        </div>
      </div>
    </GameContext.Provider>
  );
}

export default App;
