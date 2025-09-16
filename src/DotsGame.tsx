import { useContext, useEffect, useMemo, useState } from 'react';
import { GameContext, Polygon } from './GameState';
import { useQuery } from '@tanstack/react-query';
import { getWasmModule } from './wasm_interface';
import { DOT_SPACING } from './constants';
import { PolygonSVG } from './Polygon';

const DOT_RADIUS = 2;
const DOT_RADIUS_HOVER = 4;

function EmptySpace({
  onClick,
  x,
  y,
}: {
  x: number;
  y: number;
  onClick?: () => void;
}) {
  const [hovered, setHovered] = useState(false);
  return (
    <g>
      <rect
        x={x * DOT_SPACING}
        y={y * DOT_SPACING}
        width={DOT_SPACING}
        height={DOT_SPACING}
        fill="transparent"
        style={{ pointerEvents: 'all' }}
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
        onClick={onClick}
      />
      <circle
        cx={x * DOT_SPACING + DOT_SPACING / 2}
        cy={y * DOT_SPACING + DOT_SPACING / 2}
        r={hovered ? DOT_RADIUS_HOVER : DOT_RADIUS}
        fill={hovered ? '#6b7280' : '#d1d5db'}
        style={{
          transition: 'r 0.15s, fill 0.15s',
          pointerEvents: 'none',
        }}
      />
    </g>
  );
}

function Dot({
  value,
  ...coords
}: {
  x: number;
  y: number;
  value: number;
  onClick?: () => void;
}) {
  if (value === 0) return <EmptySpace {...coords} />;
  const { x, y, onClick } = coords;
  if (value === 1) {
    // Red player: red circle
    return (
      <circle
        cx={x * DOT_SPACING + DOT_SPACING / 2}
        cy={y * DOT_SPACING + DOT_SPACING / 2}
        r={DOT_RADIUS_HOVER * 1.5}
        fill="#ef4444"
      />
    );
  }
  if (value === 2) {
    // Blue player: blue rhombus
    const cx = x * DOT_SPACING + DOT_SPACING / 2;
    const cy = y * DOT_SPACING + DOT_SPACING / 2;
    const r = DOT_RADIUS_HOVER * 1.5;
    return (
      <polygon
        points={`
            ${cx},${cy - r}
            ${cx + r},${cy}
            ${cx},${cy + r}
            ${cx - r},${cy}
          `}
        fill="#3b82f6"
      />
    );
  }
  return null;
}

function Grid({
  dots,
  height,
  width,
}: {
  dots: Uint8Array;
  height: number;
  width: number;
}) {
  const [currentPlayer, setCurrentPlayer] = useState<'red' | 'blue'>('red');
  const game = useContext(GameContext);
  const onClick = (i: number) => {
    console.log(`Clicked on empty space at (${i})`);
    if (currentPlayer === 'red') {
      dots[i] = 1;
      setCurrentPlayer('blue');
    } else {
      dots[i] = 2;
      setCurrentPlayer('red');
    }
    // Future: handle placing a dot
  };
  return (
    <>
      <h1 className="text-2xl font-bold mb-4 text-black">
        Current Player: {currentPlayer}
      </h1>
      <svg
        width={width * DOT_SPACING}
        height={height * DOT_SPACING}
        style={{ display: 'block' }}
      >
        <defs>
          {/* Red player: diagonal stripes */}
          <pattern
            id="red-stripes"
            patternUnits="userSpaceOnUse"
            width="8"
            height="8"
            patternTransform="rotate(45)"
          >
            <rect width="8" height="8" fill="#fecaca" />
            <line
              x1="0"
              y1="0"
              x2="0"
              y2="8"
              stroke="#ef4444"
              strokeWidth="2"
            />
          </pattern>
          {/* Blue player: dots */}
          <pattern
            id="blue-dots"
            patternUnits="userSpaceOnUse"
            width="8"
            height="8"
          >
            <rect width="8" height="8" fill="#dbeafe" />
            <circle cx="4" cy="4" r="2" fill="#3b82f6" />
          </pattern>
        </defs>
        {/* TS refuses me to change type of dots in map */}
        {[...dots].map((dot, index) => {
          return (
            <Dot
              key={`${index}`}
              x={index % width}
              y={Math.floor(index / width)}
              value={dot}
              onClick={() => onClick(index)}
            />
          );
        })}
        {game!.polygons.map((polygon) => (
          <PolygonSVG key={polygon.id} polygon={polygon} />
        ))}
      </svg>
    </>
  );
}

export function DotsGame({ className }: { className?: string }) {
  const [height, width] = [48, 48];
  const { data, error } = useQuery({
    queryKey: ['wasm'],
    queryFn: getWasmModule,
  });
  const [gameState, setGameState] = useState<Uint8Array | null>(null);
  useEffect(() => {
    if (data) {
      const array = new Uint8Array(height * width);
      array.fill(0);
      setGameState(array);
    }
    return () => {
      console.log('Cleaning up WASM module');
      setGameState(null);
    };
  }, [data]);
  if (!gameState) return <div>Loading WASM...</div>;
  if (error)
    return (
      <div className="text-red-500">
        Error loading WASM: {(error as Error).message}
      </div>
    );
  return (
    <div className={className}>
      <Grid dots={gameState} {...{ height, width }} />
    </div>
  );
}
