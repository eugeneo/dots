import { useEffect, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Game, getWasmModule } from './wasm_interface';
import { DOT_SPACING } from './constants';
import { on } from 'events';

const DOT_RADIUS = 2;
const DOT_RADIUS_HOVER = 4;

type PlayerDotProps = {
  x: number;
  y: number;
  player: number;
  player_colors: string[];
  style?: React.CSSProperties | undefined;
};

function PlayerDot({ x, y, player, player_colors, style }: PlayerDotProps) {
  const cx = (x + 0.5) * DOT_SPACING;
  const cy = (y + 0.5) * DOT_SPACING;
  const r = DOT_RADIUS_HOVER * 1.5;
  const fill = player_colors[player] || 'gray';
  switch (player) {
    case 0:
      return <circle {...{ cx, cy, r, fill, style }} />;
    case 1:
      return (
        <polygon
          points={`
            ${cx},${cy - r} ${cx + r},${cy} ${cx},${cy + r} ${cx - r},${cy}
          `}
          {...{ fill, style }}
        />
      );
    default:
      return null;
  }
}

function EmptySpace({
  onClick,
  x,
  y,
  current_player,
}: {
  x: number;
  y: number;
  onClick?: () => void;
  current_player: number;
}) {
  const colors = ['#f7a1a1', '#9ec1fa'];
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
      {!hovered ? (
        <circle
          cx={x * DOT_SPACING + DOT_SPACING / 2}
          cy={y * DOT_SPACING + DOT_SPACING / 2}
          r={DOT_RADIUS}
          fill="#d1d5db"
          style={{
            transition: 'r 0.15s, fill 0.15s',
            pointerEvents: 'none',
          }}
        />
      ) : (
        <PlayerDot
          player={current_player}
          x={x}
          y={y}
          player_colors={colors}
          style={{
            transition: 'r 0.15s, fill 0.15s',
            pointerEvents: 'none',
          }}
        />
      )}
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
  current_player: number;
}) {
  if (value === 0) {
    return <EmptySpace {...coords} />;
  } else {
    return (
      <PlayerDot
        player={value - 1}
        {...coords}
        player_colors={['#ef4444', '#3b82f6']}
      />
    );
  }
}

function Grid({
  dots,
  height,
  width,
  onTurn,
}: {
  dots: Game;
  height: number;
  width: number;
  onTurn?: () => void;
}) {
  const [{ field, current_player }, setState] = useState({
    field: dots.field(),
    current_player: 0,
  });
  const onClick = (i: number) => {
    setState(({ current_player }) => {
      dots.doTurn(i, current_player + 1);
      return {
        field: dots.field(),
        current_player: (current_player + 1) % 2,
      };
    });
    onTurn?.();
  };
  return (
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
          <line x1="0" y1="0" x2="0" y2="8" stroke="#ef4444" strokeWidth="2" />
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
      {[...field].map((dot, index) => {
        return (
          <Dot
            key={`${index}`}
            x={index % width}
            y={Math.floor(index / width)}
            value={dot}
            onClick={() => onClick(index)}
            current_player={current_player}
          />
        );
      })}
    </svg>
  );
}

export function DotsGame({ className }: { className?: string }) {
  const [height, width] = [48, 48];
  const {
    data: m,
    error,
    isLoading,
  } = useQuery({
    queryKey: ['wasm'],
    queryFn: getWasmModule,
  });
  const [gameState, setGameState] = useState<Game | null>(null);
  const [regions, setRegions] = useState<any>('Boop');
  useEffect(() => {
    if (m) {
      setGameState(new m.Game(height, width));
    }
    return () => {
      console.log('Cleaning up WASM module');
      setGameState(null);
    };
  }, [isLoading, m]);
  if (!gameState) return <div>Loading WASM...</div>;
  if (error)
    return (
      <div className="text-red-500">
        Error loading WASM: {(error as Error).message}
      </div>
    );
  return (
    <div className={className}>
      <div className="flex justify-between items-center mb-4 text-5xl text-gray-700">
        <span className="font-bold text-red-500">
          {gameState.playerScore(0)}
        </span>
        {String(regions)}
        <span className="font-bold text-blue-500">
          {gameState.playerScore(1)}
        </span>
      </div>
      <Grid
        dots={gameState}
        {...{ height, width }}
        onTurn={() => {
          setRegions(Object.keys(gameState.regions()).join(', '));
        }}
      />
    </div>
  );
}
