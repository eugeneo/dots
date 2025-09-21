import { useState } from 'react';

import { DOT_SPACING } from './constants';

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

export function Dots({
  current_player,
  field,
  onEmptyClick,
  width,
}: {
  current_player: number;
  field: Uint8Array;
  onEmptyClick: (index: number) => void;
  width: number;
}) {
  return (
    <>
      {[...field].map((dot, index) => {
        return (
          <Dot
            key={`${index}`}
            x={index % width}
            y={Math.floor(index / width)}
            value={dot}
            onClick={() => onEmptyClick(index)}
            current_player={current_player}
          />
        );
      })}
    </>
  );
}
