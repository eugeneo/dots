import { Region } from './wasm_interface';

const patterns = [
  {
    pattern: (id: string) => (
      <pattern
        key={id}
        id={id}
        patternUnits="userSpaceOnUse"
        width="16"
        height="16"
        patternTransform="rotate(45)"
      >
        <rect width="16" height="16" fill="#fecaca" />
        <circle cx="8" cy="8" r="4" fill="#ef4444" />
      </pattern>
    ),
    color: '#ef4444',
  },
  {
    pattern: (id: string) => (
      <pattern
        key={id}
        id={id}
        patternUnits="userSpaceOnUse"
        width="16"
        height="16"
      >
        <rect width="16" height="16" fill="#dbeafe" />
        <polygon points="8,4 12,8 8,12 4,8" fill="#3b82f6" opacity="0.7" />
      </pattern>
    ),
    color: '#3b82f6',
  },
];

export function Regions({ regions }: { regions: Region[] }) {
  return (
    <>
      <defs>
        {patterns.map(({ pattern }, i) => pattern(`player-${i + 1}`))}
      </defs>
      {regions.map((region, index) => (
        <path
          key={index}
          d={region.shape}
          fill={`url(#player-${region.player + 1})`}
          opacity="0.5"
          stroke={patterns[region.player]?.color || 'black'}
          strokeWidth="4"
        />
      ))}
    </>
  );
}
