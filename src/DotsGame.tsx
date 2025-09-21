import { useEffect, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Game, getWasmModule, Region } from './wasm_interface';

import { DOT_SPACING } from './constants';
import { Dots } from './Dots';
import { Regions } from './Regions';

function* ProcessRegions(regions: ReturnType<Game['regions']>) {
  for (let i = 0; i < regions.size(); ++i) {
    yield regions.get(i);
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
  const [{ field, current_player, regions }, setState] = useState({
    field: dots.field(),
    current_player: 0,
    regions: [] as Region[],
  });
  const onClick = (i: number) => {
    setState(({ current_player }) => {
      dots.doTurn(i, current_player + 1);
      return {
        field: dots.field(),
        current_player: (current_player + 1) % 2,
        regions: [...ProcessRegions(dots.regions())],
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
      <Regions regions={regions} />
      <Dots
        current_player={current_player}
        field={field}
        onEmptyClick={onClick}
        width={width}
      />
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
  const [scores, setScores] = useState([0, 0]);
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
        <span className="font-bold text-red-500">{scores[0]}</span>
        <span className="font-bold text-blue-500">{scores[1]}</span>
      </div>
      <Grid
        dots={gameState}
        {...{ height, width }}
        onTurn={() =>
          setScores([gameState.playerScore(0), gameState.playerScore(1)])
        }
      />
    </div>
  );
}
