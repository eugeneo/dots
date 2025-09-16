import './App.css';

import { GameContext } from './GameState';
import { DotsGame } from './DotsGame';
import { GetString } from './wasm_interface';
import {
  QueryClient,
  QueryClientProvider,
  useQuery,
} from '@tanstack/react-query';

const queryClient = new QueryClient();

function WasmDisplay() {
  const { data, isLoading, error } = useQuery({
    queryKey: ['wasmResult'],
    queryFn: GetString,
  });
  if (isLoading) return <div>Loading WASM...</div>;
  if (error)
    return (
      <div className="text-red-500">Error: {(error as Error).message}</div>
    );
  return <div>{data}</div>;
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <GameContext.Provider
        value={{
          occupied_spaces: new Set(),
          dots: new Map(),
          polygons: [
            {
              id: 1,
              points: [
                [10, 10],
                [11, 10],
                [12, 10],
                [11, 11],
                [10, 11],
              ],
              player: 'red',
            },
            {
              id: 2,
              points: [
                [22, 22],
                [23, 22],
                [23, 23],
                [22, 23],
              ],
              player: 'blue',
            },
          ],
        }}
      >
        <WasmDisplay />
        <h1 className="text-3xl font-bold mb-4">Крапки (Dots) Game</h1>
        <DotsGame className="overflow-auto max-h-[80vh] max-w-full border rounded bg-white p-4 shadow-lg" />
      </GameContext.Provider>
    </QueryClientProvider>
  );
}

export default App;
