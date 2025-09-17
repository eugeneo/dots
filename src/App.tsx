import './App.css';

import { DotsGame } from './DotsGame';
import {
  QueryClient,
  QueryClientProvider,
  useQuery,
} from '@tanstack/react-query';

const queryClient = new QueryClient();

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <DotsGame className="overflow-auto max-w-full border rounded bg-white shadow-lg" />
    </QueryClientProvider>
  );
}

export default App;
