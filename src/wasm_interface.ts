/**
 * Component that loads WASM for handling game logic.
 */

export type Region = { shape: string; player: number };

export interface Game {
  field(): Uint8Array;
  doTurn(index: number, player: number): void;
  playerScore(player: number): number;
  regions(): { size(): number; get(i: number): Region };
  suggest(): number;
}

interface WasmModule {
  concat_strings(a: string, b: string): string;
  Game: {
    new (height: number, width: number): Game;
  };
}

let modulePromise: Promise<WasmModule> | null = null;

export function getWasmModule(): Promise<WasmModule> {
  if (modulePromise) {
    return modulePromise;
  }
  const {
    promise: wasmPromise,
    resolve,
    reject,
  } = Promise.withResolvers<WasmModule>();
  modulePromise = wasmPromise;
  let script: HTMLScriptElement | null = document.querySelector(
    'script[src="/hello-world.js"]'
  );
  if (!script) {
    script = document.createElement('script');
    script.src = '/hello-world.js';
    script.async = true;
    document.body.appendChild(script);
  }
  script.onload = () => {
    (window as any)['Module'].onRuntimeInitialized = () => {
      resolve((window as any)['Module']);
    };
  };
  script.onerror = (e) => {
    reject(new Error('Failed to load WASM script', { cause: e }));
  };
  return modulePromise;
}
