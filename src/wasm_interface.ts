/**
 * Component that loads WASM for handling game logic.
 */

interface WasmModule {
  concat_strings(a: string, b: string): string;
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
    resolve((window as any)['Module']);
  };
  script.onerror = (e) => {
    reject(new Error('Failed to load WASM script'));
  };
  return modulePromise;
}

export async function GetString(): Promise<string> {
  const module = await getWasmModule();
  return module.concat_strings('Hello, ', 'world!');
}
