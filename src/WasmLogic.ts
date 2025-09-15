/**
 * Component that loads WASM for handling game logic.
 */

import { useEffect, useState } from "react";

export function useHelloWorldWasm() {
  const [wasmResult, setWasmResult] = useState<string | null>(null);
  useEffect(() => {
    let cancelled = false;
    let script: HTMLScriptElement | null = document.querySelector('script[src="/hello-world.js"]');
    let added = false;
    if (!script) {
      script = document.createElement('script');
      script.src = '/hello-world.js';
      script.async = true;
      document.body.appendChild(script);
      added = true;
    }
    script.onload = () => {
      setTimeout(() => {
        let result = 'WASM loaded';
        try {
          const mod = (window as any)['Module'];
          console.log('Module:', Object.keys(mod || {}).join(', '));
          if (mod && typeof mod.concat_strings === 'function') {
            result = mod.concat_strings("a", "2");
          }
        } catch (e) {
          result = 'WASM loaded, but call failed';
          console.error(e);
        }
        if (!cancelled) setWasmResult(result);
      }, 100);
    };
    script.onerror = () => {
      if (!cancelled) setWasmResult('Failed to load WASM');
    };
    return () => {
      cancelled = true;
      if (added && script) document.body.removeChild(script);
    };
  }, []);
  return wasmResult;
}