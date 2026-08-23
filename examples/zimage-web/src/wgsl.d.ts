/**
 * Matches `build.mjs`'s esbuild `loader: { ".wgsl": "text" }` — every
 * `import x from "*.wgsl"` in `browser-runtime.ts` becomes a plain string at
 * bundle time. `tsc` (this demo's own `tsconfig.json`, run by `npm run
 * lint`) needs this declaration to type-check those imports at all; esbuild
 * itself does not read this file, it just does the substitution.
 */
declare module "*.wgsl" {
  const source: string;
  export default source;
}
