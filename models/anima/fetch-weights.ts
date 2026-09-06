/**
 * `web-xpu-ops/models/anima/fetch-weights` — the browser weight loader.
 *
 * Kept off the main barrel on purpose: it needs the Cache API and a server
 * that honours `Range`, and a host with its own storage (an OPFS mirror, a
 * peer-to-peer swarm) wants the forwards without this policy attached.
 */
export * from "../../examples/anima-web/src/fetch-weights.js";
