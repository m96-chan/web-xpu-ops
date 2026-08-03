export {
  flashAttention,
  DEFAULT_TILE,
  type FlashAttentionArgs,
  type FlashAttentionResult,
} from "./reference.js";
// The mask is `ops/attention`'s — `FlashAttentionArgs` already extends
// `AttentionArgs`, so re-exporting the helpers keeps one mask type across the
// three attention ops (#77).
export { keyPaddingBias, type MaskShape } from "../attention/reference.js";
