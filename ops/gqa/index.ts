export {
  groupedAttention,
  defaultScale,
  kvCacheBytes,
  kvHeadOf,
  type GroupedAttentionArgs,
  type GroupedAttentionResult,
} from "./reference.js";
// The mask is `ops/attention`'s, and so are its helpers. Re-exported rather than
// redefined so a caller masking a GQA dispatch and an unfused one cannot end up
// with two mask types (#77).
export { keyPaddingBias, type MaskShape } from "../attention/reference.js";
