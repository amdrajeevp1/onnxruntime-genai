# Qwen3-VL Modeling Diff: One-Slide Report

## Reference vs Modified

| | **Golden** (HF transformers v4.57.6) | **Modified** (`pytorch_backup/modeling_qwen3_vl.py`) |
|---|-------------------------------------|--------------------------------------------------|
| **Source** | [modeling_qwen3_vl.py](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py) | Same file + one conditional (see below) |
| **Vision DeepStack** | Always run at layers ∈ `deepstack_visual_indexes` | Run only when ¬(export): see condition |
| **Vision ONNX outputs** | 3 (pooled + 2 deepstack) if traced as-is | 1 (`pooled_embeds`) when export flag set |

---

## Only code difference: DeepStack branch

**Code snippet changes (Qwen3VLVisionModel.forward, ~lines 745–753):**

| | **Golden** | **Modified** |
|---|-------------|--------------|
| **Condition** | `if layer_num in self.deepstack_visual_indexes:` | `if layer_num in self.deepstack_visual_indexes and not (getattr(self, "_skip_deepstack_export", False) or torch.jit.is_tracing()):` |
| **Body** | `deepstack_feature = self.deepstack_merger_list[...](hidden_states)`<br>`deepstack_feature_lists.append(deepstack_feature)` | Same as Golden |

- **Condition (Modified):** Run DeepStack merger at layer ℓ **iff** ℓ ∈ deepstack_visual_indexes **and** ¬( _skip_deepstack_export ∨ torch.jit.is_tracing() ).
- **Golden:** same, but without the second clause (always runs when ℓ ∈ deepstack_visual_indexes).

**Copy-paste for PowerPoint (plain text):**
Run DeepStack merger at layer ℓ iff ℓ ∈ deepstack_visual_indexes and ¬(_skip_deepstack_export ∨ torch.jit.is_tracing()).

- **Export:** builder sets `_skip_deepstack_export = True` → DeepStack not traced → ONNX has single output path → one output tensor.

---

## Shared math / behavior (unchanged)

**Vision RoPE (both):**
- θ = 10⁴; inv_freq_i = θ^(-2i/d); freqs = seq ⊗ inv_freq (outer product); then cos/sin applied to (q, k).
- No precomputed table; dynamic seqlen.

**Vision data flow (both):**
- pixel_values → patch_embed → pos_embed + rotary → blocks → merger → pooled_embeds; DeepStack branches (when run) tap intermediate layers and merge.

**Text / MRoPE (both):**
- position_ids shape (3, B, S) for T/H/W or batch/seq; interleaved MRoPE; cos, sin from inv_freq and position_ids.

**Output contract (both):**
- Vision forward returns (hidden_states, deepstack_feature_lists); ONNX export wrapper uses outputs[0] only.

---

## Image sizes accepted by the vision model

The vision encoder does **not** take raw pixels; it takes **patch features** and a **grid**. Image size is fixed by preprocessing.

- **Input:** pixel_values ∈ R^(N × 1536), image_grid_thw ∈ Z^(M × 3) with (T, H, W) per image.
- **Patch layout:** Spatial patch size p = 16, temporal p_t = 2; 1536 = 3 × p_t × p² (RGB × time × spatial).
- **Grid ↔ resolution:** For one image: grid_h = ceil(H_img/p), grid_w = ceil(W_img/p), N = T · grid_h · grid_w.
- **This pipeline:** All images resized to 384×384 before the processor → grid (1, 24, 24), N = 576 patches. One nominal resolution: 384×384.
- **Other grids:** ONNX uses dynamic N; other (T, H, W) (e.g. 20×20) are possible if the processor outputs them and export used matching --vision_grid.

**Short:** “Accepted image sizes” = any original image, but the pipeline resizes to 384×384; the vision model then sees a fixed grid (e.g. 24×24) and fixed patch dimension 1536.

---

## Bottom line

| Item | Equation / fact |
|------|------------------|
| **Diff** | Modified = golden + condition \(\neg(\texttt{\_skip\_deepstack\_export} \lor \texttt{tracing})\) on DeepStack. |
| **Export** | One ONNX output: \(\text{pooled\_embeds} \in \mathbb{R}^{\text{num\_merged\_patches} \times 2560}\). |
| **Runtime** | When not exporting, behavior and weights match golden. |
