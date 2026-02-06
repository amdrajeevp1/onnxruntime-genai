# Qwen3-VL Modeling Diff: One-Slide Report

## Reference vs Modified

| | **Golden** (HF transformers v4.57.6) | **Modified** (`pytorch_backup/modeling_qwen3_vl.py`) |
|---|-------------------------------------|--------------------------------------------------|
| **Source** | [modeling_qwen3_vl.py](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py) | Same file + one conditional (see below) |
| **Vision DeepStack** | Always run at layers ∈ `deepstack_visual_indexes` | Run only when ¬(export): see condition |
| **Vision ONNX outputs** | 3 (pooled + 2 deepstack) if traced as-is | 1 (`pooled_embeds`) when export flag set |

---

## Only code difference: DeepStack branch

- **Condition (ours):** run DeepStack merger at layer \(\ell\) **iff**
  \[
  \ell \in \text{deepstack\_visual\_indexes}
  \quad\text{and}\quad
  \neg\,\bigl(\texttt{\_skip\_deepstack\_export} \lor \texttt{torch.jit.is\_tracing()}\bigr).
  \]
- **Golden:** same, but without the second clause (always runs when \(\ell \in \text{deepstack\_visual\_indexes}\)).
- **Export:** builder sets `_skip_deepstack_export = True` → DeepStack not traced → ONNX has single output path → one output tensor.

---

## Shared math / behavior (unchanged)

**Vision RoPE (both):**
- \(\theta = 10^4\), \(\text{inv\_freq}_i = \theta^{-2i/d}\), \(\text{freqs} = \text{seq}_{1:L} \otimes \text{inv\_freq}\) (outer), then \(\cos/\sin\) applied to \((q,k)\).
- No precomputed table; dynamic \(\text{seqlen}\).

**Vision data flow (both):**
- \(\text{pixel\_values} \to \text{patch\_embed} \to \text{pos\_embed} + \text{rotary} \to \text{blocks} \to \text{merger} \to \text{pooled\_embeds}\); DeepStack branches (when run) tap intermediate layers and merge.

**Text / MRoPE (both):**
- \(\text{position\_ids}\) shape \((3, B, S)\) (T/H/W or batch/seq); interleaved MRoPE; \(\cos,\sin\) from \(\text{inv\_freq}\) and \(\text{position\_ids}\).

**Output contract (both):**
- Vision forward returns \((\text{hidden\_states}, \text{deepstack\_feature\_lists})\); ONNX export wrapper uses \(\text{outputs}[0]\) only.

---

## Image sizes accepted by the vision model

The vision encoder does **not** take raw pixels; it takes **patch features** and a **grid**. Image size is fixed by preprocessing.

| Item | Meaning |
|------|--------|
| **Input** | \(\text{pixel\_values} \in \mathbb{R}^{N \times 1536}\), \(\text{image\_grid\_thw} \in \mathbb{Z}^{M \times 3}\) with \((T, H, W)\) per image. |
| **Patch layout** | Spatial patch size \(p = 16\), temporal \(p_t = 2\); \(1536 = 3 \times p_t \times p^2\) (RGB × time × spatial). |
| **Grid ↔ resolution** | For one image: \(\text{grid\_h} = \lceil H_{\text{img}}/p\rceil\), \(\text{grid\_w} = \lceil W_{\text{img}}/p\rceil\), \(N = T \cdot \text{grid\_h} \cdot \text{grid\_w}\). |
| **This pipeline** | All images **resized to 384×384** before the processor → grid \((1, 24, 24)\), \(N = 576\) patches. So one nominal resolution: **384×384**. |
| **Other grids** | ONNX uses dynamic \(N\); other \((T,H,W)\) (e.g. 20×20) are possible if the processor outputs them and export used matching \(\texttt{--vision\_grid}\). |

- **Short:** “Accepted image sizes” = **any** original image, but the pipeline **resizes to 384×384**; the vision model then sees a fixed grid (e.g. 24×24) and fixed patch dimension 1536.

---

## Bottom line

| Item | Equation / fact |
|------|------------------|
| **Diff** | Modified = golden + condition \(\neg(\texttt{\_skip\_deepstack\_export} \lor \texttt{tracing})\) on DeepStack. |
| **Export** | One ONNX output: \(\text{pooled\_embeds} \in \mathbb{R}^{\text{num\_merged\_patches} \times 2560}\). |
| **Runtime** | When not exporting, behavior and weights match golden. |
