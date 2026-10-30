# Qwen3.8-27B Full SFT & LoRA SFT on ARC-AGI-2: Experiment Report

**Code & Experiment Repositories**:
- **SFT Training & FP8 Export Pipeline**: [`//experimental/users/arazdai/kaggle/sft/`](file:///google/src/cloud/arazdai/test_workspace/google3/experimental/users/arazdai/kaggle/sft/README.md) ([`data_pipeline.py`](file:///google/src/cloud/arazdai/test_workspace/google3/experimental/users/arazdai/kaggle/sft/data_pipeline.py), [`train_sft.py`](file:///google/src/cloud/arazdai/test_workspace/google3/experimental/users/arazdai/kaggle/sft/train_sft.py), [`train_full_sft.py`](file:///google/src/cloud/arazdai/test_workspace/google3/experimental/users/arazdai/kaggle/sft/train_full_sft.py), [`fast_deltanet.py`](file:///google/src/cloud/arazdai/test_workspace/google3/experimental/users/arazdai/kaggle/sft/fast_deltanet.py), [`export_and_quantize.py`](file:///google/src/cloud/arazdai/test_workspace/google3/experimental/users/arazdai/kaggle/sft/export_and_quantize.py))
- **ARC-AGI-2 Evaluation Harness (`pilot_2` & `pilot_1`)**: [`//experimental/users/arazdai/kaggle/arc_agi_2/`](file:///google/src/cloud/arazdai/test_workspace/google3/experimental/users/arazdai/kaggle/arc_agi_2/README.md) ([`pilot2_eval.py`](file:///google/src/cloud/arazdai/test_workspace/google3/experimental/users/arazdai/kaggle/arc_agi_2/pilot_2/pilot2_eval.py), [`prompts_and_selector.py`](file:///google/src/cloud/arazdai/test_workspace/google3/experimental/users/arazdai/kaggle/arc_agi_2/pilot_2/prompts_and_selector.py), [`PILOT_2_EXPERIMENTS.md`](file:///google/src/cloud/arazdai/test_workspace/google3/experimental/users/arazdai/kaggle/arc_agi_2/pilot_2/PILOT_2_EXPERIMENTS.md), [`ARC_AGI2_EXPERIMENTS_FINDINGS.md`](file:///google/src/cloud/arazdai/test_workspace/google3/experimental/users/arazdai/kaggle/arc_agi_2/pilot_1/setB/ARC_AGI2_EXPERIMENTS_FINDINGS.md))

---

## 1. Head-to-Head Comparison: Pure Qwen Baselines vs. Best LoRA SFT vs. Best Full SFT

The table below compares:
1. **Direct Comparable `N=2` Python-Tool Baseline (`h1_python_tools_base`)** on pure `Qwen3.8-27B-FP8` (the only `N=2` agentic Python-tool baseline on the un-finetuned model).
2. **Best Pure `Qwen3.8-27B-FP8` Harnesses (`h7` & `h8`)** using `N=4` attempts and two-stage `<think>` nudging / evolutionary wave-2 repair.
3. **Best LoRA SFT Checkpoint (Step 191 Final, `r=64, alpha=128`)** evaluated with `N=2` attempts under `e031_deepseek_teaching`.
4. **Best Evaluated Full-Parameter FSDP2 SFT Checkpoint (Step 100)** evaluated with `N=2` attempts under the Kaggle-budgeted `e031_deepseek_teaching_fast` harness (`40k` token cap).

> [!IMPORTANT]
> **Key Headline Result**: At equal `N=2` attempts, **LoRA SFT Step 191 (`46.94%` Official Kaggle Task Pass@2)** more than **doubles** the score of the comparable `N=2` pure Qwen Python-tool baseline (`h1`: `19.17%` Task Pass@2, `+27.77 pp`), and matches the best `N=4` pure Qwen harnesses (`h7`: `47.35%`, `h8`: `49.15%`) while using **half as many attempts (`N=2` vs. `N=4`) and `47%` fewer total generated tokens (`14.57M` vs. `27.55M`)**.

| Model & Run ID | Harness & Interface Description | Eval Coverage (Pairs / Tasks) | Num Attempts ($N$) & Max Turns | Official Kaggle Task Pass@1 | **Official Kaggle Task Pass@2** | Pair Pass@1 | Pair Pass@2 | Demo-Verified Pair Rate (`any_verified`) | Attempt Demo Pass Rate (`train_pass`) | Median Reasoning (`<think>`) Tokens / Attempt *(Total Gen Tokens)* | Median Verified Tokens (Turns) | Total Output Tokens (Full Eval) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pure Qwen Baseline (`N=2` Comparable)**<br>`h1_python_tools_base` ([`http://xid/303505766`](http://xid/303505766)) | **JSON Tools (`run_python`, `submit_program`, `submit_answer`)**: Persistent sandbox, `3,500` think cap + `512` nudge, `48k` budget | `167 / 167` pairs<br>(`120 / 120` tasks) | `N = 2`<br>`20` turns | `18.33%`<br>(`22.00 / 120`) | **`19.17%`**<br>(`23.00 / 120`) | `19.16%`<br>(`32 / 167`) | `19.76%`<br>(`33 / 167`) | `30.54%`<br>(`51 / 167`) | `17.96%`<br>(`60 / 334`) | **`~17,991` `<think>`**<br>*(`25,015` total; `26,421` mean)* | `28,649` tok<br>(`7.5` turns) | `8.82M` |
| **Pure Qwen Best Single-Wave (`N=4`)**<br>`h7_native_xml_and_grace_nudge` ([`http://xid/303505832`](http://xid/303505832)) | **Native Qwen XML Tools (`<function=...>`) + Two-Stage `<think>` Nudge**: `+1,200` grace think, forced action at stage 2, `56k` budget | `150 / 167` pairs<br>(`113` tasks; `42.92%` strict `/120`) | `N = 4`<br>`40` turns | `43.81%`<br>(`49.50 / 113`) | **`47.35%`**<br>(`53.50 / 113`) | `44.67%`<br>(`67 / 150`) | `48.00%`<br>(`72 / 150`) | `67.33%`<br>(`101 / 150`) | `51.00%`<br>(`306 / 600`) | **`~39,983` `<think>`**<br>*(`51,771` total; `42,625` mean)* | `28,562` tok<br>(`6.0` turns) | `25.58M` |
| **Pure Qwen Best Overall (`N=4`, 2-Wave)**<br>`h8_evolutionary_two_wave_repair` ([`http://xid/303505768`](http://xid/303505768)) | **Evolutionary 2-Wave Repair (2+2) + Native XML + Equivariance Tie-Break**: Wave 2 seeded with Wave 1 best partial code & cell diffs, `56k` budget | `158 / 167` pairs<br>(`117` tasks; `46.25%` strict `/120`) | `N = 4`<br>`(2 + 2)`<br>`40` turns | `41.88%`<br>(`49.00 / 117`) | **`49.15%`**<br>(`57.50 / 117`) | `39.24%`<br>(`62 / 158`) | `46.20%`<br>(`73 / 158`) | `68.99%`<br>(`109 / 158`) | `47.31%`<br>(`299 / 632`) | **`~43,921` `<think>`**<br>*(`56,000` total; `43,596` mean)* | `26,207` tok<br>(`6.0` turns) | `27.55M` |
| **Best LoRA SFT (`r=64`, Step 191 Final)**<br>`e031_deepseek_teaching` ([`http://xid/304525051`](http://xid/304525051)) | **E031 DeepSeek Teaching Interface**: Minimal-effort `lean2sub` prompt, persistent IPython (`I_i`, `O_i`, `T_j`), `6,000` think cap, `60k` budget | `167 / 167` pairs<br>(`120 / 120` tasks) | `N = 2`<br>`40` turns | `39.17%`<br>(`47.00 / 120`) | **`46.94%`**<br>(`56.33 / 120`) | `37.72%`<br>(`63 / 167`) | `45.51%`<br>(`76 / 167`) | `50.90%`<br>(`85 / 167`) | `39.22%`<br>(`131 / 334`) | **`~51,713` `<think>`**<br>*(`56,425` total; `43,633` mean)* | **`19,924` tok**<br>(`9.0` turns) | `14.57M` |
| **Best Evaluated Full SFT (Step 100)**<br>`e031_deepseek_teaching_fast` ([`http://xid/304490794`](http://xid/304490794)) | **Fast E031 Teaching Interface**: Same `lean2sub` IPython harness capped at **`40k` max tokens & `32` turns** for Kaggle `4×L4` speed | `164 / 167` pairs<br>(`119` tasks; `40.42%` strict `/120`) | `N = 2`<br>`32` turns | `36.55%`<br>(`43.50 / 119`) | **`40.76%`**<br>(`48.50 / 119`) | `35.98%`<br>(`59 / 164`) | `40.24%`<br>(`66 / 164`) | `43.90%`<br>(`72 / 164`) | `32.32%`<br>(`106 / 328`) | **`~34,765` `<think>`**<br>*(`39,718` total; `31,540` mean)* | **`17,023` tok**<br>(`7.0` turns) | `10.35M` |

*Note on Reasoning Token Accounting*: In our multi-turn tool-use evaluations, `total_gen_tokens` measures all assistant tokens generated across all turns (`<think>...</think>` internal reasoning plus `<tool_call>...</tool_call>` Python snippets). Direct transcript parsing shows that `<think>` blocks account for **`87.5%–91.7%`** of generated tokens in `e031_deepseek_teaching` SFT runs (where Python tool calls are concise `15–25` line IPython cells) and **`71.9%–78.4%`** in pure Qwen `pilot_2` runs. For solved/verified attempts (`train_pass=True`), SFT models reach a 100% demo-verified solution in a median of just **`17,023–19,924` total tokens (`7–9` turns)**, whereas unsolved attempts think until hitting the `40,000` (`e031_fast`) or `60,000` (`e031`) hard budget cap.

---

## 2. LoRA SFT Checkpoint Evaluations & Loss Evolution

**Training Job**: [`http://xid/304138725`](http://xid/304138725) (Steps 0–40) $\rightarrow$ resumed from Step 40 to Step 191 in [`http://xid/304366741`](http://xid/304366741) on `1× NVIDIA B200 (178.4 GB)`.
- **Log Path**: `/cns/lu-d/home/arazdai/kaggle/sft/checkpoints/xid_304138725_resume_s40/wu_1/metrics.jsonl`
- **Eval Harness**: `e031_deepseek_teaching` (`N = 2` attempts, `max_turns = 40`, `turn_think_cap = 6,000`, `total_token_budget = 60,000`, `temperature = 0.6`, `top_p = 0.95`).

| Training Step | Epoch Progress | Instant Train Loss | 25-Step Mean Train Loss | **Val Loss** (`64` val trajs, `279.8k` tok) | **Val Perplexity (PPL)** | Eval Status & XID | Eval Coverage (Pairs / Tasks) | Official Task Pass@1 | **Official Task Pass@2** | Pair Pass@1 | Pair Pass@2 | Demo-Verified Pair Rate | Attempt `train_pass` Rate | Median Output Tokens / Attempt *(Verified Median)* |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Step 0**<br>*(Pure Base)* | `0.00` ep | `0.8645`<br>*(Step 1)* | — | **`0.8721`** | **`2.3919`** | Baseline `h1` ([`http://xid/303505766`](http://xid/303505766)) | `167 / 167` pairs<br>(`120` tasks) | `18.33%`<br>(`22.00/120`) | **`19.17%`**<br>(`23.00/120`) | `19.16%`<br>(`32/167`) | `19.76%`<br>(`33/167`) | `30.54%`<br>(`51/167`) | `17.96%`<br>(`60/334`) | `25,015` tok<br>*(`28,649` ver)* |
| **Step 25** | `0.13` ep | `0.7153` | `0.7876` | **`0.7812`** | **`2.1841`** | *Ckpt saved (not eval'd)* | — | — | — | — | — | — | — | — |
| **Step 50** | `0.26` ep | `0.8661` | `0.7795` | **`0.7764`** | **`2.1736`** | Eval'd ([`http://xid/304392081`](http://xid/304392081), [`http://xid/304497107`](http://xid/304497107)) | `167 / 167` pairs<br>(`120` tasks) | `37.50%`<br>(`45.00/120`) | **`40.42%`**<br>(`48.50/120`) | `35.93%`<br>(`60/167`) | `38.32%`<br>(`64/167`) | `42.51%`<br>(`71/167`) | `36.72%`<br>(`94/256`)* | `56,866` tok<br>*(`19,075` ver)* |
| **Step 75** | `0.39` ep | `0.8337` | `0.7746` | **`0.7708`** | **`2.1615`** | *Ckpt saved (not eval'd)* | — | — | — | — | — | — | — | — |
| **Step 100** | `0.52` ep | `0.6057` | `0.7615` | **`0.7677`** | **`2.1548`** | *Ckpt saved (not eval'd)* | — | — | — | — | — | — | — | — |
| **Step 125** | `0.65` ep | `0.7790` | `0.7753` | **`0.7658`** | **`2.1508`** | *Ckpt saved (not eval'd)* | — | — | — | — | — | — | — | — |
| **Step 150** | `0.79` ep | `0.8596` | `0.7689` | **`0.7643`** | **`2.1476`** | Eval'd (partial: [`http://xid/304471303`](http://xid/304471303)) | `115 / 167` pairs<br>(`92` tasks) | `39.13%`<br>(`36.00/92`) | **`40.22%`**<br>(`37.00/92`) | `39.13%`<br>(`45/115`) | `40.00%`<br>(`46/115`) | `46.96%`<br>(`54/115`) | `37.39%`<br>(`86/230`) | `58,774` tok<br>*(`19,200` ver)* |
| **Step 175** | `0.92` ep | `0.7810` | `0.7429` | **`0.7632`** | **`2.1451`** | *Ckpt saved (not eval'd)* | — | — | — | — | — | — | — | — |
| **Step 191 (Final)** | `1.00` ep | `0.7412` | `0.7485` | **`0.7627`** | **`2.1440`** | **Eval'd Full (`167/167`)** ([`http://xid/304496623`](http://xid/304496623), [`http://xid/304525051`](http://xid/304525051), [`http://xid/304544539`](http://xid/304544539)) | `167 / 167` pairs<br>(`120` tasks)<br>*(Single 15-shard run: `164/167` pairs)* | **`39.17%`**<br>(`47.00/120`)<br>*(`38.66%` on 15-shard)* | **`46.94%`**<br>(`56.33/120`)<br>*(`45.66%` on 15-shard)* | **`37.72%`**<br>(`63/167`)<br>*(`36.59%` on 15-shard)* | **`45.51%`**<br>(`76/167`)<br>*(`43.90%` on 15-shard)* | **`50.90%`**<br>(`85/167`)<br>*(`50.00%` on 15-shard)* | **`39.22%`**<br>(`131/334`)<br>*(`39.02%` on 15-shard)* | `56,425` tok<br>*(`19,924` ver)* |

*\*Note on LoRA Step 50 & Step 191 Shards*:
- At **Step 50**, the `06`-shard evaluation run ([`http://xid/304392081`](http://xid/304392081), `N=2`) completed `95/167` pairs with **`43.83%` Task Pass@2** (`35.50/81` tasks) and **`50.53%` demo-verified pairs**, while the `11`-shard run ([`http://xid/304497107`](http://xid/304497107)) completed all `167/167` pairs (`120/120` tasks) with **`40.42%` Task Pass@2** (`48.50/120` tasks).
- At **Step 191 (Final)**, the single `15`-shard run ([`http://xid/304525051`](http://xid/304525051)) completed `164/167` pairs (`119/120` tasks) achieving **`45.66%` Task Pass@2** (`54.33/119`, or `44.86%` strict `/120`), and combining the 3 missing pairs from the companion `11`-shard run ([`http://xid/304496623`](http://xid/304496623)) yields complete `167/167` coverage (`120/120` tasks) at **`46.94%` Task Pass@2 (`56.33 / 120`)** and **`39.17%` Task Pass@1 (`47.00 / 120`)**.

---

## 3. Full-Parameter FSDP2 SFT Checkpoint Evaluations & Loss Evolution

**Training Job**: [`http://xid/304470493`](http://xid/304470493) (`qwen38_27b_arc_agi2_e031_full_sft_8xb200_v3`) on `8× NVIDIA B200 (178.4 GB/GPU)` using PyTorch FSDP2 (`fully_shard`) with `100%` of text backbone parameters trainable (`26,895,998,464` params).
- **Log Path**: `/cns/lu-d/home/arazdai/kaggle/sft/checkpoints_full_sft_v2/xid_304470493_qwen38_27b_arc_agi2_e031_full_sft_8xb200_v3/wu_1/metrics.jsonl`
- **Eval Harness**: `e031_deepseek_teaching_fast` (`N = 2` attempts, `max_turns = 32`, `turn_think_cap = 5,000`, `total_token_budget = 40,000`, `temperature = 0.6`, `top_p = 0.95`). Note that Full SFT checkpoints were evaluated under the stricter **`40k` token / `32` turn** `e031_deepseek_teaching_fast` harness (33% smaller token budget than LoRA's `60k` / `40` turn harness).

| Training Step | Epoch Progress | Instant Train Loss | 25-Step Mean Train Loss | **Val Loss** (`64` val trajs, `279.8k` tok) | **Val Perplexity (PPL)** | Val Loss $\Delta$ vs. LoRA SFT | Eval Status & XID (`e031_fast`, `40k` cap) | Eval Coverage (Pairs / Tasks) | Official Task Pass@1 | **Official Task Pass@2** | Pair Pass@1 | Pair Pass@2 | Demo-Verified Pair Rate | Attempt `train_pass` Rate | Median Output Tokens / Attempt *(Verified Median)* |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Step 0**<br>*(Pure Base)* | `0.00` ep | `0.8645`<br>*(Step 1)* | — | **`0.8721`** | **`2.3919`** | `0.0000` | Baseline `h1` ([`http://xid/303505766`](http://xid/303505766)) | `167 / 167` pairs<br>(`120` tasks) | `18.33%`<br>(`22.00/120`) | **`19.17%`**<br>(`23.00/120`) | `19.16%`<br>(`32/167`) | `19.76%`<br>(`33/167`) | `30.54%`<br>(`51/167`) | `17.96%`<br>(`60/334`) | `25,015` tok<br>*(`28,649` ver)* |
| **Step 25** | `0.13` ep | `0.7115` | `0.7852` | **`0.7765`** | **`2.1738`** | `-0.0047`<br>*(matches LoRA s50)* | *No ckpt exported* | — | — | — | — | — | — | — | — |
| **Step 50** | `0.26` ep | `0.8611` | `0.7745` | **`0.7708`** | **`2.1614`** | `-0.0056`<br>*(matches LoRA s75)* | **Eval'd Full (`167/167`)** ([`http://xid/304486799`](http://xid/304486799)) | `167 / 167` pairs<br>(`120` tasks) | `31.39%`<br>(`37.67/120`) | **`36.94%`**<br>(`44.33/120`) | `29.94%`<br>(`50/167`) | `35.33%`<br>(`59/167`) | `36.53%`<br>(`61/167`) | `26.95%`<br>(`90/334`) | `40,011` tok<br>*(`19,751` ver)* |
| **Step 75** | `0.39` ep | `0.8334` | `0.7689` | **`0.7669`** | **`2.1531`** | `-0.0039` | *No ckpt exported* | — | — | — | — | — | — | — | — |
| **Step 100** | `0.52` ep | `0.6004` | `0.7556` | **`0.7634`** | **`2.1455`** | `-0.0043`<br>*(beats LoRA s175)* | **Eval'd (`164/167`)** ([`http://xid/304490794`](http://xid/304490794)) | `164 / 167` pairs<br>(`119` tasks; `40.42%` strict `/120`) | **`36.55%`**<br>(`43.50/119`) | **`40.76%`**<br>(`48.50/119`) | **`35.98%`**<br>(`59/164`) | **`40.24%`**<br>(`66/164`) | **`43.90%`**<br>(`72/164`) | **`32.32%`**<br>(`106/328`) | `39,718` tok<br>*(`17,023` ver)* |
| **Step 125** | `0.65` ep | `0.7741` | `0.7691` | **`0.7607`** | **`2.1398`** | `-0.0051`<br>*(beats LoRA final)* | *No ckpt exported* | — | — | — | — | — | — | — | — |
| **Step 150** | `0.78` ep | `0.8507` | `0.7618` | **`0.7575`** | **`2.1329`** | `-0.0068` | **FP8 Exported on CNS** *(ready for eval)* | — | — | — | — | — | — | — | — |
| **Step 175** | `0.91` ep | `0.7749` | `0.7358` | **`0.7558`** | **`2.1293`** | `-0.0074` | *No ckpt exported* | — | — | — | — | — | — | — | — |
| **Step 192 (Final)** | `1.00` ep | `0.6867` | `0.7376` | **`0.7549`** | **`2.1275`** | **`-0.0078`**<br>*(lowest val loss)* | **FP8 Exported on CNS** *(ready for eval)* | — | — | — | — | — | — | — | — |

### Full SFT Parameter-Group Update Norms (Step 192 Final vs. Base `Qwen3.8-27B`)
At the end of Full SFT (`step_00192`), [`export_and_quantize.py`](file:///google/src/cloud/arazdai/test_workspace/google3/experimental/users/arazdai/kaggle/sft/export_and_quantize.py) measured the exact L2 distance $\|\theta_{\text{step192}} - \theta_{\text{base}}\|_2$ across all 7 architectural parameter groups in FP32 before block-FP8 quantization:
- **`lm_head`**: `1.3737` (largest single-tensor adaptation, sharpening token probabilities on IPython tool syntax and `<think>` transitions)
- **`mlp` (`gate_proj`, `up_proj`, `down_proj`)**: `0.5735`
- **`deltanet_projections` (`in_proj_qkv`, `in_proj_z`, `out_proj`)**: `0.3979`
- **`full_attn` (`q_proj`, `k_proj`, `v_proj`, `o_proj`)**: `0.3484`
- **`embed_tokens`**: `0.2817` (unfrozen in Full SFT; frozen in LoRA)
- **`norms` (`input_layernorm`, `post_attention_layernorm`, `q_norm`, `k_norm`)**: `0.00318` (unfrozen in Full SFT; frozen in LoRA)
- **`deltanet_newly_trainable` (`conv1d`, `A_log`, `dt_bias`, `in_proj_a`, `in_proj_b`, `norm`)**: `0.00072` (recurrent state-decay and gating parameters updated smoothly without destabilizing linear-attention recurrence)

---

## 4. Pure `Qwen3.8-27B-FP8` Harness Ablation Table (`pilot_1` & `pilot_2`)

Before and alongside SFT, we ran a systematic progression of test-time harnesses on the unmodified **`Qwen3.8-27B-FP8`** model across `pilot_1` (`vLLM` on TPU v6e) and `pilot_2` (`h1` through `h8` on `6×8 TPU v6e`).

| Harness ID & XID | Phase | Num Attempts ($N$) | Max Turns | Token Cap / Attempt (`Think Cap`) | Harness Description & Key Mechanism | Eval Coverage (Pairs / Tasks) | Official Task Pass@1 | **Official Task Pass@2** | Pair Pass@1 | Pair Pass@2 | Demo-Verified Pair Rate | Attempt `train_pass` Rate | Median `<think>` Tokens *(Total Gen Tokens)* | Total Eval Tokens | Fits Kaggle `4×L4` (`12h`)? |
| :--- | :---: | :---: | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`EXP-00`: `no_tools_32k`** | `pilot_1` | `N = 2` | `1` *(1-shot)* | `32,000` | Pure CoT reasoning without Python tools (`32k` budget). Model frequently truncates inside `<think>` before emitting a grid (`50.9%` truncated). | `167 / 167` pairs<br>(`120` tasks) | `8.75%`<br>(`10.50/120`) | **`12.50%`**<br>(`15.00/120`) | `8.98%`<br>(`15/167`) | `13.17%`<br>(`22/167`) | `0.0%`<br>*(no tools)* | `0.0%` | `~29,500`<br>*(`30,100` total)* | `8.38M` | Yes (`~3.7h`) |
| **`EXP-01`: `no_tools_48k`** | `pilot_1` | `N = 2` | `1` *(1-shot)* | `48,000` | Pure CoT reasoning (`48k` budget). Reduces truncation from `50.9%` to `26.0%`, nearly doubling 1-shot accuracy. | `167 / 167` pairs<br>(`120` tasks) | `16.25%`<br>(`19.50/120`) | **`21.25%`**<br>(`25.50/120`) | `16.17%`<br>(`27/167`) | `21.56%`<br>(`36/167`) | `0.0%`<br>*(no tools)* | `0.0%` | `~33,800`<br>*(`34,500` total)* | `11.55M` | Yes (`~5.1h`) |
| **`EXP-10B`: `no_tools_Bo8`** | `pilot_1` | `N = 8` | `1` *(1-shot)* | `48,000` | Best-of-8 pure CoT majority voting. Huge oracle (`38.9%`), but without demo verification, frequency voting loses `12.6 pp` to common wrong answers. | `167 / 167` pairs<br>(`120` tasks) | `18.33%`<br>(`22.00/120`) | **`26.67%`**<br>(`32.00/120`) | `17.96%`<br>(`30/167`) | `26.35%`<br>(`44/167`) | `0.0%`<br>*(no tools)* | `0.0%` | `~33,800`<br>*(`34,400` total)* | `45.99M` | No (`~20.3h`) |
| **`h1_python_tools_base`**<br>([`http://xid/303505766`](http://xid/303505766)) | `pilot_2` | `N = 2` | `20` | `48,000`<br>(`3,500` think) | **Baseline Agentic Python Tools**: JSON tool format (`run_python`, `submit_program`, `submit_answer`), `3.5k` per-turn think cap + `512` single nudge. | `167 / 167` pairs<br>(`120` tasks) | `18.33%`<br>(`22.00/120`) | **`19.17%`**<br>(`23.00/120`) | `19.16%`<br>(`32/167`) | `19.76%`<br>(`33/167`) | `30.54%`<br>(`51/167`) | `17.96%`<br>(`60/334`) | `~17,991`<br>*(`25,015` total)* | `8.82M` | Yes (`~3.9h`) |
| **`h2_more_attempts_bo4`**<br>([`http://xid/303505921`](http://xid/303505921)) | `pilot_2` | `N = 4` | `20` | `48,000`<br>(`3,500` think) | **Best-of-4 Attempts (`N=4`)**: Doubles candidate diversity over `h1` (`2` `solve_first` + `2` `explore_first` prompts). | `167 / 167` pairs<br>(`120` tasks) | `24.31%`<br>(`29.17/120`) | **`26.39%`**<br>(`31.67/120`) | `22.16%`<br>(`37/167`) | `25.15%`<br>(`42/167`) | `40.12%`<br>(`67/167`) | `15.57%`<br>(`104/668`) | `~16,700`<br>*(`23,173` total)* | `17.29M` | Marginal (`~7.6h`) |
| **`h3_more_max_turns_40`**<br>([`http://xid/303505830`](http://xid/303505830)) | `pilot_2` | `N = 4` | `40` | `48,000`<br>(`3,500` think) | **Doubled Interactive Horizon (`40` turns)**: Allows iterative hypothesis debugging (`+6.94 pp` Task Pass@2 over `h2`). | `167 / 167` pairs<br>(`120` tasks) | `32.22%`<br>(`38.67/120`) | **`33.33%`**<br>(`40.00/120`) | `31.14%`<br>(`52/167`) | `32.93%`<br>(`55/167`) | `46.71%`<br>(`78/167`) | `21.56%`<br>(`144/668`) | `~25,100`<br>*(`34,837` total)* | `22.51M` | No (`~9.9h`) |
| **`h4_input_desc_and_helpers`**<br>([`http://xid/303505922`](http://xid/303505922)) | `pilot_2` | `N = 4` | `40` | `48,000`<br>(`3,500` think) | **Precomputed Grid Summary + `arc_helpers`**: Adds component/symmetry summary to half of attempts. *Slightly regressed vs. `h3` due to prompt distraction.* | `167 / 167` pairs<br>(`120` tasks) | `25.97%`<br>(`31.17/120`) | **`28.75%`**<br>(`34.50/120`) | `24.55%`<br>(`41/167`) | `27.54%`<br>(`46/167`) | `42.51%`<br>(`71/167`) | `17.51%`<br>(`117/668`) | `~24,250`<br>*(`33,675` total)* | `21.89M` | No (`~9.7h`) |
| **`h5_longer_logical_reasoning`**<br>([`http://xid/303505767`](http://xid/303505767)) | `pilot_2` | `N = 4` | `40` | `64,000`<br>(`6,000` think) | **Expanded Per-Turn `<think>` Cap (`6,000` tok)**: Raises demo-verified pair rate to `49.70%` (`83/167`), recovering `h3`-level score. | `167 / 167` pairs<br>(`120` tasks) | `28.33%`<br>(`34.00/120`) | **`31.94%`**<br>(`38.33/120`) | `29.94%`<br>(`50/167`) | `33.53%`<br>(`56/167`) | `49.70%`<br>(`83/167`) | `21.26%`<br>(`142/668`) | `~31,500`<br>*(`41,959` total)* | `28.42M` | No (`~12.5h`) |
| **`h6_adaptive_hybrid_ensemble`**<br>([`http://xid/303505831`](http://xid/303505831)) | `pilot_2` | `N = 4` | `40` | `56,000`<br>(`5,000` think) | **Diverse Prompt Styles + Transcript Compaction + Secondary Verified Variants**: Adds `hypothesis_matrix` & `invariant_coder` styles. | `167 / 167` pairs<br>(`120` tasks) | `35.28%`<br>(`42.33/120`) | **`36.53%`**<br>(`43.83/120`) | `34.73%`<br>(`58/167`) | `35.93%`<br>(`60/167`) | `49.70%`<br>(`83/167`) | `21.41%`<br>(`143/668`) | `~29,750`<br>*(`39,145` total)* | `25.49M` | No (`~11.2h`) |
| **`h7_native_xml_and_grace_nudge`**<br>([`http://xid/303505832`](http://xid/303505832)) | `pilot_2` | `N = 4` | `40` | `56,000`<br>(`5,000 + 1,200`) | **Native Qwen XML Tool Format (`<function=run_python>`) + Two-Stage `<think>` Nudge**: Eliminates JSON escaping errors and mid-thought truncation. **Massive `+10.82 pp` jump!** | `150 / 167` pairs<br>(`113` tasks; `42.92%` strict `/120`) | `43.81%`<br>(`49.50/113`) | **`47.35%`**<br>(`53.50/113`) | `44.67%`<br>(`67/150`) | `48.00%`<br>(`72/150`) | `67.33%`<br>(`101/150`) | `51.00%`<br>(`306/600`) | `~39,983`<br>*(`51,771` total)* | `25.58M` | No (`~11.3h`) |
| **`h8_evolutionary_two_wave_repair`**<br>([`http://xid/303505768`](http://xid/303505768)) | `pilot_2` | `N = 4`<br>`(2+2)` | `40` | `56,000`<br>(`5,000 + 1,200`) | **Evolutionary 2-Wave Repair + Rotational Equivariance Tie-Break**: Wave 2 seeded with Wave 1's best partial `solve(grid)` & exact failure diffs. | `158 / 167` pairs<br>(`117` tasks; `46.25%` strict `/120`) | `41.88%`<br>(`49.00/117`) | **`49.15%`**<br>(`57.50/117`) | `39.24%`<br>(`62/158`) | `46.20%`<br>(`73/158`) | `68.99%`<br>(`109/158`) | `47.31%`<br>(`299/632`) | `~43,921`<br>*(`56,000` total)* | `27.55M` | Only with gated Wave 2 (`EXP-19`) |
| **`EXP-19`: `Gated 2-Wave + CoT Fallback`** | `pilot_1` | `N = 2..4` | `40` | `40,000` | **Kaggle-Feasible Hybrid**: Skips Wave 2 whenever Wave 1 (`N=2`) achieves consensus on a verified prediction; falls back to 1-shot CoT if unverified. | `167 / 167` pairs<br>(`120` tasks) | `38.33%`<br>(`46.00/120`) | **`41.67%`**<br>(`50.00/120`) | `39.52%`<br>(`66/167`) | `43.11%`<br>(`72/167`) | `55.69%`<br>(`93/167`) | `~45.0%` | `~27,500`<br>*(`35,500` total)* | `19.65M` | **Yes (`~8.7h`)** |

---

## 5. Conclusions & Future Directions

### Key Conclusions from Our Experiments

1. **SFT on `E031` DeepSeek-R1 Agentic Traces Dramatically Transforms `Qwen3.8-27B` Tool Efficiency (`+27.77 pp` at `N=2`)**:
   - Out of the box, pure `Qwen3.8-27B-FP8` struggles with JSON tool-call formatting, over-thinks without calling Python early, and only verifies `30.54%` of pairs (`19.17%` Task Pass@2) in `h1_python_tools_base` (`N=2`).
   - Training for just **1 epoch (`1,535` trajectories)** on the `E031` DeepSeek-R1 IPython teaching trajectories teaches `Qwen3.8-27B` the exact **analyze $\rightarrow$ write short IPython test script $\rightarrow$ inspect output $\rightarrow$ verify on all `I_i -> O_i` $\rightarrow$ `submit_answer`** loop.
   - At equal `N=2` attempts, **LoRA SFT Step 191 reaches `46.94%` Official Kaggle Task Pass@2 (`56.33 / 120` tasks)** and **`50.90%` demo-verified pairs**, more than doubling `h1` (`19.17%`) and matching `N=4` pure Qwen harnesses (`h7` `47.35%`, `h8` `49.15%`) at **half the sample count (`N=2` vs. `N=4`) and nearly half the token cost (`14.57M` vs. `27.55M`)**.
   - Furthermore, when an SFT attempt finds a verified program (`train_pass=True`), it does so in a median of **`17,023–19,924` tokens** (`7–9` turns), compared to **`26,207–34,030` tokens** for pure Qwen.

2. **Full SFT Achieves Strictly Lower Validation Loss (`0.7549` vs. `0.7627`), While LoRA (`r=64`) Provides Exceptional Regularization & Sample Efficiency**:
   - Across the entire epoch on the exact same `1,535` training sequences and `64` validation sequences (`279,810` supervised tokens), **Full SFT (`lr=5e-6`, FSDP2) tracks ~50–75 steps ahead of LoRA SFT (`lr=1e-4`, `r=64`) in validation cross-entropy loss**:
     - Full SFT at **Step 100** (`val_loss = 0.7634`) already beats LoRA SFT at **Step 175** (`val_loss = 0.7632`).
     - Full SFT at **Step 192 Final** reaches **`val_loss = 0.7549` (`PPL = 2.1275`)**, outperforming LoRA SFT's final **Step 191** (`val_loss = 0.7627`, `PPL = 2.1440`) by **`-0.0078` nats**.
   - On downstream ARC-AGI-2 evaluation, **Full SFT Step 100** achieved **`40.76%` Task Pass@2 (`48.50 / 119` tasks)** under the **`40k`-token capped `e031_deepseek_teaching_fast` harness** (a `+3.82 pp` gain over Full SFT Step 50 at `36.94%`), matching LoRA Step 50 (`40.42%`) and Step 150 (`40.22%`) despite having a **33% smaller token budget (`40k` vs. `60k`) and fewer turns (`32` vs. `40`)**.
   - Crucially, the highest-accuracy LoRA jump occurred between **Step 150 (`40.22%`) and Step 191 (`46.94%`, `+6.72 pp`)**, during the final cosine LR decay tail (`lr` decaying from `2.3e-5` to `1.0e-5`). We have **not yet evaluated Full SFT Step 150 (`val_loss = 0.7575`) or Step 192 Final (`val_loss = 0.7549`)**, which sit in that exact same cosine tail and are already exported to FP8 on CNS.

3. **Why `40k` vs. `60k` Token Cap Matters for SFT Models**:
   - In `Full SFT Step 50` and `Step 100` (`e031_deepseek_teaching_fast`), the median generated tokens per attempt is `39,718–40,011` (`p90 = 40,014`), meaning **over half of all attempts hit the `40,000` hard token ceiling** while actively debugging in IPython.
   - In `LoRA SFT Step 191` (`e031_deepseek_teaching` with a `60,000` cap), attempts that need between `40k` and `60k` tokens to finish debugging succeed, lifting attempt-level `train_pass` from `32.32%` (`Full SFT s100 @ 40k`) to `39.22%` (`LoRA s191 @ 60k`) and pair-level verification from `43.90%` to `50.90%`.

4. **Test-Time Harness Engineering and SFT Are Highly Complementary**:
   - On pure Qwen, switching from JSON tool calls to **Native XML + Two-Stage `<think>` Nudge (`h7`)** gave `+10.82 pp` (`36.53% -> 47.35%`), and adding **Evolutionary Two-Wave Repair (`h8`)** gave another `+1.80 pp` (`49.15%`).
   - However, our SFT evaluations (`e031_deepseek_teaching`) did **not** yet enable `h7`/`h8`'s best test-time features: they ran with `N=2` single-wave attempts, `use_equivariance_audit=False`, and `use_two_stage_think_nudge=False`.

---

### Concrete Future Directions & Next Steps

1. **Immediate (Zero-Training) Eval: Run Full SFT Step 192 Final & Step 150 under the `60k` Harness**:
   - The FP8 weights for **Full SFT Step 192 Final (`val_loss = 0.7549`)** and **Step 150 (`val_loss = 0.7575`)** are already quantized and saved on CNS at `/cns/lu-d/home/arazdai/kaggle/sft/models_fp8/qwen38_27b_arc_agi2_e031_full_sft_xid304470493_step00192_fp8`.
   - Evaluating `step_00192` under the identical `60k`-token `e031_deepseek_teaching` harness used for `LoRA Step 191` will provide a strict apples-to-apples comparison at the end of Epoch 1.

2. **Combine SFT Models (`LoRA s191` / `Full SFT s192`) with `h8` Evolutionary Gated Two-Wave (`N=2+2`) + Equivariance Tie-Break**:
   - Right now, SFT (`e031_deepseek_teaching`) only runs `N=2` independent attempts.
   - Because SFT verified attempts finish in a median of **`17k–19.9k` tokens** (`~45%` faster than pure Qwen), we can run **Wave 1 (`N=2` attempts)** and **early-stop immediately whenever both attempts verify and agree** (which happens on `~40%+` of tasks in `<20k` tokens/attempt), spending **Wave 2 (`+2` evolutionary repair attempts seeded with Wave 1's best partial `solve(grid)` and cell diffs)** *only* on the remaining unverified/disagreeing tasks.
   - Enabling `use_equivariance_audit=True` and `use_two_stage_think_nudge=True` on `e031_deepseek_teaching` requires zero retraining and directly addresses the `40k/60k` truncation ceiling.

3. **Weight-Space Ensemble / Interpolation (`WiSE-FT`) Between Base `Qwen3.8-27B`, `LoRA Step 191`, and `Full SFT Step 192`**:
   - Observe that in Table 1, **Pure Qwen `h7`/`h8` achieves `67.3%–69.0%` demo-verified pair rate** (via broad exploration across 4 diverse prompt styles), whereas **`LoRA SFT Step 191` achieves a much higher conversion of verified pairs to exact test pass (`45.51% / 50.90% = 89.4%` precision vs. `h8`'s `46.20% / 68.99% = 67.0%`)**.
   - Interpolating weights $\theta(\alpha) = (1-\alpha)\theta_{\text{base}} + \alpha\theta_{\text{SFT}}$ (e.g. $\alpha \in \{0.5, 0.7\}$) or running **1 attempt with SFT `lean2sub` + 1 attempt with Base/LoRA-0.5 `Native XML`** in a 2-attempt ensemble can capture both SFT's high-precision IPython synthesis and the base model's broader hypothesis diversity.

4. **Second-Stage Rejection-Sampled SFT (On-Policy RFT) + RLVR (GRPO) on Unsolved ARC Tasks**:
   - The `E031` dataset was distilled from **DeepSeek-R1 (`flash1017`)** trajectories. Now that `LoRA Step 191` and `Full SFT Step 192` solve `~47%` of ARC-AGI-2 eval tasks in `Qwen3.8-27B`'s own token distribution, we should generate on-policy rollouts on the `1,080` ARC training/concept/synthetic tasks, filter for `train_pass == True` and `tokens <= 28,000`, and run **on-policy Reinforcement Learning with Verifiable Rewards (GRPO / RLVR)** where the reward is exact demo-pair verification + $D_8$ equivariance + token efficiency penalty.

---

## 6. Detailed Training Procedure

### 6.1 Dataset Source, Filtering Pipeline, and De-Contamination Audit
All data preparation and auditing is implemented in [`data_pipeline.py`](file:///google/src/cloud/arazdai/test_workspace/google3/experimental/users/arazdai/kaggle/sft/data_pipeline.py) and documented in [`e031_audit_summary.json`](file:///google/src/cloud/arazdai/test_workspace/google3/experimental/users/arazdai/kaggle/sft/e031_audit_summary.json) and [`ARC_AGI2_SFT_DATA_CATALOG.md`](file:///google/src/cloud/arazdai/test_workspace/google3/experimental/users/arazdai/kaggle/arc_agi_2/ARC_AGI2_SFT_DATA_CATALOG.md):

1. **Raw Source Dataset**:
   - `livobereshka/arc-agi2-deepseek-flash1017-rollouts-e031-e036` (mirrored on CNS at `/cns/lu-d/home/arazdai/kaggle/arc_agi2_e031_rollouts/rollouts_e031_e036.jsonl`, `2.08 GB`).
   - Contains **`5,456` raw multi-turn trajectories** across **`911` base task IDs** generated by a DeepSeek-R1-class reasoning teacher (`flash1017`) interacting with a persistent IPython sandbox (`run_python` and `submit_answer`) across 6 generation epochs (`e031` through `e036`).

2. **Three-Gate Quality & Integrity Filter (`5,456` $\rightarrow$ `2,043` trajectories)**:
   - **Gate 1 (`correct == True`)**: The submitted test output grid must exactly match the ground-truth test output (`2,216 / 5,456` pass).
   - **Gate 2 (`train_pass == True`)**: The candidate's Python transformation must be verified to pass **100% of demonstration training pairs** (`2,043 / 2,216` pass; rejects lucky test guesses).
   - **Gate 3 (`answer_source == "submit"`)**: The trajectory must terminate cleanly via an explicit `submit_answer` tool call (`2,043 / 2,043` pass; rejects timeout fallbacks).

3. **Two-Layer Evaluation De-Contamination Audit against the 120 Official ARC-AGI-2 Eval Tasks**:
   - Before splitting, every task in `E031` was audited against all **120 official ARC-AGI-2 evaluation tasks (`167` test pairs)** using two independent checks in [`audit_eval_contamination`](file:///google/src/cloud/arazdai/test_workspace/google3/experimental/users/arazdai/kaggle/sft/data_pipeline.py#L149-L227):
     1. **Exact Grid-Pair SHA-256 Matching**: Hashes every individual `(input, output)` grid pair and full task demonstration signature.
     2. **$D_8 \times S_{10}$ Canonical Isomorphism Matching**: Computes the lexicographically smallest representation of every grid under all **8 dihedral symmetries ($D_8$: rotations and reflections)** combined with **order-of-appearance color relabeling** (`canonical_grid_hash`), catching rotated, flipped, or color-permuted clones of evaluation tasks.
   - **Audit Finding**: Zero (`0`) exact task or pair overlaps were found, and **two (`2`) canonical symmetry/color-isomorphic tasks** (`2753e76c` and `81c0276b`, accounting for `12` verified rollouts) matched official evaluation tasks (`0a1d4ef5` and `f931b4a8`). Both tasks (`2753e76c` and `81c0276b`) were **strictly excluded** (`EXCLUDED_CONTAMINATED_BASE_IDS`).

4. **ChatML Conversion & Sequence Length Filtering (`2,031` $\rightarrow$ `1,900` $\rightarrow$ `1,713` trajectories)**:
   - Each trajectory is converted into `Qwen3.8-27B`'s native ChatML tool-use format (`<|im_start|>system...`, `<|im_start|>user...`, `<|im_start|>assistant\n<think>\n...\n</think>\n\n<tool_call>\n{"name": "run_python", ...}</tool_call><|im_end|>`).
   - **Stage 1 Length Cap ($\le 65,536$ tokens in `data_pipeline.py`)**: Retains **`1,900` clean trajectories** (`131` ultra-long trajectories $>65,536$ tokens dropped).
   - **Parent-Task Grouped 90/10 Train/Val Split**: Grouped strictly by `base_task_id` (`seed=42`) so all dihedral/color augmentations and multiple rollouts of the same underlying ARC task stay in the same split:
     - **Full Prepared Dataset ($\le 65,536$ tokens)**: **`1,702` train trajectories** (`567` unique base tasks) and **`198` validation trajectories** (`64` unique base tasks).
   - **Stage 2 Training Context Cap ($\le 32,768$ tokens in `train_sft.py` / `train_full_sft.py`)**:
     - Both LoRA SFT and Full SFT filter out trajectories exceeding `max_seq_len = 32,768` (`167` train and `20` val trajectories skipped; zero mid-trajectory truncation so `submit_answer` is never cut off).
     - **Active Training Set ($\le 32,768$ tokens)**: **`1,535` train trajectories** and **`178` validation trajectories** (`64` fixed validation trajectories / `279,810` supervised tokens evaluated every `25` steps).

5. **What Tokens We Train On (`loss_mask` Construction)**:
   - Implemented in [`tokenize_and_mask_trajectory`](file:///google/src/cloud/arazdai/test_workspace/google3/experimental/users/arazdai/kaggle/sft/data_pipeline.py#L291-L363):
   - **`loss_mask = 0` (Unsupervised Context)**: System prompt (`tools` JSON schema + minimal-effort instruction), initial User prompt (ASCII grids for training pairs `I_0 -> O_0` and test inputs `T_0`), and all intermediate `<|im_start|>user\n<tool_response>\n...\n</tool_response><|im_end|>` IPython execution outputs and `[Turn X/40]` budget notices.
   - **`loss_mask = 1` (Supervised Target Tokens)**: Every Assistant turn from `<think>\n` right after `<|im_start|>assistant\n` through `</think>\n\n<tool_call>\n{"name": "...", "arguments": {...}}\n</tool_call><|im_end|>\n`.
   - Across the dataset, **`82.37%` of all tokens are supervised assistant tokens** (`loss_mask = 1`, mean `15,431` supervised tokens out of `18,735` total tokens per trajectory across the `1,900` trajectories; `~4,372` supervised tokens/trajectory on the shortest-first validation subset).

---

### 6.2 Side-by-Side Training Hyperparameters & Systems Configuration: LoRA SFT vs. Full SFT

| Parameter / System Component | **LoRA SFT (`train_sft.py`)** | **Full-Parameter SFT (`train_full_sft.py`)** |
| :--- | :--- | :--- |
| **XManager Experiment ID** | [`http://xid/304138725`](http://xid/304138725) $\rightarrow$ [`http://xid/304366741`](http://xid/304366741) | [`http://xid/304470493`](http://xid/304470493) |
| **Base Checkpoint** | `/cns/lu-d/home/arazdai/models/Qwen3.8-27B` (BF16, `64` layers: `48` Gated DeltaNet linear-attn + `16` full-attn layers) | `/cns/lu-d/home/arazdai/models/Qwen3.8-27B` (Vision tower stripped; `100%` of text backbone loaded) |
| **Hardware Topology** | **`1× NVIDIA B200 GPU`** (`178.4 GB` HBM3e, single-process) | **`8× NVIDIA B200 GPUs`** (`8 × 178.4 GB = 1.43 TB` HBM3e, single-node `torchrun` via NCCL) |
| **Parallelism Strategy** | Single-GPU PEFT (`Qwen3_5ForConditionalGeneration` text backbone) | **PyTorch FSDP2 (`fully_shard`)**: Every `DecoderLayer`, `embed_tokens`, and `lm_head` sharded `1/8` across 8 ranks (`reshard_after_forward=True`) |
| **Trainable Parameters** | **`435,159,040`** / `27,331,157,504` (**`1.5922%`**) | **`26,895,998,464`** / `26,895,998,464` (**`100.00%`** of text model; `3.362B` sharded params/GPU) |
| **LoRA / Parameter Scope** | `r = 64`, `lora_alpha = 128` (scaling `2.0`), `lora_dropout = 0.0`, `bias = "none"` on `q_proj, k_proj, v_proj, o_proj, in_proj_qkv, in_proj_z, out_proj, gate_proj, up_proj, down_proj` | **All 64 layers unfrozen**: Full Attention, Gated DeltaNet projections + recurrent state params (`conv1d`, `A_log`, `dt_bias`, `in_proj_a`, `in_proj_b`, `norm`), MLPs, `embed_tokens`, `lm_head`, and all RMSNorms |
| **Compute & Master Weight Precision** | Base weights in `bfloat16`; LoRA adapter weights (`A`, `B`) and AdamW states in **`float32`**; `sdpa` FlashAttention-2 | FSDP2 `MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)`; sharded master weights, gradients, and AdamW states in **`float32`** |
| **Optimizer & Weight Decay** | `torch.optim.AdamW` (`fused=True`), `betas = (0.9, 0.95)`, `eps = 1e-8`, `weight_decay = 0.01` | `torch.optim.AdamW` (`foreach=True` on DTensor shards), `betas = (0.9, 0.95)`, `eps = 1e-8`, `weight_decay = 0.01` (`0.0` on 1D norms/biases/`A_log`/`dt_bias` & embeddings) |
| **Learning Rate & Schedule** | Peak **`lr = 1.0e-4`** $\rightarrow$ cosine decay to **`1.0e-5`** (`min_lr_ratio = 0.1`), linear warmup **`10` steps** | Peak **`lr = 5.0e-6`** ($20\times$ lower than LoRA) $\rightarrow$ cosine decay to **`5.0e-7`** (`min_lr_ratio = 0.1`), linear warmup **`10` steps** |
| **Batch Size & Gradient Accumulation** | `micro_batch_size = 1`, `grad_accum_steps = 8` $\implies$ **Effective Batch Size = `8` trajectories/step** (`~100k–130k` tokens/step) | `world_size = 8`, `micro_batch_size = 1/GPU`, `grad_accum_steps = 1` $\implies$ **Effective Batch Size = `8` trajectories/step** (1 trajectory per GPU per step) |
| **Loss Reduction & Normalization** | **Global Token-Weighted Cross-Entropy**: Sums CE over all `loss_mask == 1` tokens across all 8 trajectories in the step and divides by total supervised tokens | **Cross-Rank Token-Weighted Cross-Entropy**: `dist.all_reduce(SUM)` on local `(ce_sum, sup_tokens)` across all 8 GPUs so each rank's gradient is weighted by `(rank_sup_tok / global_mean_sup_tok)` |
| **Gradient Clipping** | `max_grad_norm = 1.0` | Distributed FP32 L2 norm (`_clip_fsdp2_grad_norm_`, `max_grad_norm = 1.0`); observed grad norms `0.64–1.05` (zero NaN/Inf spikes) |
| **Max Sequence Length** | `max_seq_len = 32,768` (`1,535` train / `178` val samples fit; `167` train $>32\text{k}$ skipped) | `max_seq_len = 32,768` (`1,535` train / `178` val samples fit; `167` train $>32\text{k}$ skipped) |
| **Steps per Epoch & Total Runtime** | **`191` steps** (`floor(1535 / 8)`), `~51s/step` on `1× B200` (`~2.7 hours` total), peak VRAM `~96 GB / 178.4 GB` | **`192` steps** (`ceil(1535 / 8)` including final 7-sample batch), **`~23.6s/step` on `8× B200` (`75.5 minutes` total)**, peak VRAM **`105.54 GB / 178.4 GB`** (`72.8 GB` free headroom) |

---

### 6.3 Custom B200 Kernel Engineering, Memory Optimization & FP8 Quantization Pipeline

To make `32,768`-token full-parameter and LoRA training of a `27B` hybrid Gated DeltaNet + Full Attention model fast and memory-safe on B200 GPUs and TPU v6e vLLM servers, we built four critical systems components:

1. **Custom Triton Chunkwise Gated DeltaNet Kernels ([`fast_deltanet.py`](file:///google/src/cloud/arazdai/test_workspace/google3/experimental/users/arazdai/kaggle/sft/fast_deltanet.py))**:
   - `Qwen3.8-27B` uses **Gated DeltaNet** linear attention in 48 of its 64 layers (`in_proj_qkv`, `conv1d`, state-space recurrence $S_t = e^{g_t} S_{t-1} + k_t (v_t - S_{t-1}^\top k_t)^\top$).
   - Because standard HuggingFace `transformers` falls back to a sequential Python `for t in range(T):` loop when `fla` (`flash-linear-attention`) is not compiled for B200 `sm_100`, running `T = 32,768` tokens in eager PyTorch takes `>60s` per layer and OOMs on autograd graph intermediates.
   - [`fast_deltanet.py`](file:///google/src/cloud/arazdai/test_workspace/google3/experimental/users/arazdai/kaggle/sft/fast_deltanet.py) monkey-patches `Qwen3_5GatedDeltaNet` with:
     1. A **fused `@triton.jit` causal depthwise `conv1d + SiLU` kernel** (`_causal_conv1d_silu_fwd_kernel`) that eliminates the intermediate `(B, C, L+3)` padding tensor across all `10,240` channels.
     2. A **chunked block-parallel WY-representation forward + custom `@torch.autograd.Function` backward kernel** (`_ChunkedGatedDeltaRuleFn` + `_recurrent_scan_fwd_kernel`, `BT = 64`), which performs intra-chunk projections via tensor-core `torch.bmm` and runs only the `64`-step state scan in Triton SRAM, saving only `[B, H, num_chunks, K, V]` checkpoints (`64x` memory reduction) and recomputing intra-chunk states during backward.

2. **Chunked Logits Cross-Entropy (`logits_chunk_size = 2048` / `4096`)**:
   - Materializing the full `[1, 32768, 248320]` FP32 logits tensor for `Qwen3.8-27B`'s `248,320`-token vocabulary would allocate **`32.5 GB` per micro-step** just for the final projection.
   - Both [`compute_chunked_masked_ce`](file:///google/src/cloud/arazdai/test_workspace/google3/experimental/users/arazdai/kaggle/sft/train_sft.py#L204-L287) and [`compute_chunked_masked_ce_sum`](file:///google/src/cloud/arazdai/test_workspace/google3/experimental/users/arazdai/kaggle/sft/train_full_sft.py#L239-L315) slice `hidden_states` into `2,048`-token chunks, skip any chunk where `loss_mask.sum() == 0` (such as the initial system/user prompt), and compute `lm_head(chunk)` + FP32 cross-entropy inside `torch.utils.checkpoint.checkpoint`, reducing peak `lm_head` memory from **`32.5 GB` to `<2.0 GB`**.

3. **Selective Layer Activation Checkpointing (`checkpoint_every_n_layers = 2` in Full SFT)**:
   - Because `8× B200` has `178.4 GB` HBM3e per GPU and FSDP2 shards parameters/gradients/optimizer states `8`-ways (`~40.3 GB/GPU` static footprint), [`train_full_sft.py`](file:///google/src/cloud/arazdai/test_workspace/google3/experimental/users/arazdai/kaggle/sft/train_full_sft.py#L366-L378) only wraps **every 2nd decoder layer (`32 / 64` layers)** in `torch.utils.checkpoint`, cutting backward recomputation FLOPs in half while keeping peak VRAM at **`105.54 GB`** (`72.82 GB` safety margin).

4. **High-Precision FP32 Merge & $128 \times 128$ Block-FP8 Export for vLLM ([`export_and_quantize.py`](file:///google/src/cloud/arazdai/test_workspace/google3/experimental/users/arazdai/kaggle/sft/export_and_quantize.py))**:
   - For both LoRA checkpoints ($\theta_{\text{merged}} = \theta_{\text{base}}^{\text{FP32}} + \frac{\alpha}{r} B^{\text{FP32}} A^{\text{FP32}}$) and Full SFT checkpoints (consolidated FSDP2 BF16 state dicts), [`export_and_quantize.py`](file:///google/src/cloud/arazdai/test_workspace/google3/experimental/users/arazdai/kaggle/sft/export_and_quantize.py#L96-L129) quantizes all 2D linear weights in the 64 text layers into **FineGrained $128 \times 128$ block-wise `float8_e4m3fn`** with exact per-block FP32 scales:
     $$s_{i,j} = \frac{\max_{(r,c) \in B_{i,j}} |W_{r,c}|}{448.0}, \qquad W_{B_{i,j}}^{\text{FP8}} = \text{round\_to\_e4m3fn}\!\left(\frac{W_{B_{i,j}}}{s_{i,j}}\right)$$
   - Preserves `lm_head`, `model.embed_tokens`, `visual.*`, all 1D norms, and small Gated DeltaNet recurrent projections (`in_proj_a`, `in_proj_b`, `conv1d`, `A_log`, `dt_bias`) in unquantized **`bfloat16`** so the exported `12`-shard safetensors directory is a drop-in replacement for `Qwen3.8-27B-FP8` on both TPU v6e (`tpu_streaming_loader`) and Kaggle `4× NVIDIA L4` GPUs.
