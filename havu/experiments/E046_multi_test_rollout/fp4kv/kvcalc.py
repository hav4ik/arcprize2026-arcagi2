"""Re-derive the fork's V4.1 KV pool from its spec code (no vLLM import). Mirrors
_get_packed_kv_cache_groups (kv_cache_utils.py:2056-2205) for the E128 config."""
from math import ceil
GiB = 2**30
MLEN, BS, INFLIGHT = 163840, 128, 2 * 2048          # max-model-len, block, async 2 x max_num_batched_tokens
def pool(swa_row=528, glob_row=288, idx_row=68, n_swa_layers=40, avail=20983 * 113920):
    mla = [BS // 2 * glob_row] * 3 + [BS * glob_row] + [BS // 2 * idx_row] * 3 + [BS * idx_row]  # layers 2,8,14 | 20
    bpb = sum(mla)                                   # MLA group = widest group = pool bytes per block
    swa_page = 32 * swa_row                          # SWA block 32 tokens
    per_blk = max(bpb // swa_page, 1); n_swa_groups = ceil(n_swa_layers / per_blk)
    ring_page = 8 * 1024 * 4                         # CircularBufferSpec, 3 layers, 1 block/request
    swa_adm = ceil((128 - 1 + INFLIGHT) / 32) + 1    # SlidingWindowSpec.max_admission_blocks_per_request
    bpr = MLEN // BS + n_swa_groups * swa_adm + 1
    nb = avail // bpb
    return dict(bytes_per_block=bpb, B_per_ctx_token=bpb / BS, swa_groups=n_swa_groups, blocks_per_req=bpr,
                num_blocks=nb, reported_tokens=int(nb / bpr * MLEN), raw_ctx_slots=nb * BS,
                decode_blocks_per_stream=n_swa_groups * 5 + 1)
base = pool()
print("today      ", base)
print("check 1,554,183:", base["reported_tokens"] == 1554183, " pool GiB", 20983 * 113920 / GiB)
print("SWA FP4    ", pool(swa_row=288))
for util in (0.938, 0.945):   # Kaggle L4 22.03 GiB; min-rank KV 2.2262 GiB at 0.945
    av = int((2.2262 - (0.945 - util) * 22.03) * GiB)
    print(util, "today", pool(avail=av)["num_blocks"], pool(avail=av)["reported_tokens"], " swafp4", pool(swa_row=288, avail=av)["reported_tokens"])
# LRU churn: cached blocks appended per 128 processed tokens under each SWA caching policy (7 SWA groups, need=4 blocks)
for name, frac in (("interval128(dense)", 1.0), ("interval256", .5), ("interval512", .25), ("interval1024", .125), ("bounded replay", 0)):
    print(name, 1 + 7 * 4 * frac)
