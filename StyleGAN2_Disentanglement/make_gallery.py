#!/usr/bin/env python3
"""
Generate gallery figures for the report.
Reads Exp25 + Exp26 results and makes composites for embedding in the report.
"""
import json, cv2, numpy as np
from pathlib import Path

REPORT_FIG = Path(__file__).parent / "figures"
EXP25 = Path("/mnt/data0/naimul/ExperimentRoom/Experiment25")
EXP26 = Path("/mnt/data0/naimul/ExperimentRoom/Experiment26")

def load(path, size=None):
    img = cv2.imread(str(path))
    if img is None:
        img = np.full((size or 256, size or 256, 3), 60, np.uint8)
    if size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LANCZOS4)
    return img

def label_bar(text_list, cell_w, cell_h=28, bg=(30,30,30), fg=(220,220,220)):
    bar = np.full((cell_h, cell_w * len(text_list), 3), bg[0], np.uint8)
    bar[:] = bg
    for i, t in enumerate(text_list):
        cv2.putText(bar, t, (i*cell_w + 6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, fg, 1, cv2.LINE_AA)
    return bar

def metric_bar(pairs_info, cell_w, cell_h=22):
    bar = np.full((cell_h, cell_w * len(pairs_info), 3), 20, np.uint8)
    for i, r in enumerate(pairs_info):
        color = (60,180,60) if r["id_sim_source"] >= 0.95 else (60,130,200)
        txt = f"s={r['id_sim_source']:.3f} l={r['id_sim_target']:.3f}"
        cv2.putText(bar, txt, (i*cell_w + 3, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv2.LINE_AA)
    return bar

# ── 1. Exp26 swap gallery — 10 selected pairs (best 3, mid 4, worst 3) ────────
with open(EXP26/"results_real/metrics/real_metrics.json") as f:
    rows26 = json.load(f)

sl = sorted(rows26, key=lambda r: r["id_sim_target"])
sel = sl[:3] + sl[len(sl)//2-2:len(sl)//2+2] + sl[-3:]
lbls = ["Best#1","Best#2","Best#3","Mid#1","Mid#2","Mid#3","Mid#4","Worst#1","Worst#2","Worst#3"]

CELL = 224
cols = []
for r in sel:
    pid = r["pair"]
    src  = load(EXP26/"results_real/aligned"/f"{pid}_src.png",  CELL)
    tgt  = load(EXP26/"results_real/aligned"/f"{pid}_tgt.png",  CELL)
    swap = load(EXP26/"results_real/final"/f"{pid}_swap.png",    CELL)
    diff = np.abs(tgt.astype(np.float32) - swap.astype(np.float32)).mean(axis=2)
    diff_vis = cv2.applyColorMap(np.clip(diff*5,0,255).astype(np.uint8), cv2.COLORMAP_HOT)
    col_img = np.vstack([src, tgt, swap, diff_vis])
    cols.append(col_img)

grid = np.hstack(cols)
top_bar = label_bar(["Source","Target","FSGeomNet","Diff×5"], CELL)
top_bar_full = np.tile(top_bar, (1, len(sel)//4 + 1, 1))[:, :grid.shape[1]]

# Per-column metric bar
mbar = np.full((22, grid.shape[1], 3), 20, np.uint8)
for i, r in enumerate(sel):
    lk_color = (60,200,60) if r["id_sim_target"] < 0.08 else (60,130,200) if r["id_sim_target"] < 0.15 else (60,60,200)
    txt = f"s={r['id_sim_source']:.3f} lk={r['id_sim_target']:.3f}"
    cv2.putText(mbar, txt, (i*CELL+4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, lk_color, 1)

# Header row
hdr = np.full((30, grid.shape[1], 3), 15, np.uint8)
for i, lbl in enumerate(lbls):
    cv2.putText(hdr, lbl, (i*CELL+4, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

final = np.vstack([top_bar_full, hdr, grid, mbar])
out = REPORT_FIG/"exp26/gallery_swap_10pairs.png"
cv2.imwrite(str(out), final)
print(f"saved {out.name}  {final.shape}")

# ── 2. Exp26 full 30-pair compact strip (src | swap only) ─────────────────────
all_rows = sorted(rows26, key=lambda r: int(r["pair"].replace("pair","")))
THUMB = 128
strips = []
for r in all_rows:
    pid  = r["pair"]
    src  = load(EXP26/"results_real/aligned"/f"{pid}_src.png",  THUMB)
    swap = load(EXP26/"results_real/final"/f"{pid}_swap.png",    THUMB)
    pair_strip = np.vstack([src, swap])
    # colour-code border by leakage
    col = (0,180,0) if r["id_sim_target"] < 0.08 else (0,150,200) if r["id_sim_target"] < 0.15 else (0,60,200)
    cv2.rectangle(pair_strip, (0,0), (THUMB-1, THUMB*2-1), col, 2)
    strips.append(pair_strip)

# Arrange in 6 rows × 5 cols
ROWS, COLS = 6, 5
rows_imgs = []
for r in range(ROWS):
    row_strips = strips[r*COLS:(r+1)*COLS]
    row_img = np.hstack(row_strips)
    rows_imgs.append(row_img)
grid30 = np.vstack(rows_imgs)

# Legend
legend = np.full((28, grid30.shape[1], 3), 20, np.uint8)
cv2.putText(legend, "green border: leak<0.08  blue: leak<0.15  dark: leak>=0.15    top=source  bottom=swap",
            (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200,200,200), 1)
title = np.full((32, grid30.shape[1], 3), 15, np.uint8)
cv2.putText(title, "Experiment 26 — All 30 Real-to-Real Swaps (src top, swap bottom) — sorted by pair index",
            (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230,230,230), 1)

strip30 = np.vstack([title, grid30, legend])
out2 = REPORT_FIG/"exp26/all_30_pairs_strip.png"
cv2.imwrite(str(out2), strip30)
print(f"saved {out2.name}  {strip30.shape}")

# ── 3. Side-by-side Exp25 vs Exp26 comparison (same 10 target images) ─────────
with open(EXP25/"results/metrics/inference_metrics.json") as f:
    rows25 = json.load(f)

# find pairs that use the same target file as Exp26 pairs
tgt_to_e25 = {r["target"]: r for r in rows25}
tgt_to_e26 = {r["target"]: r for r in rows26}
shared_tgts = [t for t in tgt_to_e26 if t in tgt_to_e25][:8]

if len(shared_tgts) >= 4:
    CELL2 = 200
    cols_cmp = []
    for tgt in shared_tgts[:8]:
        r25 = tgt_to_e25[tgt]; r26 = tgt_to_e26[tgt]
        tgt_img  = load(EXP26/"real_images/targets"/tgt,          CELL2)
        swap25   = load(EXP25/"results/final"/f"{r25['pair']}_fsgeomnet.png", CELL2)
        swap26   = load(EXP26/"results_real/final"/f"{r26['pair']}_swap.png",  CELL2)
        col_img  = np.vstack([tgt_img, swap25, swap26])
        cols_cmp.append(col_img)
    cmp_grid = np.hstack(cols_cmp)
    lrow = label_bar(["Target","Exp25 swap","Exp26 swap"], CELL2)
    lrow_full = np.tile(lrow, (1, len(shared_tgts)//3+1, 1))[:, :cmp_grid.shape[1]]
    cmp_final = np.vstack([lrow_full, cmp_grid])
    out3 = REPORT_FIG/"exp26/exp25_vs_exp26_comparison.png"
    cv2.imwrite(str(out3), cmp_final)
    print(f"saved {out3.name}  {cmp_final.shape}")
else:
    print("not enough shared targets for comparison panel")

# ── 4. W-space layer swap visualisation from existing outputs ──────────────────
# Copy the pair swaps from the W-space experiment
wswap_src = Path("/mnt/data0/naimul/StyleGAN2/outputs/layer_swap")
for fn in ["pair00_swaps.jpg","pair01_swaps.jpg"]:
    src_p = wswap_src / fn
    if src_p.exists():
        import shutil
        shutil.copy(str(src_p), str(REPORT_FIG / fn))
        print(f"copied {fn}")

print("\nAll gallery figures generated.")
