import os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

# reuse load_features / FEATURE_COLS / SUBJECT_RE from tsne_features.py
_ns = {}
exec(open("/userdata/msharma/audio-speech-analysis/tsne_features.py").read().split('if __name__ == "__main__"')[0], _ns)
load_features = _ns["load_features"]; FEATURE_COLS = _ns["FEATURE_COLS"]; SUBJECT_RE = _ns["SUBJECT_RE"]; UD = _ns["UD"]

PARENT = f"{UD}/sub-RCS08-stage-1_audio-audiotype_preproc_spectral_gating_100_percent_metadata_and_plots"
PREFIX = "sub-RCS08_stage-1_audio-audiotype_preproc_spectral_gating_100_percent"
SCORES = ["pain_nrs_s1_daily", "pain_vas_s1_daily", "relief_vas_s1_daily", "mpq_s1_daily", "mood_vas_s1_weekly"]

feats = load_features(PARENT, PREFIX, SUBJECT_RE)
feats["id"] = feats["id"].astype(str)
merged = pd.read_csv(f"{PARENT}/{PREFIX}_pain_mood_correlation/merged_scores_features.csv")
scores = merged[["audio_id", "audio_uploaded"] + SCORES].rename(columns={"audio_id": "id"})
scores["id"] = scores["id"].astype(str)
scores["ts"] = pd.to_datetime(scores["audio_uploaded"], errors="coerce")
df = feats.merge(scores, on="id", how="inner").dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

X = df.reindex(columns=FEATURE_COLS).apply(pd.to_numeric, errors="coerce").to_numpy(float).copy()
col_med = np.nanmedian(X, axis=0)
inds = np.where(np.isnan(X)); X[inds] = np.take(col_med, inds[1])
Xs = StandardScaler().fit_transform(X)

n = len(df)
perplexity = float(min(30, max(5, n // 4)))
params = dict(n_components=2, perplexity=perplexity, learning_rate="auto",
              init="pca", metric="euclidean", max_iter=1000, random_state=42)
emb = TSNE(**params).fit_transform(Xs)
df["tsne_x"], df["tsne_y"] = emb[:, 0], emb[:, 1]
df["day_number"] = (df["ts"] - df["ts"].min()).dt.total_seconds() / 86400.0

md = f"{PARENT}/{PREFIX}_tsne_metadata"; pl = f"{PARENT}/{PREFIX}_tsne_plots"
os.makedirs(md, exist_ok=True); os.makedirs(pl, exist_ok=True)
df[["id", "ts", "day_number", "tsne_x", "tsne_y"] + SCORES].to_csv(f"{md}/tsne_embedding.csv", index=False)
json.dump({**params, "perplexity": perplexity, "n_samples": n, "n_features": len(FEATURE_COLS),
           "feature_columns": FEATURE_COLS, "preprocessing": "median-impute NaN then StandardScaler (z-score)",
           "sklearn_TSNE": "sklearn.manifold.TSNE"}, open(f"{md}/tsne_params.json", "w"), indent=2, default=str)

panels = [("day_number", "Day number", "viridis"),
          ("pain_nrs_s1_daily", "Pain NRS (daily)", "viridis_r"),
          ("pain_vas_s1_daily", "Pain VAS (daily)", "viridis_r"),
          ("mood_vas_s1_weekly", "Mood VAS (weekly)", "viridis_r")]
fig, axes = plt.subplots(2, 2, figsize=(13, 11))
for ax, (col, label, cmap) in zip(axes.ravel(), panels):
    c = pd.to_numeric(df[col], errors="coerce"); ok = c.notna()
    ax.scatter(df.loc[~ok, "tsne_x"], df.loc[~ok, "tsne_y"], c="lightgray", s=18, alpha=0.5, label="no score")
    sc = ax.scatter(df.loc[ok, "tsne_x"], df.loc[ok, "tsne_y"], c=c[ok], s=22, cmap=cmap)
    fig.colorbar(sc, ax=ax, label=label)
    ax.set_title(f"t-SNE colored by {label}  (n={int(ok.sum())}/{n})", fontsize=11)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
fig.suptitle(f"RCS08 Stage 1 (chronic pain): t-SNE of all acoustic features (perplexity={perplexity:.0f})", fontsize=14)
fig.tight_layout(); fig.savefig(f"{pl}/tsne_4panel.png", dpi=150); plt.close(fig)
print(f"RCS08 t-SNE: n={n} -> {pl}/tsne_4panel.png")
