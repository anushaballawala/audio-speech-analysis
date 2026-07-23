import pandas as pd, numpy as np
UD = "/userdata/msharma"
S2 = f"{UD}/sub-PR05-stage-2_audio-audiotype_preproc_spectral_gating_100_percent_metadata_and_plots/sub-PR05_stage-2_audio-audiotype_preproc_spectral_gating_100_percent_hamd_correlation/correlations.csv"
S3 = f"{UD}/sub-PR05-stage-3_audio-audiotype_preproc_spectral_gating_100_percent_metadata_and_plots/sub-PR05_stage-3_audio-audiotype_preproc_spectral_gating_100_percent_hamd_correlation/correlations.csv"

a = pd.read_csv(S2); b = pd.read_csv(S3)
print(f"Stage 2: {a['feature'].nunique()} features x {a['score'].nunique()} scores, n_recordings~{a['n'].max()}")
print(f"Stage 3: {b['feature'].nunique()} features x {b['score'].nunique()} scores, n_recordings~{b['n'].max()}")

m = a.merge(b, on=["score", "feature"], suffixes=("_s2", "_s3"))
m["dr"] = m["pearson_r_s3"] - m["pearson_r_s2"]
m["sig_s2"] = m["pearson_p_s2"] < 0.05
m["sig_s3"] = m["pearson_p_s3"] < 0.05

print(f"\npairs compared: {len(m)}")
print(f"significant (p<.05) in Stage2: {m.sig_s2.sum()}  |  Stage3: {m.sig_s3.sum()}  |  both: {(m.sig_s2 & m.sig_s3).sum()}")
print(f"mean |r| Stage2: {m.pearson_r_s2.abs().mean():.3f}  Stage3: {m.pearson_r_s3.abs().mean():.3f}")
corr_of_r = m[["pearson_r_s2", "pearson_r_s3"]].corr().iloc[0, 1]
print(f"correlation between the two r-maps (do the patterns agree?): {corr_of_r:.3f}")
signflip = m[(m.pearson_r_s2 * m.pearson_r_s3 < 0) & (m.sig_s2 | m.sig_s3)]
print(f"significant sign flips (opposite direction, sig in >=1 stage): {len(signflip)}")

def show(df, cols, title):
    print(f"\n{title}")
    print(df[cols].to_string(index=False))

top2 = a.reindex(a.pearson_r.abs().sort_values(ascending=False).index).head(8)
top3 = b.reindex(b.pearson_r.abs().sort_values(ascending=False).index).head(8)
show(top2, ["score", "feature", "pearson_r", "pearson_p"], "=== Stage 2 strongest correlations ===")
show(top3, ["score", "feature", "pearson_r", "pearson_p"], "=== Stage 3 strongest correlations ===")

big = m.reindex(m.dr.abs().sort_values(ascending=False).index).head(12)
show(big, ["score", "feature", "pearson_r_s2", "pearson_r_s3", "dr"], "=== biggest Stage2->Stage3 differences (dr = r_s3 - r_s2) ===")

if len(signflip):
    show(signflip.reindex(signflip.dr.abs().sort_values(ascending=False).index).head(10),
         ["score", "feature", "pearson_r_s2", "pearson_r_s3"], "=== significant sign flips ===")

# per-score agreement
print("\n=== per-score: mean|r| and r-map agreement ===")
rows = []
for s in sorted(m.score.unique()):
    ms = m[m.score == s]
    rows.append({"score": s, "mean|r|_s2": round(ms.pearson_r_s2.abs().mean(), 3),
                 "mean|r|_s3": round(ms.pearson_r_s3.abs().mean(), 3),
                 "sig_s2": int(ms.sig_s2.sum()), "sig_s3": int(ms.sig_s3.sum()),
                 "r_map_agree": round(ms[["pearson_r_s2", "pearson_r_s3"]].corr().iloc[0, 1], 3)})
print(pd.DataFrame(rows).to_string(index=False))
