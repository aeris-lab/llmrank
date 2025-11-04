# =============================================================
# AAAI Study – Extended Jupyter Notebook Version
# =============================================================
#
# Title: Retrieval Without Consensus: Quantifying Inter-LLM Divergence
# Author: Eyhab Al‑Masri et al.
# Version: Extended Reproducibility Notebook
# Dataset: llm_data_extracted.xlsx (previously variant_report_c.xlsx)
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from scipy.stats import kendalltau, spearmanr
from pathlib import Path

# --------------------------------------------------------------
# 1. Configuration & Paths
# --------------------------------------------------------------
TOP_K = 10
INPUT_XLSX = Path("llm_data_extracted.xlsx")
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)
OUT_FILE = OUTPUT_DIR / "pairwise_metrics.csv"

print(f"📘 Using input file: {INPUT_XLSX.name}")

# --------------------------------------------------------------
# 2. Data Loading and Inspection
# --------------------------------------------------------------
print("\n🔍 Loading dataset and inspecting contents…")
xls = pd.ExcelFile(INPUT_XLSX)
print(f"Found {len(xls.sheet_names)} query sheets:")
for s in xls.sheet_names:
    print(f" • {s}")

# Preview first sheet
df_preview = xls.parse(xls.sheet_names[0])
print("\n📊 Preview of first query sheet:")
print(df_preview.head())

# --------------------------------------------------------------
# 3. Metric Functions
# --------------------------------------------------------------
def average_overlap(r1, r2, k=TOP_K):
    """Compute Average Overlap (AO) between two rankings."""
    def _overlap(d):
        return len(set(r1[:d]) & set(r2[:d])) / d
    return np.mean([_overlap(d) for d in range(1, k + 1)])


def jaccard_similarity(r1, r2, k=TOP_K):
    s1, s2 = set(r1[:k]), set(r2[:k])
    return len(s1 & s2) / len(s1 | s2) if (s1 or s2) else np.nan


def rbo_score(S, T, p=0.9, k=TOP_K):
    S, T = S[:k], T[:k]
    x_d = 0.0
    for d in range(1, k + 1):
        overlap = len(set(S[:d]) & set(T[:d])) / d
        x_d += overlap * (p ** (d - 1))
    return (1 - p) * x_d


def kendall_tau_corr(r1, r2, k=TOP_K):
    r1k, r2k = r1[:k], r2[:k]
    common = [s for s in r1k if s in r2k]
    if len(common) < 2:
        return np.nan
    rank1 = [r1k.index(s) + 1 for s in common]
    rank2 = [r2k.index(s) + 1 for s in common]
    tau, _ = kendalltau(rank1, rank2)
    return tau


def spearman_corr(r1, r2, k=TOP_K):
    r1k, r2k = r1[:k], r2[:k]
    common = [s for s in r1k if s in r2k]
    if len(common) < 2:
        return np.nan
    rank1 = [r1k.index(s) + 1 for s in common]
    rank2 = [r2k.index(s) + 1 for s in common]
    rho, _ = spearmanr(rank1, rank2)
    return rho

# --------------------------------------------------------------
# 4. Load All Queries from Excel
# --------------------------------------------------------------
def load_queries_from_excel(path):
    xls = pd.ExcelFile(path)
    queries = {}
    for sheet in xls.sheet_names:
        df = xls.parse(sheet)
        llm_col = next(c for c in df.columns if "llm" in c.lower())
        service_col = next(c for c in df.columns if "service" in c.lower())
        rank_col = next(c for c in df.columns if "rank" in c.lower())
        df = df.dropna(subset=[rank_col])
        df[rank_col] = pd.to_numeric(df[rank_col], errors="coerce")
        df = df.sort_values(rank_col)
        llm_groups = {}
        for llm, grp in df.groupby(llm_col):
            llm_groups[llm.strip()] = list(grp[service_col].values)
        queries[sheet] = llm_groups
    return queries

print("\n📘 Loading all query sheets …")
queries = load_queries_from_excel(INPUT_XLSX)
print(f"✅ Loaded {len(queries)} queries.")

# --------------------------------------------------------------
# 5. Pairwise Metric Computation
# --------------------------------------------------------------
def compute_pairwise_metrics(queries):
    records = []
    for qname, qdata in queries.items():
        llms = list(qdata.keys())
        for l1, l2 in combinations(llms, 2):
            r1, r2 = qdata[l1], qdata[l2]
            record = {
                "Query": qname,
                "LLM1": l1,
                "LLM2": l2,
                "AO": round(average_overlap(r1, r2), 3),
                "Jaccard": round(jaccard_similarity(r1, r2), 3),
                "RBO": round(rbo_score(r1, r2), 3),
                "KendallTau": round(kendall_tau_corr(r1, r2), 3),
                "Spearman": round(spearman_corr(r1, r2), 3)
            }
            records.append(record)
    return pd.DataFrame(records)

print("\n⚙️ Computing pairwise metrics …")
df_metrics = compute_pairwise_metrics(queries)
print(f"✅ Computed {len(df_metrics)} LLM pair comparisons.")

# --------------------------------------------------------------
# 6. Summary Statistics
# --------------------------------------------------------------
summary = df_metrics.describe().T
print("\n📈 Summary statistics:")
print(summary)

# --------------------------------------------------------------
# 7. Visualization Section
# --------------------------------------------------------------
sns.set(style="whitegrid", font_scale=1.1)

# Heatmap of average Kendall Tau per LLM pair
pivot_tau = df_metrics.pivot_table(index="LLM1", columns="LLM2", values="KendallTau", aggfunc="mean")
plt.figure(figsize=(7, 5))
sns.heatmap(pivot_tau, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'Avg Kendall Tau'})
plt.title("Average Kendall Tau Across LLM Pairs")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "kendalltau_heatmap.png", dpi=300)
plt.show()

# Histogram of AO scores
plt.figure(figsize=(6, 4))
sns.histplot(df_metrics["AO"], bins=10, kde=True, color="seagreen")
plt.xlabel("Average Overlap (AO)")
plt.title("Distribution of AO Across All LLM Pairs")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ao_histogram.png", dpi=300)
plt.show()

# --------------------------------------------------------------
# 8. Save Output
# --------------------------------------------------------------
df_metrics.to_csv(OUT_FILE, index=False)
print(f"\n💾 Results saved to {OUT_FILE.resolve()}")

# Display top results
print("\nTop 10 Results:")
print(df_metrics.head(10))

# --------------------------------------------------------------
# 9. Interpretation Summary
# --------------------------------------------------------------
print("\n🧩 Interpretation Summary:")
print("High AO and Jaccard values indicate strong overlap in API discovery across models.")
print("High Kendall Tau and Spearman correlations suggest consistent ranking logic.")
print("Lower RBO in creative domains typically reflects divergent reasoning structures.")
