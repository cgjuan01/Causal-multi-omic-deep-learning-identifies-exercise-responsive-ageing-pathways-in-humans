#!/usr/bin/env python3
import numpy as np
import pandas as pd

BASE = "/Users/ciara/Downloads/LDaware_MR_proteins"

IN_NODE  = f"{BASE}/GNN_nodes_EXERCISEONLY_MR_SC_STRUCT_PANTHER_INTERPRO_UNIPROT_AFv6_PROTEINMRfixed.tsv"
OUT_NODE = f"{BASE}/GNN_nodes_EXERCISEONLY_MR_SC_STRUCT_PANTHER_INTERPRO_UNIPROT_AFv6_PROTEINMRfixed_MTIbetaRecalc.tsv"

df = pd.read_csv(IN_NODE, sep="\t")
if "gene_symbol" not in df.columns:
    raise ValueError("Missing gene_symbol")

# ---- 1) Promote *_fixed into canonical protein MR columns (fill only)
for col in ["beta", "se", "p", "nsnp"]:
    fixed = f"protein_MR_{col}_fixed"
    base  = f"protein_MR_{col}"
    if fixed in df.columns:
        if base not in df.columns:
            df[base] = np.nan
        df[base] = df[fixed].combine_first(df[base])

# ---- 2) Define layers used in MTI (beta-based)
layers = {
    "protein": "protein_MR_beta",
    "cpg":     "cpg_MR_beta",
    "glycan":  "glycan_MR_beta",
    "sc":      "sc_MR_beta",
}

# ---- 3) Recompute beta_std per layer (beta / SD(beta) within layer)
beta_std_cols = []
for layer, bcol in layers.items():
    outcol = f"{layer}_beta_std_recalc"
    beta_std_cols.append(outcol)

    if bcol not in df.columns:
        df[outcol] = np.nan
        continue

    x = pd.to_numeric(df[bcol], errors="coerce").to_numpy(dtype=float)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd == 0:
        df[outcol] = np.nan
    else:
        df[outcol] = x / sd

# ---- 4) Recompute MTI components
M = df[beta_std_cols].to_numpy(dtype=float)
present = np.isfinite(M)
df["MTI_n_layers_recalc"] = present.sum(axis=1).astype(int)

df["MTI_sumsq_recalc"] = np.nansum(M**2, axis=1)
df["MTI_score_recalc"] = np.sqrt(df["MTI_sumsq_recalc"].to_numpy(dtype=float))

# ---- 5) Optional: overwrite canonical MTI columns to keep pipeline consistent
# If you prefer to preserve old MTI, comment out the 3 lines below.
df["MTI_n_layers"] = df["MTI_n_layers_recalc"]
df["MTI_sumsq"]    = df["MTI_sumsq_recalc"]
df["MTI_score"]    = df["MTI_score_recalc"]

df.to_csv(OUT_NODE, sep="\t", index=False)
print("✅ Wrote:", OUT_NODE)

# ---- Quick sanity check
check = ["SIRT1","SIRT2","B4GALT1","FUT8","ST6GAL1","MGAT3","SIRT6"]
cols_show = ["gene_symbol","protein_MR_beta","protein_MR_p","MTI_n_layers","MTI_score"]
print("\nSanity check:")
print(df.loc[df["gene_symbol"].isin(check), cols_show].to_string(index=False))

