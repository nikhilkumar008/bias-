"""
============================================================
  AI Fairness Pipeline – Disparate Impact Remover (DIR)
  Research Paper Methodology
  Dataset: UCI Adult Income (adult.csv)
============================================================
METHODOLOGY:
  1. Load & preprocess Adult Income dataset
  2. Train Logistic Regression + Decision Tree (BASELINE – biased)
  3. Apply Disparate Impact Remover (AIF360) on training data
  4. Re-train same models on de-biased data (POST-BIAS)
  5. Compare Disparate Impact (DI) ratio & Accuracy
  6. Generate all publication-quality graphs
============================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve, auc
)
from sklearn.preprocessing import StandardScaler

from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import DisparateImpactRemover
from aif360.metrics import BinaryLabelDatasetMetric

# ─────────────────────────────────────────────
# STEP 1 — LOAD & PREPROCESS
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1 — Loading and Preprocessing Dataset")
print("=" * 60)

df = pd.read_csv("adult.csv")
df.replace("?",  np.nan, inplace=True)
df.replace(" ?", np.nan, inplace=True)
df.dropna(inplace=True)

# Strip whitespace from all string columns
df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)

# Encode target and protected attribute
df["income"] = df["income"].map({"<=50K": 0, ">50K": 1})
df["sex"]    = df["sex"].map({"Male": 1, "Female": 0})

df.dropna(subset=["income", "sex"], inplace=True)
df["income"] = df["income"].astype(int)
df["sex"]    = df["sex"].astype(int)

print(f"  Rows after cleaning : {len(df)}")
print(f"  Income distribution :\n{df['income'].value_counts()}")
print(f"  Sex distribution    :\n{df['sex'].value_counts()}")

# ─────────────────────────────────────────────
# STEP 2 — FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 — Feature Engineering")
print("=" * 60)

TARGET    = "income"
PROTECTED = "sex"

categorical_cols = [
    "workclass", "education", "marital.status",
    "occupation", "relationship", "race", "native.country"
]
categorical_cols = [c for c in categorical_cols if c in df.columns]

X = df.drop(columns=[TARGET])
y = df[TARGET]

X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
print(f"  Feature matrix shape : {X.shape}")

# ─────────────────────────────────────────────
# STEP 3 — TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train size : {X_train.shape[0]} | Test size : {X_test.shape[0]}")

# ─────────────────────────────────────────────
# STEP 4 — BASELINE BIAS MEASUREMENT (AIF360)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4 — Measuring Baseline Bias with AIF360")
print("=" * 60)

def make_bld(X_df, y_series):
    temp = X_df.copy()
    temp[TARGET] = y_series.values
    temp = temp.astype(float)
    return BinaryLabelDataset(
        df=temp,
        label_names=[TARGET],
        protected_attribute_names=[PROTECTED],
        favorable_label=1,
        unfavorable_label=0
    )

privileged_groups   = [{PROTECTED: 1}]
unprivileged_groups = [{PROTECTED: 0}]

bld_train = make_bld(X_train, y_train)

metric_before = BinaryLabelDatasetMetric(
    bld_train,
    privileged_groups=privileged_groups,
    unprivileged_groups=unprivileged_groups
)
print(f"  [BEFORE DIR] Dataset Disparate Impact : {metric_before.disparate_impact():.4f}")
print(f"  [BEFORE DIR] Statistical Parity Diff  : {metric_before.mean_difference():.4f}")

# ─────────────────────────────────────────────
# STEP 5 — APPLY DISPARATE IMPACT REMOVER
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5 — Applying Disparate Impact Remover (repair_level=1.0)")
print("=" * 60)

dir_remover  = DisparateImpactRemover(repair_level=1.0)
bld_repaired = dir_remover.fit_transform(bld_train)

df_repaired      = bld_repaired.convert_to_dataframe()[0]
X_train_repaired = df_repaired.drop(columns=[TARGET])
y_train_repaired = df_repaired[TARGET].astype(int)

metric_after = BinaryLabelDatasetMetric(
    bld_repaired,
    privileged_groups=privileged_groups,
    unprivileged_groups=unprivileged_groups
)
print(f"  [AFTER  DIR] Dataset Disparate Impact : {metric_after.disparate_impact():.4f}")
print(f"  [AFTER  DIR] Statistical Parity Diff  : {metric_after.mean_difference():.4f}")

# ─────────────────────────────────────────────
# STEP 6 — TRAIN & EVALUATE MODELS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6 — Training & Evaluating Models")
print("=" * 60)

def run_experiment(X_tr, y_tr, X_te, y_te, model_type="lr"):
    scaler      = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_te_scaled = scaler.transform(X_te)

    if model_type == "lr":
        model = LogisticRegression(max_iter=2000, random_state=42)
        model.fit(X_tr_scaled, y_tr)
        preds = model.predict(X_te_scaled)
        proba = model.predict_proba(X_te_scaled)[:, 1]
    else:
        model = DecisionTreeClassifier(max_depth=8, random_state=42)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        proba = model.predict_proba(X_te)[:, 1]

    temp         = X_te.copy()
    temp["pred"] = preds
    sr_male      = temp[temp[PROTECTED] == 1]["pred"].mean()
    sr_female    = temp[temp[PROTECTED] == 0]["pred"].mean()
    di           = sr_female / sr_male if sr_male > 0 else np.nan

    temp["actual"] = y_te.values
    male_mask   = temp[PROTECTED] == 1
    female_mask = temp[PROTECTED] == 0
    tpr_male   = ((temp.loc[male_mask,   "pred"]==1)&(temp.loc[male_mask,   "actual"]==1)).sum() / (temp.loc[male_mask,   "actual"]==1).sum()
    tpr_female = ((temp.loc[female_mask, "pred"]==1)&(temp.loc[female_mask, "actual"]==1)).sum() / (temp.loc[female_mask, "actual"]==1).sum()
    eod = tpr_female - tpr_male

    acc = accuracy_score(y_te, preds)
    cm  = confusion_matrix(y_te, preds)

    return {"di": di, "acc": acc, "eod": eod,
            "preds": preds, "proba": proba, "cm": cm}

print("\n  Logistic Regression ...")
lr_pre  = run_experiment(X_train,          y_train,          X_test, y_test, "lr")
lr_post = run_experiment(X_train_repaired, y_train_repaired, X_test, y_test, "lr")

print("  Decision Tree ...")
dt_pre  = run_experiment(X_train,          y_train,          X_test, y_test, "dt")
dt_post = run_experiment(X_train_repaired, y_train_repaired, X_test, y_test, "dt")

# ─────────────────────────────────────────────
# PRINT RESULTS TABLE
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
print(f"  {'Model':<22} {'Stage':<8} {'DI':>8} {'Accuracy':>10} {'EOD':>8}")
print("  " + "-" * 58)
for name, stage, r in [
    ("Logistic Regression", "Before", lr_pre),
    ("Logistic Regression", "After",  lr_post),
    ("Decision Tree",       "Before", dt_pre),
    ("Decision Tree",       "After",  dt_post),
]:
    print(f"  {name:<22} {stage:<8} {r['di']:>8.4f} {r['acc']:>10.4f} {r['eod']:>8.4f}")
print("\n  DI > 0.8 = Fair  |  EOD closer to 0 = Fairer")

# ─────────────────────────────────────────────
# STEP 7 — GRAPHS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7 — Generating Graphs")
print("=" * 60)

COLORS = {
    "before":    "#E74C3C",
    "after":     "#2ECC71",
    "fair_line": "#F39C12",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "figure.dpi": 150,
})

# ── GRAPH 1: Disparate Impact ─────────────────
fig1, axes = plt.subplots(1, 2, figsize=(12, 5))
fig1.suptitle("Disparate Impact (DI) — Before vs After DIR\n"
              "(DI = Selection Rate Female / Male  |  Threshold = 0.8)",
              fontsize=13, fontweight="bold")
for ax, (name, pre, post) in zip(axes, [
    ("Logistic Regression", lr_pre["di"], lr_post["di"]),
    ("Decision Tree",       dt_pre["di"], dt_post["di"]),
]):
    bars = ax.bar(["Before DIR", "After DIR"], [pre, post],
                  color=[COLORS["before"], COLORS["after"]],
                  edgecolor="black", width=0.4)
    ax.axhline(0.8, color=COLORS["fair_line"], linewidth=2,
               linestyle="--", label="Fairness Threshold (0.8)")
    ax.set_title(name); ax.set_ylabel("Disparate Impact Ratio")
    ax.set_ylim(0, 1.2); ax.legend(fontsize=9)
    for bar, val in zip(bars, [pre, post]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                f"{val:.4f}", ha="center", fontweight="bold", fontsize=11)
plt.tight_layout()
fig1.savefig("graph1_disparate_impact.png", bbox_inches="tight")
print("  graph1_disparate_impact.png saved")

# ── GRAPH 2: Accuracy ─────────────────────────
fig2, axes = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle("Model Accuracy — Before vs After DIR",
              fontsize=13, fontweight="bold")
for ax, (name, pre_acc, post_acc) in zip(axes, [
    ("Logistic Regression", lr_pre["acc"], lr_post["acc"]),
    ("Decision Tree",       dt_pre["acc"], dt_post["acc"]),
]):
    bars = ax.bar(["Before DIR", "After DIR"], [pre_acc, post_acc],
                  color=[COLORS["before"], COLORS["after"]],
                  edgecolor="black", width=0.4)
    ax.set_title(name); ax.set_ylabel("Accuracy"); ax.set_ylim(0.75, 0.90)
    for bar, val in zip(bars, [pre_acc, post_acc]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001,
                f"{val:.4f}", ha="center", fontweight="bold", fontsize=11)
plt.tight_layout()
fig2.savefig("graph2_accuracy.png", bbox_inches="tight")
print("  graph2_accuracy.png saved")

# ── GRAPH 3: Trade-off Scatter ────────────────
fig3, ax = plt.subplots(figsize=(9, 6))
ax.set_title("Fairness–Accuracy Trade-off\n(Ideal: High Accuracy + DI ≥ 0.8)",
             fontweight="bold")
points = [
    ("LR Before", lr_pre["di"],  lr_pre["acc"],  COLORS["before"], "o"),
    ("LR After",  lr_post["di"], lr_post["acc"], COLORS["after"],  "o"),
    ("DT Before", dt_pre["di"],  dt_pre["acc"],  COLORS["before"], "s"),
    ("DT After",  dt_post["di"], dt_post["acc"], COLORS["after"],  "s"),
]
for label, di_v, acc_v, color, marker in points:
    ax.scatter(di_v, acc_v, c=color, marker=marker,
               s=200, edgecolors="black", linewidths=1.2, zorder=5)
    ax.annotate(label, (di_v, acc_v),
                textcoords="offset points", xytext=(8, 4), fontsize=10)
ax.axvline(0.8, color=COLORS["fair_line"], linestyle="--", linewidth=2)
ax.set_xlabel("Disparate Impact (DI)"); ax.set_ylabel("Accuracy")
red_patch   = mpatches.Patch(color=COLORS["before"], label="Before DIR")
green_patch = mpatches.Patch(color=COLORS["after"],  label="After DIR")
fair_line   = plt.Line2D([0],[0],color=COLORS["fair_line"],linestyle="--",
                          linewidth=2, label="Fairness Threshold (0.8)")
ax.legend(handles=[red_patch, green_patch, fair_line], fontsize=10)
plt.tight_layout()
fig3.savefig("graph3_tradeoff_scatter.png", bbox_inches="tight")
print("  graph3_tradeoff_scatter.png saved")

# ── GRAPH 4: ROC Curves ───────────────────────
fpr_lr_pre,  tpr_lr_pre,  _ = roc_curve(y_test, lr_pre["proba"])
fpr_lr_post, tpr_lr_post, _ = roc_curve(y_test, lr_post["proba"])
fpr_dt_pre,  tpr_dt_pre,  _ = roc_curve(y_test, dt_pre["proba"])
fpr_dt_post, tpr_dt_post, _ = roc_curve(y_test, dt_post["proba"])

fig4, axes = plt.subplots(1, 2, figsize=(13, 5))
fig4.suptitle("ROC Curves — Before vs After DIR", fontweight="bold")
for ax, (title, fp, tp, fp2, tp2) in zip(axes, [
    ("Logistic Regression", fpr_lr_pre, tpr_lr_pre, fpr_lr_post, tpr_lr_post),
    ("Decision Tree",       fpr_dt_pre, tpr_dt_pre, fpr_dt_post, tpr_dt_post),
]):
    ax.plot(fp,  tp,  color=COLORS["before"], lw=2,
            label=f"Before DIR (AUC={auc(fp,  tp):.3f})")
    ax.plot(fp2, tp2, color=COLORS["after"],  lw=2,
            label=f"After DIR  (AUC={auc(fp2, tp2):.3f})")
    ax.plot([0,1],[0,1],"k--",alpha=0.4,label="Random")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(title); ax.legend(fontsize=9)
plt.tight_layout()
fig4.savefig("graph4_roc_curves.png", bbox_inches="tight")
print("  graph4_roc_curves.png saved")

# ── GRAPH 5: Confusion Matrices ──────────────
fig5, axes = plt.subplots(2, 4, figsize=(18, 8))
fig5.suptitle("Confusion Matrices — All Models Before & After DIR",
              fontsize=14, fontweight="bold")
for ax, (title, cm) in zip(axes[0], [
    ("LR — Before DIR", lr_pre["cm"]),
    ("LR — After DIR",  lr_post["cm"]),
    ("DT — Before DIR", dt_pre["cm"]),
    ("DT — After DIR",  dt_post["cm"]),
]):
    ConfusionMatrixDisplay(cm, display_labels=["<=50K",">50K"]).plot(
        ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title, fontsize=11)

for ax, (title, cm) in zip(axes[1], [
    ("LR — Before (Norm)", lr_pre["cm"]),
    ("LR — After  (Norm)", lr_post["cm"]),
    ("DT — Before (Norm)", dt_pre["cm"]),
    ("DT — After  (Norm)", dt_post["cm"]),
]):
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    disp = ConfusionMatrixDisplay(cm_norm, display_labels=["<=50K",">50K"])
    disp.plot(ax=ax, colorbar=False, cmap="Greens")
    ax.set_title(title, fontsize=11)
    for t in disp.text_.ravel():
        t.set_text(f"{float(t.get_text()):.2f}")
plt.tight_layout()
fig5.savefig("graph5_confusion_matrices.png", bbox_inches="tight")
print("  graph5_confusion_matrices.png saved")

# ── GRAPH 6: Selection Rate by Gender ─────────
fig6, axes = plt.subplots(1, 2, figsize=(13, 5))
fig6.suptitle("Positive Prediction Rate by Gender — Before vs After DIR",
              fontweight="bold")
def sel_rates(result):
    temp = X_test.copy()
    temp["pred"] = result["preds"]
    return (temp[temp[PROTECTED]==1]["pred"].mean(),
            temp[temp[PROTECTED]==0]["pred"].mean())

for ax, (title, pre_r, post_r) in zip(axes, [
    ("Logistic Regression", lr_pre, lr_post),
    ("Decision Tree",       dt_pre, dt_post),
]):
    m_pre, f_pre   = sel_rates(pre_r)
    m_post, f_post = sel_rates(post_r)
    x = np.arange(2); w = 0.3
    b1 = ax.bar(x-w/2, [m_pre, f_pre],   w, label="Before DIR",
                color=COLORS["before"], edgecolor="black")
    b2 = ax.bar(x+w/2, [m_post, f_post], w, label="After DIR",
                color=COLORS["after"],  edgecolor="black")
    ax.set_xticks(x); ax.set_xticklabels(["Male","Female"])
    ax.set_ylabel("Positive Prediction Rate")
    ax.set_ylim(0, 0.65); ax.set_title(title); ax.legend()
    for bar in list(b1)+list(b2):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f"{bar.get_height():.3f}", ha="center", fontsize=9,
                fontweight="bold")
plt.tight_layout()
fig6.savefig("graph6_selection_rate_by_gender.png", bbox_inches="tight")
print("  graph6_selection_rate_by_gender.png saved")

# ── GRAPH 7: Dashboard ────────────────────────
fig7 = plt.figure(figsize=(16, 10))
fig7.suptitle("AI Fairness Pipeline — Full Summary Dashboard",
              fontsize=15, fontweight="bold", y=1.01)
gs = GridSpec(2, 3, figure=fig7, hspace=0.45, wspace=0.35)

labels     = ["LR\nBefore","LR\nAfter","DT\nBefore","DT\nAfter"]
bar_colors = [COLORS["before"],COLORS["after"],COLORS["before"],COLORS["after"]]

# DI
ax1 = fig7.add_subplot(gs[0,0])
di_vals = [lr_pre["di"],lr_post["di"],dt_pre["di"],dt_post["di"]]
bars = ax1.bar(labels, di_vals, color=bar_colors, edgecolor="black")
ax1.axhline(0.8, color=COLORS["fair_line"], linestyle="--", lw=2,
            label="Threshold 0.8")
ax1.set_title("Disparate Impact"); ax1.set_ylim(0,1.2); ax1.legend(fontsize=8)
for b,v in zip(bars,di_vals):
    ax1.text(b.get_x()+b.get_width()/2, b.get_height()+0.02,
             f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")

# Accuracy
ax2 = fig7.add_subplot(gs[0,1])
acc_vals = [lr_pre["acc"],lr_post["acc"],dt_pre["acc"],dt_post["acc"]]
bars = ax2.bar(labels, acc_vals, color=bar_colors, edgecolor="black")
ax2.set_title("Accuracy"); ax2.set_ylim(0.75, 0.90)
for b,v in zip(bars,acc_vals):
    ax2.text(b.get_x()+b.get_width()/2, b.get_height()+0.001,
             f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")

# EOD
ax3 = fig7.add_subplot(gs[0,2])
eod_vals = [lr_pre["eod"],lr_post["eod"],dt_pre["eod"],dt_post["eod"]]
bars = ax3.bar(labels, eod_vals, color=bar_colors, edgecolor="black")
ax3.axhline(0, color="black", lw=1)
ax3.set_title("Equal Opportunity Diff\n(0 = perfectly fair)")
for b,v in zip(bars,eod_vals):
    ax3.text(b.get_x()+b.get_width()/2, b.get_height()+(0.003 if v>=0 else -0.009),
             f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")

# ROC LR
ax4 = fig7.add_subplot(gs[1,0])
ax4.plot(fpr_lr_pre, tpr_lr_pre, color=COLORS["before"], lw=2,
         label=f"Before (AUC={auc(fpr_lr_pre,tpr_lr_pre):.3f})")
ax4.plot(fpr_lr_post,tpr_lr_post,color=COLORS["after"],  lw=2,
         label=f"After  (AUC={auc(fpr_lr_post,tpr_lr_post):.3f})")
ax4.plot([0,1],[0,1],"k--",alpha=0.4)
ax4.set_title("ROC – Log. Regression")
ax4.set_xlabel("FPR"); ax4.set_ylabel("TPR"); ax4.legend(fontsize=8)

# ROC DT
ax5 = fig7.add_subplot(gs[1,1])
ax5.plot(fpr_dt_pre, tpr_dt_pre, color=COLORS["before"], lw=2,
         label=f"Before (AUC={auc(fpr_dt_pre,tpr_dt_pre):.3f})")
ax5.plot(fpr_dt_post,tpr_dt_post,color=COLORS["after"],  lw=2,
         label=f"After  (AUC={auc(fpr_dt_post,tpr_dt_post):.3f})")
ax5.plot([0,1],[0,1],"k--",alpha=0.4)
ax5.set_title("ROC – Decision Tree")
ax5.set_xlabel("FPR"); ax5.set_ylabel("TPR"); ax5.legend(fontsize=8)

# Summary table
ax6 = fig7.add_subplot(gs[1,2]); ax6.axis("off")
tbl = ax6.table(
    cellText=[
        ["LR","Before",f"{lr_pre['di']:.3f}", f"{lr_pre['acc']:.3f}", f"{lr_pre['eod']:.3f}"],
        ["LR","After", f"{lr_post['di']:.3f}",f"{lr_post['acc']:.3f}",f"{lr_post['eod']:.3f}"],
        ["DT","Before",f"{dt_pre['di']:.3f}", f"{dt_pre['acc']:.3f}", f"{dt_pre['eod']:.3f}"],
        ["DT","After", f"{dt_post['di']:.3f}",f"{dt_post['acc']:.3f}",f"{dt_post['eod']:.3f}"],
    ],
    colLabels=["Model","Stage","DI","Acc","EOD"],
    loc="center", cellLoc="center"
)
tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.2,1.7)
ax6.set_title("Metrics Summary", fontsize=11, pad=10)

plt.tight_layout()
fig7.savefig("graph7_dashboard.png", bbox_inches="tight")
print("  graph7_dashboard.png saved")

print("\n" + "=" * 60)
print("ALL DONE! 7 graphs generated successfully.")
print("=" * 60)
