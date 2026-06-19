# =======================================================================
# HealthStat Pro — Python/Dash Version
# Requirements:
#   pip install dash dash-bootstrap-components pandas numpy scipy
#               scikit-learn statsmodels plotly lifelines xgboost shap
#               dash-ag-grid openpyxl
# =======================================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
from scipy import stats as sp_stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import (
    Ridge, Lasso, ElasticNet, LogisticRegression
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error, roc_auc_score
from sklearn.manifold import TSNE
import xgboost as xgb
import shap
import io
import base64
import re

import dash
from dash import dcc, html, dash_table, Input, Output, State, ctx
import dash_bootstrap_components as dbc

# ─────────────────────────────────────────────────────────────
# 1.  SIMULATED DATA
# ─────────────────────────────────────────────────────────────

np.random.seed(123)
N = 100_000

def generate_data(n=N):
    age        = np.clip(np.random.normal(62, 15, n), 18, 95)
    bmi        = np.clip(np.random.normal(29,  7, n), 17, 50)
    gender     = np.random.choice(["Male", "Female"],   n, p=[0.48, 0.52])
    smoking    = np.random.choice(["No", "Yes"],         n, p=[0.72, 0.28])
    diabetes   = np.random.choice(["No", "Yes"],         n, p=[0.65, 0.35])
    hypert     = np.random.choice(["No", "Yes"],         n, p=[0.55, 0.45])
    insurance  = np.random.choice(
        ["Private", "Medicare", "Medicaid", "Uninsured"],
        n, p=[0.40, 0.30, 0.20, 0.10])
    admission  = np.random.choice(
        ["Elective", "Emergency", "Urgent", "Trauma"],
        n, p=[0.40, 0.35, 0.18, 0.07])
    severity   = np.random.choice(
        ["Mild", "Moderate", "Severe", "Critical"],
        n, p=[0.40, 0.35, 0.20, 0.05])

    adm_add = np.where(admission == "Emergency", 9,
              np.where(admission == "Urgent",    5,
              np.where(admission == "Trauma",   15, 0)))
    sev_add = np.where(severity == "Critical", 20,
              np.where(severity == "Severe",   10,
              np.where(severity == "Moderate",  4, 0)))

    los = np.round(np.clip(
        4 + 0.08*age + 0.12*bmi
        + 2.5*(smoking == "Yes") + 3.0*(diabetes == "Yes")
        + 2.0*(hypert  == "Yes") + adm_add + sev_add
        + np.random.normal(0, 4, n), 1, None), 1)

    cost = np.clip(np.round(
        1500 + 120*los + 800*(admission == "Emergency")
        + 2000*(severity == "Critical")
        + np.where(insurance == "Uninsured", 500, 0)
        + np.random.normal(0, 1200, n)), 500, None)

    logit = (-6.5 + 0.04*age + 0.06*bmi
             + 1.2*(smoking == "Yes") + 1.5*(diabetes == "Yes")
             + 0.8*(hypert  == "Yes") + 1.8*(los > 20)
             + 1.0*(insurance == "Uninsured"))
    rp          = 1 / (1 + np.exp(-logit))
    readmitted  = np.random.binomial(1, rp, n)
    tte         = np.where(
        readmitted == 1,
        np.random.exponential(40,  n),
        np.random.exponential(333, n))
    tte         = np.clip(np.round(tte, 1), 1, None)
    event       = (tte < 365).astype(int)
    satisfaction = np.clip(np.round(
        8 - 0.02*age - 0.1*los
        + np.where(insurance == "Private", 1, 0)
        - 1.5*(admission == "Emergency")
        + np.random.normal(0, 1.2, n)), 1, 10)

    df = pd.DataFrame({
        "PatientID":           np.arange(1, n+1),
        "Age":                 age.round(1),
        "Gender":              gender,
        "BMI":                 bmi.round(1),
        "Smoking":             smoking,
        "Diabetes":            diabetes,
        "Hypertension":        hypert,
        "Insurance":           insurance,
        "Admission":           admission,
        "Severity":            severity,
        "Length_of_Stay":      los,
        "Treatment_Cost":      cost,
        "Readmitted_90d":      np.where(readmitted == 1, "Yes", "No"),
        "Time_to_Readmission": tte,
        "Event_Status":        event,
        "Satisfaction":        satisfaction.astype(int),
    })
    return df

SIM_DATA = generate_data()

# Built-in time-series
np.random.seed(42)
_months    = pd.date_range("2019-01-01", periods=60, freq="MS")
_t         = np.arange(60)
TS_DATA    = pd.DataFrame({
    "Date": _months,
    "Bed_Occupancy": np.clip(np.round(
        72 + 8*np.sin(2*np.pi*_t/12) + 0.15*_t
        + np.random.normal(0, 4, 60), 1), 40, 98),
    "Monthly_Admissions": np.clip(np.round(
        800 + 100*np.sin(2*np.pi*_t/12) + 3*_t
        + np.random.normal(0, 50, 60)).astype(int), 200, None),
})


# ─────────────────────────────────────────────────────────────
# 2.  HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────

def assess_data_quality(df: pd.DataFrame) -> pd.DataFrame:
    nr = len(df)
    rows = []
    for col in df.columns:
        s     = df[col]
        miss  = s.isna().sum()
        pct   = round(miss / nr * 100, 1)
        if pd.api.types.is_numeric_dtype(s):
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr    = q3 - q1
            out    = int(((s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)).sum())
        else:
            out = None
        rows.append({
            "Column": col,
            "Type":    str(s.dtype),
            "Missing": miss,
            "Missing_Pct": pct,
            "Outliers": out,
        })
    return pd.DataFrame(rows)


def clean_data(df: pd.DataFrame,
               method: str = "median",
               remove_outliers: bool = False) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            na_n = df[col].isna().sum()
            if 0 < na_n < len(df) * 0.5:
                fill = (df[col].median() if method == "median"
                        else df[col].mean())
                df[col].fillna(fill, inplace=True)
        else:
            na_n = df[col].isna().sum()
            if 0 < na_n < len(df) * 0.5:
                md = df[col].mode()
                if len(md):
                    df[col].fillna(md[0], inplace=True)
    if remove_outliers:
        for col in df.select_dtypes(include=np.number).columns:
            lo, hi = df[col].quantile(0.01), df[col].quantile(0.99)
            df[col] = df[col].clip(lo, hi)
    return df.dropna()


def check_pii(df: pd.DataFrame) -> dict:
    pii_name_patterns = [
        "ssn", "social_security", "phone", "email", "address",
        "name", "first_name", "last_name", "dob", "date_of_birth",
        "mrn", "medical_record"
    ]
    name_hits = [
        c for c in df.columns
        if any(p in c.lower() for p in pii_name_patterns)
    ]
    content_hits = []
    char_cols = df.select_dtypes(include="object").columns
    for col in char_cols:
        sample = df[col].dropna().head(100).astype(str)
        if sample.str.match(r"^\d{3}-\d{2}-\d{4}$").any():
            content_hits.append(f"{col} (SSN-like)")
        if sample.str.contains(
            r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
        ).any():
            content_hits.append(f"{col} (email-like)")
        if sample.str.match(
            r"^\(?\d{3}\)?[\-.\s]?\d{3}[\-.\s]?\d{4}$"
        ).any():
            content_hits.append(f"{col} (phone-like)")
    return {"name_hits": name_hits, "content_hits": content_hits}


def read_uploaded_file(contents: str, filename: str,
                        max_mb: float = 50) -> pd.DataFrame:
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    size_mb = len(decoded) / (1024**2)
    if size_mb > max_mb:
        raise ValueError(
            f"File too large ({size_mb:.1f} MB). Max {max_mb} MB.")
    ext = filename.rsplit(".", 1)[-1].lower()
    buf = io.BytesIO(decoded)
    if ext == "csv":
        return pd.read_csv(buf)
    elif ext == "tsv":
        return pd.read_csv(buf, sep="\t")
    elif ext in ("xlsx", "xls"):
        return pd.read_excel(buf)
    elif ext == "json":
        return pd.read_json(buf)
    else:
        raise ValueError(f"Unsupported format: {ext}")


def get_numeric_cols(df: pd.DataFrame) -> list:
    return df.select_dtypes(include=np.number).columns.tolist()


def encode_predictors(df: pd.DataFrame,
                       preds: list) -> pd.DataFrame:
    """One-hot encode categorical predictors."""
    cat = [c for c in preds if not pd.api.types.is_numeric_dtype(df[c])]
    num = [c for c in preds if  pd.api.types.is_numeric_dtype(df[c])]
    parts = [df[num]]
    for c in cat:
        dummies = pd.get_dummies(df[c], prefix=c, drop_first=True)
        parts.append(dummies)
    return pd.concat(parts, axis=1).astype(float)


# ─────────────────────────────────────────────────────────────
# 3.  INTERPRETATION GENERATORS
# ─────────────────────────────────────────────────────────────

def interpret_lm(model_result) -> str:
    try:
        r2  = round(model_result.rsquared, 3)
        ar2 = round(model_result.rsquared_adj, 3)
        sig = model_result.pvalues[model_result.pvalues < 0.05].drop(
            "const", errors="ignore")
        txt = (f"**R² = {r2}** (Adj = {ar2}). "
               f"Explains ~{round(r2*100,1)}% of variance.\n\n")
        if len(sig):
            txt += "**Key predictors (p<0.05):**\n\n"
            for nm, pv in sig.nsmallest(5).items():
                coef = model_result.params[nm]
                direction = "increases" if coef > 0 else "decreases"
                txt += (f"- **{nm}** {direction} outcome by "
                        f"{abs(round(coef, 3))} (p={pv:.2e})\n")
        else:
            txt += "No significant predictors at p<0.05.\n"
        return txt
    except Exception as e:
        return f"Interpretation error: {e}"


def interpret_logistic(model_result, auc=None) -> str:
    txt = "**Logistic Regression:**\n\n"
    if auc is not None:
        perf = ("Excellent" if auc > 0.9 else
                "Good"      if auc > 0.8 else
                "Acceptable" if auc > 0.7 else "Poor")
        txt += f"- AUC = {round(auc, 3)} ({perf})\n\n"
    pv  = model_result.pvalues
    sig = pv[(pv < 0.05) & (pv.index != "const") & (pv.index != "Intercept")]
    if len(sig):
        txt += "**Significant Odds Ratios:**\n\n"
        for nm in sig.nsmallest(5).index:
            or_ = round(np.exp(model_result.params[nm]), 2)
            dir_ = (f"{round((or_-1)*100,1)}% increased odds"
                    if or_ > 1 else
                    f"{round((1-or_)*100,1)}% decreased odds")
            txt += f"- **{nm}**: OR={or_} ({dir_})\n"
    return txt


def interpret_rf(model, feature_names) -> str:
    imp  = model.feature_importances_
    idx  = np.argsort(imp)[::-1]
    txt  = "**Random Forest:**\n\n**Top 5 Features:**\n\n"
    for i in range(min(5, len(idx))):
        txt += f"- **{feature_names[idx[i]]}**: {round(imp[idx[i]], 4)}\n"
    return txt


def interpret_survival(cox_result) -> str:
    txt = (f"**Cox Model:** Concordance = "
           f"{round(cox_result.concordance_index_, 3)}\n\n")
    hr_ = np.exp(cox_result.params_)
    pv  = cox_result.summary["p"]
    sig = pv[pv < 0.05]
    if len(sig):
        txt += "**Significant Hazard Ratios:**\n\n"
        for nm in sig.nsmallest(5).index:
            h = round(hr_[nm], 2)
            d = (f"{round((h-1)*100,1)}% increased hazard"
                 if h > 1 else
                 f"{round((1-h)*100,1)}% decreased hazard")
            txt += f"- **{nm}**: HR={h} ({d})\n"
    return txt


def interpret_clustering(km) -> str:
    sizes = np.bincount(km.labels_)
    inertia_pct = round(
        (1 - km.inertia_ /
         np.sum((StandardScaler().fit_transform(
             np.zeros((1, km.cluster_centers_.shape[1]))
         )))**2) * 100, 1)
    return (f"**K-Means (k=4):**\n\n"
            f"- Cluster sizes: {', '.join(map(str, sizes))}\n"
            f"- Inertia: {round(km.inertia_, 1)}\n")


def interpret_policy(cost_df: pd.DataFrame) -> str:
    if cost_df is None or len(cost_df) == 0:
        return ""
    hc  = cost_df.loc[cost_df["Mean_Cost"].idxmax(), "Insurance"]
    lc  = cost_df.loc[cost_df["Mean_Cost"].idxmin(), "Insurance"]
    gap = round(cost_df["Mean_Cost"].max() - cost_df["Mean_Cost"].min())
    txt = (f"**Cost Gap:** {hc} patients cost "
           f"${gap:,} more than {lc}.\n\n")
    if "Readmit_Rate" in cost_df.columns:
        hr_ = cost_df.loc[cost_df["Readmit_Rate"].idxmax(), "Insurance"]
        txt += (f"**Highest readmission:** {hr_} at "
                f"{round(cost_df['Readmit_Rate'].max(), 1)}%.\n")
    return txt


# ─────────────────────────────────────────────────────────────
# 4.  ANALYSIS ENGINE
# ─────────────────────────────────────────────────────────────

class AnalysisEngine:
    """
    Runs all 25 analytical modules and stores results.
    Each method returns a dict with 'figures', 'tables', 'text'.
    """

    def __init__(self, df: pd.DataFrame,
                 outcomes: list, predictors: list):
        self.df   = df.copy()
        self.outcomes  = outcomes
        self.predictors = predictors
        self.results   = {}

    # ── helper ────────────────────────────────────────────────
    def _build_df(self, outvar):
        cols = [outvar] + self.predictors
        cols = [c for c in cols if c in self.df.columns]
        sub  = self.df[cols].dropna()
        return sub if len(sub) >= 50 else None

    def _encode(self, df, preds):
        return encode_predictors(df, preds)

    # ── 1. Descriptive Statistics ─────────────────────────────
    def descriptive(self):
        cols = [c for c in self.outcomes + self.predictors
                if c in self.df.columns]
        sub  = self.df[cols].dropna()
        rows = []
        for col in cols:
            s = sub[col]
            if pd.api.types.is_numeric_dtype(s):
                rows.append({
                    "Variable": col, "Type": "Numeric",
                    "Mean (SD)": f"{s.mean():.2f} ({s.std():.2f})",
                    "Median":    f"{s.median():.2f}",
                    "Min":       f"{s.min():.1f}",
                    "Max":       f"{s.max():.1f}",
                    "Missing %": f"{s.isna().mean()*100:.1f}%",
                })
            else:
                vc = s.value_counts()
                top = vc.index[0] if len(vc) else "—"
                rows.append({
                    "Variable": col, "Type": "Categorical",
                    "Mean (SD)": f"Mode: {top}",
                    "Median": f"{len(vc)} levels",
                    "Min": "—", "Max": "—",
                    "Missing %": f"{s.isna().mean()*100:.1f}%",
                })
        tbl = pd.DataFrame(rows)
        self.results["descriptive"] = {"table": tbl}
        return tbl

    # ── 2. Visualization (returns factory) ────────────────────
    def make_plot(self, plot_type, x_var, color_var=None):
        df_p = self.df.copy()
        if len(df_p) > 5000:
            df_p = df_p.sample(5000, random_state=1)
        kw = dict(color=color_var) if color_var and color_var != "None" else {}
        try:
            if plot_type == "histogram":
                return px.histogram(df_p, x=x_var, **kw,
                                    nbins=40, opacity=0.7,
                                    title=f"Distribution of {x_var}")
            elif plot_type == "boxplot":
                if kw:
                    return px.box(df_p, x=color_var, y=x_var, **kw,
                                  title=f"{x_var} by {color_var}")
                return px.box(df_p, y=x_var, title=f"Box: {x_var}")
            elif plot_type == "violin":
                if kw:
                    return px.violin(df_p, x=color_var, y=x_var,
                                     box=True, **kw,
                                     title=f"Violin: {x_var}")
                return px.violin(df_p, y=x_var, box=True,
                                 title=f"Violin: {x_var}")
            elif plot_type == "scatter":
                ov = self.outcomes[0] if self.outcomes else x_var
                if ov in df_p.columns:
                    return px.scatter(df_p, x=x_var, y=ov,
                                      trendline="ols", opacity=0.3, **kw,
                                      title=f"{x_var} vs {ov}")
            elif plot_type == "density":
                return px.histogram(df_p, x=x_var, histnorm="density",
                                    **kw, marginal="rug",
                                    title=f"Density: {x_var}")
            elif plot_type == "bar":
                vc  = df_p[x_var].value_counts().reset_index()
                vc.columns = [x_var, "Count"]
                return px.bar(vc, x=x_var, y="Count",
                              title=f"Count: {x_var}")
            elif plot_type == "group_comparison":
                ov = self.outcomes[0] if self.outcomes else None
                if ov and color_var and color_var != "None":
                    return px.box(df_p, x=color_var, y=ov,
                                  color=color_var,
                                  title=f"{ov} by {color_var}")
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(text=str(e),
                               xref="paper", yref="paper",
                               x=0.5, y=0.5, showarrow=False)
            return fig
        fig = go.Figure()
        fig.add_annotation(text="Select a valid variable combination.",
                           xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False)
        return fig

    # ── 3. Correlation Matrix ─────────────────────────────────
    def correlation(self):
        num = self.df[self.outcomes + self.predictors].select_dtypes(
            include=np.number)
        if num.shape[1] < 2:
            return None, "Need 2+ numeric variables."
        corr = num.corr()
        fig  = px.imshow(corr, text_auto=".2f",
                         color_continuous_scale="RdBu_r",
                         zmin=-1, zmax=1,
                         title="Correlation Matrix")
        self.results["corr"] = corr
        return fig, corr

    # ── 4. Hypothesis Testing ─────────────────────────────────
    def hypothesis_testing(self):
        outvar = self.outcomes[0]
        sub    = self._build_df(outvar)
        if sub is None:
            return {}
        y   = sub[outvar]
        out = {}
        for p in self.predictors:
            if p not in sub.columns:
                continue
            x = sub[p]
            if pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_numeric_dtype(x):
                grps = [g.values for _, g in y.groupby(x)]
                if len(grps) == 2:
                    t_stat, t_p = sp_stats.ttest_ind(*grps)
                    w_stat, w_p = sp_stats.mannwhitneyu(*grps)
                    out[p] = {"test": "t-test / Mann-Whitney",
                              "t_p": t_p, "w_p": w_p}
                elif len(grps) > 2:
                    f, anova_p = sp_stats.f_oneway(*grps)
                    kw_h, kw_p = sp_stats.kruskal(*grps)
                    out[p] = {"test": "ANOVA / Kruskal-Wallis",
                              "anova_p": anova_p, "kw_p": kw_p}
            elif not pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_numeric_dtype(x):
                ct   = pd.crosstab(y, x)
                chi2, chi_p, _, _ = sp_stats.chi2_contingency(ct)
                out[p] = {"test": "Chi-squared", "chi_p": chi_p}
        self.results["hypothesis"] = out
        return out

    # ── 5. Linear Regression ──────────────────────────────────
    def linear_regression(self):
        models = {}
        for outvar in self.outcomes:
            sub = self._build_df(outvar)
            if sub is None:
                continue
            if not pd.api.types.is_numeric_dtype(sub[outvar]):
                continue
            X = self._encode(sub, self.predictors)
            X = sm.add_constant(X)
            y = sub[outvar]
            try:
                mdl = sm.OLS(y, X).fit()
                models[f"Linear: {outvar}"] = {
                    "result": mdl,
                    "interp": interpret_lm(mdl),
                    "type":   "linear",
                }
            except Exception as e:
                pass
        self.results["linear"] = models
        return models

    # ── 6. Polynomial Regression ──────────────────────────────
    def polynomial_regression(self):
        models = {}
        for outvar in self.outcomes:
            sub = self._build_df(outvar)
            if sub is None:
                continue
            if not pd.api.types.is_numeric_dtype(sub[outvar]):
                continue
            num_preds = [c for c in self.predictors
                         if c in sub.columns and
                         pd.api.types.is_numeric_dtype(sub[c])]
            cat_preds = [c for c in self.predictors
                         if c in sub.columns and
                         not pd.api.types.is_numeric_dtype(sub[c])]
            X_base = self._encode(sub, self.predictors)
            for col in num_preds:
                if col in sub.columns:
                    X_base[f"{col}_sq"] = sub[col].values ** 2
            X = sm.add_constant(X_base)
            y = sub[outvar]
            try:
                mdl = sm.OLS(y, X).fit()
                models[f"Polynomial: {outvar}"] = {
                    "result": mdl,
                    "interp": interpret_lm(mdl),
                    "type":   "polynomial",
                }
            except Exception:
                pass
        self.results["polynomial"] = models
        return models

    # ── 7. Stepwise (Forward AIC-like via p-value) ────────────
    def stepwise_regression(self):
        models = {}
        for outvar in self.outcomes:
            sub = self._build_df(outvar)
            if sub is None:
                continue
            if not pd.api.types.is_numeric_dtype(sub[outvar]):
                continue
            y        = sub[outvar]
            X_full   = self._encode(sub, self.predictors)
            remaining = list(X_full.columns)
            selected  = []
            current_score = np.inf
            while remaining:
                scores = {}
                for cand in remaining:
                    cols = selected + [cand]
                    X_try = sm.add_constant(X_full[cols])
                    try:
                        mdl = sm.OLS(y, X_try).fit()
                        scores[cand] = mdl.aic
                    except Exception:
                        pass
                if not scores:
                    break
                best_cand  = min(scores, key=scores.get)
                best_score = scores[best_cand]
                if best_score < current_score:
                    current_score = best_score
                    selected.append(best_cand)
                    remaining.remove(best_cand)
                else:
                    break
            if selected:
                X_sel = sm.add_constant(X_full[selected])
                mdl   = sm.OLS(y, X_sel).fit()
                models[f"Stepwise: {outvar}"] = {
                    "result":   mdl,
                    "interp":   interpret_lm(mdl),
                    "type":     "stepwise",
                    "selected": selected,
                }
        self.results["stepwise"] = models
        return models

    # ── 8. Hierarchical Regression ────────────────────────────
    def hierarchical_regression(self):
        models = {}
        for outvar in self.outcomes:
            sub = self._build_df(outvar)
            if sub is None:
                continue
            if not pd.api.types.is_numeric_dtype(sub[outvar]):
                continue
            y    = sub[outvar]
            X_all = self._encode(sub, self.predictors)
            blocks = {
                "Demographics": [c for c in X_all.columns
                                  if any(k in c for k in
                                         ["Age","Gender","BMI"])],
                "Clinical":     [c for c in X_all.columns
                                  if any(k in c for k in
                                         ["Smoking","Diabetes",
                                          "Hypertension","Severity"])],
                "System":       [c for c in X_all.columns
                                  if any(k in c for k in
                                         ["Admission","Insurance"])],
            }
            cumulative = []; prev_r2 = 0.0
            interp = "**Hierarchical Blocks:**\n\n"
            last_mdl = None
            for bname, bcols in blocks.items():
                bcols = [c for c in bcols if c in X_all.columns]
                if not bcols:
                    continue
                cumulative += bcols
                X_b = sm.add_constant(X_all[cumulative])
                try:
                    mdl    = sm.OLS(y, X_b).fit()
                    delta  = round((mdl.rsquared - prev_r2)*100, 1)
                    interp += (f"- **{bname}**: R²="
                               f"{round(mdl.rsquared,3)} (Δ=+{delta}%)\n")
                    prev_r2  = mdl.rsquared
                    last_mdl = mdl
                except Exception:
                    pass
            if last_mdl is not None:
                models[f"Hierarchical: {outvar}"] = {
                    "result": last_mdl,
                    "interp": interp,
                    "type":   "hierarchical",
                }
        self.results["hierarchical"] = models
        return models

    # ── 9. Ridge / Lasso / ElasticNet ─────────────────────────
    def penalised_regression(self):
        outvar = self.outcomes[0]
        sub    = self._build_df(outvar)
        if sub is None or not pd.api.types.is_numeric_dtype(sub[outvar]):
            return {}
        X = self._encode(sub, self.predictors)
        y = sub[outvar].values
        sc  = StandardScaler()
        Xs  = sc.fit_transform(X)
        kf  = KFold(5, shuffle=True, random_state=42)
        out = {}
        for name, mdl in [("Ridge",       Ridge(alpha=1.0)),
                           ("Lasso",       Lasso(alpha=0.1, max_iter=5000)),
                           ("ElasticNet",  ElasticNet(alpha=0.1, l1_ratio=0.5,
                                                       max_iter=5000))]:
            cv_r2 = cross_val_score(mdl, Xs, y, cv=kf,
                                    scoring="r2").mean()
            mdl.fit(Xs, y)
            out[name] = {"model": mdl, "cv_r2": round(cv_r2, 4),
                         "features": list(X.columns)}
        self.results["penalised"] = out
        return out

    # ── 10. Logistic Regression ───────────────────────────────
    def logistic_regression(self):
        models = {}
        for outvar in self.outcomes:
            sub = self._build_df(outvar)
            if sub is None:
                continue
            y = sub[outvar]
            if y.nunique() != 2:
                continue
            y_bin = (y == y.unique()[1]).astype(int)
            X = self._encode(sub, self.predictors)
            X_sm = sm.add_constant(X)
            try:
                mdl = sm.Logit(y_bin, X_sm).fit(disp=False,
                                                  maxiter=200)
                prob = mdl.predict(X_sm)
                auc  = roc_auc_score(y_bin, prob)
                models[f"Logistic: {outvar}"] = {
                    "result": mdl,
                    "auc":    round(auc, 4),
                    "interp": interpret_logistic(mdl, auc),
                    "type":   "logistic",
                }
            except Exception:
                pass
        self.results["logistic"] = models
        return models

    # ── 11. Poisson / Negative Binomial ───────────────────────
    def poisson_regression(self):
        models = {}
        for outvar in self.outcomes:
            sub = self._build_df(outvar)
            if sub is None:
                continue
            y = sub[outvar]
            if not pd.api.types.is_numeric_dtype(y):
                continue
            if not (y >= 0).all():
                continue
            X = sm.add_constant(self._encode(sub, self.predictors))
            try:
                pois = sm.GLM(y, X,
                              family=sm.families.Poisson()).fit()
                disp = pois.deviance / pois.df_resid
                models[f"Poisson: {outvar}"] = {
                    "result": pois,
                    "disp":   round(disp, 3),
                    "type":   "poisson",
                    "interp": (
                        f"**Poisson** — Dispersion ratio: {round(disp,2)} "
                        f"{'(OVERDISPERSED — use NegBin)' if disp > 1.5 else '(OK)'}\n"
                    ),
                }
                nb = sm.GLM(y, X,
                            family=sm.families.NegativeBinomial()).fit()
                models[f"NegBin: {outvar}"] = {
                    "result": nb, "type": "negbin",
                    "interp": "**Negative Binomial** fitted.\n",
                }
            except Exception:
                pass
        self.results["poisson"] = models
        return models

    # ── 12. Quantile Regression (median) ─────────────────────
    def quantile_regression(self):
        models = {}
        for outvar in self.outcomes:
            sub = self._build_df(outvar)
            if sub is None or not pd.api.types.is_numeric_dtype(sub[outvar]):
                continue
            X = sm.add_constant(self._encode(sub, self.predictors))
            y = sub[outvar]
            try:
                mdl = sm.QuantReg(y, X).fit(q=0.5)
                models[f"Quantile(50th): {outvar}"] = {
                    "result": mdl, "type": "quantile",
                    "interp": "**Quantile (median) regression** — "
                              "more resistant to outliers than OLS.\n",
                }
            except Exception:
                pass
        self.results["quantile"] = models
        return models

    # ── 13. Robust Regression (RLM) ───────────────────────────
    def robust_regression(self):
        models = {}
        for outvar in self.outcomes:
            sub = self._build_df(outvar)
            if sub is None or not pd.api.types.is_numeric_dtype(sub[outvar]):
                continue
            X = sm.add_constant(self._encode(sub, self.predictors))
            y = sub[outvar]
            try:
                mdl = sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit()
                models[f"Robust: {outvar}"] = {
                    "result": mdl, "type": "robust",
                    "interp": "**Robust (Huber M-estimator)** — "
                              "downweights influential outliers automatically.\n",
                }
            except Exception:
                pass
        self.results["robust"] = models
        return models

    # ── 14. Random Forest ─────────────────────────────────────
    def random_forest(self):
        outvar = self.outcomes[0]
        sub    = self._build_df(outvar)
        if sub is None or not pd.api.types.is_numeric_dtype(sub[outvar]):
            return None
        if len(sub) > 20000:
            sub = sub.sample(20000, random_state=42)
        X = self._encode(sub, self.predictors)
        y = sub[outvar].values
        rf = RandomForestRegressor(n_estimators=300, random_state=42,
                                    n_jobs=-1)
        rf.fit(X, y)
        cv_r2 = cross_val_score(rf, X, y, cv=5,
                                scoring="r2").mean()
        self.results["rf"] = {
            "model":    rf,
            "features": list(X.columns),
            "cv_r2":    round(cv_r2, 4),
            "interp":   interpret_rf(rf, list(X.columns)),
        }
        return self.results["rf"]

    # ── 15. XGBoost (5-fold CV) ───────────────────────────────
    def xgboost_model(self):
        outvar = self.outcomes[0]
        sub    = self._build_df(outvar)
        if sub is None or not pd.api.types.is_numeric_dtype(sub[outvar]):
            return None
        if len(sub) > 30000:
            sub = sub.sample(30000, random_state=42)
        X = self._encode(sub, self.predictors)
        y = sub[outvar].values
        dtrain = xgb.DMatrix(X, label=y,
                             feature_names=list(X.columns))
        cv_res = xgb.cv(
            {"max_depth": 5, "eta": 0.1, "subsample": 0.8,
             "objective": "reg:squarederror", "seed": 42},
            dtrain, num_boost_round=300, nfold=5, verbose_eval=False,
            early_stopping_rounds=20,
        )
        best_n = int(cv_res["test-rmse-mean"].idxmin()) + 1
        mdl    = xgb.train(
            {"max_depth": 5, "eta": 0.1, "subsample": 0.8,
             "objective": "reg:squarederror", "seed": 42},
            dtrain, num_boost_round=best_n,
        )
        # SHAP
        explainer  = shap.TreeExplainer(mdl)
        sn         = min(2000, len(X))
        X_shap     = X.sample(sn, random_state=1)
        shap_vals  = explainer.shap_values(X_shap)
        best_rmse  = round(cv_res["test-rmse-mean"].min(), 2)
        self.results["xgb"] = {
            "model":      mdl,
            "shap_vals":  shap_vals,
            "shap_X":     X_shap,
            "features":   list(X.columns),
            "best_iter":  best_n,
            "cv_rmse":    best_rmse,
            "interp": (f"**XGBoost (5-fold CV):**\n\n"
                       f"- Best iteration: {best_n}\n"
                       f"- CV RMSE: {best_rmse}\n"),
        }
        return self.results["xgb"]

    # ── 16. SVR ───────────────────────────────────────────────
    def svr_model(self):
        outvar = self.outcomes[0]
        sub    = self._build_df(outvar)
        if sub is None or not pd.api.types.is_numeric_dtype(sub[outvar]):
            return None
        if len(sub) > 10000:
            sub = sub.sample(10000, random_state=42)
        X  = self._encode(sub, self.predictors)
        y  = sub[outvar].values
        sc = StandardScaler()
        Xs = sc.fit_transform(X)
        svr = SVR(kernel="rbf")
        cv_r2 = cross_val_score(svr, Xs, y, cv=5,
                                scoring="r2").mean()
        svr.fit(Xs, y)
        pred = svr.predict(Xs)
        rmse = round(np.sqrt(mean_squared_error(y, pred)), 2)
        r2   = round(r2_score(y, pred), 3)
        self.results["svr"] = {
            "model": svr, "rmse": rmse, "r2": r2,
            "cv_r2": round(cv_r2, 4),
            "interp": f"**SVR (RBF kernel):** R²={r2}, RMSE={rmse}\n",
        }
        return self.results["svr"]

    # ── 17. KNN ───────────────────────────────────────────────
    def knn_model(self):
        outvar = self.outcomes[0]
        sub    = self._build_df(outvar)
        if sub is None or not pd.api.types.is_numeric_dtype(sub[outvar]):
            return None
        if len(sub) > 10000:
            sub = sub.sample(10000, random_state=42)
        X  = self._encode(sub, self.predictors)
        y  = sub[outvar].values
        sc = StandardScaler()
        Xs = sc.fit_transform(X)
        k  = max(3, round(np.sqrt(len(Xs))))
        knn = KNeighborsRegressor(n_neighbors=k, n_jobs=-1)
        cv_r2 = cross_val_score(knn, Xs, y, cv=5,
                                scoring="r2").mean()
        knn.fit(Xs, y)
        pred = knn.predict(Xs)
        rmse = round(np.sqrt(mean_squared_error(y, pred)), 2)
        r2   = round(r2_score(y, pred), 3)
        self.results["knn"] = {
            "model": knn, "k": k, "rmse": rmse, "r2": r2,
            "cv_r2": round(cv_r2, 4),
            "interp": f"**KNN (k={k}):** R²={r2}, RMSE={rmse}\n",
        }
        return self.results["knn"]

    # ── 18. Neural Net (MLP) ──────────────────────────────────
    def neural_net(self):
        outvar = self.outcomes[0]
        sub    = self._build_df(outvar)
        if sub is None or not pd.api.types.is_numeric_dtype(sub[outvar]):
            return None
        if len(sub) > 15000:
            sub = sub.sample(15000, random_state=42)
        X  = self._encode(sub, self.predictors)
        y  = sub[outvar].values
        sc = StandardScaler()
        Xs = sc.fit_transform(X)
        nn = MLPRegressor(hidden_layer_sizes=(64, 32),
                          max_iter=300, random_state=42,
                          early_stopping=True)
        cv_r2 = cross_val_score(nn, Xs, y, cv=5,
                                scoring="r2").mean()
        nn.fit(Xs, y)
        pred = nn.predict(Xs)
        rmse = round(np.sqrt(mean_squared_error(y, pred)), 2)
        r2   = round(r2_score(y, pred), 3)
        self.results["nn"] = {
            "model": nn, "rmse": rmse, "r2": r2,
            "cv_r2": round(cv_r2, 4),
            "interp": f"**Neural Net (MLP 64-32):** R²={r2}, RMSE={rmse}\n",
        }
        return self.results["nn"]

    # ── 19. Survival Analysis ─────────────────────────────────
    def survival_analysis(self):
        try:
            from lifelines import KaplanMeierFitter, CoxPHFitter
        except ImportError:
            return None
        df = self.df.copy()
        needed = ["Time_to_Readmission", "Event_Status"]
        if not all(c in df.columns for c in needed):
            return None
        cpreds = [c for c in
                  ["Age","BMI","Gender","Smoking","Diabetes",
                   "Hypertension","Insurance","Admission","Severity"]
                  if c in df.columns]
        sdf = df[needed + cpreds].dropna()
        sdf = sdf[sdf["Time_to_Readmission"] > 0]
        if len(sdf) > 20000:
            sdf = sdf.sample(20000, random_state=42)
        if len(sdf) < 50:
            return None
        # KM
        kmf   = KaplanMeierFitter()
        km_fig = go.Figure()
        if "Admission" in sdf.columns:
            for grp in sdf["Admission"].unique():
                g = sdf[sdf["Admission"] == grp]
                kmf.fit(g["Time_to_Readmission"], g["Event_Status"],
                        label=grp)
                km_fig.add_trace(go.Scatter(
                    x=kmf.survival_function_.index,
                    y=kmf.survival_function_.iloc[:, 0],
                    name=grp, mode="lines"))
        else:
            kmf.fit(sdf["Time_to_Readmission"], sdf["Event_Status"])
            km_fig.add_trace(go.Scatter(
                x=kmf.survival_function_.index,
                y=kmf.survival_function_.iloc[:, 0],
                name="Overall", mode="lines"))
        km_fig.update_layout(title="Kaplan-Meier Survival Curves",
                              xaxis_title="Days",
                              yaxis_title="Survival Probability")
        # Cox
        cox_df = sdf.copy()
        for c in cox_df.select_dtypes(include="object").columns:
            cox_df[c] = pd.Categorical(cox_df[c]).codes
        cox   = CoxPHFitter()
        cox.fit(cox_df, "Time_to_Readmission", "Event_Status")
        self.results["survival"] = {
            "km_fig": km_fig, "cox": cox,
            "interp": interpret_survival(cox),
        }
        return self.results["survival"]

    # ── 20. K-Means Clustering ────────────────────────────────
    def clustering(self):
        cols = [c for c in self.outcomes + self.predictors
                if c in self.df.columns and
                pd.api.types.is_numeric_dtype(self.df[c]) and
                c != "PatientID"]
        if len(cols) < 2:
            return None
        sub = self.df[cols].dropna()
        if len(sub) > 5000:
            sub = sub.sample(5000, random_state=42)
        sc  = StandardScaler()
        Xs  = sc.fit_transform(sub)
        km  = KMeans(n_clusters=4, random_state=42, n_init=25)
        km.fit(Xs)
        pca = PCA(n_components=2, random_state=42)
        pc  = pca.fit_transform(Xs)
        fig = px.scatter(x=pc[:, 0], y=pc[:, 1],
                         color=km.labels_.astype(str),
                         title="K-Means Patient Clusters (k=4, PCA view)",
                         labels={"x": "PC1", "y": "PC2",
                                 "color": "Cluster"})
        self.results["cluster"] = {
            "km": km, "fig": fig,
            "interp": interpret_clustering(km),
        }
        return self.results["cluster"]

    # ── 21. PCA / t-SNE ───────────────────────────────────────
    def pca_tsne(self):
        cols = [c for c in self.outcomes + self.predictors
                if c in self.df.columns and
                pd.api.types.is_numeric_dtype(self.df[c]) and
                c != "PatientID"]
        if len(cols) < 3:
            return None
        sub = self.df[cols].dropna()
        if len(sub) > 5000:
            sub = sub.sample(5000, random_state=42)
        Xs  = StandardScaler().fit_transform(sub)
        pca = PCA(n_components=min(len(cols), 10), random_state=42)
        pc  = pca.fit_transform(Xs)
        var_exp = pca.explained_variance_ratio_[:2] * 100
        pca_fig = px.scatter(x=pc[:, 0], y=pc[:, 1], opacity=0.3,
                             title="PCA (first 2 components)",
                             labels={
                                 "x": f"PC1 ({var_exp[0]:.1f}%)",
                                 "y": f"PC2 ({var_exp[1]:.1f}%)"})
        tsne_fig = None
        if len(sub) <= 5000:
            tsne = TSNE(n_components=2, perplexity=30,
                        random_state=42, n_jobs=-1)
            emb  = tsne.fit_transform(Xs)
            tsne_fig = px.scatter(x=emb[:, 0], y=emb[:, 1],
                                  opacity=0.3,
                                  title="t-SNE Embedding",
                                  labels={"x": "D1", "y": "D2"})
        self.results["pca"] = {
            "pca_fig":  pca_fig,
            "tsne_fig": tsne_fig,
            "var_exp":  var_exp,
            "interp":   (f"**PCA:** First 2 PCs explain "
                         f"{var_exp.sum():.1f}% of variance.\n"),
        }
        return self.results["pca"]

    # ── 22. Outlier Detection ─────────────────────────────────
    def outlier_detection(self):
        outvar = self.outcomes[0]
        sub    = self._build_df(outvar)
        if sub is None or not pd.api.types.is_numeric_dtype(sub[outvar]):
            return None
        X  = sm.add_constant(self._encode(sub, self.predictors))
        y  = sub[outvar]
        mdl = sm.OLS(y, X).fit()
        infl = mdl.get_influence()
        cooks_d, _  = infl.cooks_distance
        leverage    = infl.hat_matrix_diag
        rstud       = infl.resid_studentized_external
        threshold_c = 4 / len(sub)
        threshold_l = 2 * leverage.mean()
        outlier_mask = (
            (cooks_d > threshold_c) |
            (np.abs(rstud) > 3) |
            (leverage > threshold_l)
        )
        n_out = outlier_mask.sum()
        idx   = np.where(outlier_mask)[0]
        fig   = go.Figure()
        fig.add_trace(go.Bar(y=cooks_d[idx], name="Cook's D",
                              marker_color="crimson"))
        fig.update_layout(title="Cook's Distance (Flagged Outliers)",
                          xaxis_title="Observation index",
                          yaxis_title="Cook's D")
        self.results["outliers"] = {
            "fig":   fig,
            "n_out": int(n_out),
            "pct":   round(n_out / len(sub) * 100, 1),
            "interp": (
                f"**Outlier Detection:** Found {n_out} potential outliers "
                f"({round(n_out/len(sub)*100,1)}%) — Cook's D > {threshold_c:.4f}, "
                f"|studentized residual| > 3, or leverage > {threshold_l:.4f}.\n"
            ),
        }
        return self.results["outliers"]

    # ── 23. Cost & Disparity Analysis ────────────────────────
    def cost_disparity(self):
        df = self.df.copy()
        if not all(c in df.columns for c in
                   ["Insurance", "Treatment_Cost"]):
            return None
        grp = df.groupby("Insurance").agg(
            Mean_Cost   =("Treatment_Cost", "mean"),
            Median_Cost =("Treatment_Cost", "median"),
            SD_Cost     =("Treatment_Cost", "std"),
            N           =("Treatment_Cost", "count"),
            Mean_LOS    =("Length_of_Stay", "mean")
            if "Length_of_Stay" in df.columns else
            ("Treatment_Cost", lambda x: np.nan),
        ).reset_index()
        if "Readmitted_90d" in df.columns:
            rr = df.groupby("Insurance")["Readmitted_90d"].apply(
                lambda x: (x == "Yes").mean() * 100
            ).reset_index(name="Readmit_Rate")
            grp = grp.merge(rr, on="Insurance")
        fig = px.bar(grp, x="Insurance", y="Mean_Cost",
                     color="Insurance",
                     error_y=grp["SD_Cost"] / np.sqrt(grp["N"]),
                     title="Mean Treatment Cost by Insurance",
                     labels={"Mean_Cost": "Mean Cost ($)"})
        self.results["cost"] = {
            "table": grp, "fig": fig,
            "interp": interpret_policy(grp),
        }
        return self.results["cost"]

    # ── 24. Causal Inference (PSM) ────────────────────────────
    def causal_inference(self):
        outvar = self.outcomes[0]
        sub    = self._build_df(outvar)
        if sub is None or not pd.api.types.is_numeric_dtype(sub[outvar]):
            return None
        treat_var = None
        for p in self.predictors:
            if p in sub.columns and sub[p].nunique() == 2:
                treat_var = p
                break
        if treat_var is None:
            return None
        df_c = sub.copy()
        df_c["_treat"] = (pd.Categorical(df_c[treat_var]).codes
                          if not pd.api.types.is_numeric_dtype(df_c[treat_var])
                          else df_c[treat_var].values)
        confs = [c for c in self.predictors
                 if c != treat_var and c in df_c.columns]
        if not confs:
            return None
        X_ps = self._encode(df_c, confs)
        lr   = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_ps, df_c["_treat"])
        df_c["_ps"] = lr.predict_proba(X_ps)[:, 1]
        treated = df_c[df_c["_treat"] == 1].copy()
        control = df_c[df_c["_treat"] == 0].copy()
        if len(treated) > 5000:
            treated = treated.sample(5000, random_state=42)
        mt, mc = [], []
        c_ps = control["_ps"].values
        c_y  = control[outvar].values
        for _, row in treated.iterrows():
            best = np.argmin(np.abs(c_ps - row["_ps"]))
            mt.append(row[outvar])
            mc.append(c_y[best])
        mt, mc = np.array(mt), np.array(mc)
        diff   = mt - mc
        ate    = diff.mean()
        se     = diff.std() / np.sqrt(len(diff))
        ci     = (ate - 1.96*se, ate + 1.96*se)
        fig    = px.histogram(df_c, x="_ps",
                               color=df_c["_treat"].map(
                                   {0: "Control", 1: "Treated"}),
                               barmode="overlay", opacity=0.6,
                               title="Propensity Score Overlap",
                               labels={"_ps": "Propensity Score",
                                       "color": treat_var})
        sig = "Statistically significant" if abs(ate) > 1.96*se \
            else "Not significant"
        self.results["causal"] = {
            "ate": round(ate, 4), "se": round(se, 4),
            "ci":  (round(ci[0], 4), round(ci[1], 4)),
            "n_matched": len(mt),
            "treat_var": treat_var,
            "outcome":   outvar,
            "fig":       fig,
            "interp": (
                f"**Causal (PSM):** Treatment={treat_var}, "
                f"Outcome={outvar}\n"
                f"- ATE={round(ate,3)} (SE={round(se,3)})\n"
                f"- 95% CI: [{round(ci[0],3)}, {round(ci[1],3)}]\n"
                f"- {len(mt)} matched pairs\n"
                f"- {sig} at p<0.05\n"
            ),
        }
        return self.results["causal"]

    # ── 25. Time-Series Forecasting ───────────────────────────
    def time_series_forecast(self, target="Bed_Occupancy",
                              ts_df=None):
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
        except ImportError:
            return None
        if ts_df is None:
            ts_df = TS_DATA.copy()
        if target not in ts_df.columns:
            target = "Bed_Occupancy"
        y     = ts_df[target].values
        n     = len(y)
        if n < 24:
            return None
        n_tr  = round(n * 0.8)
        y_tr  = y[:n_tr]
        y_te  = y[n_tr:]
        h_te  = len(y_te)

        def fit_arima(ys, h):
            mdl = SARIMAX(ys, order=(1, 1, 1),
                          seasonal_order=(1, 1, 1, 12),
                          enforce_stationarity=False,
                          enforce_invertibility=False).fit(disp=False)
            return mdl, mdl.forecast(h)

        def fit_ets(ys, h):
            mdl = ExponentialSmoothing(
                ys, seasonal="add", seasonal_periods=12).fit()
            return mdl, mdl.forecast(h)

        arima_tr, fc_arima_te = fit_arima(y_tr, h_te)
        rmse_arima = np.sqrt(mean_squared_error(y_te, fc_arima_te))
        try:
            ets_tr, fc_ets_te = fit_ets(y_tr, h_te)
            rmse_ets = np.sqrt(mean_squared_error(y_te, fc_ets_te))
        except Exception:
            ets_tr, fc_ets_te, rmse_ets = None, None, np.inf

        if rmse_arima <= rmse_ets:
            best_mdl, best_name = fit_arima(y, 12)[0], "SARIMA(1,1,1)(1,1,1,12)"
            best_rmse = rmse_arima
        else:
            best_mdl, best_name = fit_ets(y, 12)[0], "ETS (Holt-Winters)"
            best_rmse = rmse_ets

        fc    = best_mdl.forecast(12)
        ci95  = getattr(best_mdl, "get_forecast", None)
        fc_df = pd.DataFrame({
            "Period":   range(1, 13),
            "Forecast": np.round(fc, 1),
        })

        # Build figure
        hist_dates = list(ts_df["Date"].astype(str))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist_dates, y=y,
                                  name="Historical", mode="lines"))
        fig.add_trace(go.Scatter(
            x=[f"Month+{i}" for i in range(1, 13)],
            y=np.round(fc, 1),
            name=f"Forecast ({best_name})", mode="lines",
            line=dict(dash="dash")))
        fig.update_layout(title=f"12-Month Forecast — {target}",
                          xaxis_title="Date",
                          yaxis_title=target)

        comp_df = pd.DataFrame({
            "Model":        ["SARIMA", "ETS"],
            "Holdout_RMSE": [round(rmse_arima, 2),
                             round(rmse_ets, 2) if np.isfinite(rmse_ets)
                             else None],
            "Selected":     [rmse_arima <= rmse_ets,
                             rmse_ets < rmse_arima],
        })

        self.results["ts"] = {
            "fig":        fig,
            "fc_df":      fc_df,
            "comp_df":    comp_df,
            "best_name":  best_name,
            "best_rmse":  round(best_rmse, 2),
            "interp": (
                f"**Forecast ({target}):**\n\n"
                f"- Best model: **{best_name}**\n"
                f"- Holdout RMSE: {round(best_rmse,2)}\n"
                f"- SARIMA RMSE: {round(rmse_arima,2)}\n"
                f"- ETS RMSE: "
                f"{round(rmse_ets,2) if np.isfinite(rmse_ets) else 'N/A'}\n"
                f"- Mean 12-period forecast: {round(np.mean(fc),1)}\n"
            ),
        }
        return self.results["ts"]


# ─────────────────────────────────────────────────────────────
# 5.  DASH APP
# ─────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.COSMO,
                           dbc.icons.FONT_AWESOME],
    suppress_callback_exceptions=True,
    title="HealthStat Pro",
)

# ── KPI values ────────────────────────────────────────────────
_avg_los   = round(SIM_DATA["Length_of_Stay"].mean(), 1)
_readmit   = round((SIM_DATA["Readmitted_90d"] == "Yes").mean()*100, 1)
_avg_cost  = int(SIM_DATA["Treatment_Cost"].mean())

SIDEBAR = dbc.Card([
    dbc.CardHeader(html.B("📊 HealthStat Pro",
                           className="text-white"),
                   style={"background": "linear-gradient(90deg,#1a2b4a,#2563eb)"}),
    dbc.CardBody([

        # Upload
        dcc.Upload(
            id="upload-data",
            children=html.Div([
                html.I(className="fas fa-cloud-upload-alt fa-2x mb-2",
                       style={"color":"#94a3b8"}),
                html.Div("Drag & drop or click to upload"),
                html.Div("CSV, Excel, JSON — max 50 MB",
                         className="text-muted small"),
            ], className="text-center"),
            style={
                "border": "3px dashed #94a3b8",
                "borderRadius": "12px", "padding": "24px",
                "cursor": "pointer", "background": "#f8fafc",
            },
        ),
        html.Div([
            dbc.Button("Clear / Use Sample", id="btn-clear",
                       color="outline-secondary", size="sm",
                       className="w-100 mt-2"),
        ]),
        html.Div(id="upload-status", className="mt-2"),
        html.Hr(),

        # Outcome selector
        html.Label("Outcome Variable(s)", className="fw-semibold"),
        dcc.Dropdown(id="dd-outcome", multi=True,
                     placeholder="Select outcome(s)…"),
        html.Div(id="outcome-hint", className="small text-muted mt-1"),

        # Predictors
        html.Label("Predictors", className="fw-semibold mt-2"),
        dcc.Dropdown(id="dd-predictors", multi=True,
                     placeholder="Select predictors…"),

        html.Hr(),

        # Module selection
        html.Label("Analysis Modules", className="fw-semibold"),
        dcc.Dropdown(
            id="dd-modules",
            multi=True,
            options=[
                {"label": "Descriptive Statistics",  "value": "descriptive"},
                {"label": "Visualization",            "value": "visualization"},
                {"label": "Correlation Matrix",       "value": "correlation"},
                {"label": "Hypothesis Testing",       "value": "hypothesis"},
                {"label": "Linear Regression",        "value": "linear"},
                {"label": "Polynomial Regression",    "value": "polynomial"},
                {"label": "Stepwise Regression",      "value": "stepwise"},
                {"label": "Hierarchical Regression",  "value": "hierarchical"},
                {"label": "Ridge/Lasso/ElasticNet",   "value": "penalised"},
                {"label": "Logistic Regression",      "value": "logistic"},
                {"label": "Poisson/NegBin",           "value": "poisson"},
                {"label": "Quantile Regression",      "value": "quantile"},
                {"label": "Robust Regression",        "value": "robust"},
                {"label": "Random Forest",            "value": "rf"},
                {"label": "XGBoost",                  "value": "xgb"},
                {"label": "SVR",                      "value": "svr"},
                {"label": "KNN",                      "value": "knn"},
                {"label": "Neural Net (MLP)",         "value": "nn"},
                {"label": "Survival Analysis",        "value": "survival"},
                {"label": "Clustering (K-Means)",     "value": "cluster"},
                {"label": "PCA / t-SNE",              "value": "pca"},
                {"label": "Outlier Detection",        "value": "outliers"},
                {"label": "Cost & Disparity",         "value": "cost"},
                {"label": "Causal Inference (PSM)",   "value": "causal"},
                {"label": "Time-Series Forecast",     "value": "ts"},
            ],
            value=["descriptive", "visualization", "linear"],
        ),
        html.Hr(),

        dbc.Button([html.I(className="fas fa-rocket me-2"),
                    "RUN ANALYSIS"],
                   id="btn-run", color="success", size="lg",
                   className="w-100 fw-bold"),
        dbc.Progress(id="run-progress", value=0,
                     className="mt-2", style={"height": "6px"},
                     animated=True, striped=True),
        html.Div(id="run-status", className="small text-muted mt-1"),
    ]),
], className="sticky-top shadow-sm", style={"borderRadius":"12px"})

# ── Main result tabs ──────────────────────────────────────────
TABS = dbc.Tabs(id="result-tabs", active_tab="tab-overview", children=[
    dbc.Tab(label="Overview",       tab_id="tab-overview"),
    dbc.Tab(label="Data",           tab_id="tab-data"),
    dbc.Tab(label="Quality",        tab_id="tab-quality"),
    dbc.Tab(label="Summary",        tab_id="tab-summary"),
    dbc.Tab(label="Plots",          tab_id="tab-plots"),
    dbc.Tab(label="Correlation",    tab_id="tab-corr"),
    dbc.Tab(label="Regression",     tab_id="tab-regression"),
    dbc.Tab(label="Diagnostics",    tab_id="tab-diagnostics"),
    dbc.Tab(label="Survival",       tab_id="tab-survival"),
    dbc.Tab(label="Clustering",     tab_id="tab-cluster"),
    dbc.Tab(label="Policy",         tab_id="tab-policy"),
    dbc.Tab(label="SHAP",           tab_id="tab-shap"),
    dbc.Tab(label="Compare",        tab_id="tab-compare"),
    dbc.Tab(label="Advanced",       tab_id="tab-advanced"),
    dbc.Tab(label="Causal",         tab_id="tab-causal"),
    dbc.Tab(label="Forecast",       tab_id="tab-forecast"),
])

app.layout = dbc.Container([

    # Navbar
    dbc.Navbar(
        dbc.Container([
            dbc.NavbarBrand([
                html.I(className="fas fa-heartbeat me-2"),
                html.Span("HealthStat", className="fw-bold"),
                dbc.Badge("Pro", color="info", className="ms-1"),
            ]),
        ]),
        color="#1a2b4a", dark=True, className="mb-3",
    ),

    # KPI row
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Div("Total Patients",
                     className="small text-muted"),
            html.Div(f"{N:,}",
                     className="fs-4 fw-bold text-primary"),
        ])), width=3),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Div("Avg LOS (days)", className="small text-muted"),
            html.Div(str(_avg_los),
                     className="fs-4 fw-bold text-success"),
        ])), width=3),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Div("Readmission Rate", className="small text-muted"),
            html.Div(f"{_readmit}%",
                     className="fs-4 fw-bold text-danger"),
        ])), width=3),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Div("Avg Cost", className="small text-muted"),
            html.Div(f"${_avg_cost:,}",
                     className="fs-4 fw-bold text-warning"),
        ])), width=3),
    ], className="mb-3 g-2"),

    # Main layout
    dbc.Row([
        dbc.Col(SIDEBAR, width=3),
        dbc.Col([
            TABS,
            html.Div(id="tab-content", className="mt-3 p-2"),
        ], width=9),
    ]),

    # Hidden stores
    dcc.Store(id="store-data"),
    dcc.Store(id="store-results"),
    dcc.Store(id="store-engine-state"),

], fluid=True)


# ─────────────────────────────────────────────────────────────
# 6.  CALLBACKS
# ─────────────────────────────────────────────────────────────

# ── Load data ─────────────────────────────────────────────────
@app.callback(
    Output("store-data",    "data"),
    Output("upload-status", "children"),
    Output("dd-outcome",    "options"),
    Output("dd-predictors", "options"),
    Output("dd-outcome",    "value"),
    Output("dd-predictors", "value"),
    Input("upload-data", "contents"),
    Input("btn-clear",   "n_clicks"),
    State("upload-data", "filename"),
    prevent_initial_call=False,
)
def load_data(contents, _clear, filename):
    triggered = ctx.triggered_id
    df = SIM_DATA.copy()
    status = dbc.Alert(
        f"Built-in dataset: {len(df):,} patients",
        color="success", className="small py-1")

    if triggered == "upload-data" and contents:
        try:
            df     = read_uploaded_file(contents, filename)
            pii    = check_pii(df)
            hits   = pii["name_hits"] + pii["content_hits"]
            status = dbc.Alert(
                f"Loaded: {filename} ({len(df):,} rows)"
                + (f" | PII detected: {', '.join(hits)}" if hits else ""),
                color="warning" if hits else "success",
                className="small py-1")
        except Exception as e:
            df     = SIM_DATA.copy()
            status = dbc.Alert(str(e), color="danger",
                               className="small py-1")

    cols    = df.columns.tolist()
    opts    = [{"label": c, "value": c} for c in cols]
    def_out = ["Length_of_Stay"] if "Length_of_Stay" in cols else [cols[0]]
    def_prd = [c for c in
               ["Age","Gender","BMI","Smoking","Diabetes",
                "Hypertension","Insurance","Admission","Severity"]
               if c in cols] or cols[1:5]

    return (df.to_json(date_format="iso", orient="split"),
            status, opts, opts, def_out, def_prd)


# ── Run analysis ──────────────────────────────────────────────
@app.callback(
    Output("store-results",    "data"),
    Output("run-status",       "children"),
    Output("run-progress",     "value"),
    Input("btn-run",           "n_clicks"),
    State("store-data",        "data"),
    State("dd-outcome",        "value"),
    State("dd-predictors",     "value"),
    State("dd-modules",        "value"),
    prevent_initial_call=True,
)
def run_analysis(n_clicks, data_json, outcomes, predictors, modules):
    if not n_clicks or not data_json or not outcomes or not predictors:
        return dash.no_update, "Select variables first.", 0

    df  = pd.read_json(data_json, orient="split")
    eng = AnalysisEngine(df, outcomes, predictors)

    # Run selected modules
    completed = []
    total     = len(modules)

    _map = {
        "descriptive":  eng.descriptive,
        "visualization": lambda: None,  # handled in tab callback
        "correlation":  eng.correlation,
        "hypothesis":   eng.hypothesis_testing,
        "linear":       eng.linear_regression,
        "polynomial":   eng.polynomial_regression,
        "stepwise":     eng.stepwise_regression,
        "hierarchical": eng.hierarchical_regression,
        "penalised":    eng.penalised_regression,
        "logistic":     eng.logistic_regression,
        "poisson":      eng.poisson_regression,
        "quantile":     eng.quantile_regression,
        "robust":       eng.robust_regression,
        "rf":           eng.random_forest,
        "xgb":          eng.xgboost_model,
        "svr":          eng.svr_model,
        "knn":          eng.knn_model,
        "nn":           eng.neural_net,
        "survival":     eng.survival_analysis,
        "cluster":      eng.clustering,
        "pca":          eng.pca_tsne,
        "outliers":     eng.outlier_detection,
        "cost":         eng.cost_disparity,
        "causal":       eng.causal_inference,
        "ts":           eng.time_series_forecast,
    }

    errors = []
    for mod in modules:
        fn = _map.get(mod)
        if fn:
            try:
                fn()
                completed.append(mod)
            except Exception as e:
                errors.append(f"{mod}: {str(e)[:60]}")

    # Serialise only what can go in Store
    # (figures, numpy arrays etc. kept in engine — we store a flag)
    summary = {
        "modules_run": completed,
        "errors":      errors,
        "descriptive": (eng.results.get("descriptive", {})
                        .get("table", pd.DataFrame()).to_json(orient="split")
                        if "descriptive" in eng.results else None),
    }

    pct = round(len(completed) / total * 100) if total else 0
    msg = f"✓ {len(completed)}/{total} modules complete"
    if errors:
        msg += f" | {len(errors)} errors"

    # Store engine results in a global cache (simple approach for demo)
    _ENGINE_CACHE["results"] = eng.results
    _ENGINE_CACHE["df"]      = df

    return summary, msg, pct


# Global cache (single-user demo — for production use server-side sessions)
_ENGINE_CACHE: dict = {}


# ── Tab content ───────────────────────────────────────────────
@app.callback(
    Output("tab-content", "children"),
    Input("result-tabs",  "active_tab"),
    State("store-data",   "data"),
    State("store-results","data"),
    State("dd-outcome",   "value"),
    State("dd-predictors","value"),
    prevent_initial_call=False,
)
def render_tab(active_tab, data_json, results_store, outcomes, predictors):

    res = _ENGINE_CACHE.get("results", {})
    df  = _ENGINE_CACHE.get("df", SIM_DATA)

    def not_run(name):
        return dbc.Alert(
            [html.I(className="fas fa-circle-pause me-2"),
             f"'{name}' not yet run. Select it and click RUN."],
            color="secondary")

    def interp_box(text):
        return dbc.Card(dbc.CardBody([
            html.H6([html.I(className="fas fa-brain me-2"),
                     "Interpretation"]),
            dcc.Markdown(text),
        ]), className="mt-3 border-start border-primary border-3")

    # ── Overview ──────────────────────────────────────────────
    if active_tab == "tab-overview":
        return dbc.Card(dbc.CardBody([
            html.H4("Healthcare Intelligence Platform"),
            html.P("25 analysis modules across regression, machine learning, "
                   "survival, clustering, causal inference, and forecasting. "
                   "Upload your data or use the built-in 100K patient dataset."),
            html.Hr(),
            dbc.Row([
                dbc.Col([
                    html.H6("Supported Modules"),
                    html.Ul([html.Li(m) for m in [
                        "Descriptive Statistics",
                        "Visualization (8 plot types)",
                        "Correlation Matrix",
                        "Hypothesis Testing (t / ANOVA / χ²)",
                        "Linear, Polynomial, Stepwise, Hierarchical",
                        "Ridge, Lasso, ElasticNet",
                        "Logistic, Poisson, NegBin, Quantile, Robust",
                        "Random Forest, XGBoost, SVR, KNN, Neural Net",
                        "SHAP explainability",
                        "Survival (KM + Cox)",
                        "K-Means Clustering",
                        "PCA / t-SNE",
                        "Outlier Detection",
                        "Cost & Disparity Analysis",
                        "Causal Inference (PSM)",
                        "Time-Series Forecasting (SARIMA/ETS)",
                    ]]),
                ], width=6),
                dbc.Col([
                    html.H6("Data Format"),
                    html.P("One row per patient. Headers in first row. "
                           "CSV, Excel, or JSON. Max 50 MB."),
                    html.H6("Special columns (auto-detected)"),
                    html.Ul([
                        html.Li(html.Code("Time_to_Readmission — survival")),
                        html.Li(html.Code("Event_Status — 0/1 event")),
                        html.Li(html.Code("Treatment_Cost — cost analysis")),
                        html.Li(html.Code("Insurance — disparity")),
                        html.Li(html.Code("Date — time-series")),
                    ]),
                ], width=6),
            ]),
        ]))

    # ── Data preview ──────────────────────────────────────────
    if active_tab == "tab-data":
        if data_json is None:
            df = SIM_DATA
        else:
            df = pd.read_json(data_json, orient="split")
        return html.Div([
            dbc.Alert([
                html.I(className="fas fa-info-circle me-2"),
                f"Preview: {len(df):,} rows × {df.shape[1]} columns."
            ], color="info", className="small py-1"),
            dash_table.DataTable(
                data=df.head(500).to_dict("records"),
                columns=[{"name": c, "id": c} for c in df.columns],
                page_size=20, filter_action="native",
                sort_action="native", style_table={"overflowX":"auto"},
                style_header={"fontWeight":"bold",
                               "background":"#1a2b4a","color":"white"},
                style_cell={"fontSize":"12px"},
            ),
        ])

    # ── Data Quality ──────────────────────────────────────────
    if active_tab == "tab-quality":
        if data_json is None:
            df = SIM_DATA
        else:
            df = pd.read_json(data_json, orient="split")
        qdf = assess_data_quality(df)
        return html.Div([
            dbc.Alert("Red rows: >5% missing. Outlier count uses IQR×1.5 rule.",
                      color="info", className="small py-1"),
            dash_table.DataTable(
                data=qdf.to_dict("records"),
                columns=[{"name": c, "id": c} for c in qdf.columns],
                style_data_conditional=[
                    {"if": {"filter_query": "{Missing_Pct} > 5",
                             "column_id": "Missing_Pct"},
                     "backgroundColor": "#fef2f2"},
                    {"if": {"filter_query": "{Missing_Pct} > 0 && {Missing_Pct} <= 5",
                             "column_id": "Missing_Pct"},
                     "backgroundColor": "#fffbeb"},
                ],
                style_header={"fontWeight":"bold",
                               "background":"#1a2b4a","color":"white"},
            ),
        ])

    # ── Descriptive Summary ───────────────────────────────────
    if active_tab == "tab-summary":
        if "descriptive" not in res:
            return not_run("Descriptive Statistics")
        tbl = res["descriptive"].get("table", pd.DataFrame())
        return dash_table.DataTable(
            data=tbl.to_dict("records"),
            columns=[{"name": c, "id": c} for c in tbl.columns],
            style_header={"fontWeight":"bold","background":"#1a2b4a",
                          "color":"white"},
            style_cell={"fontSize":"13px"},
        )

    # ── Plots ─────────────────────────────────────────────────
    if active_tab == "tab-plots":
        if data_json is None:
            df_p = SIM_DATA
        else:
            df_p = pd.read_json(data_json, orient="split")
        eng_tmp = AnalysisEngine(df_p,
                                  outcomes or ["Length_of_Stay"],
                                  predictors or ["Age"])
        num_cols = get_numeric_cols(df_p)
        cat_cols = df_p.select_dtypes(include="object").columns.tolist()
        all_cols = df_p.columns.tolist()
        return html.Div([
            dbc.Row([
                dbc.Col(dcc.Dropdown(
                    id="plot-type",
                    options=[
                        {"label":"Histogram",         "value":"histogram"},
                        {"label":"Box Plot",          "value":"boxplot"},
                        {"label":"Violin",            "value":"violin"},
                        {"label":"Scatter",           "value":"scatter"},
                        {"label":"Density",           "value":"density"},
                        {"label":"Bar (categorical)", "value":"bar"},
                        {"label":"Group Comparison",  "value":"group_comparison"},
                    ],
                    value="histogram",
                ), width=3),
                dbc.Col(dcc.Dropdown(
                    id="plot-xvar",
                    options=[{"label":c,"value":c} for c in all_cols],
                    value=num_cols[0] if num_cols else all_cols[0],
                ), width=3),
                dbc.Col(dcc.Dropdown(
                    id="plot-color",
                    options=[{"label":"None","value":"None"}] +
                            [{"label":c,"value":c} for c in cat_cols],
                    value="None",
                ), width=3),
            ], className="mb-3 g-2"),
            dcc.Graph(id="plot-main",
                      figure=eng_tmp.make_plot("histogram",
                                               num_cols[0] if num_cols
                                               else all_cols[0])),
        ])

    # ── Correlation ───────────────────────────────────────────
    if active_tab == "tab-corr":
        if "corr" not in res:
            return not_run("Correlation Matrix")
        corr = res["corr"]
        fig  = px.imshow(corr, text_auto=".2f",
                         color_continuous_scale="RdBu_r",
                         zmin=-1, zmax=1,
                         title="Correlation Matrix")
        return html.Div([dcc.Graph(figure=fig)])

    # ── Regression ────────────────────────────────────────────
    if active_tab == "tab-regression":
        all_models = {}
        for key in ["linear","polynomial","stepwise",
                    "hierarchical","logistic","poisson",
                    "quantile","robust"]:
            if key in res:
                all_models.update(res[key])
        pen = res.get("penalised", {})
        if not all_models and not pen:
            return not_run("Any Regression")
        rows = []
        for nm, m in all_models.items():
            r = m.get("result")
            if r is None:
                continue
            try:
                tbl = r.summary2().tables[1].reset_index()
                tbl.columns = ["Variable","Coef","SE","t/z",
                               "p","CI_low","CI_high"]
                tbl.insert(0, "Model", nm)
                rows.append(tbl)
            except Exception:
                pass
        interps = []
        for nm, m in all_models.items():
            interps.append(html.Div([
                html.H6(f"▶ {nm}"),
                dcc.Markdown(m.get("interp",""))
            ], className="mb-2"))
        if pen:
            pen_txt = "**Ridge/Lasso/ElasticNet (5-fold CV R²):**\n\n"
            for nm, m in pen.items():
                pen_txt += f"- {nm}: CV R²={m['cv_r2']}\n"
            interps.append(dcc.Markdown(pen_txt))
        out = [interp_box(""), html.Div(interps)]
        if rows:
            combined = pd.concat(rows, ignore_index=True)
            out = [dash_table.DataTable(
                data=combined.to_dict("records"),
                columns=[{"name":c,"id":c} for c in combined.columns],
                style_data_conditional=[{
                    "if": {"filter_query": "{p} < 0.05"},
                    "backgroundColor": "#eff6ff",
                }],
                style_header={"fontWeight":"bold",
                               "background":"#1a2b4a","color":"white"},
                style_table={"overflowX":"auto"},
                page_size=20, filter_action="native",
                sort_action="native",
            )] + out
        return html.Div(out)

    # ── Diagnostics ───────────────────────────────────────────
    if active_tab == "tab-diagnostics":
        lin = {k: v for k, v in res.get("linear", {}).items()}
        if not lin:
            return not_run("Linear Regression (for diagnostics)")
        nm, m = next(iter(lin.items()))
        r     = m["result"]
        fitted   = r.fittedvalues
        residuals = r.resid
        fig = make_subplots(rows=2, cols=2,
                            subplot_titles=["Residuals vs Fitted",
                                            "Q-Q Plot",
                                            "Scale-Location",
                                            "Residuals vs Leverage"])
        fig.add_trace(go.Scatter(x=fitted, y=residuals,
                                  mode="markers", opacity=0.3,
                                  name="Residuals",
                                  marker=dict(color="#2563eb")),
                      row=1, col=1)
        fig.add_hline(y=0, row=1, col=1,
                      line_dash="dash", line_color="red")
        # Q-Q
        (osm, osr) = sp_stats.probplot(residuals, dist="norm")
        fig.add_trace(go.Scatter(x=osm[0], y=osm[1],
                                  mode="markers", opacity=0.3,
                                  name="Q-Q",
                                  marker=dict(color="#059669")),
                      row=1, col=2)
        fig.add_trace(go.Scatter(x=osm[0],
                                  y=osm[0]*osr[0]+osr[1],
                                  mode="lines", name="Q-Q line",
                                  line=dict(color="red")),
                      row=1, col=2)
        # Scale-Location
        sqrt_abs_res = np.sqrt(np.abs(residuals))
        fig.add_trace(go.Scatter(x=fitted, y=sqrt_abs_res,
                                  mode="markers", opacity=0.3,
                                  name="Scale-Loc",
                                  marker=dict(color="#d97706")),
                      row=2, col=1)
        fig.update_layout(height=650, title=f"Diagnostics: {nm}",
                          showlegend=False)
        # VIF text
        X = r.model.exog
        feat = r.model.exog_names
        vif_txt = "**VIF:**\n\n"
        try:
            for i, f in enumerate(feat):
                if f == "const":
                    continue
                v = variance_inflation_factor(X, i)
                vif_txt += f"- {f}: {round(v, 2)}\n"
        except Exception as e:
            vif_txt += str(e)
        return html.Div([
            dcc.Graph(figure=fig),
            html.Hr(),
            interp_box(vif_txt),
        ])

    # ── Survival ──────────────────────────────────────────────
    if active_tab == "tab-survival":
        if "survival" not in res:
            return not_run("Survival Analysis")
        sv = res["survival"]
        return html.Div([
            dcc.Graph(figure=sv["km_fig"]),
            interp_box(sv.get("interp", "")),
        ])

    # ── Clustering ────────────────────────────────────────────
    if active_tab == "tab-cluster":
        if "cluster" not in res:
            return not_run("Clustering (K-Means)")
        cl = res["cluster"]
        return html.Div([
            dcc.Graph(figure=cl["fig"]),
            interp_box(cl.get("interp", "")),
        ])

    # ── Policy / Cost ─────────────────────────────────────────
    if active_tab == "tab-policy":
        if "cost" not in res:
            return not_run("Cost & Disparity Analysis")
        co = res["cost"]
        return html.Div([
            dcc.Graph(figure=co["fig"]),
            html.Hr(),
            dash_table.DataTable(
                data=co["table"].round(1).to_dict("records"),
                columns=[{"name":c,"id":c}
                         for c in co["table"].columns],
                style_header={"fontWeight":"bold",
                               "background":"#1a2b4a",
                               "color":"white"},
            ),
            interp_box(co.get("interp", "")),
        ])

    # ── SHAP ──────────────────────────────────────────────────
    if active_tab == "tab-shap":
        if "xgb" not in res:
            return not_run("XGBoost (required for SHAP)")
        xr = res["xgb"]
        sv = xr["shap_vals"]
        fx = xr["shap_X"]
        feats = xr["features"]
        imp = np.abs(sv).mean(axis=0)
        ord_ = np.argsort(imp)[::-1][:15]
        bar = go.Figure(go.Bar(
            y=[feats[i] for i in ord_[::-1]],
            x=imp[ord_[::-1]],
            orientation="h",
            marker_color="#2563eb",
        ))
        bar.update_layout(title="SHAP Feature Importance (mean |SHAP|)",
                          xaxis_title="Mean |SHAP|")
        return html.Div([
            dcc.Graph(figure=bar),
            interp_box(xr.get("interp", "")),
        ])

    # ── Model Comparison ─────────────────────────────────────
    if active_tab == "tab-compare":
        rows = []
        for key, label in [
            ("linear",    "Linear"),
            ("polynomial","Polynomial"),
            ("stepwise",  "Stepwise"),
            ("logistic",  "Logistic"),
            ("rf",        "Random Forest"),
            ("xgb",       "XGBoost"),
            ("svr",       "SVR"),
            ("knn",       "KNN"),
            ("nn",        "Neural Net"),
        ]:
            if key in res:
                d = res[key]
                if isinstance(d, dict) and "cv_r2" in d:
                    rows.append({"Model": label,
                                 "CV R²": d["cv_r2"],
                                 "RMSE":  d.get("rmse","—"),
                                 "AUC":   d.get("auc","—")})
                elif isinstance(d, dict):
                    for nm, m in d.items():
                        r = m.get("result")
                        if r and hasattr(r, "rsquared"):
                            rows.append({"Model": nm,
                                         "CV R²": round(r.rsquared,4),
                                         "RMSE": "—",
                                         "AUC":  "—"})
                        elif r and hasattr(m, "auc"):
                            rows.append({"Model": nm,
                                         "CV R²":"—",
                                         "RMSE": "—",
                                         "AUC":  m.get("auc","—")})
        if not rows:
            return not_run("Model Comparison (run 2+ models)")
        tbl = pd.DataFrame(rows)
        return dash_table.DataTable(
            data=tbl.to_dict("records"),
            columns=[{"name":c,"id":c} for c in tbl.columns],
            style_header={"fontWeight":"bold",
                           "background":"#1a2b4a","color":"white"},
        )

    # ── Advanced ─────────────────────────────────────────────
    if active_tab == "tab-advanced":
        parts = []
        if "hypothesis" in res:
            hyp = res["hypothesis"]
            rows_ = []
            for nm, h in hyp.items():
                r = {"Predictor": nm, "Test": h["test"]}
                for k, v in h.items():
                    if k.endswith("_p"):
                        r[k] = round(v, 6)
                rows_.append(r)
            if rows_:
                parts.append(html.H5("Hypothesis Testing"))
                parts.append(dash_table.DataTable(
                    data=rows_,
                    columns=[{"name":c,"id":c}
                             for c in rows_[0].keys()],
                    style_header={"fontWeight":"bold",
                                   "background":"#1a2b4a",
                                   "color":"white"},
                ))
        if "outliers" in res:
            ot = res["outliers"]
            parts.append(html.Hr())
            parts.append(html.H5("Outlier Detection"))
            parts.append(dcc.Graph(figure=ot["fig"]))
            parts.append(interp_box(ot.get("interp","")))
        if "pca" in res:
            pc = res["pca"]
            parts.append(html.Hr())
            parts.append(html.H5("PCA"))
            parts.append(dcc.Graph(figure=pc["pca_fig"]))
            if pc.get("tsne_fig"):
                parts.append(dcc.Graph(figure=pc["tsne_fig"]))
            parts.append(interp_box(pc.get("interp","")))
        if not parts:
            return not_run("Advanced Analyses")
        return html.Div(parts)

    # ── Causal ────────────────────────────────────────────────
    if active_tab == "tab-causal":
        if "causal" not in res:
            return not_run("Causal Inference")
        ca = res["causal"]
        stats_txt = (
            f"**ATE:** {ca['ate']}  \n"
            f"**SE:** {ca['se']}  \n"
            f"**95% CI:** [{ca['ci'][0]}, {ca['ci'][1]}]  \n"
            f"**Matched pairs:** {ca['n_matched']}"
        )
        return html.Div([
            dcc.Graph(figure=ca["fig"]),
            interp_box(ca.get("interp", "")),
        ])

    # ── Time-Series Forecast ─────────────────────────────────
    if active_tab == "tab-forecast":
        if "ts" not in res:
            return not_run("Time-Series Forecasting")
        ts = res["ts"]
        return html.Div([
            dcc.Graph(figure=ts["fig"]),
            html.H5("12-Period Forecast Table"),
            dash_table.DataTable(
                data=ts["fc_df"].to_dict("records"),
                columns=[{"name":c,"id":c}
                         for c in ts["fc_df"].columns],
                style_header={"fontWeight":"bold",
                               "background":"#1a2b4a",
                               "color":"white"},
            ),
            html.H5("Model Comparison", className="mt-3"),
            dash_table.DataTable(
                data=ts["comp_df"].to_dict("records"),
                columns=[{"name":c,"id":c}
                         for c in ts["comp_df"].columns],
                style_data_conditional=[{
                    "if": {"filter_query":"{Selected} eq True"},
                    "backgroundColor": "#ecfdf5",
                }],
                style_header={"fontWeight":"bold",
                               "background":"#1a2b4a",
                               "color":"white"},
            ),
            interp_box(ts.get("interp", "")),
        ])

    return html.Div("Select a tab.")


# ── Dynamic plot update ───────────────────────────────────────
@app.callback(
    Output("plot-main", "figure"),
    Input("plot-type",  "value"),
    Input("plot-xvar",  "value"),
    Input("plot-color", "value"),
    State("store-data", "data"),
    State("dd-outcome", "value"),
    State("dd-predictors","value"),
    prevent_initial_call=True,
)
def update_plot(plot_type, x_var, color_var, data_json,
                outcomes, predictors):
    if data_json:
        df_p = pd.read_json(data_json, orient="split")
    else:
        df_p = SIM_DATA
    eng = AnalysisEngine(df_p,
                          outcomes or ["Length_of_Stay"],
                          predictors or ["Age"])
    return eng.make_plot(
        plot_type or "histogram",
        x_var or df_p.columns[0],
        color_var if color_var != "None" else None,
    )


# ── Outcome type hint ─────────────────────────────────────────
@app.callback(
    Output("outcome-hint", "children"),
    Input("dd-outcome", "value"),
    State("store-data", "data"),
    prevent_initial_call=True,
)
def outcome_hint(outcomes, data_json):
    if not outcomes:
        return ""
    df = pd.read_json(data_json, orient="split") \
        if data_json else SIM_DATA
    hints = []
    for ov in outcomes:
        if ov not in df.columns:
            continue
        col = df[ov]
        if pd.api.types.is_numeric_dtype(col):
            nu = col.nunique()
            msg = ("→ Binary numeric (logistic or linear)"
                   if nu == 2 else
                   "→ Continuous (linear / quantile / robust)")
        elif hasattr(col, "cat") or col.dtype == "object":
            msg = (f"→ Binary factor (logistic)"
                   if col.nunique() == 2 else
                   f"→ {col.nunique()} levels")
        else:
            msg = "→ Character"
        hints.append(html.Div(f"{ov}: {msg}",
                               className="small text-muted"))
    return html.Div(hints)


# ─────────────────────────────────────────────────────────────
# 7.  RUN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    #app.run(debug=False, host="127.0.0.1", port=8050)
    app.run(host="0.0.0.0", port=10000)
