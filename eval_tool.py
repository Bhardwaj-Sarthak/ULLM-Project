from __future__ import annotations
import re
import json
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional
import math
import statistics
import sys
import traceback

#!/usr/bin/env python
"""
Lightweight evaluation tool inspired by DataSciBench.

Scores a generated notebook + agent message history on:
- Data Understanding (DU)
- Data Preparation (DP)
- Modelling (M)
- Evaluation (E)

Usage:
    python eval_tool.py --notebook notebook.html --history history.json --save-dir artifacts_dir

Inputs:
    notebook_html: Raw HTML content (string) OR path via --notebook
    message_history: JSON list of messages (role/content) OR path via --history
    save_dir: Directory where artifacts (models, metrics, plots) may reside.

No external dependencies beyond standard library.
Heuristic (rule-based) scoring: 0.0 - 1.0 per dimension.
"""


# --------------------------- Data Classes ---------------------------

@dataclass
class DimensionScore:
        score: float
        evidence: List[str]
        rationale: str

@dataclass
class EvaluationReport:
        data_understanding: DimensionScore
        data_preparation: DimensionScore
        modelling: DimensionScore
        evaluation: DimensionScore
        overall_score: float
        summary: str

        def to_dict(self):
                return {
                        "data_understanding": asdict(self.data_understanding),
                        "data_preparation": asdict(self.data_preparation),
                        "modelling": asdict(self.modelling),
                        "evaluation": asdict(self.evaluation),
                        "overall_score": self.overall_score,
                        "summary": self.summary,
                }

# --------------------------- Extraction Helpers ---------------------------

CODE_PATTERN = re.compile(r"<code[^>]*>(.*?)</code>|<pre[^>]*>(.*?)</pre>", re.DOTALL | re.IGNORECASE)

def extract_code_from_html(html: str) -> str:
        snippets = []
        for m in CODE_PATTERN.finditer(html):
                g = m.group(1) or m.group(2) or ""
                # Unescape minimal HTML entities
                g = (g.replace("&lt;", "<")
                             .replace("&gt;", ">")
                             .replace("&amp;", "&"))
                snippets.append(g)
        return "\n\n".join(snippets)

def load_text_maybe_path(value: str) -> str:
        p = Path(value)
        if p.exists() and p.is_file():
                return p.read_text(encoding="utf-8", errors="ignore")
        return value

def load_json_maybe_path(value: str) -> Any:
        p = Path(value)
        if p.exists() and p.is_file():
                return json.loads(p.read_text(encoding="utf-8", errors="ignore"))
        # else try to parse raw string
        return json.loads(value)

# --------------------------- Heuristic Feature Detection ---------------------------

def _count_patterns(patterns: List[str], text: str) -> int:
        return sum(1 for pat in patterns if re.search(pat, text))

def pattern_hits(patterns: List[str], text: str) -> List[str]:
        hits = []
        for pat in patterns:
                if re.search(pat, text):
                        hits.append(pat)
        return hits

# Define signals per dimension
DU_PATTERNS = {
        "basic_inspection": [r"\.head\(", r"\.tail\(", r"\.sample\(", r"\.shape", r"\.info\("],
        "summary_stats": [r"\.describe\(", r"\.nunique\(", r"value_counts\(", r"\.corr\("],
        "missing_overview": [r"isnull\(\)\.sum\(", r"isna\(\)\.sum\("],
        "visual_eda": [r"hist\(", r"pairplot\(", r"heatmap\(", r"scatter", r"seaborn", r"plt\.plot", r"plt\.figure"]
}

DP_PATTERNS = {
        "missing_handling": [r"fillna\(", r"dropna\(", r"SimpleImputer", r"KNNImputer", r"IterativeImputer"],
        "scaling": [r"StandardScaler", r"MinMaxScaler", r"RobustScaler", r"Normalizer"],
        "encoding": [r"get_dummies\(", r"OneHotEncoder", r"LabelEncoder", r"OrdinalEncoder"],
        "splitting": [r"train_test_split\("],
        "feature_engineering": [r"FeatureUnion", r"ColumnTransformer", r"PolynomialFeatures", r"feature_importances_", r"selectKBest", r"SelectKBest", r"PCA\("],
        "pipeline": [r"Pipeline\(", r"make_column_selector", r"make_column_transformer"]
}

M_PATTERNS = {
        "classical_models": [
                r"LogisticRegression", r"RandomForest", r"GradientBoosting",
                r"XGBClassifier", r"XGBRegressor", r"LGBMClassifier", r"LGBMRegressor",
                r"CatBoost", r"SVC", r"KNeighbors", r"LinearRegression", r"Ridge\(", r"Lasso\("
        ],
        "ml_training": [r"\.fit\("],
        "dl_frameworks": [r"torch\.", r"tensorflow", r"keras", r"Sequential\("],
        "cv": [r"cross_val_score", r"GridSearchCV", r"RandomizedSearchCV", r"StratifiedKFold", r"KFold\("]
}

E_PATTERNS = {
        "classification_metrics": [r"accuracy_score", r"precision_score", r"recall_score", r"f1_score", r"classification_report", r"roc_auc_score", r"confusion_matrix"],
        "regression_metrics": [r"mean_squared_error", r"r2_score", r"mean_absolute_error", r"mape", r"median_absolute_error"],
        "model_diagnostics": [r"learning_curve", r"validation_curve", r"permutation_importance", r"shap", r"PartialDependenceDisplay", r"roc_curve", r"precision_recall_curve"],
        "interpretability": [r"feature_importances_", r"shap", r"LIME", r"permutation_importance"]
}

def score_dimension(code: str, message_text: str) -> Dict[str, DimensionScore]:
        # Combine code & messages for richer detection
        combined = code + "\n" + message_text
        results = {}

        def compute_score(group_patterns: Dict[str, List[str]], name: str) -> DimensionScore:
                evid = []
                sub_scores = []
                for group, pats in group_patterns.items():
                        hits = pattern_hits(pats, combined)
                        if hits:
                                evid.extend([f"{group}:{h}" for h in hits])
                        # group contribution: min(1, hits_count / expected)
                        group_score = min(1.0, len(hits) / max(1, len(pats)))
                        sub_scores.append(group_score)
                if not sub_scores:
                        score_val = 0.0
                else:
                        # Weighted: reward breadth (coverage) and depth
                        coverage = sum(1 for s in sub_scores if s > 0) / len(sub_scores)
                        depth = statistics.fmean(sub_scores)
                        score_val = round((0.6 * coverage + 0.4 * depth), 3)
                coverage_str = f"{coverage:.2f}" if sub_scores else "0.00"
                depth_str = f"{depth:.2f}" if sub_scores else "0.00"
                rationale = f"{name} coverage={coverage_str}, depth={depth_str}"
                return DimensionScore(score=score_val, evidence=evid, rationale=rationale)

        results["DU"] = compute_score(DU_PATTERNS, "Data Understanding")
        results["DP"] = compute_score(DP_PATTERNS, "Data Preparation")
        results["M"]  = compute_score(M_PATTERNS, "Modelling")
        results["E"]  = compute_score(E_PATTERNS, "Evaluation")
        return results

# --------------------------- Message Analysis ---------------------------

def flatten_messages(message_history: List[Dict[str, Any]]) -> str:
        parts = []
        for m in message_history:
                role = m.get("role", "")
                content = m.get("content", "")
                if isinstance(content, list):
                        content = " ".join(str(x) for x in content)
                parts.append(f"[{role}] {content}")
        return "\n".join(parts)

def intent_bonus(text: str) -> Dict[str, float]:
        bonuses = {"DU":0.0,"DP":0.0,"M":0.0,"E":0.0}
        lower = text.lower()
        if any(k in lower for k in ["explore","understand","eda","overview","distribution","correlation"]):
                bonuses["DU"] += 0.05
        if any(k in lower for k in ["clean","preprocess","prepare","impute","encode","scal","feature"]):
                bonuses["DP"] += 0.05
        if any(k in lower for k in ["train","model","fit","hyperparameter","tune","architecture"]):
                bonuses["M"] += 0.05
        if any(k in lower for k in ["evaluate","metric","accuracy","f1","roc","mse","r2","performance"]):
                bonuses["E"] += 0.05
        return bonuses

# --------------------------- Artifact Inspection ---------------------------

def inspect_artifacts(save_dir: str) -> Dict[str, float]:
        """
        Look for presence of artifacts that indicate robust workflow.
        Example: metrics.json, model.* (pkl, joblib), figures, reports.
        """
        bonuses = {"DU":0.0,"DP":0.0,"M":0.0,"E":0.0}
        p = Path(save_dir)
        if not p.exists() or not p.is_dir():
                return bonuses
        files = list(p.rglob("*"))
        names = [f.name.lower() for f in files if f.is_file()]

        # Data dictionary or profiling
        if any("profile" in n or "eda" in n for n in names):
                bonuses["DU"] += 0.05

        # Preprocessing pipeline object
        if any("pipeline" in n for n in names):
                bonuses["DP"] += 0.05

        # Model file
        if any(n.endswith((".pkl",".joblib",".pt",".h5")) for n in names):
                bonuses["M"] += 0.05

        # Metrics json
        metric_files = [f for f in files if f.name.lower().startswith("metrics") and f.suffix in (".json",".txt")]
        if metric_files:
                try:
                        for mf in metric_files:
                                txt = mf.read_text(encoding="utf-8")
                                if any(k in txt.lower() for k in ["accuracy","f1","roc","mse","r2","mae","precision","recall"]):
                                        bonuses["E"] += 0.05
                                        break
                except Exception:
                        pass

        # Plots
        if any(n.endswith((".png",".jpg",".jpeg",".svg")) for n in names):
                bonuses["DU"] = min(0.1, bonuses["DU"] + 0.05)  # small extra for visualization artifacts
        return bonuses

# --------------------------- Main Evaluation Logic ---------------------------

def clamp(v: float, lo: float=0.0, hi: float=1.0) -> float:
        return max(lo, min(hi, v))

def evaluate_session(
        notebook_html: str,
        message_history: List[Dict[str, Any]],
        save_dir: Optional[str] = None
) -> EvaluationReport:
        try:
                code = extract_code_from_html(notebook_html)
                messages_text = flatten_messages(message_history)

                dimension_scores = score_dimension(code, messages_text)

                # Apply intent and artifact bonuses
                ib = intent_bonus(messages_text)
                ab = inspect_artifacts(save_dir) if save_dir else {"DU":0,"DP":0,"M":0,"E":0}

                du = dimension_scores["DU"]
                dp = dimension_scores["DP"]
                m  = dimension_scores["M"]
                e  = dimension_scores["E"]

                du.score = clamp(du.score + ib["DU"] + ab["DU"])
                dp.score = clamp(dp.score + ib["DP"] + ab["DP"])
                m.score  = clamp(m.score  + ib["M"]  + ab["M"])
                e.score  = clamp(e.score  + ib["E"]  + ab["E"])

                overall = round(statistics.fmean([du.score, dp.score, m.score, e.score]), 3)

                summary_parts = []
                def qual(dim: DimensionScore, label: str):
                        if dim.score >= 0.8: lvl = "strong"
                        elif dim.score >= 0.6: lvl = "good"
                        elif dim.score >= 0.4: lvl = "fair"
                        else: lvl = "weak"
                        summary_parts.append(f"{label}:{lvl} ({dim.score})")

                qual(du, "DU")
                qual(dp, "DP")
                qual(m,  "M")
                qual(e,  "E")

                summary = "; ".join(summary_parts)

                return EvaluationReport(
                        data_understanding=du,
                        data_preparation=dp,
                        modelling=m,
                        evaluation=e,
                        overall_score=overall,
                        summary=summary
                )
        except Exception as ex:
                tb = traceback.format_exc()
                err_report = EvaluationReport(
                        data_understanding=DimensionScore(0.0, [], "error"),
                        data_preparation=DimensionScore(0.0, [], "error"),
                        modelling=DimensionScore(0.0, [], "error"),
                        evaluation=DimensionScore(0.0, [], "error"),
                        overall_score=0.0,
                        summary=f"Evaluation failed: {ex}"
                )
                # Attach traceback for debugging in evidence
                err_report.data_understanding.evidence.append(tb)
                return err_report

# --------------------------- CLI ---------------------------

def parse_args(argv=None):
        ap = argparse.ArgumentParser(description="Evaluate generated notebook & agent session.")
        ap.add_argument("--notebook", required=True, help="Notebook HTML content or path.")
        ap.add_argument("--history", required=True, help="JSON list of messages or path.")
        ap.add_argument("--save-dir", required=False, default=None, help="Directory with artifacts.")
        ap.add_argument("--out", required=False, default=None, help="Output JSON path.")
        return ap.parse_args(argv)

def main(argv=None):
        args = parse_args(argv)
        notebook_html = load_text_maybe_path(args.notebook)
        message_history = load_json_maybe_path(args.history)
        if not isinstance(message_history, list):
                raise ValueError("message_history must be a list of dicts")

        report = evaluate_session(
                notebook_html=notebook_html,
                message_history=message_history,
                save_dir=args.save_dir
        )
        out_json = json.dumps(report.to_dict(), indent=2)
        if args.out:
                Path(args.out).write_text(out_json, encoding="utf-8")
        print(out_json)

# --------------------------- Import Hook Convenience ---------------------------

if __name__ == "__main__":
        main()