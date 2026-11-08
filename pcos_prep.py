"""
pcos_prep.py
============
Shared preprocessing toolkit for the PCOS classification project.

It is imported by BOTH the training script (``train_pipeline.py``) and the
Streamlit app (``app.py``).  The two custom scikit-learn transformers defined
here are stored inside every saved model pipeline, so this file must sit next
to ``app.py`` when the app is deployed.

Contents
--------
* ``harmonise_headers``     - map raw Kaggle headers / variants to PascalCase names
* ``repair_typos``          - fix numeric cells that were typed as text
* ``load_raw_pcos``         - read + merge the two raw source files
* ``ClinicalFeatureBuilder`` - adds engineered clinical features (row-wise, no fitting)
* ``PercentileClipper``     - winsorises numeric columns at train-set percentiles
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# --------------------------------------------------------------------------- #
#  Column vocabulary
# --------------------------------------------------------------------------- #
LABEL = "PCOS"

# raw header (any spacing / case)  ->  project column name
_RAW_TO_PROJECT = {
    "PCOS (Y/N)": LABEL,
    "Age (yrs)": "Age",
    "Weight (Kg)": "WeightKg",
    "Height(Cm)": "HeightCm",
    "BMI": "BMI",
    "Blood Group": "BloodGroup",
    "Pulse rate(bpm)": "PulseBpm",
    "RR (breaths/min)": "RespRate",
    "Hb(g/dl)": "Haemoglobin",
    "Cycle(R/I)": "CycleType",
    "Cycle length(days)": "PeriodDays",
    "Marraige Status (Yrs)": "YearsMarried",
    "Pregnant(Y/N)": "Pregnant",
    "No. of aborptions": "Abortions",
    "No. of abortions": "Abortions",
    "I beta-HCG(mIU/mL)": "BetaHCG_I",
    "II beta-HCG(mIU/mL)": "BetaHCG_II",
    "FSH(mIU/mL)": "FSH",
    "LH(mIU/mL)": "LH",
    "FSH/LH": "FSHtoLH",
    "Hip(inch)": "HipIn",
    "Waist(inch)": "WaistIn",
    "Waist:Hip Ratio": "WaistHipRatio",
    "TSH (mIU/L)": "TSH",
    "AMH(ng/mL)": "AMH",
    "PRL(ng/mL)": "Prolactin",
    "Vit D3 (ng/mL)": "VitD3",
    "PRG(ng/mL)": "Progesterone",
    "RBS(mg/dl)": "RBS",
    "Weight gain(Y/N)": "WeightGain",
    "hair growth(Y/N)": "HairGrowth",
    "Skin darkening (Y/N)": "SkinDarkening",
    "Hair loss(Y/N)": "HairLoss",
    "Pimples(Y/N)": "Pimples",
    "Fast food (Y/N)": "FastFood",
    "Reg.Exercise(Y/N)": "RegExercise",
    "BP _Systolic (mmHg)": "SystolicBP",
    "BP _Diastolic (mmHg)": "DiastolicBP",
    "Follicle No. (L)": "FollicleCountL",
    "Follicle No. (R)": "FollicleCountR",
    "Avg. F size (L) (mm)": "FollicleSizeL",
    "Avg. F size (R) (mm)": "FollicleSizeR",
    "Endometrium (mm)": "EndometriumMm",
}

# identifiers / junk columns that carry no signal
NON_PREDICTIVE = {"Sl. No", "Patient File No.", "Unnamed: 44"}

# the 41 raw predictors the saved pipelines expect (in this order)
INPUT_COLUMNS = [
    "Age", "WeightKg", "HeightCm", "BMI", "BloodGroup", "PulseBpm", "RespRate",
    "Haemoglobin", "CycleType", "PeriodDays", "YearsMarried", "Pregnant",
    "Abortions", "BetaHCG_I", "BetaHCG_II", "FSH", "LH", "FSHtoLH", "HipIn",
    "WaistIn", "WaistHipRatio", "TSH", "AMH", "Prolactin", "VitD3",
    "Progesterone", "RBS", "WeightGain", "HairGrowth", "SkinDarkening",
    "HairLoss", "Pimples", "FastFood", "RegExercise", "SystolicBP",
    "DiastolicBP", "FollicleCountL", "FollicleCountR", "FollicleSizeL",
    "FollicleSizeR", "EndometriumMm",
]

# Y/N symptom & lifestyle flags (0/1)
FLAG_COLUMNS = [
    "Pregnant", "WeightGain", "HairGrowth", "SkinDarkening", "HairLoss",
    "Pimples", "FastFood", "RegExercise",
]
SYMPTOM_FLAGS = ["WeightGain", "HairGrowth", "SkinDarkening", "HairLoss", "Pimples"]
NOMINAL_COLUMNS = ["BloodGroup"]

# columns created by ClinicalFeatureBuilder
ENGINEERED = ["IrregularCycle", "FollicleTotal", "LHtoFSH", "SymptomScore"]

# after feature building: continuous columns get clipped / imputed / scaled
CONTINUOUS_COLUMNS = [
    c for c in INPUT_COLUMNS
    if c not in FLAG_COLUMNS + NOMINAL_COLUMNS + ["CycleType"]
] + ["FollicleTotal", "LHtoFSH", "SymptomScore"]
BINARY_COLUMNS = FLAG_COLUMNS + ["IrregularCycle"]


def _header_key(text: str) -> str:
    """Lower-case and keep only letters/digits, so 'BP _Systolic (mmHg)' -> 'bpsystolicmmhg'."""
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


_HEADER_LOOKUP = {_header_key(k): v for k, v in _RAW_TO_PROJECT.items()}
_HEADER_LOOKUP.update({_header_key(v): v for v in _RAW_TO_PROJECT.values()})
_DROP_KEYS = {_header_key(c) for c in NON_PREDICTIVE}


def harmonise_headers(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename any recognised header (raw Kaggle or project style) to the project name
    and drop identifier columns. Unknown columns are kept untouched."""
    keep = [c for c in frame.columns if _header_key(c) not in _DROP_KEYS]
    out = frame[keep].copy()
    out.columns = [_HEADER_LOOKUP.get(_header_key(c), str(c).strip()) for c in out.columns]
    return out.loc[:, ~out.columns.duplicated()]


def repair_typos(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce every predictor to numeric.

    The raw file has two typed-in errors:
      * ``BetaHCG_II`` = ``'1.99.'`` (trailing full stop) -> 1.99
      * ``AMH``        = ``'a'``                         -> NaN (imputed later)
    """
    out = frame.copy()
    for col in out.columns:
        if out[col].dtype == object or pd.api.types.is_string_dtype(out[col]):
            cleaned = out[col].astype(str).str.strip().str.rstrip(".")
            out[col] = pd.to_numeric(cleaned, errors="coerce")
    return out


def load_raw_pcos(xlsx_path: str | Path, infertility_csv: str | Path | None = None) -> pd.DataFrame:
    """Read the main workbook (sheet ``Full_new``) and, if given, cross-check it against
    ``PCOS_infertility.csv`` (merged on the serial number).

    The infertility file repeats PCOS label, both beta-HCG readings and AMH for the same
    541 patients; we use it as an integrity check and to fill any gaps in the workbook.
    """
    book = pd.read_excel(xlsx_path, sheet_name="Full_new")
    book.columns = [" ".join(str(c).split()) for c in book.columns]

    if infertility_csv is not None:
        extra = pd.read_csv(infertility_csv)
        extra.columns = [" ".join(str(c).split()) for c in extra.columns]
        shared = ["PCOS (Y/N)", "I beta-HCG(mIU/mL)", "II beta-HCG(mIU/mL)", "AMH(ng/mL)"]
        joined = book.merge(extra[["Sl. No"] + shared], on="Sl. No", how="left",
                            suffixes=("", "__inf"), validate="one_to_one")
        if (joined["PCOS (Y/N)"] != joined["PCOS (Y/N)__inf"]).any():
            raise ValueError("PCOS labels disagree between the two source files")
        for col in shared[1:]:
            backup = pd.to_numeric(joined[f"{col}__inf"], errors="coerce")
            # only fill where the workbook is empty (never overwrite a value)
            joined[col] = joined[col].where(joined[col].notna(), backup)
        book = joined.drop(columns=[c for c in joined.columns if c.endswith("__inf")])

    tidy = repair_typos(harmonise_headers(book))
    return tidy[INPUT_COLUMNS + [LABEL]]


# --------------------------------------------------------------------------- #
#  Custom transformers (pickled inside the saved pipelines)
# --------------------------------------------------------------------------- #
class ClinicalFeatureBuilder(BaseEstimator, TransformerMixin):
    """Stateless, row-wise feature engineering.

    * ``IrregularCycle`` : 1 if ``CycleType`` >= 4 (4 = irregular; one record has 5)
    * ``FollicleTotal``  : left + right follicle counts (Rotterdam criterion proxy)
    * ``LHtoFSH``        : LH / FSH (clinicians read LH:FSH > 2 as PCOS-suggestive)
    * ``SymptomScore``   : number of the five Y/N PCOS symptoms reported
    ``CycleType`` itself is dropped because ``IrregularCycle`` replaces it.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = pd.DataFrame(X).copy()
        rows["IrregularCycle"] = (rows["CycleType"] >= 4).astype(float)
        rows.loc[rows["CycleType"].isna(), "IrregularCycle"] = np.nan
        rows["FollicleTotal"] = rows["FollicleCountL"] + rows["FollicleCountR"]
        rows["LHtoFSH"] = rows["LH"] / rows["FSH"].clip(lower=0.05)
        rows["SymptomScore"] = rows[SYMPTOM_FLAGS].sum(axis=1, min_count=1)
        return rows.drop(columns=["CycleType"])

    def get_feature_names_out(self, input_features=None):
        base = [c for c in (input_features if input_features is not None else INPUT_COLUMNS)
                if c != "CycleType"]
        return np.asarray(base + ENGINEERED, dtype=object)


class PercentileClipper(BaseEstimator, TransformerMixin):
    """Winsorise each column to the [lower_pct, upper_pct] percentiles learned on the
    training data.  This tames impossible entries (pulse 13 bpm, BP 12 mmHg,
    FSH 5052 mIU/mL, ...) without hand-written rules and without dropping rows."""

    def __init__(self, lower_pct: float = 1.0, upper_pct: float = 99.0):
        self.lower_pct = lower_pct
        self.upper_pct = upper_pct

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        self.floor_ = np.nanpercentile(values, self.lower_pct, axis=0)
        self.ceiling_ = np.nanpercentile(values, self.upper_pct, axis=0)
        self.n_features_in_ = values.shape[1]
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        values = np.asarray(X, dtype=float)
        return np.clip(values, self.floor_, self.ceiling_)

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        return getattr(self, "feature_names_in_", np.asarray(
            [f"x{i}" for i in range(self.n_features_in_)], dtype=object))
