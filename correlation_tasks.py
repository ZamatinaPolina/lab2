"""Задачи второй части лабораторной: корреляционный анализ."""
from __future__ import annotations

from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput

import pandas as pd

FEATURES = ["FSIQ", "VIQ", "PIQ", "Weight", "Height"]

def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    """Проанализируйте brainsize.txt.

    Разделите наблюдения по полу и для каждой группы вычислите корреляции
    признаков FSIQ, VIQ, PIQ, Weight, Height с MRI_Count методом Пирсона.
    В strongest_mri_feature верните название признака с наибольшим модулем
    корреляции с MRI_Count среди объединённых результатов двух групп.
    """
    df = pd.read_csv(data.csv_path, sep="\t", na_values=["NA"])
 
    women = df[df["Gender"] == "Female"]
    men = df[df["Gender"] == "Male"]
 
    women_corr = _mri_correlations(women)
    men_corr = _mri_correlations(men)
 
    candidates = [(abs(v), name) for corr in (women_corr, men_corr) for name, v in corr.items()]
    strongest = max(candidates)[1]
 
    return BrainCorrelationSummary(
        men_count=int(len(men)),
        women_count=int(len(women)),
        women_mri_correlation=women_corr,
        men_mri_correlation=men_corr,
        strongest_mri_feature=strongest,
    )


def _mri_correlations(group: pd.DataFrame) -> dict[str, float]:
    """Корреляция Пирсона каждого признака с MRI_Count внутри одной группы."""
    correlations = {}

    for feature in FEATURES:
        column = group[feature]
        mri = group["MRI_Count"]
        r = column.corr(mri, method="pearson")
        correlations[feature] = float(r)

    return correlations
