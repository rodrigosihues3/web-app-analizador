"""Utilidades compartidas para identificar columnas numéricas."""

from typing import Any
import pandas as pd


def obtener_columnas_numericas(df: pd.DataFrame) -> list[str]:
    """Retorna la lista de nombres de columnas numéricas del DataFrame."""
    return [str(col) for col in df.select_dtypes(include="number").columns]


def es_columna_numerica(serie: pd.Series) -> bool:
    """Indica si una columna es numérica (int o float) excluyendo booleanos."""
    return bool(
        pd.api.types.is_numeric_dtype(serie)
        and not pd.api.types.is_bool_dtype(serie)
    )
