"""Utilidades funcionales compartidas para el núcleo analítico."""

from __future__ import annotations

import pandas as pd


def obtener_columnas_numericas(df: pd.DataFrame) -> list[str]:
    """Retorna los nombres de columnas numéricas del DataFrame (excluye booleanos)."""
    # select_dtypes(include="number") filtra columnas int/float y omite bool
    return [str(col) for col in df.select_dtypes(include="number").columns]


def es_columna_numerica(serie: pd.Series) -> bool:
    """Indica si una Series es numérica real (int/float) y no booleana."""
    # En Python/Pandas bool es subtipo numérico, por eso se excluye explícitamente
    return bool(
        pd.api.types.is_numeric_dtype(serie)
        and not pd.api.types.is_bool_dtype(serie)
    )


def calcular_limites_iqr(
    serie: pd.Series,
) -> tuple[float, float, float, float, float]:
    """Calcula cuartiles y límites de atípicos con la regla 1.5 * IQR.

    Args:
        serie: Columna numérica de Pandas.

    Returns:
        Tupla ``(q1, q3, iqr, limite_inferior, limite_superior)``.

    Raises:
        ValueError: Si la serie no contiene valores numéricos válidos.
    """
    serie_valida = serie.dropna().astype(float)
    if serie_valida.empty:
        raise ValueError("La columna no contiene valores numéricos válidos.")

    # quantile(0.25) y quantile(0.75) obtienen el 25% y 75% (mismos cuartiles de describe())
    q1 = float(serie_valida.quantile(0.25))
    q3 = float(serie_valida.quantile(0.75))
    iqr = q3 - q1
    limite_inferior = q1 - 1.5 * iqr
    limite_superior = q3 + 1.5 * iqr
    return q1, q3, iqr, limite_inferior, limite_superior
