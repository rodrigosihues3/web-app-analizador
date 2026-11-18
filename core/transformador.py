"""Módulo funcional de limpieza y transformación inmutable de DataFrames."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

ESTRATEGIAS_NULOS_VALIDAS: frozenset[str] = frozenset(
    {"eliminar", "media", "mediana", "moda", "ninguna"}
)


def _es_columna_numerica(serie: pd.Series) -> bool:
    """Retorna True si la serie es numérica continua/discreta y no booleana."""
    return bool(
        pd.api.types.is_numeric_dtype(serie)
        and not pd.api.types.is_bool_dtype(serie)
    )


def _remover_duplicados(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Retorna un nuevo DataFrame sin filas duplicadas y su registro."""
    cantidad_duplicados = int(df.duplicated().sum())
    if cantidad_duplicados == 0:
        return df.copy(deep=True), "Sin registros duplicados."

    df_sin_duplicados = df.drop_duplicates().reset_index(drop=True)
    return (
        df_sin_duplicados,
        f"Duplicados eliminados: {cantidad_duplicados} fila(s).",
    )


def _calcular_valor_imputacion_numerica(
    serie: pd.Series, estrategia: str
) -> float:
    """Calcula el valor escalar de imputación para una serie numérica."""
    serie_valida = serie.dropna().astype(float)
    if serie_valida.empty:
        return 0.0

    try:
        if estrategia == "media":
            return float(serie_valida.mean())
        if estrategia == "mediana":
            return float(serie_valida.median())
        if estrategia == "moda":
            modas = serie_valida.mode(dropna=True)
            return float(modas.iloc[0]) if not modas.empty else 0.0
    except (ValueError, TypeError, ZeroDivisionError, IndexError):
        return 0.0

    return 0.0


def _calcular_valor_imputacion_categorica(serie: pd.Series) -> Any:
    """Calcula la moda o 'Desconocido' para columnas categóricas/texto."""
    serie_valida = serie.dropna()
    if serie_valida.empty:
        return "Desconocido"

    try:
        modas = serie_valida.mode()
        if not modas.empty:
            return modas.iloc[0]
    except (ValueError, TypeError, IndexError):
        return "Desconocido"

    return "Desconocido"


def _tratar_nulos(
    df: pd.DataFrame, estrategia: str
) -> tuple[pd.DataFrame, list[str]]:
    """Aplica la estrategia de valores nulos sin mutar el DataFrame origen."""
    df_nuevo = df.copy(deep=True)
    logs: list[str] = []

    if estrategia == "ninguna":
        logs.append("Valores faltantes conservados sin cambios.")
        return df_nuevo, logs

    if estrategia == "eliminar":
        filas_con_nulos = int(df_nuevo.isna().any(axis=1).sum())
        if filas_con_nulos == 0:
            logs.append("Sin filas con valores faltantes.")
            return df_nuevo, logs

        df_filtrado = df_nuevo.dropna().reset_index(drop=True)
        logs.append(
            f"Eliminadas {filas_con_nulos} fila(s) con datos faltantes."
        )
        return df_filtrado, logs

    columnas_con_nulos = [
        col for col in df_nuevo.columns if int(df_nuevo[col].isna().sum()) > 0
    ]

    if not columnas_con_nulos:
        logs.append("Sin valores faltantes por completar.")
        return df_nuevo, logs

    for col in columnas_con_nulos:
        n_nulos = int(df_nuevo[col].isna().sum())
        serie = df_nuevo[col]

        if _es_columna_numerica(serie):
            valor_imputado = _calcular_valor_imputacion_numerica(
                serie, estrategia
            )
            if pd.api.types.is_integer_dtype(serie):
                if float(valor_imputado).is_integer():
                    df_nuevo[col] = serie.fillna(int(valor_imputado))
                else:
                    df_nuevo[col] = serie.astype("float64").fillna(
                        valor_imputado
                    )
            else:
                df_nuevo[col] = serie.fillna(valor_imputado)

            logs.append(
                f"'{col}': {n_nulos} dato(s) faltante(s) completado(s) con "
                f"{estrategia} ({round(valor_imputado, 4)})."
            )
        elif pd.api.types.is_datetime64_any_dtype(serie):
            modas_fecha = serie.dropna().mode()
            if not modas_fecha.empty:
                fecha_imp = modas_fecha.iloc[0]
                df_nuevo[col] = serie.fillna(fecha_imp)
                logs.append(
                    f"'{col}': {n_nulos} fecha(s) faltante(s) completada(s) "
                    f"con {fecha_imp}."
                )
            else:
                logs.append(
                    f"'{col}': sin fechas válidas para completar."
                )
        else:
            valor_cat = _calcular_valor_imputacion_categorica(serie)
            if isinstance(serie.dtype, pd.CategoricalDtype):
                if valor_cat not in serie.cat.categories:
                    serie = serie.cat.add_categories([valor_cat])
            df_nuevo[col] = serie.fillna(valor_cat)
            logs.append(
                f"'{col}': {n_nulos} dato(s) faltante(s) completado(s) con "
                f"'{valor_cat}'."
            )

    return df_nuevo, logs


def _recortar_outliers_iqr(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Acota valores atípicos mediante ``numpy.clip`` según límites IQR."""
    df_nuevo = df.copy(deep=True)
    logs: list[str] = []

    columnas_numericas = [
        col for col in df_nuevo.columns if _es_columna_numerica(df_nuevo[col])
    ]

    for col in columnas_numericas:
        serie = df_nuevo[col]
        try:
            valores_validos = serie.dropna().to_numpy(dtype=float)
            if valores_validos.size == 0:
                continue

            q1 = float(np.percentile(valores_validos, 25))
            q3 = float(np.percentile(valores_validos, 75))
            iqr = q3 - q1
            lim_inf = q1 - 1.5 * iqr
            lim_sup = q3 + 1.5 * iqr

            serie_float = serie.astype("float64")
            mascara_outliers = (serie_float < lim_inf) | (
                serie_float > lim_sup
            )
            cantidad_outliers = int(mascara_outliers.sum())

            if cantidad_outliers > 0:
                df_nuevo[col] = np.clip(serie_float, lim_inf, lim_sup)
                logs.append(
                    f"'{col}': {cantidad_outliers} valor(es) extremo(s) ajustado(s) "
                    f"al rango [{round(lim_inf, 2)}, {round(lim_sup, 2)}]."
                )
        except (ValueError, TypeError, IndexError) as exc:
            logs.append(
                f"No fue posible ajustar extremos en '{col}': {exc}."
            )

    if not logs:
        logs.append("Sin valores extremos fuera de rango.")

    return df_nuevo, logs


def limpiar_dataset(
    df: pd.DataFrame,
    eliminar_duplicados: bool = True,
    estrategia_nulos: str = "mediana",
    recortar_outliers: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Aplica limpieza funcional sin mutar el DataFrame original.

    Args:
        df: DataFrame de entrada que permanece intacto.
        eliminar_duplicados: Si es True, elimina registros duplicados exactos.
        estrategia_nulos: Estrategia en ``{"eliminar", "media", "mediana",
            "moda", "ninguna"}``.
        recortar_outliers: Si es True, acota atípicos IQR usando ``numpy.clip``.

    Returns:
        Tupla ``(df_limpio, logs)`` con el nuevo DataFrame depurado y la
        bitácora cronológica de transformaciones aplicadas.

    Raises:
        ValueError: Si ``df`` no es un DataFrame válido o la estrategia de nulos
            no es reconocida.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("El parámetro 'df' debe ser un pandas.DataFrame.")

    estrategia = estrategia_nulos.strip().lower()
    if estrategia not in ESTRATEGIAS_NULOS_VALIDAS:
        raise ValueError(
            f"Estrategia de nulos inválida: '{estrategia_nulos}'. "
            f"Opciones válidas: {sorted(ESTRATEGIAS_NULOS_VALIDAS)}."
        )

    df_actual = df.copy(deep=True)
    logs: list[str] = []

    if eliminar_duplicados:
        df_actual, log_dup = _remover_duplicados(df_actual)
        logs.append(log_dup)
    else:
        logs.append("Duplicados conservados.")

    df_actual, logs_nulos = _tratar_nulos(df_actual, estrategia)
    logs.extend(logs_nulos)

    if recortar_outliers:
        df_actual, logs_outliers = _recortar_outliers_iqr(df_actual)
        logs.extend(logs_outliers)

    return df_actual, logs
