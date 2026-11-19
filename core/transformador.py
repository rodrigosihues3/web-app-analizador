"""Módulo funcional de limpieza y transformación inmutable de DataFrames."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from core.utils import (
    calcular_limites_iqr,
    es_columna_numerica,
    obtener_columnas_numericas,
)

ESTRATEGIAS_NULOS_VALIDAS: frozenset[str] = frozenset(
    {"eliminar", "media", "mediana", "moda", "ninguna"}
)


def _remover_duplicados(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Retorna un nuevo DataFrame sin filas duplicadas y su registro."""
    cantidad_duplicados = int(df.duplicated().sum())
    if cantidad_duplicados == 0:
        return df.copy(deep=True), "Sin registros duplicados."

    # reset_index(drop=True) regenera el índice 0..N-1 tras quitar filas repetidas
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
        serie = df_nuevo[col]
        n_nulos = int(serie.isna().sum())

        if es_columna_numerica(serie):
            valor = _calcular_valor_imputacion_numerica(serie, estrategia)
            # Si la columna es entera y el valor imputado tiene decimales, convierte a float64
            if pd.api.types.is_integer_dtype(serie) and not float(valor).is_integer():
                df_nuevo[col] = serie.astype("float64").fillna(valor)
            else:
                relleno = int(valor) if pd.api.types.is_integer_dtype(serie) else valor
                df_nuevo[col] = serie.fillna(relleno)

            logs.append(
                f"'{col}': {n_nulos} dato(s) faltante(s) completado(s) con "
                f"{estrategia} ({round(valor, 4)})."
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
                logs.append(f"'{col}': sin fechas válidas para completar.")
        else:
            valor_cat = _calcular_valor_imputacion_categorica(serie)
            # En columnas tipo Categorical es obligatorio registrar una categoría nueva antes de fillna
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

    for col in obtener_columnas_numericas(df_nuevo):
        try:
            _, _, _, lim_inf, lim_sup = calcular_limites_iqr(df_nuevo[col])
            serie_float = df_nuevo[col].astype("float64")
            cantidad_outliers = int(
                ((serie_float < lim_inf) | (serie_float > lim_sup)).sum()
            )

            if cantidad_outliers > 0:
                # np.clip limita de forma vectorizada cualquier valor fuera de [lim_inf, lim_sup]
                df_nuevo[col] = np.clip(serie_float, lim_inf, lim_sup)
                logs.append(
                    f"'{col}': {cantidad_outliers} valor(es) extremo(s) ajustado(s) "
                    f"al rango [{round(lim_inf, 2)}, {round(lim_sup, 2)}]."
                )
        except (ValueError, TypeError, IndexError):
            continue

    if not logs:
        logs.append("Sin valores extremos fuera de rango.")

    return df_nuevo, logs


def limpiar_dataset(
    df: pd.DataFrame,
    eliminar_duplicados: bool = True,
    estrategia_nulos: str = "mediana",
    recortar_outliers: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Aplica limpieza funcional sin mutar el DataFrame original."""
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
