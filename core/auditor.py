"""Módulo de auditoría estructural de datasets (Enfoque POO)."""

from pathlib import Path
from typing import Any
import zipfile

import pandas as pd


class DatasetAuditor:
    """Audita la estructura y calidad de un archivo CSV o Excel."""

    EXTENSIONES_SOPORTADAS = (".csv", ".xlsx")

    def __init__(self, ruta_archivo: str) -> None:
        """Carga el archivo tabular desde disco y conserva una copia en self.df_raw."""
        if not ruta_archivo or not isinstance(ruta_archivo, str):
            raise ValueError("La ruta del archivo debe ser una cadena no vacía.")

        ruta = Path(ruta_archivo)
        if not ruta.exists() or not ruta.is_file():
            raise FileNotFoundError(
                f"No se encontró el archivo especificado: '{ruta_archivo}'."
            )

        extension = ruta.suffix.lower()
        if extension not in self.EXTENSIONES_SOPORTADAS:
            raise ValueError(
                f"Extensión '{extension}' no soportada. "
                "Formatos permitidos: .csv, .xlsx."
            )

        self.ruta_archivo = str(ruta)
        df_cargado = self._cargar_archivo(ruta, extension)

        if df_cargado.empty and len(df_cargado.columns) == 0:
            raise pd.errors.EmptyDataError(
                f"El archivo '{ruta.name}' no contiene columnas ni registros."
            )

        self.df_raw = df_cargado.copy(deep=True)

    def _leer_csv_con_delimitador(self, ruta: Path, encoding: str) -> pd.DataFrame:
        """Lee un CSV soportando coma, punto y coma o tabulación."""
        df = pd.read_csv(ruta, encoding=encoding)
        if len(df.columns) == 1:
            encabezado = str(df.columns[0])
            if ";" in encabezado or "\t" in encabezado:
                return pd.read_csv(
                    ruta, encoding=encoding, sep=None, engine="python"
                )
        return df

    def _cargar_archivo(self, ruta: Path, extension: str) -> pd.DataFrame:
        """Lee un archivo CSV o Excel manejando errores comunes de formato."""
        if extension == ".csv":
            try:
                return self._leer_csv_con_delimitador(ruta, "utf-8")
            except UnicodeDecodeError:
                try:
                    return self._leer_csv_con_delimitador(ruta, "latin-1")
                except (
                    UnicodeDecodeError,
                    pd.errors.EmptyDataError,
                    pd.errors.ParserError,
                    OSError,
                ) as exc:
                    raise ValueError(
                        f"No fue posible decodificar el CSV '{ruta.name}' "
                        f"con utf-8 ni latin-1: {exc}"
                    ) from exc
            except pd.errors.EmptyDataError as exc:
                raise pd.errors.EmptyDataError(
                    f"El archivo CSV '{ruta.name}' está vacío."
                ) from exc
            except (pd.errors.ParserError, OSError, ValueError) as exc:
                raise ValueError(
                    f"Error al analizar el archivo CSV '{ruta.name}': {exc}"
                ) from exc

        try:
            return pd.read_excel(ruta, engine="openpyxl")
        except pd.errors.EmptyDataError as exc:
            raise pd.errors.EmptyDataError(
                f"El archivo Excel '{ruta.name}' está vacío."
            ) from exc
        except (ValueError, OSError, zipfile.BadZipFile) as exc:
            raise ValueError(
                f"Error al leer el archivo Excel '{ruta.name}': {exc}"
            ) from exc

    def _inferir_tipo_logico(self, serie: pd.Series) -> str:
        """Clasifica una columna como numerica, fecha o categorica."""
        if pd.api.types.is_datetime64_any_dtype(serie):
            return "fecha"
        if pd.api.types.is_bool_dtype(serie):
            return "categorica"
        if pd.api.types.is_numeric_dtype(serie):
            return "numerica"

        serie_valida = serie.dropna()
        if serie_valida.empty:
            return "categorica"

        if pd.api.types.is_object_dtype(serie) or pd.api.types.is_string_dtype(serie):
            muestra = serie_valida.astype(str).str.strip().head(30)
            primer_valor = muestra.iloc[0]
            if "-" in primer_valor or "/" in primer_valor:
                try:
                    parseadas = pd.to_datetime(
                        muestra, format="mixed", errors="coerce"
                    )
                    if float(parseadas.notna().mean()) >= 0.8:
                        return "fecha"
                except (ValueError, TypeError):
                    return "categorica"

        return "categorica"

    def obtener_muestra(self, n: int = 3) -> list[dict[str, Any]]:
        """Retorna las primeras n filas como lista de diccionarios."""
        return self.df_raw.head(n).to_dict(orient="records")

    def auditar_estructura(self) -> dict[str, Any]:
        """Retorna el diagnóstico de filas, columnas, nulos y duplicados."""
        filas = int(self.df_raw.shape[0])
        columnas = int(self.df_raw.shape[1])
        memoria_kb = round(
            float(self.df_raw.memory_usage(deep=True).sum()) / 1024.0, 2
        )

        esquema: list[dict[str, Any]] = []
        for col in self.df_raw.columns:
            serie = self.df_raw[col]
            nulos = int(serie.isna().sum())
            porcentaje_nulos = (
                round((nulos / filas) * 100.0, 2) if filas > 0 else 0.0
            )

            esquema.append(
                {
                    "columna": str(col),
                    "dtype": str(serie.dtype),
                    "tipo_logico": self._inferir_tipo_logico(serie),
                    "nulos": nulos,
                    "porcentaje_nulos": porcentaje_nulos,
                    "valores_unicos": int(serie.nunique(dropna=True)),
                }
            )

        duplicados = int(self.df_raw.duplicated().sum())
        columnas_varianza_cero = [
            str(col)
            for col in self.df_raw.columns
            if int(self.df_raw[col].nunique(dropna=True)) <= 1
        ]

        return {
            "filas": filas,
            "columnas": columnas,
            "memoria_kb": memoria_kb,
            "esquema": esquema,
            "duplicados": duplicados,
            "columnas_varianza_cero": columnas_varianza_cero,
        }
