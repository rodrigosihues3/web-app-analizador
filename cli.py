"""Interfaz de línea de comandos (CLI) para demostración analítica sin navegador."""

import argparse
from pathlib import Path
import sys
import time
from typing import Any

from core.auditor import DatasetAuditor
from core.estadistica import analizar_valores_clave, resumen_descriptivo
from core.exportador import exportar_dataframe, generar_reporte_pdf
from core.sintetizador import generar_contexto_ejecutivo
from core.transformador import limpiar_dataset
from core.visualizador import generar_graficos_estaticos_pdf


def _detectar_columnas(
    auditoria: dict[str, Any],
) -> tuple[str | None, str | None]:
    """Selecciona automáticamente la primera columna categórica y numérica útil."""
    esquema = auditoria.get("esquema", [])
    constantes = set(auditoria.get("columnas_varianza_cero", []))

    cats = [
        str(col["columna"])
        for col in esquema
        if col.get("tipo_logico") == "categorica"
        and 2 <= int(col.get("valores_unicos", 0)) <= 12
    ] or [
        str(col["columna"])
        for col in esquema
        if col.get("tipo_logico") in {"categorica", "fecha"}
    ]
    nums = [
        str(col["columna"])
        for col in esquema
        if col.get("tipo_logico") == "numerica"
    ]
    nums_utiles = [c for c in nums if c not in constantes] or nums

    return (
        cats[0] if cats else None,
        nums_utiles[0] if nums_utiles else None,
    )


def _construir_parser() -> argparse.ArgumentParser:
    """Define los argumentos de línea de comandos para el analizador."""
    parser = argparse.ArgumentParser(
        description=(
            "Analizador Multiparadigma de Datos — Ejecución en consola "
            "(auditoría, limpieza funcional, estadística y exportación PDF/CSV)."
        )
    )
    parser.add_argument(
        "--archivo",
        required=True,
        help="Ruta al archivo tabular (.csv o .xlsx).",
    )
    parser.add_argument(
        "--nulos",
        default="mediana",
        choices=[
            "eliminar",
            "media",
            "mediana",
            "moda",
            "personalizado",
            "ninguna",
        ],
        help="Estrategia para tratar valores nulos (por defecto: mediana).",
    )
    parser.add_argument(
        "--valor-nulos",
        default="0",
        help="Valor manual de reemplazo cuando --nulos es 'personalizado' (por defecto: 0).",
    )
    parser.add_argument(
        "--sin-duplicados",
        action="store_true",
        help="Elimina filas duplicadas exactas.",
    )
    parser.add_argument(
        "--salida",
        default="salida_cli",
        help="Carpeta destino para guardar el CSV limpio y el PDF (por defecto: salida_cli).",
    )
    parser.add_argument(
        "--categoria",
        default="",
        help="Columna categórica opcional para agrupar valores clave.",
    )
    parser.add_argument(
        "--metrica",
        default="",
        help="Columna numérica opcional para evaluar valores clave.",
    )
    return parser


def _imprimir_tabla_estadisticas(estadisticas: list[dict[str, Any]]) -> None:
    """Imprime la tabla de estadísticas descriptivas alineada en consola."""
    if not estadisticas:
        print("  (Sin columnas numéricas para resumir)")
        return

    encabezado = (
        f"  {'Columna':<20} {'Media':>9} {'Mediana':>9} {'Moda':>9} "
        f"{'Desv.Est.':>10} {'Mín':>9} {'Máx':>9}"
    )
    separador = "  " + "-" * (len(encabezado) - 2)
    print(encabezado)
    print(separador)

    for st in estadisticas:
        nombre = str(st["columna"])[:20]
        print(
            f"  {nombre:<20} "
            f"{st['media']:>9.2f} "
            f"{st['mediana']:>9.2f} "
            f"{st['moda']:>9.2f} "
            f"{st['desviacion_estandar']:>10.2f} "
            f"{st['minimo']:>9.2f} "
            f"{st['maximo']:>9.2f}"
        )


def main() -> int:
    """Ejecuta el flujo analítico completo desde la terminal."""
    args = _construir_parser().parse_args()

    try:
        inicio = time.perf_counter()
        auditor = DatasetAuditor(args.archivo)
        auditoria = auditor.auditar_estructura()
        contexto = generar_contexto_ejecutivo(
            auditoria, auditor.obtener_muestra(3)
        )
        tiempo_s = time.perf_counter() - inicio

        total_nulos = sum(int(c["nulos"]) for c in auditoria["esquema"])

        print("\n=== 1. DIAGNOSTICO DE AUDITORIA INICIAL ===")
        print(f"  Archivo       : {Path(args.archivo).name}")
        print(
            f"  Dimensiones   : {auditoria['filas']} filas x {auditoria['columnas']} columnas"
        )
        print(f"  Memoria RAM   : {auditoria['memoria_kb']:.2f} KB")
        print(f"  Valores nulos : {total_nulos}")
        print(f"  Duplicados    : {auditoria['duplicados']}")
        print(f"  Tiempo        : {tiempo_s:.2f} s")

        df_limpio, logs = limpiar_dataset(
            auditor.df_raw,
            eliminar_duplicados=args.sin_duplicados,
            estrategia_nulos=args.nulos,
            valor_nulos=args.valor_nulos,
        )

        print("\n=== 2. BITACORA DE LIMPIEZA APLICADA ===")
        for idx, paso in enumerate(logs, start=1):
            print(f"  {idx}. {paso}")

        estadisticas = resumen_descriptivo(df_limpio)
        print("\n=== 3. ESTADISTICA DESCRIPTIVA ===")
        _imprimir_tabla_estadisticas(estadisticas)

        cat_def, met_def = _detectar_columnas(auditoria)
        col_cat = args.categoria.strip() or cat_def
        col_met = args.metrica.strip() or met_def

        valores_clave: dict[str, Any] | None = None
        if col_cat and col_met:
            try:
                valores_clave = analizar_valores_clave(
                    df_limpio, col_categoria=col_cat, col_metrica=col_met
                )
            except (KeyError, ValueError, ZeroDivisionError):
                valores_clave = None

        imagenes = generar_graficos_estaticos_pdf(df_limpio)

        directorio = Path(args.salida)
        directorio.mkdir(parents=True, exist_ok=True)
        nombre_base = Path(args.archivo).stem

        ruta_csv = directorio / f"{nombre_base}_limpio.csv"
        ruta_csv.write_bytes(
            exportar_dataframe(df_limpio, formato="csv").getvalue()
        )

        ruta_pdf = directorio / f"{nombre_base}_reporte.pdf"
        pdf_buf = generar_reporte_pdf(
            contexto=contexto,
            auditoria=auditoria,
            logs_limpieza=logs,
            estadisticas=estadisticas,
            valores_clave=valores_clave,
            imagenes_graficos=imagenes,
        )
        ruta_pdf.write_bytes(pdf_buf.getvalue())

        print("\n=== 4. ARCHIVOS GENERADOS ===")
        print(f"  CSV limpio    : {ruta_csv.resolve()}")
        print(f"  Reporte PDF   : {ruta_pdf.resolve()}")
        print("")
        return 0
    except (
        FileNotFoundError,
        KeyError,
        ValueError,
        ZeroDivisionError,
        TimeoutError,
        OSError,
    ) as exc:
        print(
            f"[ERROR] No fue posible completar la ejecución: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
