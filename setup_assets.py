"""Script de aprovisionamiento local de activos estáticos (Modo Offline-First)."""

from __future__ import annotations

from pathlib import Path
import urllib.error
import urllib.request

BASE_DIR: Path = Path(__file__).resolve().parent
STATIC_DIR: Path = BASE_DIR / "web" / "static"
TAMANO_MINIMO_BYTES: int = 10 * 1024  # 10 KB

ACTIVOS_ESTATICOS: tuple[tuple[str, Path], ...] = (
    (
        "https://unpkg.com/tailwindcss@2.2.19/dist/tailwind.min.css",
        STATIC_DIR / "css" / "tailwind.min.css",
    ),
    (
        "https://unpkg.com/htmx.org@1.9.12/dist/htmx.min.js",
        STATIC_DIR / "js" / "htmx.min.js",
    ),
    (
        "https://cdn.plot.ly/plotly-2.30.0.min.js",
        STATIC_DIR / "js" / "plotly.min.js",
    ),
)


def descargar_activo(url: str, destino: Path, timeout: float = 60.0) -> int:
    """Descarga un archivo remoto y valida que supere el tamaño mínimo.

    Args:
        url: Dirección HTTPS de origen del activo estático.
        destino: Ruta local donde se almacenará el archivo.
        timeout: Tiempo máximo de espera en segundos para la descarga.

    Returns:
        Tamaño final del archivo descargado en bytes.

    Raises:
        RuntimeError: Si ocurre un error de red, sistema de archivos o si el
            archivo descargado pesa menos de 10 KB.
    """
    destino.parent.mkdir(parents=True, exist_ok=True)
    peticion = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AnalizadorMultiparadigma/1.0"
            )
        },
    )

    try:
        with urllib.request.urlopen(peticion, timeout=timeout) as respuesta:
            contenido: bytes = respuesta.read()
    except (
        urllib.error.HTTPError,
        urllib.error.URLError,
        TimeoutError,
        OSError,
        ValueError,
    ) as exc:
        raise RuntimeError(
            f"Fallo al descargar el activo desde '{url}': {exc}"
        ) from exc

    try:
        destino.write_bytes(contenido)
        tamano_bytes = int(destino.stat().st_size)
    except OSError as exc:
        raise RuntimeError(
            f"No fue posible escribir el archivo en '{destino}': {exc}"
        ) from exc

    if tamano_bytes <= TAMANO_MINIMO_BYTES:
        raise RuntimeError(
            f"El archivo descargado '{destino.name}' tiene un tamaño inválido "
            f"({tamano_bytes} bytes <= {TAMANO_MINIMO_BYTES} bytes)."
        )

    return tamano_bytes


def configurar_activos_locales() -> dict[str, int]:
    """Ejecuta la descarga y verificación de todos los activos Offline-First.

    Returns:
        Diccionario con la ruta relativa del archivo y su peso en bytes.
    """
    resumen: dict[str, int] = {}
    for url, ruta_destino in ACTIVOS_ESTATICOS:
        ruta_relativa = str(ruta_destino.relative_to(BASE_DIR))
        print(f"[SETUP] Descargando {ruta_relativa} desde {url} ...")
        tamano = descargar_activo(url, ruta_destino)
        tamano_kb = round(tamano / 1024.0, 2)
        print(f"[OK] {ruta_relativa} almacenado ({tamano_kb} KB).")
        resumen[ruta_relativa] = tamano

    return resumen


if __name__ == "__main__":
    configurar_activos_locales()
