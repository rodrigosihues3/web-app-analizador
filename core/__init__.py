"""Capa de dominio analítico del Analizador Multiparadigma de Datos.

Este paquete implementa la lógica pura del sistema sin dependencias de
frameworks web, combinando Programación Orientada a Objetos (auditoría de
estado estructural) y Programación Funcional (transformación y estadística
inmutables).
"""

from core.auditor import DatasetAuditor
from core.estadistica import (
    analizar_valores_clave,
    matriz_correlacion,
    resumen_descriptivo,
)
from core.exportador import exportar_dataframe, generar_reporte_pdf
from core.sintetizador import generar_contexto_ejecutivo
from core.transformador import limpiar_dataset
from core.visualizador import (
    generar_boxplot_plotly,
    generar_graficos_estaticos_pdf,
    generar_heatmap_plotly,
    generar_histograma_plotly,
)

__all__ = [
    "DatasetAuditor",
    "limpiar_dataset",
    "resumen_descriptivo",
    "analizar_valores_clave",
    "matriz_correlacion",
    "generar_histograma_plotly",
    "generar_boxplot_plotly",
    "generar_heatmap_plotly",
    "generar_graficos_estaticos_pdf",
    "generar_contexto_ejecutivo",
    "exportar_dataframe",
    "generar_reporte_pdf",
]
