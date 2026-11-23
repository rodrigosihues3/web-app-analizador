"""Lógica del sistema sin dependencias de frameworks web"""

from core.auditor import DatasetAuditor
from core.estadistica import (
    analizar_valores_clave,
    extraer_metricas_grafico,
    matriz_correlacion,
    resumen_descriptivo,
)
from core.exportador import exportar_dataframe, generar_reporte_pdf
from core.sintetizador import explicar_grafico_ia, generar_contexto_ejecutivo
from core.transformador import limpiar_dataset
from core.visualizador import (
    generar_barras_categoria_plotly,
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
    "extraer_metricas_grafico",
    "generar_histograma_plotly",
    "generar_barras_categoria_plotly",
    "generar_boxplot_plotly",
    "generar_heatmap_plotly",
    "generar_graficos_estaticos_pdf",
    "generar_contexto_ejecutivo",
    "explicar_grafico_ia",
    "exportar_dataframe",
    "generar_reporte_pdf",
]
