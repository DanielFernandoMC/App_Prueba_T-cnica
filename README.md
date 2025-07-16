# App Prueba Técnica Bancolombia
## Prueba técnica para vacante de Analista de Datos II.
README - Dashboard Ejecutivo de Movimientos y Saldos Bancarios

Nombre del Proyecto: Dashboard Ejecutivo - Análisis de Movimientos y Saldos Bancolombia 
Versión 01 
Fecha: Julio 2025
Autor: Daniel Manosalva

📅 Propósito General

El presente dashboard fue diseñado para proveer una herramienta interactiva, automatizada y visualmente amigable para la toma de decisiones estratégicas basadas en los movimientos y saldos de las cuentas corporativas. El sistema permite analizar desde una visión agregada hasta niveles de detalle por negocio, cuenta o tercero, integrando capacidades de análisis, predicción y detección de anomalías.

📊 Principales Características

1. Visión General

Evolución temporal de saldos por negocio y cuenta.

Comparativos semanales y mensuales de movimientos (débitos y créditos).

Rankings de cuentas y negocios con mayores movimientos netos.

Comparación de saldos vs movimientos ajustados.

Tendencias generales.

2. Análisis Detallado

Filtros por negocio, cuenta y rango de fechas.

Gráficos de distribución de movimientos, saldos y netos.

Rankings de terceros y cuentas más activas.

Descarga de datos en formato CSV.

3. Exploración Avanzada

Detección de anomalías por porcentaje de débito sobre saldo (>70%).

Clasificación de riesgo con modelos de Machine Learning (Random Forest, árboles de decisión).

Análisis de correlaciones entre variables clave.

Predicción de saldos semanales mediante modelos Holt-Winters.

Identificación de cuentas dormidas (sin movimiento en >30 días).



🔗 Integración de Datos y Reglas de Negocio

Se realiza una limpieza exhaustiva de los datos importados desde un archivo Excel con dos hojas: movimientos y saldos.

Las reglas de negocio incluyen:

Multiplicación/división de débitos y créditos según condiciones específicas de comentario, documento y cuenta.

Cálculo de indicadores como neto, porcentaje debito sobre saldo y riesgo.

🧰 Librerías Utilizadas

Python (pandas, matplotlib, plotly, scikit-learn, statsmodels)

Streamlit como framework principal para la visualización web

Excel como fuente de datos estructurados

Machine Learning para clasificación de riesgo (Random Forest, Decision Tree)

Modelos de serie de tiempo (Holt-Winters) para pronósticos de saldo


🛍️ Recomendaciones Finales

El dashboard puede ser extendido para incluir validación de alertas, conexiones en tiempo real a bases de datos y automatización de reportes periódicos.

URL de la App (Ejemplo Local): http://localhost:8501/
Contacto: danielmanosalva951117@hotmail.com

Este proyecto representa una solución integral para el monitoreo de información financiera crítica, permitiendo a la organización anticiparse a riesgos, detectar oportunidades y tomar decisiones basadas en evidencia.
