# Manual de Usuario: Sistema de Análisis SARIMA - Metodología UFPS

Este proyecto provee una herramienta interactiva en Python para realizar análisis econométricos de series de tiempo trimestrales utilizando modelos SARIMAX (Metodología Box-Jenkins). Está especialmente diseñado para el análisis de la Tasa de Desocupación (TD) con rezagos estacionales.

## 1. Estructura del Proyecto

El sistema está organizado de forma modular para facilitar su mantenimiento:

*   **`arima_ufps_interactivo_corregido.py`**: El archivo principal (punto de entrada). Coordina la carga de datos, estimación y generación de reportes.
*   **`data_loader.py`**: Gestiona la lectura de archivos (CSV, XLSX, XLS), la detección de formatos (vertical/horizontal) y los menús interactivos.
*   **`econometric_engine.py`**: Contiene el motor de cálculo estadístico, las pruebas de estacionariedad (ADF), ACF/PACF y la selección del mejor modelo mediante el criterio AICc.
*   **`results_manager.py`**: Encargado de exportar los resultados en formato texto, Excel profesional y generar los gráficos de diagnóstico y pronóstico.

## 2. Requisitos Previos

Asegúrate de tener instaladas las siguientes librerías de Python:

```bash
pip install numpy pandas matplotlib statsmodels scikit-learn openpyxl
```

## 3. Modo de Uso

### Ejecución Básica
Para iniciar el asistente interactivo, ejecuta:

```bash
python arima_ufps_interactivo_corregido.py
```

### Argumentos de Línea de Comandos (Opcional)
Puedes saltarte algunos pasos iniciales usando argumentos:

*   `--file ruta/al/archivo.csv`: Especifica el archivo de datos.
*   `--steps N`: Define cuántos periodos trimestrales quieres proyectar (ej: 4).

### Flujo Interactivo
1.  **Selección de Archivo**: El script detectará automáticamente archivos compatibles en la carpeta.
2.  **Selección de Variable**: Podrás elegir qué columna o variable deseas analizar.
3.  **Configuración de Tiempo**: Selecciona cómo están organizados los periodos (Año, Trimestre, o ambos).
4.  **Pasos de Pronóstico**: Indica cuántos trimestres hacia el futuro deseas predecir.

## 4. Resultados Generados

Toda la salida se guarda en una carpeta dedicada llamada `resultados_[Nombre_de_la_Variable]`.

### Archivos de Salida:
1.  **`reporte_modelos.txt`**:
    *   Resultados de las pruebas de estacionariedad (ADF).
    *   Resumen de todos los modelos candidatos evaluados.
    *   **Ecuación Matemática Formal**: Al final del archivo se muestra la ecuación Box-Jenkins (SARIMA 1,1,0 x 1,1,0_4) sustituyendo los coeficientes reales obtenidos.
2.  **`reporte_econometrico_profesional.xlsx`**:
    *   Un archivo Excel con diseño institucional (Azul Marino/Gris).
    *   **Pestaña 1**: Resumen de coeficientes (Drift, Rezago Regular, Rezago Estacional) y métricas globales (AICc, RMSE, etc.).
    *   **Pestaña 2**: Tabla detallada de pronósticos con intervalos de confianza al 95%.
3.  **`grafico_diagnostico.png`**:
    *   Visualización de la serie original y el ajuste.
    *   Análisis de residuales (FAC de residuales para verificar Ruido Blanco).
    *   FAC/FACP de la serie diferenciada.
4.  **`grafico_pronostico.png`**:
    *   Gráfica de la serie histórica con la proyección futura y su banda de incertidumbre.
5.  **`pronostico_[Variable].csv`**:
    *   Tabla simple con los valores numéricos de la predicción.

## 5. Metodología Aplicada

El script automatiza:
*   La transformación de la serie mediante diferencias regulares ($d=1$) y estacionales ($D=1$).
*   La estimación de parámetros para el modelo **SARIMA(1,1,0)x(1,1,0)4**.
*   Validación estadística de los residuales mediante la prueba de Ljung-Box para asegurar que la estimación sea válida (Ruido Blanco).
