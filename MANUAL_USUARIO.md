# Manual de Usuario - Herramienta ARIMA Universal (UFPS)

Esta herramienta automatiza el análisis econométrico de series temporales utilizando modelos ARIMA. Está diseñada para ser flexible y funcionar con cualquier conjunto de datos en formato CSV (por ejemplo, exportaciones del DANE-GEIH).

## 🚀 Requisitos Previos

Asegúrate de tener instalado Python 3 y las siguientes librerías:

```bash
pip install numpy pandas statsmodels matplotlib scikit-learn
```

## 📊 Formato del Archivo CSV

El programa es inteligente y puede saltar metadatos, pero el CSV debe cumplir:
1.  **Fila de Años**: Debe existir una fila con los años (ej. 2007, 2008...).
2.  **Etiquetas de Fila**: La primera columna debe contener el nombre de la variable (ej. "Tasa de Desocupación (TD)").
3.  **Separador decimal**: El programa acepta tanto puntos (`.`) como comas (`,`).

## 🛠️ Cómo Usar el Programa

Abre una terminal en la carpeta del proyecto y usa los siguientes comandos:

### 1. Análisis por Defecto
Analiza la "Tasa de Desocupación (TD)" en el archivo `serie.csv` por defecto:
```bash
python arima_ufps.py
```

### 2. Seleccionar una Variable Específica
Usa el argumento `--variable` seguido del nombre (o parte del nombre) de la variable:
```bash
python arima_ufps.py --variable "Tasa de Ocupación"
```

### 3. Usar otro Archivo CSV
Usa el argumento `--file` para especificar la ruta del archivo:
```bash
python arima_ufps.py --file "mis_datos.csv"
```

### 4. Ajustar el Tiempo de Pronóstico
Usa el argumento `--steps` para definir cuántas unidades de tiempo (años) quieres proyectar:
```bash
python arima_ufps.py --variable "TGP" --steps 5
```

### 💡 Ayuda con los Nombres
Si no conoces el nombre exacto de la variable, escribe cualquier cosa y el programa te listará todas las opciones disponibles en el archivo:
```bash
python arima_ufps.py --variable "lista"
```

## 📁 Resultados Generados

Cada análisis crea una carpeta única (ej. `resultados_Tasa_de_Ocupación_TO/`) que contiene:

1.  **`reporte_modelos.txt`**: Documento técnico con pruebas de estacionariedad (ADF), selección del mejor modelo (AICc) y pronóstico detallado.
2.  **`pronostico_año_año.csv`**: Tabla de datos lista para importar a Excel.
3.  **`grafico_diagnostico.png`**: Imagen con 4 paneles que validan la calidad del modelo y el ruido blanco de los residuales.
4.  **`grafico_pronostico.png`**: Gráfica limpia del histórico y la proyección futura para presentaciones.

---
*Desarrollado para la cátedra de Investigación de Operaciones - UFPS*
