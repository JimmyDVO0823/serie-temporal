import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox

# ==============================================================================
# 1. CARGAR Y LIMPIAR LOS DATOS
# ==============================================================================
# Usamos el nombre exacto del archivo extraído que subiste
archivo_csv = 'ARIMA_NorteDeSantander_DANE_UFPS.xlsx - DATOS.csv'

# Saltamos las primeras 3 filas que contienen los metadatos y títulos del DANE
df = pd.read_csv(archivo_csv, skiprows=3)

# Renombramos las columnas para poder manipularlas fácilmente
df.columns = ['Unnamed', 't', 'Año', 'TD', 'Pob_Total', 'Pob_Ocupada', 'Pob_Desocupada']

# Eliminamos filas que no tengan datos numéricos reales en las variables clave
df = df.dropna(subset=['Año', 'TD'])

# Aseguramos el tipo de dato correcto (Años enteros, Tasas decimales)
df['Año'] = df['Año'].astype(int)
df['TD'] = df['TD'].astype(float)

print("=== DATOS REGIONALES DETECTADOS ===")
print(df[['Año', 'TD']].to_string(index=False))
print("-" * 50)

# ==============================================================================
# 2. GRAFICAR LA SERIE HISTÓRICA ORIGINAL
# ==============================================================================
plt.figure(figsize=(10, 5))
plt.plot(df['Año'], df['TD'], marker='o', color='#008080', linewidth=2, label='TD Histórica DANE')
plt.title('Tasa de Desocupación (TD) — Norte de Santander (2007-2025)', fontsize=12, fontweight='bold')
plt.xlabel('Año')
plt.ylabel('Porcentaje (%)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.savefig('1_serie_original_norte_santander.png', dpi=300)
print("✔ Gráfico de la serie histórica guardado como '1_serie_original_norte_santander.png'")

# ==============================================================================
# 3. MODELO 1: AJUSTAR ARIMA(1,1,0)
# ==============================================================================
modelo_110 = ARIMA(df['TD'].values, order=(1, 1, 0))
res_110 = modelo_110.fit()

print("\n==============================================================================")
print("                  RESULTADOS DEL MODELO ARIMA(1,1,0)")
print("==============================================================================")
print(res_110.summary())

# Prueba Ljung-Box para verificar si los residuos son ruido blanco
lb_110 = acorr_ljungbox(res_110.resid, lags=[1], return_df=True)
print("\nPrueba de Ljung-Box para ARIMA(1,1,0) (p-valor esperado > 0.05):")
print(lb_110)

# ==============================================================================
# 4. MODELO 2: AJUSTAR ARIMA(2,1,0) PARA COMPARACIÓN
# ==============================================================================
modelo_210 = ARIMA(df['TD'].values, order=(2, 1, 0))
res_210 = modelo_210.fit()

print("\n==============================================================================")
print("                  RESULTADOS DEL MODELO ARIMA(2,1,0)")
print("==============================================================================")
print(res_210.summary())

# ==============================================================================
# 5. GENERAR PRONÓSTICO (AÑOS 2026, 2027, 2028)
# ==============================================================================
pasos = 3
pronostico_objeto = res_110.get_forecast(steps=pasos)
valores_pronostico = pronostico_objeto.predicted_mean
intervalos_confianza = pronostico_objeto.conf_int(alpha=0.05)  # Confianza del 95%

anios_futuros = np.array([2026, 2027, 2028])

print("\n==============================================================================")
print("                 PRONÓSTICO GENERADO PARA EL TRABAJO (2026-2028)")
print("==============================================================================")
for i in range(pasos):
    print(f"Año {anios_futuros[i]}: Pronóstico = {valores_pronostico[i]:.2f}% | "
          f"Intervalo 95% = [{intervalos_confianza[i][0]:.2f}% - {intervalos_confianza[i][1]:.2f}%]")
print("-" * 78)

# ==============================================================================
# 6. GRAFICAR HISTÓRICO + PRONÓSTICO DE LA UFPS
# ==============================================================================
plt.figure(figsize=(12, 6))

# Línea de datos históricos
plt.plot(df['Año'], df['TD'], marker='o', color='#008080', label='Datos Históricos (DANE)')

# Línea de los años proyectados
plt.plot(anios_futuros, valores_pronostico, marker='s', color='#FF8C00', linestyle='--', label='Pronóstico ARIMA(1,1,0)')

# Área sombreada del intervalo de confianza
plt.fill_between(anios_futuros, intervalos_confianza[:, 0], intervalos_confianza[:, 1], 
                 color='#FF8C00', alpha=0.15, label='Intervalo de Confianza (95%)')

plt.title('Modelación y Pronóstico de la Tasa de Desocupación en Norte de Santander', fontsize=12, fontweight='bold')
plt.xlabel('Año')
plt.ylabel('Porcentaje (%)')
plt.xticks(np.append(df['Año'].values, anios_futuros), rotation=45)
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('2_pronostico_final_arima.png', dpi=300)
print("✔ Gráfico de pronóstico guardado como '2_pronostico_final_arima.png'\nProceso completado.")