import csv
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.stats.diagnostic import acorr_ljungbox

# =============================================================================
# 0. CONFIGURACIÓN
# =============================================================================
carpeta_salida = 'resultados_arima'
if not os.path.exists(carpeta_salida):
    os.makedirs(carpeta_salida)

archivo_csv = 'serie.csv'
if not os.path.exists(archivo_csv):
    archivo_csv = 'serie del dane.csv'
    if not os.path.exists(archivo_csv):
        exit(1)

años = []
td_valores = []
with open(archivo_csv, 'r', encoding='utf-8-sig', errors='ignore') as f:
    reader = csv.reader(f)
    for row in reader:
        if not row: continue
        if any(cell.strip() == '2007' for cell in row):
            años = [int(c.strip()) for c in row if c.strip().isdigit() and len(c.strip()) == 4]
        row_str = " ".join(row)
        if 'Tasa de Desocupación' in row_str:
            td_valores = []
            for cell in row:
                s = cell.strip().replace('"', '').replace(',', '.')
                try: td_valores.append(float(s))
                except ValueError: continue

min_len = min(len(años), len(td_valores))
df = pd.DataFrame({'Año': años[:min_len], 'TD': td_valores[:min_len]}).sort_values('Año')
y = df['TD'].values

# =============================================================================
# 2. MODELADO ARIMA
# =============================================================================
modelos = [(1,1,0), (2,1,0), (1,1,1)]
reporte_txt = os.path.join(carpeta_salida, 'reporte_modelos.txt')
mejor_aic = float('inf')
mejor_fit = None
mejor_nombre = ""

with open(reporte_txt, 'w', encoding='utf-8') as f:
    f.write("=========================================================\n")
    f.write("      REPORTE ARIMAS - NORTE DE SANTANDER (UFPS)         \n")
    f.write("      (Resultados formateados para COPIAR Y PEGAR)       \n")
    f.write("=========================================================\n\n")

    for p, d, q in modelos:
        nombre = f"ARIMA({p},{d},{q})"
        try:
            # trend='t' simula la constante en diferencias para d=1 en algunas versiones
            model = ARIMA(y, order=(p, d, q), trend='t')
            res = model.fit()
            
            rmse = np.sqrt(mean_squared_error(y[d:], res.fittedvalues[d:]))
            mae = mean_absolute_error(y[d:], res.fittedvalues[d:])
            lb = acorr_ljungbox(res.resid[d:], lags=[10], return_df=True)
            p_lb = lb['lb_pvalue'].values[0]
            
            f.write(f">> RESULTADOS PARA EL MODELO: {nombre}\n")
            f.write(f"Estadísticas de Bondad de Ajuste:\n")
            f.write(f"AIC:      {res.aic:.4f}\n")
            f.write(f"BIC:      {res.bic:.4f}\n")
            f.write(f"RMSE:     {rmse:.4f}\n")
            f.write(f"MAE:      {mae:.4f}\n")
            f.write(f"Ljung-Box (p-valor, 10 lags): {p_lb:.4f}\n\n")
            
            f.write(f"Estimación de Parámetros:\n")
            f.write(f"{'Parámetro':<15} | {'Valor (Coef)':<12} | {'p-valor':<10}\n")
            f.write("-" * 42 + "\n")
            
            # Acceso robusto a parámetros (Series o Dict)
            params = res.params
            pvalues = res.pvalues
            
            mapping = {'x1': 'Constante (c0)', 'ar.L1': 'phi 1 (AR1)', 'ar.L2': 'phi 2 (AR2)', 'ma.L1': 'theta 1 (MA1)'}
            
            # Si params es Series, usamos su índice
            if hasattr(params, 'index'):
                for par in params.index:
                    val = params[par]
                    pval = pvalues[par]
                    p_disp = mapping.get(par, par)
                    f.write(f"{p_disp:<15} | {val:<12.6f} | {pval:<10.4f}\n")
            else:
                # Si es un array, usamos los nombres por posición
                param_names = res.model.param_names
                for i, val in enumerate(params):
                    name = param_names[i]
                    pval = pvalues[i]
                    p_disp = mapping.get(name, name)
                    f.write(f"{p_disp:<15} | {val:<12.6f} | {pval:<10.4f}\n")
            
            f.write("\n" + "="*50 + "\n\n")
            if res.aic < mejor_aic:
                mejor_aic = res.aic
                mejor_fit = res
                mejor_nombre = nombre
        except Exception as e:
            f.write(f"⚠️ Error en {nombre}: {str(e)}\n\n")

if mejor_fit:
    fc = mejor_fit.get_forecast(steps=3)
    p_mean = fc.predicted_mean
    p_ci = fc.conf_int(alpha=0.05)
    anos_p = [2026, 2027, 2028]
    p_df = pd.DataFrame({'Año': anos_p, 'Pronóstico (%)': np.round(p_mean, 2), 'Límite Inferior': np.round(p_ci[:, 0], 2), 'Límite Superior': np.round(p_ci[:, 1], 2)})
    p_df.to_csv(os.path.join(carpeta_salida, 'pronostico_2026_2028.csv'), index=False)
    
    plt.figure(figsize=(12, 6))
    plt.plot(df['Año'], df['TD'], marker='o', color='navy', label='Histórico (DANE)')
    plt.plot(anos_p, p_mean, marker='s', ls='--', color='red', label=f'Proyección {mejor_nombre}')
    plt.fill_between(anos_p, p_ci[:, 0], p_ci[:, 1], color='red', alpha=0.1)
    plt.title(f'Proyección TD Norte de Santander - {mejor_nombre}\n(UFPS Econometría)')
    plt.xlabel('Años'); plt.ylabel('Tasa (%)'); plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(os.path.join(carpeta_salida, 'grafico_pronostico.png'), dpi=300)
    print("✅ Proceso completado.")
else:
    print("❌ Error.")