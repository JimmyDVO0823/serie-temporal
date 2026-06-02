"""
ARIMA/SARIMA interactivo y universal - Versión Especial Modelo UFPS
Modularized Version
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

# Import modular components
from data_loader import load_data_interactive, ask_int, clean_name, make_future_labels
from econometric_engine import estimate_sarima_models
from results_manager import (
    generate_text_report, 
    generate_excel_report, 
    generate_plots, 
    MAPPING_ECONO
)

def run_analysis(data, steps):
    y = np.asarray(data["series"], dtype=float)
    time_values = list(data["time"])
    time_mode = data["time_mode"]
    variable_name = data["variable_name"]
    file_path = data["file"]

    # 1. Estimation
    engine_results = estimate_sarima_models(y, steps)
    mejor = engine_results["mejor"]
    validos = engine_results["validos"]
    results = engine_results["results"]
    adf = engine_results["adf"]
    acf_pacf = engine_results["acf_pacf"]

    # 2. Preparation of results
    out_dir = f"resultados_{clean_name(variable_name)}"
    os.makedirs(out_dir, exist_ok=True)

    fc = mejor["res"].get_forecast(steps=steps)
    p_mu = np.asarray(fc.predicted_mean)
    p_ci = np.asarray(fc.conf_int(alpha=0.05))
    future_labels = make_future_labels(time_values, time_mode, steps)

    # 3. Generating reports
    generate_text_report(
        out_dir, variable_name, file_path, adf, acf_pacf, 
        results, validos, mejor, future_labels, p_mu, p_ci
    )

    pron_df = pd.DataFrame({
        "Periodo": future_labels,
        "Pronóstico": np.round(p_mu, 2),
        "LI 95%": np.round(p_ci[:, 0], 2),
        "LS 95%": np.round(p_ci[:, 1], 2),
    })
    pron_path = os.path.join(out_dir, f"pronostico_{clean_name(variable_name)}.csv")
    pron_df.to_csv(pron_path, index=False, encoding="utf-8")

    generate_excel_report(out_dir, mejor, pron_df)

    # 4. Plots
    generate_plots(
        out_dir, variable_name, file_path, y, steps, mejor, 
        future_labels, p_mu, p_ci, data["time_label"], acf_pacf
    )

    # 5. CLI Feedback
    print("\n" + "=" * 55)
    print(f"  MEJOR MODELO SELECCIONADO: {mejor['nombre']}")
    print(f"  AICc = {mejor['aicc']:.4f}   BIC = {mejor['bic']:.4f}")
    print(f"  RMSE = {mejor['rmse']:.4f}   MAE = {mejor['mae']:.4f}")
    print(f"  Ljung-Box p-valor = {mejor['p_lb']:.4f}")
    print("\n  PRONÓSTICO DE PROYECCIÓN:")
    for a, mu, li, ls in zip(future_labels, p_mu, p_ci[:, 0], p_ci[:, 1]):
        print(f"    {a}: {mu:.2f}  [IC 95%: {li:.2f} – {ls:.2f}]")
    print("=" * 55)
    print(f"\n  Archivos econométricos generados con éxito en: ./{out_dir}/")
    print(f"   • reporte_modelos.txt")
    print(f"   • {os.path.basename(pron_path)}")
    print(f"   • reporte_econometrico_profesional.xlsx")
    print(f"   • grafico_diagnostico.png")
    print(f"   • grafico_pronostico.png")

    return out_dir

def parse_args():
    parser = argparse.ArgumentParser(description="SARIMA universal interactivo - Metodología UFPS")
    parser.add_argument("--file", type=str, default="", help="Ruta al archivo CSV o Excel (opcional)")
    parser.add_argument("--variable", type=str, default="", help="Variable (opcional)")
    parser.add_argument("--steps", type=int, default=3, help="Cantidad de pasos a pronosticar")
    return parser.parse_args()

def main():
    args = parse_args()
    try:
        data = load_data_interactive(args)
        steps = args.steps if args.steps and args.steps > 0 else ask_int("¿Cuántos pasos trimestrales quieres proyectar?", default=4, min_value=1)
        run_analysis(data, steps)
    except Exception as e:
        print(f"\n⚠ ERROR EN SISTEMA: {e}")
        raise

if __name__ == "__main__":
    main()