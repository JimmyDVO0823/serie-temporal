import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_squared_error, mean_absolute_error

def estimate_sarima_models(y, steps):
    # SARIMA Estacional requiere un set de datos mínimo para calcular diferencias de rezago 4
    if len(y) < 12:
        raise ValueError("La serie tiene muy pocos datos para aplicar un análisis SARIMA con estacionalidad (Mínimo requerido: 12 datos).")

    # Diferencia Regular (d=1)
    wt = np.diff(y)
    n = len(y)

    adf_y = adfuller(y, maxlag=min(3, len(y)-2), autolag=None)
    adf_wt = adfuller(wt, maxlag=min(3, len(wt)-2), autolag=None)

    estac_y = adf_y[1] < 0.05
    estac_wt = adf_wt[1] < 0.05

    n_lags = min(8, max(1, len(wt)-1))
    acf_vals = acf(wt, nlags=n_lags, fft=False)[1:]
    pacf_vals = pacf(wt, nlags=n_lags)[1:]
    lim95 = 1.96 / np.sqrt(len(wt))

    # --- CONFIGURACIÓN ESTRUCTURA REQUERIDA POR EL PROFESOR (SARIMA) ---
    models_to_try = [
        ("SARIMA(0,1,0)x(0,1,0)4", (0, 1, 0), (0, 1, 0, 4)),
        ("SARIMA(1,1,0)x(0,1,0)4", (1, 1, 0), (0, 1, 0, 4)),
        ("SARIMA(1,1,0)x(1,1,0)4", (1, 1, 0), (1, 1, 0, 4)),
        ("SARIMA(1,1,1)x(1,1,1)4", (1, 1, 1), (1, 1, 1, 4)),
    ]

    results = []
    scale = max(float(np.nanmax(np.abs(y))), 1.0)

    for name, order, seasonal_order in models_to_try:
        try:
            res = SARIMAX(
                y,
                order=order,
                seasonal_order=seasonal_order,
                trend="c",
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
            k = len(res.params)
            denom = max(n - k - 1, 1)
            aicc = res.aic + (2 * k * (k + 1)) / denom
            
            start_idx = order[1] + (seasonal_order[1] * 4)
            
            rmse = np.sqrt(mean_squared_error(y[start_idx:], res.fittedvalues[start_idx:]))
            mae = mean_absolute_error(y[start_idx:], res.fittedvalues[start_idx:])
            lb_lag = min(10, max(1, len(res.resid[start_idx:]) - 1))
            lb = acorr_ljungbox(res.resid[start_idx:], lags=[lb_lag], return_df=True)
            p_lb = float(lb["lb_pvalue"].values[0])

            ma_ok = True
            if "ma.L1" in res.param_names:
                theta = res.params[res.param_names.index("ma.L1")]
                if abs(theta) >= 0.99:
                    ma_ok = False
            if "ma.S.L4" in res.param_names:
                theta_s = res.params[res.param_names.index("ma.S.L4")]
                if abs(theta_s) >= 0.99:
                    ma_ok = False

            fc_test = np.asarray(res.get_forecast(steps=min(steps, 3)).predicted_mean)
            forecast_ok = np.all(np.isfinite(fc_test))
            if forecast_ok and np.any(np.abs(fc_test) > scale * 50):
                forecast_ok = False

            results.append({
                "nombre": name,
                "orden": order,
                "seasonal_order": seasonal_order,
                "res": res,
                "aic": res.aic,
                "aicc": aicc,
                "bic": res.bic,
                "rmse": rmse,
                "mae": mae,
                "p_lb": p_lb,
                "ma_ok": ma_ok,
                "forecast_ok": forecast_ok,
            })
        except Exception as e:
            results.append({"nombre": name, "orden": order, "res": None, "error": str(e)})

    validos = [
        r for r in results
        if r.get("res") is not None and r.get("ma_ok", True) and r.get("forecast_ok", True)
    ]

    if not validos:
        validos = [r for r in results if r.get("res") is not None and r.get("ma_ok", True)]

    if not validos:
        raise RuntimeError("No se pudo estimar ningún modelo SARIMA válido con estos datos.")

    mejor = min(validos, key=lambda x: x["aicc"])
    
    return {
        "results": results,
        "validos": validos,
        "mejor": mejor,
        "adf": {"y": adf_y, "wt": adf_wt, "estac_y": estac_y, "estac_wt": estac_wt},
        "acf_pacf": {"acf_vals": acf_vals, "pacf_vals": pacf_vals, "lim95": lim95}
    }
