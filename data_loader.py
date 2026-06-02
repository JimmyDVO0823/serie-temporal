import os
import re
import glob
import pandas as pd
import numpy as np
from pathlib import Path

TIME_COLS = {
    "año", "anio", "year", "time", "tiempo", "t", "trimestre", "quarter",
    "fecha", "date", "periodo", "period", "mes", "month"
}

def ask_choice(prompt, options, default=None):
    if not options:
        raise ValueError("No hay opciones para elegir.")
    while True:
        print("\n" + prompt)
        for i, opt in enumerate(options, 1):
            print(f"  {i}. {opt}")
        if default is not None:
            raw = input(f"Elige una opción [1-{len(options)}] (Enter = {default}): ").strip()
        else:
            raw = input(f"Elige una opción [1-{len(options)}]: ").strip()

        if raw == "" and default is not None:
            return options[default - 1]

        try:
            idx = int(raw)
            if 1 <= idx <= len(options):
                return options[idx - 1]
        except ValueError:
            pass
        print("Entrada inválida. Intenta de nuevo.")

def ask_int(prompt, default=3, min_value=1):
    while True:
        raw = input(f"{prompt} (Enter = {default}): ").strip()
        if raw == "":
            return default
        try:
            val = int(raw)
            if val >= min_value:
                return val
        except ValueError:
            pass
        print(f"Ingresa un número entero >= {min_value}.")

def _parse_number_cell(v, default_mode="auto"):
    if pd.isna(v):
        return np.nan
    s = str(v).strip()
    if s in {"", "nan", "NaN", "-", "—"}:
        return np.nan

    if default_mode == "dot_thousands":
        s = s.replace(".", "").replace(",", ".")
    elif default_mode == "comma_decimal":
        s = s.replace(",", ".")
    elif default_mode == "dot_decimal":
        if "," in s and "." in s:
            if s.rfind(",") > s.rfind("."):
                s = s.replace(".", "").replace(",", ".")
            else:
                s = s.replace(",", "")
        elif "," in s:
            s = s.replace(",", ".")
    else:
        if re.match(r"^\d{1,3}(\.\d{3})+(,\d+)?$", s):
            s = s.replace(".", "").replace(",", ".")
        elif re.match(r"^\d{1,3}(,\d{3})+(\.\d+)?$", s):
            s = s.replace(",", "")
        elif "," in s and "." in s:
            if s.rfind(",") > s.rfind("."):
                s = s.replace(".", "").replace(",", ".")
            else:
                s = s.replace(",", "")
        elif "," in s:
            s = s.replace(",", ".")
    return pd.to_numeric(s, errors="coerce")

def normalize_numeric_series(s):
    ser = pd.Series(s, dtype="object")
    txt = ser.astype(str).str.strip()

    non_empty = txt[~txt.isin(["", "nan", "NaN", "-", "—"])]

    if len(non_empty) == 0:
        return ser.map(_parse_number_cell)

    has_comma = non_empty.str.contains(",", regex=False).any()
    has_dot = non_empty.str.contains(".", regex=False).any()

    dot_thousands = non_empty.str.match(r"^\d{1,3}(\.\d{3})+$", na=False).sum()

    if has_comma and not has_dot:
        mode = "comma_decimal"
    elif has_dot and not has_comma:
        mode = "dot_thousands" if dot_thousands >= max(2, len(non_empty) // 2) else "dot_decimal"
    else:
        mode = "auto"

    return txt.map(lambda v: _parse_number_cell(v, default_mode=mode))

def clean_name(name):
    name = str(name)
    name = re.sub(r"[^\w\s\-]+", "", name, flags=re.UNICODE)
    name = re.sub(r"\s+", "_", name.strip())
    return name[:80] if name else "serie"

def read_file_flexible(path):
    ext = Path(path).suffix.lower()
    if ext in [".xlsx", ".xls"]:
        try:
            df = pd.read_excel(path, header=None)
            return df, "excel"
        except Exception as e:
            raise RuntimeError(f"Error leyendo el archivo Excel: {e}")
    else:
        encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252"]
        last_error = None
        for enc in encodings:
            try:
                try:
                    df = pd.read_csv(path, header=None, sep=None, engine="python", encoding=enc)
                except Exception:
                    df = pd.read_csv(path, header=0, sep=None, engine="python", encoding=enc)
                return df, enc
            except Exception as e:
                last_error = e
        raise last_error

def is_year_like(v):
    try:
        n = int(float(str(v).replace(",", ".").strip()))
        return 1900 <= n <= 2100
    except Exception:
        return False

def detect_format(path):
    df0, _ = read_file_flexible(path)
    first_row = df0.iloc[0].tolist() if len(df0) else []
    year_like = sum(is_year_like(x) for x in first_row)
    first_col_text = sum(not pd.isna(x) and not is_year_like(x) for x in df0.iloc[:, 0].tolist()) if df0.shape[1] else 0
    if year_like >= max(4, len(first_row) // 2):
        return "vertical"
    
    ext = Path(path).suffix.lower()
    if ext in [".xlsx", ".xls"]:
        try:
            dfh = pd.read_excel(path)
        except Exception:
            dfh = None
    else:
        try:
            dfh = pd.read_csv(path, sep=None, engine="python", encoding="utf-8")
        except Exception:
            try:
                dfh = pd.read_csv(path, sep=None, engine="python", encoding="latin1")
            except Exception:
                dfh = None
                
    if dfh is not None and any(str(c).strip().lower() in TIME_COLS for c in dfh.columns):
        return "horizontal"
    if first_col_text > 3 and year_like < 3:
        return "horizontal"
    return "vertical"

def list_supported_files():
    extensions = ("*.csv", "*.CSV", "*.xlsx", "*.XLSX", "*.xls", "*.XLS")
    files = []
    seen = set()
    for ext in extensions:
        for p in sorted(glob.glob(ext)):
            if p not in seen:
                seen.add(p)
                files.append(p)
    return files

def choose_file_interactive():
    files = list_supported_files()
    if not files:
        raise FileNotFoundError("No se encontraron archivos CSV o Excel (.xlsx, .xls) en el directorio actual.")
    if len(files) == 1:
        print(f"Archivo detectado: {files[0]}")
        return files[0]
    return ask_choice("Selecciona el archivo para analizar:", files)

def load_vertical(path):
    df, enc = read_file_flexible(path)
    row_years = None
    for idx, row in df.iterrows():
        nums = normalize_numeric_series(row)
        years = nums.dropna()
        years = years[(years >= 1900) & (years <= 2100)]
        if len(years) >= 4:
            row_years = idx
            break
    if row_years is None:
        raise ValueError("No se pudo detectar la fila de años.")

    years_raw = normalize_numeric_series(df.iloc[row_years]).dropna()
    years = years_raw[(years_raw >= 1900) & (years_raw <= 2100)].astype(int).tolist()

    variables = []
    map_idx = []
    for idx, val in df.iloc[:, 0].items():
        if idx == row_years:
            continue
        if pd.isna(val):
            continue
        name = str(val).strip()
        if name and name.lower() != "nan":
            variables.append(name)
            map_idx.append(idx)

    if not variables:
        raise ValueError("No se detectaron variables en la primera columna.")

    chosen = ask_choice("Variables detectadas en el archivo:", variables)
    idx_var = map_idx[variables.index(chosen)]

    raw_vals = df.iloc[idx_var, 1:1 + len(years)]
    y = normalize_numeric_series(raw_vals).to_numpy(dtype=float)
    years_arr = np.array(years, dtype=int)

    mask = ~np.isnan(y)
    y = y[mask]
    years_arr = years_arr[mask].tolist()

    return {
        "format": "vertical",
        "file": path,
        "encoding": enc,
        "variable_name": chosen,
        "series": y,
        "time": years_arr,
        "time_label": "Año",
        "time_mode": "year",
    }

def quarter_to_num(q):
    q = str(q).strip().upper()
    mapping = {
        "I": 1, "1": 1, "Q1": 1,
        "II": 2, "2": 2, "Q2": 2,
        "III": 3, "3": 3, "Q3": 3,
        "IV": 4, "4": 4, "Q4": 4,
    }
    return mapping.get(q)

def build_quarter_labels(years, quarters):
    labels = []
    for y, q in zip(years, quarters):
        qn = quarter_to_num(q)
        if qn is None:
            labels.append(f"{y} {q}")
        else:
            labels.append(f"{int(y)} Q{qn}")
    return labels

def next_quarters(last_year, last_quarter_num, steps):
    out = []
    y, q = int(last_year), int(last_quarter_num)
    for _ in range(steps):
        q += 1
        if q > 4:
            q = 1
            y += 1
        out.append((y, q))
    return out

def load_horizontal(path):
    ext = Path(path).suffix.lower()
    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8")
        if df.shape[1] == 1:
            df = pd.read_csv(path, sep=None, engine="python", encoding="latin1")

    cols = list(df.columns)

    candidate_series = []
    for c in cols:
        s = normalize_numeric_series(df[c])
        non_na = s.notna().sum()
        if non_na >= max(4, len(df) // 3):
            if str(c).strip().lower() not in TIME_COLS:
                candidate_series.append(c)

    if not candidate_series:
        candidate_series = [c for c in cols if str(c).strip().lower() not in TIME_COLS]

    chosen_series = ask_choice("Columnas/variables detectadas para analizar:", candidate_series)

    time_options = []
    has_year = any(str(c).strip().lower() in {"año", "anio", "year"} for c in cols)
    has_quarter = any(str(c).strip().lower() in {"trimestre", "quarter"} for c in cols)
    has_t = any(str(c).strip().lower() == "t" for c in cols)

    if has_year and has_quarter:
        time_options.append("Año + Trimestre")
    if has_t:
        time_options.append("t")
    if has_year:
        time_options.append("Año")
        
    extra_time_candidates = [c for c in cols if str(c).strip().lower() in TIME_COLS and c != chosen_series]
    for c in extra_time_candidates:
        if c not in time_options:
            time_options.append(c)

    if not time_options:
        time_options = ["Índice de fila"]

    chosen_time = ask_choice("Opciones de tiempo disponibles:", time_options)

    y = normalize_numeric_series(df[chosen_series]).to_numpy(dtype=float)

    year_cols = [c for c in cols if str(c).strip().lower() in {"año", "anio", "year"}]
    quarter_cols = [c for c in cols if str(c).strip().lower() in {"trimestre", "quarter"}]

    if chosen_time == "Año + Trimestre" or (chosen_time.lower() in {"trimestre", "quarter"} and year_cols and quarter_cols):
        years = pd.to_numeric(df[year_cols[0]], errors="coerce")
        quarters = df[quarter_cols[0]].astype(str)
        time_labels = build_quarter_labels(years, quarters)
        time_mode = "quarter"
    elif chosen_time == "t":
        tcol = [c for c in cols if str(c).strip().lower() == "t"][0]
        tvals = pd.to_numeric(df[tcol], errors="coerce")
        time_labels = tvals.tolist()
        time_mode = "numeric"
    elif chosen_time.lower() in {"año", "anio", "year"} and year_cols:
        tvals = pd.to_numeric(df[year_cols[0]], errors="coerce")
        time_labels = tvals.tolist()
        time_mode = "year"
    elif chosen_time.lower() in {"trimestre", "quarter"} and quarter_cols:
        time_labels = df[quarter_cols[0]].astype(str).tolist()
        time_mode = "index"
    else:
        time_labels = list(range(1, len(df) + 1))
        time_mode = "index"

    mask = ~np.isnan(y)
    y = y[mask]
    time_labels = [time_labels[i] for i in range(len(time_labels)) if mask[i]]

    if time_mode == "quarter":
        years = years[mask].tolist()
        quarters = quarters[mask].tolist()
        time_labels = build_quarter_labels(years, quarters)

    return {
        "format": "horizontal",
        "file": path,
        "variable_name": chosen_series,
        "series": y,
        "time": time_labels,
        "time_label": chosen_time,
        "time_mode": time_mode,
        "raw_df": df,
    }

def load_data_interactive(args):
    file_path = args.file
    if not file_path:
        file_path = choose_file_interactive()

    if not Path(file_path).exists():
        raise FileNotFoundError(f"No existe el archivo: {file_path}")

    fmt = detect_format(file_path)
    print(f"\nFormato detectado: {fmt.upper()}")

    if fmt == "vertical":
        data = load_vertical(file_path)
    else:
        data = load_horizontal(file_path)

    return data

def make_future_labels(time_values, time_mode, steps):
    if time_mode == "year":
        last = int(time_values[-1])
        return list(range(last + 1, last + 1 + steps))
    if time_mode == "numeric":
        last = float(time_values[-1])
        if last.is_integer():
            last = int(last)
            return list(range(last + 1, last + 1 + steps))
        return [last + i + 1 for i in range(steps)]
    if time_mode == "quarter":
        m = re.match(r"(\d{4})\s*Q([1-4])", str(time_values[-1]))
        if m:
            year, q = int(m.group(1)), int(m.group(2))
            nxt = next_quarters(year, q, steps)
            return [f"{y} Q{q}" for y, q in nxt]
    start = len(time_values)
    return list(range(start + 1, start + 1 + steps))

def prepare_numeric_axis(time_values, time_mode):
    if time_mode in {"year", "numeric"}:
        return np.array(pd.to_numeric(pd.Series(time_values), errors="coerce"), dtype=float)
    return np.arange(len(time_values), dtype=float)
