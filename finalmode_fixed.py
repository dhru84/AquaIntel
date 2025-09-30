# -*- coding: utf-8 -*-

"""
finalmodel_fixed.py (3-dataset grouping)
- Datasets: Acoustic, Physical (SST+Wind), Chemical
- Safe CSV loaders (Colab or local)
- Rule-based processors for Acoustic, SST, Wind, Chemical
- Physical = combined outputs of SST and Wind
- Acoustic summary and optional folium map when coordinates exist
- Saves: acoustic_processed.csv and underwater_conditions_map.html (if possible)
"""

import os
import sys
import warnings
from typing import Optional, Tuple, List

import pandas as pd

# Optional deps for mapping; guarded import
try:
    import folium
    from folium.plugins import MarkerCluster
    HAVE_FOLIUM = True
except Exception:
    HAVE_FOLIUM = False

warnings.filterwarnings("ignore")

# ----------------------
# Runtime configuration
# ----------------------
RUN_MODE = os.environ.get("RUN_MODE", "local").strip().lower()  # 'colab' or 'local'

if RUN_MODE == "colab":
    try:
        from google.colab import files as colab_files
        IN_COLAB = True
    except Exception:
        IN_COLAB = False
        RUN_MODE = "local"
else:
    IN_COLAB = False

# ----------------------
# Utilities
# ----------------------
def log(msg: str) -> None:
    print(msg)

def ensure_dataframe(obj: Optional[pd.DataFrame]) -> pd.DataFrame:
    if obj is None:
        return pd.DataFrame()
    return obj

def safe_read_csv(path: str) -> pd.DataFrame:
    """
    Robust CSV/TXT reader with fallbacks for tabs or whitespace and without headers.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    # Try standard CSV
    try:
        return pd.read_csv(path)
    except Exception:
        pass
    # Try tab-separated
    try:
        return pd.read_csv(path, sep="\t", engine="python")
    except Exception:
        pass
    # Try whitespace-delimited, no header
    try:
        return pd.read_csv(path, sep=r"\s+", engine="python", header=None)
    except Exception as e:
        raise e

def colab_upload_one(label: str) -> Optional[str]:
    """
    Ask user to upload one file in Colab and return its local filename.
    """
    if not IN_COLAB:
        return None
    log(f"\n📂 Upload {label} dataset (CSV/TXT):")
    uploaded = colab_files.upload()
    if not uploaded:
        return None
    fname = list(uploaded.keys())[0]
    if os.stat(fname).st_size == 0:
        log(f"{fname} is empty. Skipping.")
        return None
    return fname

def load_dataset(label: str, given_path: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Load a dataset by label. In colab, if given_path is None, prompt upload.
    Locally, require given_path or default to '{label}.csv' if present.
    """
    fname = None
    if RUN_MODE == "colab" and given_path is None:
        fname = colab_upload_one(label)
    else:
        # local or explicit path
        fname = given_path

    if fname is None:
        # try default local fallback
        candidates = [f"{label}.csv", f"{label}.txt", f"{label.lower()}.csv", f"{label.lower()}.txt"]
        fname = next((c for c in candidates if os.path.exists(c)), None)

    if fname is None:
        log(f"⚠️ {label}: no file provided/found. Skipping.")
        return None

    try:
        df = safe_read_csv(fname)
        log(f"✅ {label} loaded: shape {df.shape}")
        return df
    except Exception as e:
        log(f"❌ {label} load failed: {e}")
        return None

def normalize_coords(df: pd.DataFrame, lat_candidates: List[str], lon_candidates: List[str]) -> Tuple[pd.DataFrame, bool]:
    """
    Attempt to find latitude/longitude columns by common names or indices-as-strings.
    Returns (df with Lat/Lon columns if possible, has_coords)
    """
    df = df.copy()
    lat_col = next((c for c in lat_candidates if c in df.columns), None)
    lon_col = next((c for c in lon_candidates if c in df.columns), None)

    # Also allow numeric-like column names as strings
    if lat_col is None:
        lat_col = next((c for c in df.columns if str(c).strip().lower() in ["lat", "latitude"]), None)
    if lon_col is None:
        lon_col = next((c for c in df.columns if str(c).strip().lower() in ["lon", "longitude", "long"]), None)

    # Fallback to the legacy '23','24'
    if lat_col is None and "23" in df.columns:
        lat_col = "23"
    if lon_col is None and "24" in df.columns:
        lon_col = "24"

    if lat_col is None or lon_col is None:
        return df, False

    df["Lat"] = pd.to_numeric(df[lat_col], errors="coerce")
    df["Lon"] = pd.to_numeric(df[lon_col], errors="coerce")
    df = df.dropna(subset=["Lat", "Lon"])
    return df, True

# ----------------------
# Rule-based processors
# ----------------------
def process_acoustic_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rule-based classification based on a numeric acoustic metric if present,
    else use first numeric column as proxy.
    """
    df = df.copy()
    if df.empty:
        return df

    # Try common acoustic metric column names
    candidate_cols = ["acoustic_level", "noise", "noise_db", "noiseLevel", "NoiseLevel"]
    metric_col = next((c for c in candidate_cols if c in df.columns), None)

    # Fallback: first numeric column
    if metric_col is None:
        for c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().any():
                metric_col = c
                df[c] = s
                break

    results = []
    if metric_col is not None:
        s = pd.to_numeric(df[metric_col], errors="coerce")
        for v in s.fillna(float("nan")):
            try:
                if pd.isna(v):
                    results.append("Invalid Value")
                elif v > 80:
                    results.append("High Noise - Fish Disturbance")
                elif v > 50:
                    results.append("Moderate Noise - Caution")
                else:
                    results.append("Safe Acoustic Condition")
            except Exception:
                results.append("Invalid Value")
        df["Acoustic_Result"] = results
        df["Acoustic_Metric"] = s
    else:
        df["Acoustic_Result"] = "Unknown"
        df["Acoustic_Metric"] = None

    return df

def process_sst_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Find an SST/temperature column and flag warm/cold/normal.
    """
    df = df.copy()
    if df.empty:
        return df

    temp_col = next((c for c in ["temperature", "temp", "sst", "SST", "Temp"] if c in df.columns), None)
    if temp_col is None:
        # try any numeric column as proxy
        numeric_cols = [c for c in df.columns if pd.to_numeric(df[c], errors="coerce").notna().any()]
        temp_col = numeric_cols[0] if numeric_cols else None

    if temp_col is None:
        df["SST_Prediction"] = "Unknown"
        return df

    s = pd.to_numeric(df[temp_col], errors="coerce")
    preds = []
    for v in s:
        if pd.isna(v):
            preds.append("Unknown")
        elif v > 30:
            preds.append("Warm water – suitable for tuna")
        elif v < 5:
            preds.append("Cold water – low fish activity")
        else:
            preds.append("Normal")
    df["SST_Prediction"] = preds
    df["SST_Value"] = s
    return df

def process_wind_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Find wind_speed column and flag safety.
    """
    df = df.copy()
    if df.empty:
        return df

    wind_col = next((c for c in ["wind_speed", "windspeed", "wind", "Wind_Speed"] if c in df.columns), None)
    if wind_col is None:
        # try any numeric column as proxy
        numeric_cols = [c for c in df.columns if pd.to_numeric(df[c], errors="coerce").notna().any()]
        wind_col = numeric_cols[0] if numeric_cols else None

    if wind_col is None:
        df["Wind_Prediction"] = "Unknown"
        return df

    s = pd.to_numeric(df[wind_col], errors="coerce")
    preds = []
    for v in s:
        if pd.isna(v):
            preds.append("Unknown")
        elif v > 25:
            preds.append("High wind – unsafe to go fishing")
        else:
            preds.append("Normal")
    df["Wind_Prediction"] = preds
    df["Wind_Value"] = s
    return df

def process_chemical_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Look for pH and CO2, then classify.
    """
    df = df.copy()
    if df.empty:
        return df

    pH_col = next((c for c in ["pH", "ph", "PH"] if c in df.columns), None)
    co2_col = next((c for c in ["CO2", "co2", "Co2"] if c in df.columns), None)

    # convert where present
    pH = pd.to_numeric(df[pH_col], errors="coerce") if pH_col else pd.Series([None] * len(df))
    co2 = pd.to_numeric(df[co2_col], errors="coerce") if co2_col else pd.Series([None] * len(df))

    preds = []
    for pv, cv in zip(pH, co2):
        tags = []
        if pv is not None and pd.notna(pv) and pv < 7.5:
            tags.append("Low pH – water quality concern")
        if cv is not None and pd.notna(cv) and cv > 400:
            tags.append("High CO2 – low oxygen risk")
        preds.append(tags if tags else ["Normal"])

    df["Chemical_Prediction"] = preds
    if pH_col:
        df["pH_value"] = pH
    if co2_col:
        df["CO2_value"] = co2
    return df

# ----------------------
# Summary report helpers (Acoustic)
# ----------------------
def summarize_acoustic(df: pd.DataFrame) -> dict:
    """
    Heuristic summary for acoustic signals and conditions.
    """
    if df.empty:
        return {"AcousticSignal": "No data", "SpeciesDetected": "No data",
                "SeasonRisk": "No data", "LocationRisk": "No data",
                "NoiseCondition": "No data", "OverallCondition": "No data"}

    total = len(df)
    genus_col = next((c for c in df.columns if str(c).strip().lower() in ["genus", "9", "Genus"]), None)
    family_col = next((c for c in df.columns if str(c).strip().lower() in ["family", "7", "Family"]), None)
    order_col = next((c for c in df.columns if str(c).strip().lower() in ["order", "6", "Order"]), None)

    spotted_dolphin_count = 0
    dolphin_signal_count = 0
    monsoon_count = 0
    arabian_sea_count = 0
    high_noise_count = 0

    date_col = next((c for c in df.columns if "date" in str(c).lower()), None)
    noise_col = next((c for c in ["Acoustic_Metric", "noise", "noise_db", "49"] if c in df.columns), None)

    _, has_coords = normalize_coords(df, ["Lat"], ["Lon"])

    for _, row in df.iterrows():
        if order_col and str(row.get(order_col, "")).strip() == "Cetacea":
            if family_col and str(row.get(family_col, "")).strip() == "Delphinidae":
                dolphin_signal_count += 1
        if genus_col and str(row.get(genus_col, "")).strip().lower() == "stenella":
            spotted_dolphin_count += 1
        if date_col:
            try:
                text = str(row.get(date_col, "")).strip()
                parts = [p for p in text.replace("/", "-").split("-") if p]
                mon = int(parts[1]) if len(parts) > 1 else None
                if mon and 6 <= mon <= 9:
                    monsoon_count += 1
            except Exception:
                pass
        try:
            lat_val = float(row.get("Lat")) if "Lat" in df.columns else None
            lon_val = float(row.get("Lon")) if "Lon" in df.columns else None
            if lat_val is not None and lon_val is not None:
                if 5 <= lat_val <= 25 and 55 <= lon_val <= 75:
                    arabian_sea_count += 1
        except Exception:
            pass
        try:
            if noise_col:
                nv = float(row.get(noise_col))
                if nv > 80:
                    high_noise_count += 1
        except Exception:
            pass

    def frac(x): return (x / total) if total else 0

    summary = {}
    if frac(dolphin_signal_count) > 0.5:
        summary["AcousticSignal"] = "High dolphin acoustic activity"
    elif dolphin_signal_count > 0:
        summary["AcousticSignal"] = "Moderate dolphin acoustic activity"
    else:
        summary["AcousticSignal"] = "Low dolphin acoustic activity"

    summary["SpeciesDetected"] = "Spotted Dolphin observed" if spotted_dolphin_count > 0 else "No Spotted Dolphin detected"
    summary["SeasonRisk"] = "Monsoon: High probability of detection" if frac(monsoon_count) > 0.5 else "Normal season detection probability"

    if has_coords and frac(arabian_sea_count) > 0.5:
        summary["LocationRisk"] = "Majority observations in Arabian Sea"
    elif has_coords:
        summary["LocationRisk"] = "Few observations in Arabian Sea"
    else:
        summary["LocationRisk"] = "No coordinates available"

    if frac(high_noise_count) > 0.3:
        summary["NoiseCondition"] = "High underwater noise – fish may be stressed"
    elif high_noise_count > 0:
        summary["NoiseCondition"] = "Moderate underwater noise"
    else:
        summary["NoiseCondition"] = "Low underwater noise"

    summary["OverallCondition"] = (
        "Safe fishing conditions"
        if summary["NoiseCondition"] == "Low underwater noise" and dolphin_signal_count > 0
        else "Caution advised – high noise or stress indicators"
    )
    return summary

# ----------------------
# Mapping (Acoustic)
# ----------------------
def make_map_from_acoustic(acoustic_df: pd.DataFrame, out_html: str = "underwater_conditions_map.html") -> Optional[str]:
    if not HAVE_FOLIUM:
        log("⚠️ Folium not installed, skipping map.")
        return None
    df, has_coords = normalize_coords(acoustic_df, ["Lat", "23"], ["Lon", "24"])
    if not has_coords or df.empty:
        log("⚠️ No coordinates available for mapping.")
        return None

    if "NoiseLevel" not in df.columns:
        if "Acoustic_Metric" in df.columns:
            df["NoiseLevel"] = pd.to_numeric(df["Acoustic_Metric"], errors="coerce")
        elif "49" in df.columns:
            df["NoiseLevel"] = pd.to_numeric(df["49"], errors="coerce")
        else:
            df["NoiseLevel"] = pd.NA

    temp_col = next((c for c in ["SST", "Temp", "SST_Value"] if c in df.columns), None)

    def get_color(row):
        t = row.get(temp_col, 0) if temp_col else 0
        n = row.get("NoiseLevel", 0)
        try:
            if pd.notna(t) and float(t) > 28:
                return "red"
            if pd.notna(n) and float(n) > 80:
                return "red"
            if (pd.notna(t) and float(t) > 25) or (pd.notna(n) and float(n) > 60):
                return "orange"
        except Exception:
            pass
        return "green"

    center = [df["Lat"].mean(), df["Lon"].mean()]
    m = folium.Map(location=center, zoom_start=5)
    cluster = MarkerCluster().add_to(m)

    for _, row in df.iterrows():
        lat, lon = row["Lat"], row["Lon"]
        if pd.isna(lat) or pd.isna(lon):
            continue
        noise_val = row.get("NoiseLevel", "N/A")
        try:
            popup_text = f"Location: {lat:.2f}, {lon:.2f}\nNoise Level: {noise_val}\n"
            if temp_col:
                popup_text += f"SST: {row.get(temp_col, 'N/A')}\n"
        except Exception:
            popup_text = "Record"
        folium.CircleMarker(
            location=[lat, lon],
            radius=6,
            color=get_color(row),
            fill=True,
            fill_color=get_color(row),
            fill_opacity=0.7,
            popup=popup_text
        ).add_to(cluster)

    m.save(out_html)
    log(f"✅ Map saved: {out_html}")
    return out_html

# ----------------------
# Main orchestration
# ----------------------
def main(
    acoustic_path: Optional[str] = None,
    physical_sst_path: Optional[str] = None,
    physical_wind_path: Optional[str] = None,
    chemical_path: Optional[str] = None,
    build_map: bool = True
):
    # Load
    acoustic = load_dataset("Acoustic", acoustic_path)
    sst = load_dataset("SST", physical_sst_path)
    wind = load_dataset("Wind", physical_wind_path)
    chemical = load_dataset("Chemical", chemical_path)

    # Process
    acoustic = ensure_dataframe(acoustic)
    sst = ensure_dataframe(sst)
    wind = ensure_dataframe(wind)
    chemical = ensure_dataframe(chemical)

    acoustic_proc = process_acoustic_table(acoustic)
    sst_proc = process_sst_table(sst)
    wind_proc = process_wind_table(wind)
    chemical_proc = process_chemical_table(chemical)

    # Display grouped outputs
    if not acoustic_proc.empty:
        log("\n=== Acoustic predictions (sample) ===")
        log(acoustic_proc[["Acoustic_Result"]].head().to_string(index=False))

    if not sst_proc.empty or not wind_proc.empty:
        log("\n=== Physical predictions (SST + Wind, sample) ===")
        if not sst_proc.empty:
            log("- SST:")
            log(sst_proc[["SST_Prediction"]].head().to_string(index=False))
        if not wind_proc.empty:
            log("- Wind:")
            log(wind_proc[["Wind_Prediction"]].head().to_string(index=False))

    if not chemical_proc.empty:
        log("\n=== Chemical predictions (sample) ===")
        # Truncate nested list for readability
        chem_show = chemical_proc.copy()
        if "Chemical_Prediction" in chem_show.columns:
            chem_show["Chemical_Prediction"] = chem_show["Chemical_Prediction"].astype(str)
        log(chem_show[["Chemical_Prediction"]].head().to_string(index=False))

    # Save acoustic processed
    if not acoustic_proc.empty:
        out_acoustic = "acoustic_processed.csv"
        acoustic_proc.to_csv(out_acoustic, index=False)
        log(f"\n🎯 Acoustic processing complete. Saved: {out_acoustic}")

    # Acoustic summary
    if not acoustic_proc.empty:
        summary = summarize_acoustic(acoustic_proc)
        log("\n🌊 Acoustic Summary:")
        for k, v in summary.items():
            log(f"- {k}: {v}")

    # Optional map from acoustic
    if build_map and not acoustic_proc.empty:
        make_map_from_acoustic(acoustic_proc)

    # Colab: downloads
    if IN_COLAB:
        try:
            if not acoustic_proc.empty:
                colab_files.download("acoustic_processed.csv")
            if os.path.exists("underwater_conditions_map.html"):
                colab_files.download("underwater_conditions_map.html")
        except Exception:
            pass

if __name__ == "__main__":
    """
    CLI:
    python finalmodel_fixed.py [acoustic] [physical_sst] [physical_wind] [chemical]

    Examples:
    - Acoustic only (and build map):
      python finalmodel_fixed.py acostic.csv

    - Acoustic + Physical (SST, Wind) + Chemical:
      python finalmodel_fixed.py acostic.csv sst.csv wind.csv chemical.csv
    """
    a = sys.argv[1] if len(sys.argv) > 1 else None
    sst_path = sys.argv[2] if len(sys.argv) > 2 else None
    wind_path = sys.argv[3] if len(sys.argv) > 3 else None
    chem_path = sys.argv[4] if len(sys.argv) > 4 else None
    main(acoustic_path=a, physical_sst_path=sst_path, physical_wind_path=wind_path, chemical_path=chem_path, build_map=True)
