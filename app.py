import os
from flask import Flask, send_from_directory, request, jsonify
import pandas as pd
import plotly.express as px
import numpy as np

TAXO_FILE = "taxonomy&morphology.csv"
OTOLITH_FILE = "otolith.csv"
EDNA_FILE = "edna.csv"

app = Flask(__name__, static_folder=".", static_url_path="")

def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        df.columns = [str(c).strip() for c in df.columns]
        return df
    except Exception:
        return pd.DataFrame()

def dedup(df, keys):
    for k in keys:
        if k in df.columns:
            return df.drop_duplicates(subset=[k]).copy()
    return df.copy()

def to_json_safe(data):
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, pd.Series):
        return data.tolist()
    elif isinstance(data, (np.integer, np.floating)):
        return data.item()
    elif isinstance(data, list):
        return [to_json_safe(item) for item in data]
    elif isinstance(data, dict):
        return {k: to_json_safe(v) for k, v in data.items()}
    else:
        return data

def style_treemap(fig):
    fig.update_traces(
        marker=dict(line=dict(color="white", width=2), cornerradius=6),
        tiling=dict(pad=4),
        textinfo="label+percent parent",
        textfont=dict(size=14, color="white")
    )
    fig.update_layout(
        margin=dict(t=70, l=16, r=16, b=16),
        paper_bgcolor="#F5F7FB",
        plot_bgcolor="#F5F7FB",
        font=dict(size=14, color="#111827"),
        title_font=dict(size=22, family="Segoe UI, Arial, sans-serif"),
    )
    return fig

def make_palette(values):
    base = ["#2563EB","#10B981","#F59E0B","#EF4444","#8B5CF6","#06B6D4",
            "#F97316","#22C55E","#E11D48","#A855F7","#0EA5E9","#84CC16",
            "#F43F5E","#14B8A6","#7C3AED","#0284C7","#059669","#B91C1C"]
    uniq = sorted([v for v in pd.Series(values).dropna().unique().tolist() if str(v).strip()])
    return {o: base[i % len(base)] for i,o in enumerate(uniq)}

taxo_df = load_csv(TAXO_FILE)
otolith_df = load_csv(OTOLITH_FILE)
edna_df = load_csv(EDNA_FILE)

# eDNA normalization
if not edna_df.empty:
    if "species" not in edna_df.columns:
        edna_df["species"] = edna_df["Scientific_Name"].astype(str) if "Scientific_Name" in edna_df.columns else ""
    edna_df["reads"] = pd.to_numeric(edna_df["reads"], errors="coerce").fillna(1) if "reads" in edna_df.columns else 1
    if "Latitude" in edna_df.columns and "latitude" not in edna_df.columns:
        edna_df.rename(columns={"Latitude":"latitude"}, inplace=True)
    if "Longitude" in edna_df.columns and "longitude" not in edna_df.columns:
        edna_df.rename(columns={"Longitude":"longitude"}, inplace=True)

# --------- Pages ---------
@app.route("/")
def landing():
    return send_from_directory(".", "landing.html")

@app.route("/taxonomy")
def page_taxonomy():
    return send_from_directory(".", "taxonomy.html")

@app.route("/edna")
def page_edna():
    return send_from_directory(".", "edna.html")

@app.route("/otolith")
def page_otolith():
    return send_from_directory(".", "otolith.html")

@app.route("/acoustic")
def page_acoustic():
    return send_from_directory(".", "acoustic.html")

# --------- APIs (unchanged logic) ---------
@app.get("/treemap")
def treemap_api():
    ds = request.args.get("ds","Taxonomy & Morphology")
    try:
        def ensure_cols(df, cols):
            d = df.copy()
            for c in cols:
                if c not in d.columns:
                    d[c] = "N/A"
                d[c] = d[c].astype(str).fillna("N/A").replace({"": "N/A"})
            return d

        if ds=="Taxonomy & Morphology":
            if taxo_df.empty: return jsonify({"error": f"Missing {TAXO_FILE}"})
            dfc = dedup(taxo_df, ["Scientific_Name"])
            leaf = "Common_Name" if "Common_Name" in dfc.columns else "Scientific_Name"
            dfc = ensure_cols(dfc, ["Phylum","Order","Family",leaf])
            cmap = make_palette(dfc["Order"])
            values = [1] * len(dfc)
            fig = px.treemap(dfc, path=["Phylum","Order","Family",leaf], values=values,
                             title="Marine Species Taxonomy", color="Order", color_discrete_map=cmap)
            return jsonify(to_json_safe(style_treemap(fig).to_dict()))

        elif ds=="Otolith":
            if otolith_df.empty: return jsonify({"error": f"Missing {OTOLITH_FILE}"})
            dfc = dedup(otolith_df, ["Scientific_Name"])
            leaf = "Common_Name" if "Common_Name" in dfc.columns else "Scientific_Name"
            dfc = ensure_cols(dfc, ["Phylum","Order","Family",leaf])
            cmap = make_palette(dfc["Order"])
            values = [1] * len(dfc)
            fig = px.treemap(dfc, path=["Phylum","Order","Family",leaf], values=values,
                             title="Otolith Inventory: Taxonomy", color="Order", color_discrete_map=cmap)
            return jsonify(to_json_safe(style_treemap(fig).to_dict()))

        else:  # eDNA
            if edna_df.empty: return jsonify({"error": f"Missing {EDNA_FILE}"})
            dfc = dedup(edna_df, ["species"])
            dfc = ensure_cols(dfc, ["Phylum","Order","Family","species"])
            reads = pd.to_numeric(dfc.get("reads", 1), errors="coerce").fillna(1).tolist()
            cmap = make_palette(dfc["Order"])
            fig = px.treemap(dfc, path=["Phylum","Order","Family","species"], values=reads,
                             title="Indian EEZ eDNA - Taxonomy", color="Order", color_discrete_map=cmap)
            return jsonify(to_json_safe(style_treemap(fig).to_dict()))
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"})

def pick(d: dict, keys: list[str]):
    out={}
    for k in keys:
        if k in d and str(d[k]).strip():
            out[k]=d[k]
    return out

@app.get("/search")
def search_api():
    ds = request.args.get("ds","Taxonomy & Morphology")
    q = (request.args.get("q") or "").strip().lower()
    if not q: return jsonify({"html":"<div class='warn'>Enter a species name.</div>"})
    try:
        if ds=="Taxonomy & Morphology":
            if taxo_df.empty: return jsonify({"html": f"<div class='warn'>Missing {TAXO_FILE}</div>"})
            dfc = dedup(taxo_df, ["Scientific_Name"])
            idx={}
            for i,v in dfc["Scientific_Name"].fillna("").astype(str).items(): idx[v.strip().lower()]=i
            if "Common_Name" in dfc.columns:
                for i,v in dfc["Common_Name"].fillna("").astype(str).items():
                    k=v.strip().lower()
                    if k and k not in idx: idx[k]=i
            rec=None
            if q in idx: rec = dfc.iloc[idx[q]]
            else:
                for k in idx:
                    if k.startswith(q): rec = dfc.iloc[idx[k]]; break
                if rec is None:
                    for k in idx:
                        if q in k: rec = dfc.iloc[idx[k]]; break
            if rec is None: return jsonify({"html":"<div class='warn'>No match found.</div>"})
            rd = {k:("" if pd.isna(v) else v) for k,v in rec.to_dict().items()}
            quick = pick(rd, ["Scientific_Name","Common_Name","Phylum","Class","Order","Family","Location","Depth_Range","Habitat_Type","Conservation_Status"])
            return jsonify({"html": render_card(quick, rd)})

        elif ds=="Otolith":
            if otolith_df.empty: return jsonify({"html": f"<div class='warn'>Missing {OTOLITH_FILE}</div>"})
            dfc = dedup(otolith_df, ["Scientific_Name"])
            idx={}
            for i,v in dfc["Scientific_Name"].fillna("").astype(str).items(): idx[v.strip().lower()]=i
            if "Common_Name" in dfc.columns:
                for i,v in dfc["Common_Name"].fillna("").astype(str).items():
                    k=v.strip().lower()
                    if k and k not in idx: idx[k]=i
            rec=None
            if q in idx: rec = dfc.iloc[idx[q]]
            else:
                for k in idx:
                    if k.startswith(q): rec = dfc.iloc[idx[k]]; break
                if rec is None:
                    for k in idx:
                        if q in k: rec = dfc.iloc[idx[k]]; break
            if rec is None: return jsonify({"html":"<div class='warn'>No match found.</div>"})
            rd = {k:("" if pd.isna(v) else v) for k,v in rec.to_dict().items()}
            quick = pick(rd, ["Scientific_Name","Common_Name","Phylum","Class","Order","Family","Location","Distribution","Depth_Range","Habitat_Type","Max_Length_cm","IUCN_Status"])
            return jsonify({"html": render_card(quick, rd)})

        else:  # eDNA
            if edna_df.empty: return jsonify({"html": f"<div class='warn'>Missing {EDNA_FILE}</div>"})
            dfc = dedup(edna_df, ["species"])
            idx={}
            for i,v in dfc["species"].fillna("").astype(str).items(): idx[v.strip().lower()]=i
            if "Scientific_Name" in dfc.columns:
                for i,v in dfc["Scientific_Name"].fillna("").astype(str).items():
                    k=v.strip().lower()
                    if k and k not in idx: idx[k]=i
            rec=None
            if q in idx: rec = dfc.iloc[idx[q]]
            else:
                for k in idx:
                    if k.startswith(q): rec = dfc.iloc[idx[k]]; break
                if rec is None:
                    for k in idx:
                        if q in k: rec = dfc.iloc[idx[k]]; break
            if rec is None: return jsonify({"html":"<div class='warn'>No match found.</div>"})
            rd = {k:("" if pd.isna(v) else v) for k,v in rec.to_dict().items()}
            quick = pick(rd, ["species","Scientific_Name","Phylum","Class","Order","Family","marker","asv_id","reads","sample_id","event_date","basin","latitude","longitude","depth_m","source_portal","project_accession","run_accession","dataset_doi"])
            det_html=""
            sp = rd.get("species","")
            if sp:
                det = edna_df[edna_df["species"].astype(str)==str(sp)].copy()
                cols = ["species","Phylum","Class","Order","Family","marker","asv_id","reads","sample_id","event_date","latitude","longitude","depth_m","basin","source_portal","project_accession","run_accession","dataset_doi"]
                cols = [c for c in cols if c in det.columns]
                if not det.empty:
                    det_html = "<h4>eDNA detection records (up to 25)</h4><div class='table-wrap'><table><thead><tr>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr></thead><tbody>"
                    for _,row in det.head(25).iterrows():
                        det_html += "<tr>" + "".join(f"<td>{row.get(c,'')}</td>" for c in cols) + "</tr>"
                    det_html += "</tbody></table></div>"
            return jsonify({"html": render_card(quick, rd) + det_html})
    except Exception as e:
        return jsonify({"html": f"<div class='warn'>Error: {type(e).__name__}: {e}</div>"})

def render_card(quick: dict, full: dict) -> str:
    name = quick.get("Common_Name","") or quick.get("species","") or quick.get("Scientific_Name","")
    sci  = quick.get("Scientific_Name","") or quick.get("species","")
    header = f"<h3>{name} <span class='sci'>({sci})</span></h3>"
    keys = [k for k in quick.keys() if k not in ("Common_Name","species","Scientific_Name")]
    left = keys[::2]; right = keys[1::2]
    def listify(items):
        return "".join([f"<li><span>{k.replace('_',' ')}:</span> {quick[k]}</li>" for k in items if k in quick])
    quick_html = f"""
      <div class='facts'>
        <ul>{listify(left)}</ul>
        <ul>{listify(right)}</ul>
      </div>"""
    full_html = "<details><summary>All fields</summary><ul class='all'>"
    for k in sorted(full.keys()):
        full_html += f"<li><span>{k}:</span> {full[k]}</li>"
    full_html += "</ul></details>"
    return f"<div class='card'>{header}{quick_html}{full_html}</div>"

# ========= Map helpers (India EEZ) =========
def _india_clip(lat_series: pd.Series, lon_series: pd.Series) -> pd.Series:
    lat = pd.to_numeric(lat_series, errors="coerce")
    lon = pd.to_numeric(lon_series, errors="coerce")
    main = lat.between(4.5, 24.5) & lon.between(65, 89.5)
    andam = lat.between(3.0, 14.5) & lon.between(90, 94.8)
    return (main | andam)

def _find_coord_cols(df: pd.DataFrame) -> tuple[str|None, str|None]:
    lat_names = ["Lat","LAT","lat","Latitude","latitude","lat_dd","latitude_deg","23"]
    lon_names = ["Lon","LON","lon","Longitude","longitude","LONG","lng","lon_dd","24"]
    lat_col = next((c for c in lat_names if c in df.columns), None)
    lon_col = next((c for c in lon_names if c in df.columns), None)
    return lat_col, lon_col

def _normalize_coords_df(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    lat_col, lon_col = _find_coord_cols(d)
    if not lat_col or not lon_col:
        return pd.DataFrame(columns=["latitude","longitude"])
    d["latitude"]  = pd.to_numeric(d[lat_col], errors="coerce")
    d["longitude"] = pd.to_numeric(d[lon_col], errors="coerce")
    d = d.dropna(subset=["latitude","longitude"])
    # Swap if needed (improves India coverage)
    base = _india_clip(d["latitude"], d["longitude"]).mean()
    swapped = _india_clip(d["longitude"], d["latitude"]).mean()
    if swapped > base:
        d[["latitude","longitude"]] = d[["longitude","latitude"]]
    # Clip to India EEZ envelopes
    d = d[_india_clip(d["latitude"], d["longitude"])].copy()
    return d

def _build_plotly_map(d: pd.DataFrame, color_col: str, size_col: str|None, title: str) -> dict:
    # Prepare minimal columns
    cols = ["latitude","longitude", color_col]
    if size_col and size_col in d.columns:
        cols.append(size_col)
    dd = d[[c for c in cols if c in d.columns]].copy()
    # Use species/scientific name string for color grouping
    if color_col in dd.columns:
        dd[color_col] = dd[color_col].astype(str)
    import plotly.express as _px
    fig = _px.scatter_mapbox(
        dd,
        lat="latitude",
        lon="longitude",
        color=color_col if color_col in dd.columns else None,
        size=size_col if size_col and size_col in dd.columns else None,
        hover_data=dd.columns,
        zoom=4,
        height=600,
        title=title
    )
    fig.update_layout(mapbox_style="carto-positron", margin=dict(r=0,t=40,l=0,b=0))
    return fig.to_dict()

@app.get("/mapplot")
def mapplot_api():
    # ds in {"eDNA","Otolith","Taxonomy & Morphology"}
    ds = request.args.get("ds","eDNA")
    try:
        if ds == "eDNA":
            if edna_df.empty: return jsonify({"error": f"Missing {EDNA_FILE}"})
            d = _normalize_coords_df(edna_df)
            if d.empty: return jsonify({"error":"No valid India EEZ coordinates in eDNA"})
            # prefer species + reads
            color_col = "species" if "species" in d.columns else ("Scientific_Name" if "Scientific_Name" in d.columns else None)
            size_col  = "reads" if "reads" in d.columns else None
            fig_dict = _build_plotly_map(d, color_col or "species", size_col, "eDNA detections — Indian EEZ")
            return jsonify(to_json_safe(fig_dict))

        elif ds == "Otolith":
            if otolith_df.empty: return jsonify({"error": f"Missing {OTOLITH_FILE}"})
            d = _normalize_coords_df(otolith_df)
            if d.empty: return jsonify({"error":"No valid India EEZ coordinates in Otolith"})
            color_col = "Scientific_Name" if "Scientific_Name" in d.columns else None
            fig_dict = _build_plotly_map(d, color_col or "Record", None, "Otolith records — Indian EEZ")
            return jsonify(to_json_safe(fig_dict))

        else: # Taxonomy & Morphology
            if taxo_df.empty: return jsonify({"error": f"Missing {TAXO_FILE}"})
            d = _normalize_coords_df(taxo_df)
            if d.empty: return jsonify({"error":"No valid India EEZ coordinates in Taxonomy dataset"})
            color_col = "Scientific_Name" if "Scientific_Name" in d.columns else None
            fig_dict = _build_plotly_map(d, color_col or "Record", None, "Taxonomy records — Indian EEZ")
            return jsonify(to_json_safe(fig_dict))
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"})
    
# Serve the prebuilt acoustic map HTML if available
@app.get("/acoustic/map")
def acoustic_map_file():
    fname = "attractive_ocean_map.html"
    if os.path.exists(fname):
        # return path so client can iframe-load it
        return jsonify({"file": fname})
    # Backward compatibility: also accept older filename if present
    legacy = "underwater_conditions_map.html"
    if os.path.exists(legacy):
        return jsonify({"file": legacy})
    return jsonify({"error": "No acoustic HTML found. Place attractive_ocean_map.html in the app folder."})

@app.route("/fishery")
def page_fishery():
    return send_from_directory(".", "fishery.html")

@app.route("/chatbot")
def page_chatbot():
    return send_from_directory(".", "chatbot.html")

if __name__ == "__main__":
    print("Open http://127.0.0.1:5000")
    app.run("127.0.0.1", 5000, debug=False)
