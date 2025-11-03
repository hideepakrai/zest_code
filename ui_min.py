# ui_min.py
import json
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Any

import streamlit as st
from PIL import Image
import apply_material_to_wall as amw  # must expose run_apply_material_with_inputs()

st.set_page_config(page_title="Apply Material - Minimal UI", layout="wide")

DET_PATH = Path("detections.json")

DEFAULT_JSON: Dict[str, Any] = {
    "status": "success",
    "image": "AAA1111.jpg",
    "results": [
        {
            "label": "Wall",
            "score": 1.0,
            "box": [287.3, 65.28, 632.59, 483.45],
            "polygon": [
                [287.30, 161.32],
                [287.30, 380.95],
                [549.86, 356.94],
                [600.20, 371.96],
                [600.98, 483.45],
                [632.52, 480.34],
                [632.59, 148.90],
                [548.24, 65.28]
            ]
        }
    ]
}

def parse_box_csv(s: str) -> List[float]:
    nums = [float(x.strip()) for x in s.split(",") if x.strip()]
    if len(nums) != 4:
        raise ValueError("box needs exactly 4 numbers: x_min, y_min, x_max, y_max")
    return nums

def parse_polygon_csv(s: str) -> List[List[float]]:
    nums = [float(x.strip()) for x in s.split(",") if x.strip()]
    if len(nums) < 6 or len(nums) % 2 != 0:
        raise ValueError("polygon needs even count ≥ 6 (x1,y1,x2,y2,...)")
    return [[nums[i], nums[i+1]] for i in range(0, len(nums), 2)]

# --- load once
if "base_json" not in st.session_state:
    if DET_PATH.exists():
        try:
            st.session_state.base_json = json.loads(DET_PATH.read_text(encoding="utf-8"))
        except Exception:
            st.session_state.base_json = DEFAULT_JSON.copy()
    else:
        st.session_state.base_json = DEFAULT_JSON.copy()

base = st.session_state.base_json
results = base.setdefault("results", [])

st.markdown("## 🏠 Apply Material — Minimal UI")

# --- sidebar: images
with st.sidebar:
    st.header("Inputs")
    house_file = st.file_uploader("House image", type=["png","jpg","jpeg"])
    material_file = st.file_uploader("Material / texture", type=["png","jpg","jpeg"])
    st.caption("Depth is optional; computed automatically.")

# --- compact editor
st.markdown("### Detections editor (only 3 fields)")

# pick existing result to update
options = [f"{i}: {r.get('label','(no label)')}" for i, r in enumerate(results)]
idx = st.selectbox("Select existing result to update", options, index=0 if options else 0, disabled=not options)
sel_i = int(idx.split(":")[0]) if options else None

col1, col2 = st.columns([1,2])
with col1:
    label_in = st.text_input("label", value=(results[sel_i].get("label","") if sel_i is not None else "Wall"))
with col2:
    box_in = st.text_input(
        "box (x_min, y_min, x_max, y_max)",
        value=",".join(str(x) for x in (results[sel_i].get("box",[0,0,0,0]) if sel_i is not None else [0,0,0,0]))
    )

polygon_in = st.text_input(
    "polygon (x1, y1, x2, y2, ...)",
    value=",".join(",".join(str(v) for v in pt) for pt in (results[sel_i].get("polygon", []) if sel_i is not None else [])) \
           or "17.01,314.19,38.12,307.09,102.25,318.24"
)

c1, c2, c3 = st.columns(3)
with c1:
    if st.button("✅ Update selected"):
        if sel_i is None:
            st.error("No result to update.")
        else:
            try:
                results[sel_i]["label"] = label_in.strip()
                results[sel_i]["box"] = parse_box_csv(box_in)
                results[sel_i]["polygon"] = parse_polygon_csv(polygon_in)
                st.success(f"Updated result #{sel_i}.")
            except Exception as e:
                st.error(str(e))

with c2:
    if st.button("➕ Append new"):
        try:
            results.append({
                "label": label_in.strip(),
                "box": parse_box_csv(box_in),
                "polygon": parse_polygon_csv(polygon_in),
            })
            st.success(f"Appended new result at index {len(results)-1}.")
        except Exception as e:
            st.error(str(e))

with c3:
    if st.button("💾 Save detections.json"):
        DET_PATH.write_text(json.dumps(base, indent=2), encoding="utf-8")
        st.success(f"Saved {DET_PATH.resolve()}")

with st.expander("Show detections.json (optional)", expanded=False):
    st.code(json.dumps(base, indent=2), language="json")

# --- preview & run
l, r = st.columns([1,1], gap="large")
with l:
    st.markdown("### Preview")
    if house_file: st.image(house_file, caption="House", use_container_width=True)
    if material_file: st.image(material_file, caption="Material", use_container_width=True)

with r:
    st.markdown("### Output")
    if st.button("🚀 Run"):
        if not house_file or not material_file:
            st.error("Upload both House and Material images.")
        else:
            try:
                house = Image.open(house_file).convert("RGB")
                material = Image.open(material_file).convert("RGB")
                out = amw.run_apply_material_with_inputs(
                    house_img_pil=house,
                    material_img_pil=material,
                    detections_dict=base,
                    depth_img_pil=None,
                )
                st.success("Done!")
                st.image(out, caption="Result", use_container_width=True)
                buf = BytesIO(); out.save(buf, format="PNG")
                st.download_button("⬇️ Download result", data=buf.getvalue(),
                                   file_name="result.png", mime="image/png")
            except Exception as e:
                st.exception(e)
