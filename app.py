import io
import math
import re
import zipfile
import json
import hashlib
from pathlib import Path

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image

st.set_page_config(
    page_title="Photoresist Exposure Optimizer",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Photoresist Exposure-Time Analyzer")
st.caption("7–11 sec exposure comparison · CD · pattern uniformity · defect screening")

# =========================================================
# 1. 목표 CD / 허용오차 입력 (이미지 업로드보다 먼저 표시)
# =========================================================
st.header("1. 목표 CD 및 허용오차 입력")
goal_col1, goal_col2 = st.columns(2)
with goal_col1:
    target_cd = st.number_input(
        "목표 CD (μm)", min_value=0.01, value=10.0, step=0.1,
        help="원하는 패턴 직경을 입력하세요."
    )
with goal_col2:
    tolerance_um = st.number_input(
        "허용오차 (± μm)", min_value=0.0, value=0.5, step=0.1,
        help="목표 CD에서 허용할 수 있는 최대 차이입니다."
    )
lo, hi = target_cd - tolerance_um, target_cd + tolerance_um
st.success(f"합격 범위: {lo:.3f} ~ {hi:.3f} μm")
st.caption("개별 패턴 CD가 이 범위 안이면 초록색(합격), 범위를 벗어나면 주황색(불량)으로 표시합니다.")



def load_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    return np.array(image)


def detect_red_scale_bar(image):
    """
    사진 속 빨간색 수평 스케일바를 검출합니다.
    빨간색 글자보다 길고 가로로 긴 성분을 우선 선택합니다.
    반환값: (스케일바 길이 px, 양 끝점, 미리보기 이미지) 또는 None
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # OpenCV HSV에서 빨간색은 Hue 양 끝 구간에 나타날 수 있음
    red1 = cv2.inRange(
        hsv, np.array([0, 100, 80]), np.array([12, 255, 255])
    )
    red2 = cv2.inRange(
        hsv, np.array([168, 100, 80]), np.array([179, 255, 255])
    )
    red = cv2.bitwise_or(red1, red2)

    h, w = red.shape
    kernel_width = max(25, min(100, w // 15))
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (kernel_width, 1)
    )
    horizontal = cv2.morphologyEx(
        red, cv2.MORPH_OPEN, horizontal_kernel
    )

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        horizontal, connectivity=8
    )

    candidates = []
    for i in range(1, count):
        x, y, bw, bh, area = stats[i]
        aspect = bw / max(bh, 1)
        if bw >= kernel_width and aspect >= 8:
            candidates.append((bw, x, y, bh, area))

    if not candidates:
        return None

    # 가장 긴 가로 성분을 스케일바로 간주
    bw, x, y, bh, area = max(candidates, key=lambda z: z[0])

    # 끝의 수직 캡을 포함한 전체 폭보다 캡 중심 간 거리를 사용
    # 캡 두께 추정값은 성분 높이와 연결해 보수적으로 제한
    cap_adjust = min(max(bh - 1, 0), 8)
    measured_px = float(max(1, bw - cap_adjust))

    x1 = int(round(x + cap_adjust / 2))
    x2 = int(round(x + bw - 1 - cap_adjust / 2))
    cy = int(round(y + bh / 2))

    preview = image.copy()
    cv2.line(
        preview, (x1, cy), (x2, cy),
        (0, 255, 0), 3, cv2.LINE_AA
    )
    cv2.circle(preview, (x1, cy), 5, (0, 255, 0), -1)
    cv2.circle(preview, (x2, cy), 5, (0, 255, 0), -1)

    return measured_px, (x1, cy, x2, cy), preview


def make_mask(image, h_min, h_max, s_min, v_min):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((3, 3), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    return mask


def find_reference_radius(reference_image, h_min, h_max, s_min, v_min):
    mask = make_mask(
        reference_image,
        h_min,
        h_max,
        s_min,
        v_min
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)

    moments = cv2.moments(largest)

    if moments["m00"] == 0:
        return None

    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])

    area = cv2.contourArea(largest)

    radius = math.sqrt(area / math.pi)

    return cx, cy, radius


# =========================================================
# 3. 패턴 경계 검출
# =========================================================

def detect_boundary(image, center, expected_radius, tolerance=0.20):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2LAB
    )

    L = lab[:, :, 0].astype(np.float32)

    # 노이즈 완화
    L = cv2.GaussianBlur(L, (5, 5), 1.2)

    gradient_x = cv2.Sobel(
        L,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    gradient_y = cv2.Sobel(
        L,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    gradient = cv2.magnitude(
        gradient_x,
        gradient_y
    )

    cx, cy = center

    height, width = gray.shape

    angles = np.linspace(
        0,
        2 * np.pi,
        360,
        endpoint=False
    )

    boundary_points = []

    min_radius = expected_radius * (1 - tolerance)
    max_radius = expected_radius * (1 + tolerance)

    for angle in angles:

        best_score = -1
        best_point = None

        for radius in np.arange(
            min_radius,
            max_radius,
            1.0
        ):

            x = int(cx + radius * np.cos(angle))
            y = int(cy + radius * np.sin(angle))

            if (
                x < 2 or x >= width - 2
                or y < 2 or y >= height - 2
            ):
                continue

            score = gradient[y, x]

            if score > best_score:
                best_score = score
                best_point = (x, y)

        if best_point is not None:
            boundary_points.append(best_point)

    if len(boundary_points) < 180:
        return None

    boundary_points = np.array(
        boundary_points,
        dtype=np.int32
    )

    return boundary_points


# =========================================================
# 4. CD 계산
# =========================================================

def calculate_diameters(boundary):

    points = boundary.astype(np.float32)

    if len(points) < 5:
        return None

    # 최소 외접 회전 사각형 기반 Feret 직경
    rect = cv2.minAreaRect(points)

    width, height = rect[1]

    max_feret = max(width, height)
    min_feret = min(width, height)

    # 등가직경 계산
    contour = boundary.reshape(-1, 1, 2)

    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    if area <= 0:
        return None

    equivalent_diameter = 2 * math.sqrt(
        area / math.pi
    )

    circularity = (
        4 * math.pi * area / (perimeter ** 2)
        if perimeter > 0 else 0
    )

    aspect_ratio = (
        max_feret / min_feret
        if min_feret > 0 else 0
    )

    return {
        "Equivalent Diameter (px)": equivalent_diameter,
        "Max Feret (px)": max_feret,
        "Min Feret (px)": min_feret,
        "Circularity": circularity,
        "Aspect Ratio": aspect_ratio,
        "Area (px²)": area
    }


# =========================================================
# 5. CD 라벨 표시 함수
# =========================================================

def draw_cd_label(image, x, y, text, number, index):

    font = cv2.FONT_HERSHEY_SIMPLEX

    # 글씨 크기 소폭 축소
    font_scale = 0.70
    thickness = 2

    (tw, th), baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness
    )

    height, width = image.shape[:2]

    # 기본적으로 패턴 중심 위에 표시
    tx = int(x - tw / 2)
    ty = int(y - 30)

    # 이미지 경계 밖으로 나가지 않도록 조정
    tx = max(4, min(tx, width - tw - 4))
    ty = max(th + 8, min(ty, height - baseline - 4))

    # 표시선 시작점과 끝점
    line_start = (
        int(x),
        int(y) - 5
    )

    line_end = (
        int(x),
        int(ty + 6)
    )

    # 얇은 검정색 표시선
    cv2.line(
        image,
        line_start,
        line_end,
        (0, 0, 0),
        1,
        cv2.LINE_AA
    )

    # 검은색 배경 없이 얇은 외곽선만 적용
    cv2.putText(
        image,
        text,
        (tx, ty),
        font,
        font_scale,
        (0, 0, 0),
        3,
        cv2.LINE_AA
    )

    # 노란색 CD 글씨
    cv2.putText(
        image,
        text,
        (tx, ty),
        font,
        font_scale,
        (0, 255, 255),
        thickness,
        cv2.LINE_AA
    )

    # 패턴 번호
    cv2.putText(
        image,
        str(number),
        (int(x) - 5, int(y) + 18),
        font,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )




# =========================================================
# 6. CD 라벨 표시
# =========================================================

def draw_cd_label(image, x, y, text, number, color=(0, 255, 255)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.58
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    h, w = image.shape[:2]
    tx = max(3, min(int(x - tw / 2), w - tw - 3))
    ty = max(th + 5, min(int(y - 25), h - baseline - 3))
    cv2.line(image, (int(x), max(0, int(y)-4)),
             (int(x), max(0, int(ty)+5)), (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(image, text, (tx, ty), font, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, (tx, ty), font, scale, color, thickness, cv2.LINE_AA)
    cv2.putText(image, str(number), (int(x)+3, min(h-3, int(y)+16)),
                font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)


def infer_metadata(filename):
    name = Path(filename).stem.lower()
    # 우선 7s, 8sec, 9초, exp10 등의 표기를 찾고,
    # 없으면 파일명 안의 독립된 숫자 토큰(예: sample_7_center)을 인식합니다.
    exp_match = re.search(r'(?<!\d)(11|10|9|8|7)\s*(?:s|sec|초)(?![a-z])', name)
    if not exp_match:
        exp_match = re.search(r'(?:exp|exposure)[_ -]?(11|10|9|8|7)', name)
    if not exp_match:
        exp_match = re.search(r'(?:^|[^0-9])(11|10|9|8|7)(?:[^0-9]|$)', name)
    exposure = int(exp_match.group(1)) if exp_match else None

    die_match = re.search(r'die[_ -]?(\d+)', name)
    die = f"Die {die_match.group(1)}" if die_match else "Die 1"

    if any(k in name for k in ["left", "lu", "좌상", "좌측상단"]):
        position = "Left-Upper"
    elif any(k in name for k in ["right", "rd", "우하", "우측하단"]):
        position = "Right-Down"
    elif any(k in name for k in ["center", "centre", "중앙", "중심"]):
        position = "Center"
    else:
        position = "Unspecified"
    return exposure, die, position


def analyze_one(image, um_per_pixel, params, reference_info=None, target_low=None, target_high=None):
    h_min, h_max, s_min, v_min = params["h_min"], params["h_max"], params["s_min"], params["v_min"]
    mask = make_mask(image, h_min, h_max, s_min, v_min)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result_bgr = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
    rows = []
    min_area = params["min_area"]
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]

    for contour in contours:
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        area_mask = cv2.contourArea(contour)
        mask_radius = math.sqrt(area_mask / math.pi)

        if reference_info is not None:
            ref_radius = reference_info[2]
            correction = (params["ref_diameter"] / 2) / ref_radius
            expected_radius = mask_radius * correction
        else:
            expected_radius = mask_radius

        boundary = detect_boundary(
            image, (cx, cy), expected_radius,
            tolerance=params["boundary_tolerance"]
        )
        if boundary is None:
            continue

        m = calculate_diameters(boundary)
        if m is None:
            continue

        cd_um = m["Equivalent Diameter (px)"] * um_per_pixel
        max_f = m["Max Feret (px)"] * um_per_pixel
        min_f = m["Min Feret (px)"] * um_per_pixel

        # 개별 패턴 CD가 목표 허용범위 안이면 초록색, 벗어나면 주황색
        is_pass = (
            target_low is not None and target_high is not None
            and target_low <= cd_um <= target_high
        )
        mark_color = (0, 200, 0) if is_pass else (0, 140, 255)  # BGR: green / orange
        cv2.polylines(result_bgr, [boundary.reshape(-1, 1, 2)],
                      True, mark_color, 2, cv2.LINE_AA)
        cv2.circle(result_bgr, (cx, cy), 2, mark_color, -1)
        label = f"{cd_um:.2f} um {'PASS' if is_pass else 'FAIL'}"
        draw_cd_label(result_bgr, cx, cy, label, len(rows) + 1, mark_color)

        rows.append({
            "Pattern_No": len(rows) + 1,
            "Center_X_px": cx,
            "Center_Y_px": cy,
            "CD_um": cd_um,
            "Max_Feret_um": max_f,
            "Min_Feret_um": min_f,
            "Area_um2": m["Area (px²)"] * um_per_pixel**2,
            "Circularity": m["Circularity"],
            "Aspect_Ratio": m["Aspect Ratio"],
        })

    annotated = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    return rows, annotated, mask


def classify_patterns(df):
    """간단한 영상 기반 선별 규칙. 공정 불량의 확정 판정은 아님."""
    out = df.copy()
    out["Flag_Reasons"] = ""

    for idx, row in out.iterrows():
        reasons = []
        if pd.notna(row.get("Circularity")) and row["Circularity"] < 0.75:
            reasons.append("원형도 낮음")
        if pd.notna(row.get("Aspect_Ratio")) and row["Aspect_Ratio"] > 1.20:
            reasons.append("비대칭")
        out.at[idx, "Flag_Reasons"] = "; ".join(reasons)
    out["Image_Flag"] = np.where(out["Flag_Reasons"] == "", "검토 기준 통과", "검토 필요")
    return out


def build_exposure_summary(pattern_df, image_df, target_cd=None):
    summaries = []
    for exp in sorted(image_df["Exposure_s"].dropna().unique()):
        sub = pattern_df[pattern_df["Exposure_s"] == exp]
        img_sub = image_df[image_df["Exposure_s"] == exp]
        if sub.empty:
            summaries.append({
                "Exposure_s": exp, "Image_Count": len(img_sub), "Pattern_Count": 0,
                "Mean_CD_um": np.nan, "SD_CD_um": np.nan, "CV_percent": np.nan,
                "Mean_Circularity": np.nan, "Mean_AR": np.nan,
                "Flagged_Pattern_percent": np.nan, "Target_Error_percent": np.nan
            })
            continue

        mean_cd = sub["CD_um"].mean()
        sd_cd = sub["CD_um"].std(ddof=1)
        cv = sd_cd / mean_cd * 100 if len(sub) > 1 and mean_cd else np.nan
        target_error = abs(mean_cd - target_cd) / target_cd * 100 if target_cd and target_cd > 0 else np.nan
        summaries.append({
            "Exposure_s": exp,
            "Image_Count": len(img_sub),
            "Pattern_Count": len(sub),
            "Mean_CD_um": mean_cd,
            "SD_CD_um": sd_cd,
            "CV_percent": cv,
            "Mean_Circularity": sub["Circularity"].mean(),
            "Mean_AR": sub["Aspect_Ratio"].mean(),
            "Flagged_Pattern_percent": (sub["Image_Flag"] == "검토 필요").mean() * 100,
            "Target_Error_percent": target_error
        })
    return pd.DataFrame(summaries)



with st.sidebar:
    st.header("분석 설정")
    scale_um = st.number_input("스케일바 길이 (μm)", min_value=0.1, value=50.0)
    h_min = st.slider("Hue 최소", 0, 179, 90)
    h_max = st.slider("Hue 최대", 0, 179, 160)
    s_min = st.slider("채도 최소", 0, 255, 30)
    v_min = st.slider("명도 최소", 0, 255, 30)
    min_area = st.number_input("최소 패턴 면적 (px²)", min_value=1, value=100)
    boundary_tolerance = st.slider("경계 탐색 범위 (%)", 5, 40, 20)/100
    circularity_limit = st.slider("원형도 검토 기준", 0.4, 0.99, 0.75)
    aspect_limit = st.slider("비대칭 검토 기준 (AR)", 1.0, 2.0, 1.2)

# Persistent local storage: original images, calibration and analyzed images.
ROOT = Path(__file__).resolve().parent
STORE = ROOT/"pr_saved_data"
ORIGINALS = STORE/"original_images"
ANALYZED = STORE/"analyzed_images"
for folder in (STORE, ORIGINALS, ANALYZED): folder.mkdir(parents=True, exist_ok=True)
MANIFEST = STORE/"manifest.json"
CALFILE = STORE/"calibration.json"

def read_json(path, default):
    try: return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
    except Exception: return default

def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def safe_name(v):
    return re.sub(r"[^A-Za-z0-9가-힣_.-]+", "_", str(v)).strip("._") or "image"

# Scale-bar calibration saved after first use.
st.header("2. 50 μm 스케일바 보정")
calibration = read_json(CALFILE, {})
cal_file = st.file_uploader("50 μm 스케일바가 있는 대표 이미지 (최초 1회)", type=["png","jpg","jpeg","tif","tiff"], key="calibration_upload")
um_per_pixel = calibration.get("um_per_pixel")
if cal_file:
    ci = load_image(cal_file)
    det = detect_red_scale_bar(ci)
    if det:
        px, ends, preview = det
        ratio = scale_um/px
        st.image(preview, caption="검출된 스케일바 확인", use_container_width=True)
        st.write(f"검출 길이 {px:.2f} px · 환산율 {ratio:.6f} μm/px")
        if st.button("스케일바 환산율 저장"):
            calibration={"scale_um":float(scale_um),"scale_px":float(px),"um_per_pixel":float(ratio)}
            write_json(CALFILE, calibration); um_per_pixel=ratio
            st.success("저장 완료. 다음부터 스케일바 이미지를 다시 올릴 필요가 없습니다.")
    else: st.error("빨간색 가로 스케일바를 검출하지 못했습니다.")
elif um_per_pixel:
    st.success(f"저장된 환산율 사용 중: {um_per_pixel:.6f} μm/px")
else:
    st.warning("최초 1회 스케일바 환산율을 저장해야 합니다.")

# Three separate upload slots.
st.header("3. 촬영 위치별 이미지 업로드")
st.caption("각 칸에 여러 장 업로드 가능. 파일명에 7s/8s/9s/10s/11s, Die1 등을 넣으면 자동 분류합니다.")
c1,c2,c3=st.columns(3)
with c1: center_files=st.file_uploader("① Center · 중앙",type=["png","jpg","jpeg","tif","tiff"],accept_multiple_files=True,key="center_upload")
with c2: left_files=st.file_uploader("② Left-Upper · 왼쪽 위",type=["png","jpg","jpeg","tif","tiff"],accept_multiple_files=True,key="left_upload")
with c3: right_files=st.file_uploader("③ Right-Down · 오른쪽 아래",type=["png","jpg","jpeg","tif","tiff"],accept_multiple_files=True,key="right_upload")

new=[]
for pos, files in [("Center",center_files or []),("Left-Upper",left_files or []),("Right-Down",right_files or [])]:
    for f in files:
        exp, die, _ = infer_metadata(f.name)
        if exp is None:
            st.warning(f"파일명에서 노광시간을 찾지 못했습니다: {f.name} — 저장 후 목록에서 직접 지정해 주세요.")
            exp = 9  # 임시값; 저장된 이미지 목록에서 실제 조건으로 수정하도록 안내
        new.append((f,pos,exp,die))
if new and st.button("원본 이미지 저장"):
    manifest=read_json(MANIFEST,[])
    known={x["stored"] for x in manifest}
    added=0
    for f,pos,exp,die in new:
        raw=f.getvalue(); digest=hashlib.sha256(raw).hexdigest()[:12]
        ext=Path(f.name).suffix.lower() or ".png"
        stored=f"{exp}s_{safe_name(die)}_{pos}_{safe_name(Path(f.name).stem)}_{digest}{ext}"
        if stored in known: continue
        (ORIGINALS/stored).write_bytes(raw)
        manifest.append({"stored":stored,"filename":f.name,"exposure":int(exp),"die":die,"position":pos})
        known.add(stored); added+=1
    write_json(MANIFEST,manifest)
    st.success(f"{added}장 저장 완료. 다음 실행부터 다시 업로드하지 않아도 됩니다.")
    st.rerun()

manifest=read_json(MANIFEST,[])
st.subheader("저장된 이미지 및 촬영 조건")
if manifest:
    meta=pd.DataFrame([{"파일명":x["filename"],"노광시간(초)":x["exposure"],"Die ID":x["die"],"촬영 위치":x["position"]} for x in manifest])
    edited=st.data_editor(meta,use_container_width=True,hide_index=True,column_config={
        "노광시간(초)":st.column_config.SelectboxColumn(options=[7,8,9,10,11],required=True),
        "촬영 위치":st.column_config.SelectboxColumn(options=["Center","Left-Upper","Right-Down"],required=True)
    },disabled=["파일명"],key="metadata")
    if st.button("촬영 조건 변경사항 저장"):
        for i,x in enumerate(manifest):
            x["exposure"]=int(edited.iloc[i]["노광시간(초)"]); x["die"]=str(edited.iloc[i]["Die ID"]); x["position"]=str(edited.iloc[i]["촬영 위치"])
        write_json(MANIFEST,manifest); st.success("변경사항 저장"); st.rerun()
    st.caption(f"원본 {len(manifest)}장 저장 · {ORIGINALS}")
else: st.info("아직 저장된 이미지가 없습니다.")

# Analyze all patterns; only concise exposure summaries are shown.
st.header("4. 전체 이미지 분석 및 노광시간별 결과")
if manifest and um_per_pixel:
    if st.button("저장된 전체 이미지 분석",type="primary"):
        params={"h_min":h_min,"h_max":h_max,"s_min":s_min,"v_min":v_min,"min_area":int(min_area),"boundary_tolerance":boundary_tolerance}
        records=[]
        progress=st.progress(0)
        for i,item in enumerate(manifest):
            path=ORIGINALS/item["stored"]
            if not path.exists(): continue
            image=load_image(path)
            rows,annotated,mask=analyze_one(image,float(um_per_pixel),params,None,lo,hi)
            for row in rows:
                row.update({"Exposure":int(item["exposure"]),"Die":item["die"],"Position":item["position"],"File":item["filename"]})
                records.append(row)
            outname=f'{item["exposure"]}s_{safe_name(item["die"])}_{item["position"]}_{Path(item["stored"]).stem}_analyzed.png'
            Image.fromarray(annotated).save(ANALYZED/outname)
            progress.progress((i+1)/len(manifest))
        if not records:
            st.error("패턴을 찾지 못했습니다. HSV 범위/최소 면적/경계 탐색 범위를 확인하세요.")
        else:
            d=pd.DataFrame(records)
            d["CD_status"]=np.where(d["CD_um"]<lo,"작음",np.where(d["CD_um"]>hi,"큼","정상"))
            d["shape_bad"]=(d["Circularity"]<circularity_limit)|(d["Aspect_Ratio"]>aspect_limit)
            # Equal weighting: mean within each image, then position, then 3 positions.
            im=d.groupby(["Exposure","Die","Position","File"],as_index=False).agg(
                image_cd=("CD_um","mean"), shape_rate=("shape_bad","mean"),
                out_tol_rate=("CD_status",lambda x:(x!="정상").mean()),
                circ=("Circularity","mean"), ar=("Aspect_Ratio","mean"))
            pos=im.groupby(["Exposure","Position"],as_index=False).agg(
                pos_cd=("image_cd","mean"),shape_rate=("shape_rate","mean"),
                out_tol_rate=("out_tol_rate","mean"),circ=("circ","mean"),ar=("ar","mean"))
            summ=pos.groupby("Exposure",as_index=False).agg(
                CD_um=("pos_cd","mean"),
                position_spread=("pos_cd",lambda x:float(x.max()-x.min())),
                position_count=("Position","nunique"),
                shape_rate=("shape_rate","mean"),out_tol_rate=("out_tol_rate","mean"),
                circularity=("circ","mean"),aspect_ratio=("ar","mean"))
            summ["CD판정"]=np.where(summ.CD_um < lo, "불량 (CD 작음)",
                                      np.where(summ.CD_um > hi, "불량 (CD 큼)", "합격"))
            reports=[]
            for _,r in summ.iterrows():
                issues=[]; causes=[]; remedies=[]
                if r.position_count<3:
                    issues.append("3개 위치 데이터 불완전"); causes.append("일부 촬영 위치 이미지 누락"); remedies.append("누락 위치 재촬영")
                if r.CD판정 == "불량 (CD 작음)":
                    issues.append("불량: 종합 CD가 목표 허용범위보다 작음")
                    causes.append("노광·현상 조건 또는 경계 판정 기준에 따른 치수 축소 가능성")
                    remedies.append("인접 노광시간을 비교하고 현상시간·현상액 농도 및 경계 기준 확인")
                elif r.CD판정 == "불량 (CD 큼)":
                    issues.append("불량: 종합 CD가 목표 허용범위보다 큼")
                    causes.append("노광·현상 조건 또는 경계 판정 기준에 따른 치수 확대 가능성")
                    remedies.append("인접 노광시간과 현상 조건, 마스크 접촉 상태 비교")
                else:
                    issues.append("합격: 종합 CD가 목표 허용범위 안에 있음")
                if r.position_spread>max(tolerance_um,0.1):
                    issues.append("위치별 CD 편차 큼"); causes.append("조명·초점·레벨링, PR 막 두께, 마스크 접촉 또는 현상 균일성 차이 가능성"); remedies.append("세 위치의 초점·조명·PR 두께·마스크 접촉·현상 상태 점검")
                if r.out_tol_rate>0.1:
                    issues.append("개별 패턴 중 허용오차 초과 패턴 존재"); causes.append("패턴 간 치수 편차 또는 국부 공정 불균일"); remedies.append("도포·베이크·노광·현상 조건을 고정하고 반복 Die 측정")
                if r.shape_rate>0.1:
                    issues.append("원형도 저하/비대칭 패턴 관찰"); causes.append("초점 불량, 마스크 접촉, 현상 불균일 또는 광학 경계 오검출 가능성"); remedies.append("저장된 분석 이미지를 확대해 실제 변형과 경계 오검출 구분")
                reports.append({"노광시간":f"{int(r.Exposure)}초",
                    "종합 CD":f"{r.CD_um:.3f} μm",
                    "최종 결과": "합격" if r.CD판정 == "합격" else "불량",
                    "불량 판정": " · ".join(issues),
                    "가능 원인":" · ".join(dict.fromkeys(causes)) if causes else "뚜렷한 이상 신호가 적음",
                    "개선 방향":" · ".join(dict.fromkeys(remedies)) if remedies else "현재 조건으로 반복 Die 측정해 재현성 확인"})
            # rank: target deviation, location spread, geometry flags
            summ["score"]=(summ.CD_um-target_cd).abs()/target_cd*100+summ.position_spread/target_cd*100+summ.shape_rate*100
            passing = summ[summ["CD판정"] == "합격"]
            best = (passing.sort_values("score").iloc[0]
                    if not passing.empty else summ.sort_values("score").iloc[0])
            st.session_state["reports"]=reports; st.session_state["summary"]=summ.to_dict("records")
            st.session_state["best"]=int(best.Exposure)
            st.success(
                f'현재 데이터 기준 추천 노광시간: {int(best.Exposure)}초 · '
                f'종합 CD {best.CD_um:.3f} μm · {best.CD판정}'
            )
            st.caption("세 위치를 동일 가중치로 통합한 1차 후보이며, 공정 최적 조건 확정은 반복 실험으로 검증해야 합니다.")

if "reports" in st.session_state:
    st.subheader("노광시간별 결과 요약")
    st.dataframe(pd.DataFrame(st.session_state["reports"]),use_container_width=True,hide_index=True)
    st.subheader("노광시간별 종합 CD")
    chart=pd.DataFrame(st.session_state["summary"])
    st.bar_chart(chart.set_index("Exposure")[["CD_um"]])
    st.info("합격/불량 판정은 CD 기준만 사용합니다. 원형도·비대칭은 원인 분석을 위한 참고 지표이며 CD가 합격 범위 안이면 최종 합격입니다.")
    st.subheader("노광시간별 총평")
    for report in st.session_state["reports"]:
        st.markdown(
            f'**{report["노광시간"]} — {report["최종 결과"]} / 종합 CD {report["종합 CD"]}**  \\n'
            f'{report["불량 판정"]}  \\n'
            f'가능 원인: {report["가능 원인"]}  \\n'
            f'개선 방향: {report["개선 방향"]}'
        )
    outputs=sorted(ANALYZED.glob("*_analyzed.png"))
    st.write(f"분석 이미지 {len(outputs)}장 저장됨: `{ANALYZED}`")
    st.subheader("노광시간별 패턴 판정 이미지")
    st.caption("초록색 테두리/글씨 = 목표 CD 허용범위 내, 주황색 테두리/글씨 = 허용범위 초과")
    for exposure in [7, 8, 9, 10, 11]:
        group = [p for p in outputs if re.search(rf"^{exposure}s_", p.name)]
        if group:
            with st.expander(f"{exposure}초 분석 이미지 ({len(group)}장)", expanded=(exposure == 7)):
                cols = st.columns(2)
                for idx, img_path in enumerate(group):
                    with cols[idx % 2]:
                        st.image(str(img_path), caption=img_path.name, use_container_width=True)

st.caption("불량 판정은 입력한 목표 CD±허용오차 및 영상 형상 지표에 따른 선별입니다. 광학 이미지 경계 오류가 있을 수 있어 실제 공정 불량 확정 전 대표 이미지를 확인하세요. 원본/결과 이미지는 app.py와 같은 컴퓨터의 pr_saved_data 폴더에 저장됩니다.")
