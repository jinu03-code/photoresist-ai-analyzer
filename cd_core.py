# -*- coding: utf-8 -*-
"""
Pattern CD Analyzer - 분석 함수 (화면 코드 app.py 와 분리)

원리
1) 스케일: 이미지 속 스케일바를 색과 상관없이 모양(배경과 대비되는 가는 가로 막대 + 양 끝 눈금)으로 찾고,
   막대의 바깥 끝~끝 길이를 씁니다. 사용자가 입력한 스케일바 길이(µm)와 나눠 µm/px 를 구합니다.
   막대를 못 찾거나 길이가 비어 있으면 배율 기본값(추정)을 쓰고, 둘 다 없으면 측정하지 않습니다.
2) 패드 찾기: 이미지마다 밝기·색 차이로 자동 이진화(Otsu). 패드 크기는 입력받지 않고 검출된 패드의 중앙값으로 추정합니다.
3) 경계(고정): 패드 중심에서 180개 방향으로 밝기를 읽어, 가장 어두운 값(까만 테두리)에서 배경 쪽으로
   BOUNDARY_FRAC(25%) 밝아지는 지점을 경계로 잡고, 마주 보는 두 방향을 더한 지름 90개의 평균을 CD 로 씁니다.
4) 제외 사유 (수율 계산에서 다르게 다룸)
   - 측정 한계(수율에서 뺌): 가장자리에 잘림 / 가장자리 근처 / 스케일바·주석 근처
   - 불량으로 계산(수율 분모에 포함): 크기 이상(합쳐짐·이물) / 경계 검출 실패 / 이상치
   - 패드가 아닌 작은 이물(패드 크기의 절반 미만)은 패드 수에 넣지 않고 무시합니다.
5) 원형 패드 배열만 지원합니다. 그 밖의 패턴은 '지원하지 않는 패턴'으로 알리고 측정하지 않습니다.
"""
import math
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from scipy.ndimage import binary_fill_holes, map_coordinates
from scipy.spatial import cKDTree

BOUNDARY_FRAC = 0.25            # 경계 기준 (고정): 까만 테두리 바깥 끝
MAGS = [5, 10, 20, 50, 100]
# 배율별 기본 환산율(µm/px): 10·50·100배는 지금까지 찍은 이미지의 스케일바로 잰 값, 5·20배는 계산으로 예측한 값
DEFAULT_UMPX = {5: 2.094, 10: 1.047, 20: 0.5235, 50: 0.2101, 100: 0.1047}
UMPX_NOTE = {5: "예측값", 10: "실측값", 20: "예측값", 50: "실측값", 100: "실측값"}
TYPICAL_SIZE = (1380, 775)       # 현미경 소프트웨어 원본 저장 크기(지금까지 이미지 기준)

C_USED, C_LIMIT, C_DEFECT, C_IGNORE = "사용", "측정 한계 제외", "불량으로 계산", "패드 아님(무시)"
LIMIT_REASONS = ("가장자리(잘림)", "가장자리 근처(온전하지만 측정 영역 부족)", "스케일바·주석 근처")

# 결과 그림 색 (RGB) - 밝기 차이를 둬서 색 구분이 어려워도 알아볼 수 있게 함
COL_NORMAL, COL_NEAR, COL_OUT, COL_DEFECT = (0, 175, 170), (255, 140, 0), (200, 20, 20), (255, 0, 255)


def load_rgb(src):
    return np.array(Image.open(src).convert("RGB"))


# ---------------------------------------------------------------- 스케일
def detect_scale_bar(img, de_thr=35):
    """색에 상관없이 스케일바를 찾습니다. 반환: dict(px, x, y, w, h, caps) 또는 None"""
    h, w = img.shape[:2]
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    labbg = cv2.cvtColor(cv2.medianBlur(img, 41), cv2.COLOR_RGB2LAB).astype(np.float32)
    dE = np.sqrt(((lab - labbg) ** 2).sum(axis=2))
    m = (dE > de_thr).astype(np.uint8) * 255
    kw = max(30, w // 20)
    opened = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 1)))
    n, _, st, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    cands = []
    for i in range(1, n):
        x, y, bw, bh, _a = st[i]
        if bw >= kw and bh <= 16 and bw / max(bh, 1) >= 8:
            def cap(xc):
                col = m[max(0, y - 12):y + bh + 12, max(0, xc - 2):xc + 3].max(axis=1) > 0
                return int(col.sum())
            caps = cap(x + 1) > bh + 3 and cap(x + bw - 2) > bh + 3
            cands.append((caps, bw, x, y, bh))
    if not cands:
        return None
    cands.sort(key=lambda z: (z[0], z[1]), reverse=True)
    caps, bw, x, y, bh = cands[0]
    # 막대에 패드가 겹쳐 길이가 늘어나는 경우를 막기 위해, 막대 자체의 색과 같은 픽셀만으로 길이를 다시 잼
    seg = lab[y:y + bh, x:x + bw]
    bar_col = np.median(seg[:, bw // 4: 3 * bw // 4].reshape(-1, 3), axis=0)
    best, bx0 = 0, x
    x_lo, x_hi = max(0, x - 5), min(w, x + bw + 5)
    for yy in range(y, y + bh):
        ok = np.sqrt(((lab[yy, x_lo:x_hi] - bar_col) ** 2).sum(axis=1)) < 25
        d = np.diff(np.concatenate([[0], ok.astype(int), [0]]))
        s_, e_ = np.where(d == 1)[0], np.where(d == -1)[0]
        if len(s_):
            k = int(np.argmax(e_ - s_))
            if e_[k] - s_[k] > best:
                best, bx0 = int(e_[k] - s_[k]), x_lo + int(s_[k])
    if best >= 0.8 * bw:
        x, bw = bx0, best
    return dict(px=float(bw), x=int(x), y=int(y), w=int(bw), h=int(bh), caps=bool(caps))


def size_warning(img):
    h, w = img.shape[:2]
    tw, th = TYPICAL_SIZE
    if abs(w / tw - 1) > 0.03 or abs(h / th - 1) > 0.03:
        return f"이미지 크기({w}×{h})가 평소({tw}×{th})와 달라 배율 기본값이 맞지 않을 수 있음"
    return ""


def resolve_scale(img, mag, bar_um, mag_table=None, tol=0.03):
    """이미지 1장의 스케일을 정합니다.
    반환: dict(state='정확'|'추정'|'측정 불가', um, source, warns[list], notice, bar)"""
    mag_table = mag_table or DEFAULT_UMPX
    mag = int(mag) if mag not in (None, "") and not pd.isna(mag) else None
    bar_um = float(bar_um) if bar_um not in (None, "") and not pd.isna(bar_um) and float(bar_um) > 0 else None
    default = mag_table.get(mag) if mag else None
    bar = detect_scale_bar(img)
    warns = []
    if bar is not None and bar_um:
        um = bar_um / bar["px"]
        state, source, notice = "정확", f"스케일바 자동 검출({bar['px']:.0f}px = {bar_um:g}µm)", ""
        if default and abs(um / default - 1) > tol:
            warns.append(f"스케일바로 구한 값이 배율(×{mag}) 기본값과 {um / default - 1:+.1%} 차이 - 배율과 스케일바 길이를 확인하세요")
        if not bar["caps"]:
            warns.append("스케일바 양 끝 눈금이 확인되지 않음 - 검출된 막대가 맞는지 확인하세요")
    elif default:
        why = ("이미지에서 스케일바를 찾지 못해" if (bar is None and bar_um)
               else "스케일바 길이(µm)가 비어 있어" if (bar is not None and not bar_um)
               else "스케일바를 찾지 못하고 길이도 비어 있어")
        um = float(default)
        state, source = "추정", f"배율 기본값(×{mag}, {UMPX_NOTE.get(mag, '사용자 입력')})"
        notice = f"{why} 배율 ×{mag} 기본값({um:.4f}µm/px)으로 계산한 추정치입니다. 정확도가 낮을 수 있습니다."
        if UMPX_NOTE.get(mag) == "예측값":
            notice += f" (×{mag} 기본값은 아직 실측하지 않은 예측값입니다)"
        sw = size_warning(img)
        if sw:
            warns.append(sw)
    else:
        return dict(state="측정 불가", um=None, source="", warns=[], bar=bar,
                    notice="스케일바와 배율이 모두 없어 측정하지 않았습니다. 배율을 고르거나 스케일바 길이(µm)를 입력하세요.")
    return dict(state=state, um=um, source=source, warns=warns, notice=notice, bar=bar)


# ---------------------------------------------------------------- 패드 분리
def annotation_mask(img, bar):
    """주석(빨간 글씨·선)과 스케일바, 막대 위의 숫자 글씨를 표시합니다.
    막대와 같은 색의 픽셀만 표시하므로, 글씨 주변의 온전한 패드까지 지우지 않습니다."""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    red = cv2.bitwise_or(cv2.inRange(hsv, (0, 100, 80), (12, 255, 255)),
                         cv2.inRange(hsv, (168, 100, 80), (179, 255, 255)))
    ann = cv2.dilate(red, np.ones((5, 5), np.uint8)) > 0
    if bar is not None:
        H, W = ann.shape
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
        seg = lab[bar["y"]:bar["y"] + bar["h"], bar["x"] + bar["w"] // 4: bar["x"] + 3 * bar["w"] // 4].reshape(-1, 3)
        col = np.median(seg, axis=0)
        x0, x1 = max(0, bar["x"] - 10), min(W, bar["x"] + bar["w"] + 10)
        y0, y1 = max(0, bar["y"] - 30), min(H, bar["y"] + bar["h"] + 12)
        same = np.sqrt(((lab[y0:y1, x0:x1] - col) ** 2).sum(axis=2)) < 30       # 막대·글씨와 같은 색
        ann[y0:y1, x0:x1] |= cv2.dilate(same.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    return ann


def segment_pads(img, ann):
    """가장자리 패턴 손실과 조명 불균일을 줄이는 패드 분리."""
    H, W = img.shape[:2]
    pad = max(24, min(80, int(round(min(H, W) * 0.06))))
    img_p = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
    ann_p = cv2.copyMakeBorder(ann.astype(np.uint8), pad, pad, pad, pad,
                               cv2.BORDER_CONSTANT, value=0) > 0

    lab = cv2.cvtColor(img_p, cv2.COLOR_RGB2LAB).astype(np.float32)
    feat = lab[:, :, 0] - lab[:, :, 2]
    if (~ann_p).any():
        feat[ann_p] = np.median(feat[~ann_p])
    feat8 = cv2.normalize(feat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    feat8 = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8)).apply(feat8)
    smooth = cv2.GaussianBlur(feat8, (5, 5), 0)

    ph, pw = smooth.shape
    bg_k = max(31, min(151, (min(ph, pw) // 8) | 1))
    bg = cv2.medianBlur(smooth, bg_k)
    corrected = np.clip(smooth.astype(np.int16) - bg.astype(np.int16) + 128,
                        0, 255).astype(np.uint8)

    _, otsu = cv2.threshold(corrected, 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    block = max(31, min(101, (min(ph, pw) // 18) | 1))
    adaptive = cv2.adaptiveThreshold(corrected, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, 3)

    candidates = []
    for raw in (cv2.bitwise_or(otsu, adaptive),
                cv2.bitwise_and(otsu, adaptive),
                cv2.bitwise_or(255 - otsu, 255 - adaptive)):
        # 강한 opening은 가장자리의 온전한 패턴까지 깎을 수 있어 사용하지 않음
        clean = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        clean = (binary_fill_holes(clean > 0) * 255).astype(np.uint8)
        candidates.append(clean[pad:pad + H, pad:pad + W])

    def score(mask):
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        vals = []
        for i in range(1, n):
            x, y, ww, hh, area = stats[i]
            if area < 60:
                continue
            contours, _ = cv2.findContours((labels == i).astype(np.uint8),
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            cnt = max(contours, key=cv2.contourArea)
            per = cv2.arcLength(cnt, True)
            circ = 4 * math.pi * area / (per * per) if per > 0 else 0
            ratio = max(ww, hh) / max(min(ww, hh), 1)
            if circ > 0.55 and ratio < 1.8:
                vals.append(area)
        if len(vals) < 3:
            return len(vals)
        med = float(np.median(vals))
        consistency = 1 / (1 + float(np.median(np.abs(np.asarray(vals) - med))) / max(med, 1))
        return len(vals) * consistency

    return max(candidates, key=score)


def _circle_like(mask):
    """원형 패드 배열인지 판단: 온전한(가장자리에 닿지 않은) 덩어리 중 둥근 것의 개수와 비율"""
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    H, W = mask.shape
    circ_areas, other = [], []
    for k in range(1, n):
        x, y, w, h, a = st[k]
        if a < 60 or x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1:
            continue
        cnts, _ = cv2.findContours((lab == k).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cnts, key=cv2.contourArea)
        per = cv2.arcLength(cv2.approxPolyDP(c, 1.0, True), True)
        circ = min(1.0, 4 * math.pi * a / per ** 2) if per > 0 else 0
        (_, _), (rw, rh), _ = cv2.minAreaRect(c)
        ar = max(rw, rh) / max(min(rw, rh), 1e-6)
        (_, _), rad = cv2.minEnclosingCircle(c)
        fill = a / (math.pi * rad ** 2) if rad > 0 else 0       # 둘레를 감싸는 원을 얼마나 채우는가 (원 ≈ 1, 정사각형 ≈ 0.64)
        (circ_areas if (circ > 0.8 and ar < 1.35 and fill > 0.85) else other).append(a)
    if len(circ_areas) < 3:
        return False, None
    p75 = float(np.percentile(circ_areas, 75))
    big = [a for a in circ_areas if a >= 0.6 * p75]           # 작은 이물이 많아도 패드 크기를 기준으로 삼기 위함
    ref = float(np.median(big))
    big_other = [a for a in other if a >= 0.8 * ref]            # 패드만 한 크기 이상인 둥글지 않은 덩어리만 센다
    ok = len(big) >= 0.5 * (len(big) + len(big_other))
    return ok, ref


# ---------------------------------------------------------------- 패드 1개의 경계와 지름
def measure_pad(gray_s, cx, cy, R, n_ang=180, frac=BOUNDARY_FRAC):
    ang = np.linspace(0, 2 * np.pi, n_ang, endpoint=False)
    step = 0.1
    rr = np.arange(0.4 * R, 1.8 * R, step)
    idx = lambda v: int(np.clip(np.searchsorted(rr, v), 0, len(rr) - 1))
    r = None
    for _ in range(3):                                   # 경계의 중심으로 중심을 다시 잡아 2~3회 반복
        X = cx + np.outer(np.cos(ang), rr)
        Y = cy + np.outer(np.sin(ang), rr)
        P = map_coordinates(gray_s, [Y, X], order=1, mode="nearest")
        w0, w1 = idx(0.6 * R), idx(1.2 * R)
        imin = w0 + np.argmin(P[:, w0:w1], axis=1)
        pmin = P[np.arange(n_ang), imin]
        bg = np.median(P[:, idx(1.4 * R):idx(1.75 * R) + 1])
        level = pmin + frac * (bg - pmin)
        jj = np.arange(len(rr))[None, :]
        ok = (P >= level[:, None]) & (jj > imin[:, None]) & (jj <= idx(1.7 * R))
        has = ok.any(axis=1)
        j = np.argmax(ok, axis=1)
        j0 = np.clip(j - 1, 0, len(rr) - 1)
        p0, p1 = P[np.arange(n_ang), j0], P[np.arange(n_ang), j]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(p1 != p0, (level - p0) / (p1 - p0), 0.0)
        r = np.where(has & (j > 0), rr[j0] + np.clip(t, 0, 1) * step, np.nan)
        if np.isfinite(r).mean() < 0.9:
            return None
        bx, by = cx + r * np.cos(ang), cy + r * np.sin(ang)
        ncx, ncy = np.nanmean(bx), np.nanmean(by)
        moved = math.hypot(ncx - cx, ncy - cy)
        cx, cy = ncx, ncy
        if moved < 0.2:
            break
    half = n_ang // 2
    d = r[:half] + r[half:]
    if np.isfinite(d).sum() < 0.9 * half:
        return None
    pts = np.stack([cx + r * np.cos(ang), cy + r * np.sin(ang)], axis=1)
    return dict(cx=cx, cy=cy, d_mean=float(np.nanmean(d)), d_min=float(np.nanmin(d)), d_max=float(np.nanmax(d)),
                rough=float(np.nanstd(r) / np.nanmean(r) * 100), pts=pts[np.isfinite(pts).all(axis=1)])


def judge(cd, target, tol, band):
    """정상 / 경계(허용 한계 근처) / 큼 / 작음"""
    if cd is None or pd.isna(cd):
        return ""
    dev = cd - target
    if abs(dev) <= max(tol - band, 0):
        return "정상"
    if abs(dev) <= tol:
        return "경계"
    return "큼" if dev > 0 else "작음"


# ---------------------------------------------------------------- 이미지 1장 분석
def analyze_image(img, um_px, bar=None, outlier_pct=5.0, ovality_lim=1.10, rough_lim=3.0, edge_k=1.25, ann_k=1.05):
    """반환: (패드별 DataFrame, 패드별 경계 점 목록(df 행과 같은 순서), 정보 dict)
    이미지 경계에 실제로 닿아 잘린 패드만 측정 한계로 제외
    ann_k : 패드 중심이 스케일바·글씨에서 (ann_k x 반지름) 안쪽이면 '스케일바·주석 근처'로 제외"""
    H, W = img.shape[:2]
    ann = annotation_mask(img, bar)
    mask = segment_pads(img, ann)
    info = dict(supported=True, message="", n_found=0, n_used=0, n_limit=0, n_defect=0, n_ignored=0,
                pitch_um=np.nan, pad_px=np.nan, quality="", excluded={})
    supported, ref_area = _circle_like(mask)
    if not supported:
        info.update(supported=False, message="원형 패드 배열을 찾지 못했습니다. 지원하지 않는 패턴이거나, 패드가 너무 작게 찍힌 이미지입니다. (원형 패드를 기준으로 측정하며 비원형·길쭉한 패턴은 이물 의심 후보로 분류합니다)")
        return pd.DataFrame(), [], info

    gray_s = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32), (3, 3), 0)
    n, lab, st, ce = cv2.connectedComponentsWithStats(mask, connectivity=8)
    comps = []
    for k in range(1, n):
        x, y, w, h, a = st[k]
        if a < 20:
            continue
        comps.append(dict(label=int(k), cx=float(ce[k][0]), cy=float(ce[k][1]), area=float(a),
                          touch=bool(x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1)))
    med_area = float(ref_area)                                 # 패드 1개의 기준 면적(둥근 덩어리의 중앙값)
    R0 = math.sqrt(med_area / math.pi)
    ys, xs = np.nonzero(ann)

    rows, pts_list = [], []
    for c in comps:
        R = math.sqrt(c["area"] / math.pi)
        Rm = max(R, R0)
        reason, cls = "", ""
        if c["touch"]:
            reason, cls = "가장자리(잘림)", C_LIMIT
        elif c["area"] < 0.5 * med_area:
            reason, cls = "작은 이물(패드가 아님)", C_IGNORE
        elif len(xs) and (((xs - c["cx"]) ** 2 + (ys - c["cy"]) ** 2) < (ann_k * Rm) ** 2).any():
            reason, cls = "스케일바·주석 근처", C_LIMIT
        # 이미지 경계에 실제로 닿아 잘린 덩어리만 제외합니다.
        # 경계에 가깝더라도 덩어리 전체가 이미지 안에 있으면 검출 후보로 유지합니다.
        elif c["area"] > 2.0 * med_area:
            reason, cls = "크기 이상(합쳐짐·이물)", C_DEFECT
        elif c["area"] >= 0.5 * med_area:
            # 원형 패드 기준 크기와 비슷하지만 길쭉하거나 비원형인 덩어리는
            # CD 정상 패드로 측정하지 않고 이물 의심 후보로 분류합니다.
            component = (lab == c["label"]).astype(np.uint8)
            contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                contour = max(contours, key=cv2.contourArea)
                perimeter = cv2.arcLength(contour, True)
                circularity = (4 * math.pi * c["area"] / (perimeter * perimeter)) if perimeter > 0 else 0.0
                (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
                aspect = max(rw, rh) / max(min(rw, rh), 1e-6)
                if circularity < 0.72 or aspect > 1.65:
                    reason, cls = "비원형·길쭉한 패턴(이물 의심)", C_DEFECT
        res = None
        if not reason:
            res = measure_pad(gray_s, c["cx"], c["cy"], R)
            if res is None:
                reason, cls = "경계 검출 실패", C_DEFECT
        row = dict(cx=c["cx"], cy=c["cy"], area_px=c["area"], 분류=cls or C_USED, reason=reason,
                   CD_um=np.nan, D_min_um=np.nan, D_max_um=np.nan, ovality=np.nan, rough_pct=np.nan)
        pts = None
        if res is not None:
            row.update(cx=res["cx"], cy=res["cy"], CD_um=res["d_mean"] * um_px, D_min_um=res["d_min"] * um_px,
                       D_max_um=res["d_max"] * um_px,
                       ovality=res["d_max"] / res["d_min"] if res["d_min"] > 0 else np.nan, rough_pct=res["rough"])
            pts = res["pts"]
        rows.append(row)
        pts_list.append(pts)
    df = pd.DataFrame(rows)

    # 이상치(중앙값에서 크게 벗어난 패드)는 불량으로 계산
    used = df["분류"] == C_USED
    if used.sum() >= 3:
        med = df.loc[used, "CD_um"].median()
        mad = 1.4826 * (df.loc[used, "CD_um"] - med).abs().median()
        thr = max(outlier_pct / 100 * med, 3 * mad)
        out = used & ((df["CD_um"] - med).abs() > thr)
        df.loc[out, "분류"] = C_DEFECT
        df.loc[out, "reason"] = f"이상치(중앙값 ±{thr:.2f}µm 밖)"
    used = df["분류"] == C_USED

    flags = []
    for _, r_ in df.iterrows():
        f = []
        if r_["분류"] == C_USED:
            if r_["ovality"] > ovality_lim:
                f.append("비대칭")
            if r_["rough_pct"] > rough_lim:
                f.append("반지름 변동 큼")
        flags.append(" · ".join(f))
    df["형상 검토"] = flags

    pitch = np.nan
    if used.sum() >= 7:
        pts_xy = df.loc[used, ["cx", "cy"]].to_numpy()
        dd, _ = cKDTree(pts_xy).query(pts_xy, k=2)
        pitch = float(np.median(dd[:, 1]) * um_px)
    pad_px = float(df.loc[used, "CD_um"].median() / um_px) if used.any() else np.nan
    q = []
    if used.sum() < 7:
        q.append(f"사용 패드가 {int(used.sum())}개로 적어 통계가 불안정합니다")
    if np.isfinite(pad_px) and pad_px < 30:
        q.append(f"패드가 약 {pad_px:.0f}픽셀로 작아 측정이 불확실합니다(1px = {um_px:.3f}µm). 배율을 올리세요")
    info.update(n_found=int((df["분류"] != C_IGNORE).sum()), n_used=int(used.sum()), n_limit=int((df["분류"] == C_LIMIT).sum()),
                n_defect=int((df["분류"] == C_DEFECT).sum()), n_ignored=int((df["분류"] == C_IGNORE).sum()),
                pitch_um=pitch, pad_px=pad_px, quality=" / ".join(q),
                excluded=df.loc[df["분류"] != C_USED, "reason"].str.replace(r"\(중앙값.*\)", "", regex=True).value_counts().to_dict())
    return df, pts_list, info


# ---------------------------------------------------------------- 결과 그림
def numbers_suppressed(df, show_all_numbers=False):
    """규격 밖 패드가 많으면(전체가 목표에서 벗어난 경우) 그림이 숫자로 가득 차서, 숫자 표시를 생략합니다."""
    if show_all_numbers or df is None or not len(df):
        return False
    n_used = int((df["분류"] == C_USED).sum())
    n_out = int(df["판정"].isin(["큼", "작음"]).sum())
    return n_out > max(12, 0.25 * n_used)


def render_overlay(img, df, pts_list, bar=None, show_all_numbers=False, highlight=None):
    """윤곽 색으로만 판정을 보여줍니다. df 에는 '분류','판정','CD_um' 열이 있어야 합니다.
    정상: 청록 얇게 / 경계: 주황 / 규격 밖: 진한 빨강 굵게 / 불량으로 계산한 패드: 자홍색 X.
    측정 한계로 뺀 패드는 표시하지 않습니다. 숫자는 규격 밖 패드에만 적습니다(옵션으로 전체)."""
    ov = img.copy()
    H, W = ov.shape[:2]
    t = 1  # CD 경계선은 이미지 해상도와 관계없이 1px
    fs = 0.45 * W / 1380
    skip_numbers = numbers_suppressed(df, show_all_numbers)
    for i, r in enumerate(df.itertuples()):
        cx, cy = int(round(r.cx)), int(round(r.cy))
        if r.분류 == C_USED and pts_list[i] is not None:
            col, th = {"정상": (COL_NORMAL, 1), "경계": (COL_NEAR, 1)}.get(r.판정, (COL_OUT, 1))
            cv2.polylines(ov, [np.round(pts_list[i]).astype(np.int32).reshape(-1, 1, 2)], True, col, 1, cv2.LINE_8)
            if show_all_numbers or (r.판정 in ("큼", "작음") and not skip_numbers):
                txt = f"{r.CD_um:.2f}"
                (tw, thh), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
                org = (cx - tw // 2, cy + thh // 2)
                cv2.putText(ov, txt, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(ov, txt, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)
        elif r.분류 == C_DEFECT:
            cv2.drawMarker(ov, (cx, cy), COL_DEFECT, cv2.MARKER_TILTED_CROSS, max(12, int(W / 70)), t + 1, cv2.LINE_AA)
    if highlight is not None and 0 <= highlight < len(df):
        r = df.iloc[highlight]
        p = pts_list[highlight]
        if p is not None:
            poly = np.round(p).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(ov, [poly], True, (0, 0, 0), t + 5, cv2.LINE_AA)
            cv2.polylines(ov, [poly], True, (255, 255, 255), t + 2, cv2.LINE_AA)
        else:
            c = (int(round(r["cx"])), int(round(r["cy"])))
            rad = max(12, int(math.sqrt(max(r["area_px"], 1) / math.pi) * 1.15))
            cv2.circle(ov, c, rad, (0, 0, 0), t + 5, cv2.LINE_AA)
            cv2.circle(ov, c, rad, (255, 255, 255), t + 2, cv2.LINE_AA)
    return ov


LEGEND = "청록: 정상 · 주황: 허용 한계 근처 · 빨강: 규격 밖(CD 숫자 표시) · × 표시: 불량으로 계산한 패드(합쳐짐·이물·이상치·검출 실패)"


# ---------------------------------------------------------------- 제안 (조건 변화에 따른 CD 경향에서 나온 추정)
import re

SUGGEST_NOTE = "측정 데이터의 경향에서 나온 추정이며 실험으로 확인이 필요합니다. 같은 장비 세기에서 만든 조건인지 확인하세요."


def _parse_cond(c):
    m = re.match(r"^\s*(-?\d+(?:\.\d+)?)\s*(.*)$", str(c))
    return (float(m.group(1)), m.group(2).strip()) if m else None


def _suggest_partition(part, unit, target, tol):
    """같은 패턴·배율·그룹 메모 안에서 실험 조건에 따른 평균 CD 경향을 보고 제안을 만든다."""
    part = part.sort_values("x")
    xs = part["x"].to_numpy(float)
    ys = part["평균 CD(µm)"].to_numpy(float)
    nimg = part["이미지 수"].to_numpy(int)
    sdv = part["이미지 간 편차(µm)"].to_numpy(float)
    yld = part["수율(%)"].to_numpy(float)
    item = dict(headline=None, reason=None, notes=[], trend="", nearest="",
                table=pd.DataFrame({"실험 조건": part["실험 조건"].to_list(), "평균 CD(µm)": ys, "이미지 수": nimg,
                                    "이미지 간 편차(µm)": sdv, "수율(%)": yld}))
    # 측정한 조건 중 목표에 가장 가까운 것
    k = int(np.argmin(np.abs(ys - target)))
    inside = "허용 범위 안" if abs(ys[k] - target) <= tol else "허용 범위 밖"
    item["nearest"] = (f"측정한 조건 중 목표에 가장 가까운 조건은 {xs[k]:g}{unit}입니다: CD {ys[k]:.2f}µm "
                       f"(목표와 {ys[k] - target:+.2f}µm, {inside}), 수율 {yld[k]:.0f}%")
    d = np.diff(ys)
    if not (np.all(d >= 0) or np.all(d <= 0)):                                   # 규칙 7
        item["reason"] = "실험 조건 순서대로 CD가 한 방향으로 변하지 않아 경향을 확정할 수 없습니다. 제안하지 않습니다."
        return item
    rng = float(ys.max() - ys.min())
    if np.isfinite(sdv).any():                                                   # 규칙 6
        noise = float(np.nanmax(sdv))
        if rng <= noise:
            item["reason"] = (f"조건에 따른 CD 변화({rng:.2f}µm)가 같은 조건의 이미지 간 편차(최대 {noise:.2f}µm)보다 작아 "
                              "경향을 확정할 수 없습니다. 같은 조건을 반복 측정하세요.")
            return item
    else:                                                                        # 규칙 8
        item["notes"].append("모든 조건에서 이미지가 1장이라 반복 편차를 알 수 없습니다. 변화가 우연이 아닌지 확인하지 못했습니다 - 불확실")
    single = [f"{x:g}{unit}" for x, n in zip(xs, nimg) if n == 1]
    if single and np.isfinite(sdv).any():
        item["notes"].append("이미지가 1장뿐인 조건이 있어 불확실합니다: " + ", ".join(single))
    slope = float(np.polyfit(xs, ys, 1)[0]) if len(xs) >= 2 else 0.0
    item["trend"] = f"실험 조건이 1{unit} 늘 때 평균 CD가 약 {slope:+.2f}µm 변합니다 (직선 근사)."
    lo_y, hi_y = float(ys.min()), float(ys.max())
    if lo_y <= target <= hi_y:                                                   # 규칙 4: 범위 안이면 보간
        for i in range(len(xs) - 1):
            a, b = ys[i] - target, ys[i + 1] - target
            if a * b <= 0 and ys[i] != ys[i + 1]:
                xstar = xs[i] + (target - ys[i]) * (xs[i + 1] - xs[i]) / (ys[i + 1] - ys[i])
                item["headline"] = (f"실험 조건 {xs[i]:g}{unit} ~ {xs[i + 1]:g}{unit} 사이에서 목표 CD({target:.2f}µm)에 "
                                    f"도달할 것으로 추정됩니다 (약 {xstar:.2f}{unit} 부근).")
                sp = np.diff(xs)
                if (xs[i + 1] - xs[i]) > 1.5 * float(np.median(sp)):
                    item["notes"].append("이 구간의 측정 간격이 다른 구간보다 넓어 불확실합니다(조건 사이에서 CD가 꺾일 수 있음).")
                break
    else:                                                                        # 규칙 5: 범위 밖이면 방향만
        want_smaller = target < lo_y
        increase_x = (slope < 0) if want_smaller else (slope > 0)
        item["headline"] = (f"측정한 {xs.min():g}~{xs.max():g}{unit}에서는 목표 CD({target:.2f}µm)에 도달하지 못했습니다. "
                            f"경향상 실험 조건을 {'늘리는' if increase_x else '줄이는'} 쪽이 목표에 가까워질 것으로 보이지만, "
                            "측정 범위 밖이라 값은 제시하지 않습니다.")
    return item


def suggest(groups, target, tol):
    """groups: 조건별 요약 DataFrame(열: 패턴, 배율, 실험 조건, 그룹 메모, 스케일, 이미지 수, 평균 CD(µm), 이미지 간 편차(µm), 수율(%)).
    반환: dict(items=[...], guidance=str, notes=[...]) - items 가 비어 있으면 guidance 만 보여준다."""
    out = dict(items=[], guidance="", notes=[])
    if groups is None or not len(groups):
        out["guidance"] = "조건을 2~3개 더 분석하면 제안할 수 있어요."
        return out
    g = groups[groups["실험 조건"] != ""].copy()
    n_est = int((g["스케일"] == "추정").sum())
    if n_est:                                                                    # 규칙 2
        out["notes"].append(f"스케일을 추정한 조건 {n_est}개는 제안에서 제외했습니다.")
    acc = g[g["스케일"] == "정확"]
    n_cond = acc["실험 조건"].nunique()
    if n_cond < 3:                                                               # 규칙 1
        out["guidance"] = f"조건을 2~3개 더 분석하면 제안할 수 있어요. (제안은 실험 조건이 3개 이상일 때 만들어집니다. 지금은 {n_cond}개)"
        return out
    unnumeric, small = False, False
    for key, part in acc.groupby(["패턴", "배율", "그룹 메모"], dropna=False):  # 규칙 3
        label = " · ".join(str(x) for x in key if x)
        conds = list(dict.fromkeys(part["실험 조건"]))
        if len(conds) < 3:
            small = True
            continue
        parsed = {c: _parse_cond(c) for c in conds}
        if any(v is None for v in parsed.values()) or len({v[1] for v in parsed.values()}) != 1:
            unnumeric = True
            continue
        unit = next(iter(parsed.values()))[1]
        part = part.copy()
        part["x"] = part["실험 조건"].map(lambda c: parsed[c][0])
        item = _suggest_partition(part, unit, target, tol)
        item["label"], item["unit"] = label, unit
        out["items"].append(item)
    if not out["items"]:
        out["guidance"] = ("실험 조건을 숫자와 같은 단위로 쓰면 제안을 만들 수 있어요 (예: 7초, 9초, 11초)." if unnumeric else
                           "같은 그룹 메모끼리만 비교하므로, 같은 그룹 메모(그리고 같은 패턴·배율)에서 실험 조건이 3개 이상 필요합니다." if small else
                           "조건을 2~3개 더 분석하면 제안할 수 있어요.")
    return out
