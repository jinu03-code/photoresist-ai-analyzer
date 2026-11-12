# -*- coding: utf-8 -*-
"""
Pattern CD Analyzer
실행:  streamlit run app.py   (app.py 와 cd_core.py 를 같은 폴더에 두세요)
"""
import io, re, json, html, shutil, hashlib, zipfile, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from PIL import Image

import cd_core as core

st.set_page_config(page_title="Pattern CD Analyzer", page_icon="🔬", layout="wide",
                   initial_sidebar_state="collapsed")

# ------------------------------------------------------------ 저장 폴더
ROOT = Path(__file__).resolve().parent
STORE = ROOT / "saved_data"
ORIGINALS = STORE / "original_images"
ANALYZED = STORE / "analyzed_images"
for _f in (STORE, ORIGINALS, ANALYZED):
    _f.mkdir(parents=True, exist_ok=True)
MANIFEST = STORE / "manifest.json"
CALFILE = STORE / "umpx_table.json"
IMG_TYPES = ["png", "jpg", "jpeg", "tif", "tiff", "bmp"]
MAG_LABELS = [f"×{m}" for m in core.MAGS]

ADV_DEFAULT = dict(outlier_pct=5.0, band=0.1, ovality=1.10, rough=3.0, design_pitch=0.0, show_all=False)


def read_json(path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
    except Exception:
        return default


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_name(v):
    return re.sub(r"[^A-Za-z0-9가-힣_.-]+", "_", str(v)).strip("._") or "image"


def num_or_none(v):
    try:
        if v is None or v == "" or (isinstance(v, float) and np.isnan(v)):
            return None
        return float(v)
    except Exception:
        return None


def txt(v):
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v).strip()


def mag_from_label(v):
    v = txt(v)
    return int(v.replace("×", "")) if v else None


def mag_label(m):
    return f"×{int(m)}" if m else ""


def load_umpx_table():
    table = dict(core.DEFAULT_UMPX)
    for k, v in read_json(CALFILE, {}).items():
        try:
            table[int(k)] = float(v)
        except Exception:
            pass
    return table


def infer_meta(filename):
    """파일 이름에 조건이 들어 있으면 자동으로 채웁니다. 없으면 비워 둡니다(막지 않음)."""
    low = Path(filename).stem.lower()
    mag = None
    for pat in (r"(?:x|×)\s*(\d+)", r"(\d+)\s*(?:x|배)(?![a-z])"):
        for mm in re.finditer(pat, low):
            if int(mm.group(1)) in core.MAGS:
                mag = int(mm.group(1))
                break
        if mag:
            break
    cond = ""
    m = re.search(r"(?<![\d.])(\d+(?:\.\d+)?)\s*mj(?![a-z])", low)
    if m:
        cond = f"{float(m.group(1)):g}mJ"
    else:
        m = (re.search(r"(?<![\d.])(\d+(?:\.\d+)?)\s*(?:s|sec|초)(?![a-z])", low)
             or re.search(r"(?:exp|exposure)[_ -]?(\d+(?:\.\d+)?)", low))
        if m:
            cond = f"{float(m.group(1)):g}초"
    tokens = set(re.split(r"[^a-z0-9가-힣]+", low))
    pos = ""
    if tokens & {"center", "centre", "m", "중앙", "중심"}:
        pos = "Center"
    elif tokens & {"lu", "left", "ul", "좌상"}:
        pos = "Left-Upper"
    elif tokens & {"rd", "right", "br", "우하"}:
        pos = "Right-Down"
    p = re.search(r"(?:^|[_ -])(p\d+)(?=$|[_ -])", low)
    return dict(mag=mag, cond=cond, pos=pos, pattern=p.group(1).upper() if p else "")


def pitch_label(pitch):
    if pitch is None or pd.isna(pitch):
        return "패턴 미확인"
    r = round(pitch) if pitch < 30 else round(pitch / 5) * 5
    return f"피치 {r:g}µm"


# ------------------------------------------------------------ 초기화 (고급 설정 안)
for _k, _v in (("upload_ver", 0), ("edit_ver", 0)):
    st.session_state.setdefault(_k, _v)


def clear_folder(folder, backup=True):
    """폴더 안의 파일을 비웁니다. backup=True면 삭제하지 않고 _backup 폴더로 옮깁니다."""
    files = [p for p in folder.glob("*") if p.is_file()]
    if backup and files:
        dest = STORE / "_backup" / datetime.datetime.now().strftime("%Y%m%d_%H%M%S") / folder.name
        dest.mkdir(parents=True, exist_ok=True)
        for p in files:
            shutil.move(str(p), str(dest / p.name))
    else:
        for p in files:
            p.unlink()
    return len(files)


def reset_results():
    clear_folder(ANALYZED, backup=False)
    st.session_state.pop("res", None)


def reset_images():
    clear_folder(ORIGINALS, backup=True)
    write_json(MANIFEST, [])
    reset_results()
    st.session_state["upload_ver"] += 1
    st.session_state["edit_ver"] += 1


def reset_umpx():
    if CALFILE.exists():
        CALFILE.unlink()


def run_reset(fn):
    fn()
    st.session_state["confirm_reset"] = False


# ------------------------------------------------------------ 사이드바: 고급 설정 (접혀 있음)
with st.sidebar:
    st.header("고급 설정")
    st.caption("처음에는 건드릴 필요가 없습니다. 기본값에서 바꾸면 결과에 표시됩니다.")
    outlier_pct = st.slider("이상치 기준 (중앙값 대비 %)", 2.0, 20.0, ADV_DEFAULT["outlier_pct"], 0.5)
    band = st.number_input("허용 한계 근처 폭 (µm)", min_value=0.0, value=ADV_DEFAULT["band"], step=0.05,
                           help="허용 한계 안쪽 이 폭 안의 패드는 '경계'(주황)로 표시합니다. 수율에서는 정상으로 계산합니다.")
    ovality_lim = st.slider("비대칭 검토 기준 (최대/최소 지름)", 1.02, 1.50, ADV_DEFAULT["ovality"], 0.01)
    rough_lim = st.slider("반지름 변동 검토 기준 (%)", 0.5, 10.0, ADV_DEFAULT["rough"], 0.5,
                        help="패드 중심에서 경계까지 거리가 방향마다 흔들리는 정도(%). 클수록 울퉁불퉁하거나 찌그러진 패드입니다. 표시만 하고 판정·수율에는 반영하지 않습니다.")
    design_pitch = st.number_input("설계 피치 (µm, 선택)", min_value=0.0, value=ADV_DEFAULT["design_pitch"], step=0.5,
                                   help="입력하면 측정한 피치와 비교해 스케일이 맞는지 검산합니다. 0이면 사용하지 않습니다.")
    show_all = st.checkbox("모든 패드에 CD 숫자 표시", value=ADV_DEFAULT["show_all"])

    with st.expander("공정 변수 입력", expanded=True):
        process_conditions = {
            "spin_rpm": st.number_input("Spin coating 회전속도 (rpm)", min_value=0, value=2500, step=100),
            "soft_bake_temp_c": st.number_input("Soft bake 온도 (°C)", min_value=0.0, value=100.0, step=1.0),
            "soft_bake_time_s": st.number_input("Soft bake 시간 (s)", min_value=0, value=90, step=5),
            "exposure_time_s": st.number_input("Exposure 시간 (s)", min_value=0.0, value=9.0, step=0.5),
            "irb_temp_c": st.number_input("IRB 온도 (°C)", min_value=0.0, value=110.0, step=1.0),
            "irb_time_s": st.number_input("IRB 시간 (s)", min_value=0, value=90, step=5),
            "development_time_s": st.number_input("Development 시간 (s)", min_value=0, value=60, step=5),
            "di_rinse_time_s": st.number_input("DI 세척 시간 (s)", min_value=0, value=60, step=5),
        }

    umpx_table = load_umpx_table()
    with st.expander("배율별 기본 환산율 (스케일바가 없을 때 쓰는 추정값)"):
        t_df = pd.DataFrame([{"배율": f"×{m}", "µm/px": umpx_table[m], "근거": core.UMPX_NOTE.get(m, "")} for m in core.MAGS])
        t_ed = st.data_editor(t_df, hide_index=True, disabled=["배율", "근거"], key="umpx_editor")
        if st.button("환산율 저장"):
            write_json(CALFILE, {str(core.MAGS[i]): float(r["µm/px"]) for i, (_, r) in enumerate(t_ed.iterrows())})
            st.success("저장했습니다.")
            st.rerun()
        st.caption("×5, ×20은 예측값입니다. 그 배율로 찍은 스케일바 이미지로 재서 고친 뒤 사용하세요.")

    with st.expander("초기화"):
        n_orig = len([p for p in ORIGINALS.glob("*") if p.is_file()])
        st.caption(f"저장된 원본 이미지 {n_orig}장 · 초기화하면 saved_data/_backup 폴더로 옮겨지고 삭제되지 않습니다.")
        confirm = st.checkbox("초기화를 진행합니다", key="confirm_reset")
        st.button("분석 결과만 초기화", disabled=not confirm, on_click=run_reset, args=(reset_results,))
        st.button("이미지 + 결과 전체 초기화", type="primary", disabled=not confirm, on_click=run_reset, args=(reset_images,))
        st.button("환산율 기본값으로 되돌리기", disabled=not confirm, on_click=run_reset, args=(reset_umpx,))

changed = []
if outlier_pct != ADV_DEFAULT["outlier_pct"]: changed.append(f"이상치 기준 {outlier_pct:g}%")
if band != ADV_DEFAULT["band"]: changed.append(f"허용 한계 근처 폭 {band:g}µm")
if ovality_lim != ADV_DEFAULT["ovality"]: changed.append(f"비대칭 기준 {ovality_lim:g}")
if rough_lim != ADV_DEFAULT["rough"]: changed.append(f"반지름 변동 기준 {rough_lim:g}%")
if design_pitch != ADV_DEFAULT["design_pitch"]: changed.append(f"설계 피치 {design_pitch:g}µm")
if show_all != ADV_DEFAULT["show_all"]: changed.append("모든 패드 CD 숫자 표시")
if umpx_table != core.DEFAULT_UMPX: changed.append("배율별 환산율")

# ------------------------------------------------------------ 제목과 목표
st.title("🔬 Pattern CD Analyzer")
c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    target_cd = st.number_input("목표 CD (µm)", min_value=0.01, value=10.0, step=0.1)
with c2:
    tolerance_um = st.number_input("허용오차 (± µm)", min_value=0.0, value=0.5, step=0.1)
lo, hi = target_cd - tolerance_um, target_cd + tolerance_um
with c3:
    st.markdown(f"<div style='padding-top:1.9rem;color:#555'>정상 범위 {lo:.2f} ~ {hi:.2f} µm</div>", unsafe_allow_html=True)

# ------------------------------------------------------------ 이미지 올리기
st.subheader("이미지")
manifest = read_json(MANIFEST, [])
files = st.file_uploader("이미지를 올리세요 (여러 장 가능, 파일 이름은 아무거나 괜찮습니다)", type=IMG_TYPES,
                         accept_multiple_files=True, key=f"upload_{st.session_state['upload_ver']}")
if files:
    known = {x["stored"] for x in manifest}
    added = 0
    for f in files:
        raw = f.getvalue()
        digest = hashlib.sha256(raw).hexdigest()[:12]
        stored = f"{safe_name(Path(f.name).stem)}_{digest}{Path(f.name).suffix.lower() or '.png'}"
        if stored in known:
            continue
        (ORIGINALS / stored).write_bytes(raw)
        meta = infer_meta(f.name)
        manifest.append(dict(stored=stored, filename=f.name, mag=meta["mag"], bar_um=None, cond=meta["cond"],
                             pos=meta["pos"], pattern=meta["pattern"], memo=""))
        known.add(stored)
        added += 1
    if added:
        write_json(MANIFEST, manifest)
        st.rerun()

ev = st.session_state["edit_ver"]
edited_main = edited_extra = None
if manifest:
    if len(manifest) >= 2:
        st.caption("여러 장에 같은 값 넣기")
        b1, b2, b3 = st.columns([1, 1, 1])
        with b1:
            bulk_mag = st.selectbox("배율", MAG_LABELS, index=None, placeholder="선택", key=f"bulk_mag_{ev}")
        with b2:
            bulk_bar = st.number_input("스케일바 길이 (µm)", min_value=0.0, value=None, placeholder="예: 50", key=f"bulk_bar_{ev}")
        with b3:
            st.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
            if st.button("전체에 적용"):
                for x in manifest:
                    if bulk_mag:
                        x["mag"] = mag_from_label(bulk_mag)
                    if bulk_bar:
                        x["bar_um"] = float(bulk_bar)
                write_json(MANIFEST, manifest)
                st.session_state["edit_ver"] += 1
                st.rerun()

    main_df = pd.DataFrame({
        "파일": [x["filename"] for x in manifest],
        "배율": [mag_label(x.get("mag")) or None for x in manifest],
        "스케일바(µm)": [x.get("bar_um") for x in manifest],
    })
    edited_main = st.data_editor(
        main_df, hide_index=True, key=f"main_{ev}", disabled=["파일"],
        column_config={
            "배율": st.column_config.SelectboxColumn("배율", options=MAG_LABELS),
            "스케일바(µm)": st.column_config.NumberColumn(
                "스케일바 길이 (µm)", min_value=0.0, step=1.0,
                help="이미지에 적힌 스케일바 값(µm)을 입력하세요. 지금까지는 ×10: 200, ×50: 50, ×100: 20이었습니다(현미경 설정에 따라 다를 수 있음)."),
        })
    with st.expander("추가 정보 (실험 조건 · 위치 · 패턴 · 그룹 메모, 모두 선택 사항)"):
        extra_df = pd.DataFrame({
            "파일": [x["filename"] for x in manifest],
            "실험 조건": [x.get("cond", "") for x in manifest],
            "위치": [x.get("pos", "") for x in manifest],
            "패턴": [x.get("pattern", "") for x in manifest],
            "그룹 메모": [x.get("memo", "") for x in manifest],
        })
        edited_extra = st.data_editor(
            extra_df, hide_index=True, key=f"extra_{ev}", disabled=["파일"],
            column_config={
                "실험 조건": st.column_config.TextColumn(
                    "실험 조건 (바꾼 변수의 값)",
                    help="이번 실험에서 바꾼 변수의 값을 쓰세요. 예: 7초, 9초, 11초 / 72mJ / 110°C. "
                         "한 묶음에서는 같은 변수·같은 단위로 쓰면 그래프와 제안이 만들어집니다. 고정한 나머지 조건은 '그룹 메모'에 쓰세요."),
                "패턴": st.column_config.TextColumn(help="비워 두면 측정한 피치로 자동 구분합니다."),
                "그룹 메모": st.column_config.TextColumn(help="이번 실험에서 고정한 조건. 예: IRB 110°C 90초 · 현상 60초 · 3.5mJ/s. 같은 메모끼리만 묶여 비교됩니다."),
            })
        st.caption("파일 이름에 조건이 들어 있으면 자동으로 채워집니다. 입력한 값은 [분석하기]를 누를 때 저장됩니다.")


# ------------------------------------------------------------ 분석
def analyze_all(manifest, P):
    pads_all, img_rows, imgdata = [], [], {}
    prog = st.progress(0.0, text="분석 중...")
    for i, item in enumerate(manifest):
        prog.progress((i + 1) / len(manifest), text=f"분석 중... {item['filename']}")
        path = ORIGINALS / item["stored"]
        if not path.exists():
            continue
        base = {"파일": item["filename"], "배율": mag_label(item.get("mag")), "스케일바(µm)": item.get("bar_um") or np.nan,
                "실험 조건": txt(item.get("cond")), "위치": txt(item.get("pos")), "그룹 메모": txt(item.get("memo"))}
        rec = dict(base, _stored=item["stored"], 패턴=txt(item.get("pattern")), 상태="", 안내="", 경고="", 스케일_상태="",
                   스케일_출처="", **{"µm/px": np.nan})
        try:
            img = core.load_rgb(path)
            sc = core.resolve_scale(img, item.get("mag"), item.get("bar_um"), P["umpx_table"])
            rec.update(스케일_상태=sc["state"], 스케일_출처=sc["source"], **{"µm/px": sc["um"] if sc["um"] else np.nan})
            if sc["state"] == "측정 불가":
                rec.update(상태="측정 불가", 안내=sc["notice"])
                img_rows.append(rec)
                continue
            df, pts, info = core.analyze_image(img, sc["um"], sc["bar"], P["outlier_pct"], P["ovality"], P["rough"])
            if not info["supported"]:
                rec.update(상태="지원 안 함", 안내=info["message"])
                img_rows.append(rec)
                continue
            warns = list(sc["warns"])
            if P["design_pitch"] > 0 and np.isfinite(info["pitch_um"]):
                err = info["pitch_um"] / P["design_pitch"] - 1
                if abs(err) > 0.02:
                    warns.append(f"측정한 피치({info['pitch_um']:.2f}µm)가 설계 피치보다 {err:+.1%} - 스케일을 확인하세요")
            if info["quality"]:
                warns.append(info["quality"])
            pattern = txt(item.get("pattern")) or pitch_label(info["pitch_um"])
            df["판정"] = [core.judge(v, P["target"], P["tol"], P["band"]) if c == core.C_USED else ""
                         for v, c in zip(df["CD_um"], df["분류"])]
            counted = df["분류"].isin([core.C_USED, core.C_DEFECT])
            df["pad_no"] = np.where(counted, counted.cumsum(), 0)
            used = df[df["분류"] == core.C_USED]
            n_def = int((df["분류"] == core.C_DEFECT).sum())
            in_spec = int(used["판정"].isin(["정상", "경계"]).sum())
            denom = len(used) + n_def
            mean_cd = used["CD_um"].mean() if len(used) else np.nan
            stem = Path(item["stored"]).stem
            out_name = f"{stem}_analyzed.png"
            ov = core.render_overlay(img, df, pts, sc["bar"], P["show_all"])
            Image.fromarray(ov).save(ANALYZED / out_name)
            rec.update(패턴=pattern, 상태="측정", 안내=sc["notice"], 경고=" / ".join(warns),
                       **{"사용 패드": int(len(used)), "측정 한계 제외": info["n_limit"], "불량으로 계산": n_def,
                          "무시한 작은 이물": info["n_ignored"], "평균 CD(µm)": mean_cd,
                          "CD 중앙값(µm)": used["CD_um"].median() if len(used) else np.nan,
                          "패드 SD(µm)": used["CD_um"].std() if len(used) > 1 else np.nan,
                          "수율(%)": in_spec / denom * 100 if denom else np.nan, "피치(µm)": info["pitch_um"],
                          "분석 이미지": out_name})
            img_rows.append(rec)
            df = df.assign(**{k: rec[k] for k in ("파일", "패턴", "배율", "실험 조건", "위치", "그룹 메모", "스케일_상태")})
            pads_all.append(df)
            imgdata[item["stored"]] = dict(df=df, pts=pts, out=out_name, file=item["filename"])
        except Exception as e:                                   # 한 장이 실패해도 나머지는 계속
            rec.update(상태="분석 오류", 안내=f"{type(e).__name__}: {e}")
            img_rows.append(rec)
    prog.empty()
    images = pd.DataFrame(img_rows)
    pads = pd.concat(pads_all, ignore_index=True) if pads_all else pd.DataFrame()
    groups = build_groups(pads, images, P)
    return dict(images=images, pads=pads, groups=groups, imgdata=imgdata, params=P,
                files=build_downloads(images, pads, groups, P, changed), changed=changed)


def build_groups(pads, images, P):
    if not len(pads):
        return pd.DataFrame()
    keys = ["패턴", "배율", "실험 조건", "그룹 메모", "스케일_상태"]
    pads = pads.copy()
    for k in keys:
        pads[k] = pads[k].fillna("").astype(str)
    imgs = images[images["상태"] == "측정"].copy()
    for k in keys:
        imgs[k] = imgs[k].fillna("").astype(str) if k != "스케일_상태" else imgs[k]
    rows = []
    for key, g in pads.groupby(keys, dropna=False):
        used = g[g["분류"] == core.C_USED]
        if not len(used):
            continue
        n_def = int((g["분류"] == core.C_DEFECT).sum())
        n_lim = int((g["분류"] == core.C_LIMIT).sum())
        in_spec = int(used["판정"].isin(["정상", "경계"]).sum())
        denom = len(used) + n_def
        mean_cd = used["CD_um"].mean()
        sel = imgs
        for k, v in zip(keys, key):
            sel = sel[sel[k] == v]
        img_means = sel["평균 CD(µm)"].dropna()
        pos_means = used[used["위치"] != ""].groupby("위치")["CD_um"].mean()
        label = " · ".join([x for x in (key[0], key[1], key[2], key[3]) if x])
        tone_j = core.judge(mean_cd, P["target"], P["tol"], P["band"])
        rows.append({"조건": label or "조건 미입력", "패턴": key[0], "배율": key[1], "실험 조건": key[2], "그룹 메모": key[3],
                     "스케일": key[4], "이미지 수": len(sel), "사용 패드": len(used), "측정 한계 제외": n_lim,
                     "불량으로 계산": n_def, "평균 CD(µm)": mean_cd, "패드 SD(µm)": used["CD_um"].std() if len(used) > 1 else np.nan,
                     "이미지 간 편차(µm)": img_means.std() if len(img_means) > 1 else np.nan,
                     "위치 간 편차(µm)": float(pos_means.max() - pos_means.min()) if len(pos_means) > 1 else np.nan,
                     "목표와 차이(µm)": mean_cd - P["target"], "수율(%)": in_spec / denom * 100 if denom else np.nan,
                     "신호": {"정상": "🟢", "경계": "🟡"}.get(tone_j, "🔴")})
    return pd.DataFrame(rows)


def _round_cols(df, spec):
    """CSV에 적는 값만 반올림합니다(화면·계산에 쓰는 원래 값은 바꾸지 않음)."""
    out = df.copy()
    for col, nd in spec.items():
        if col in out.columns:
            v = pd.to_numeric(out[col], errors="coerce").round(nd)
            out[col] = v.astype("Int64") if nd == 0 else v
    return out


def build_settings(P, changed_list, n_images):
    """분석 조건 기록(settings.csv). 한 줄에 항목 하나."""
    lo_, hi_ = P["target"] - P["tol"], P["target"] + P["tol"]
    rows = [
        ("분석", "프로그램", "Pattern CD Analyzer"),
        ("분석", "분석 날짜", datetime.datetime.now().strftime("%Y-%m-%d %H:%M")),
        ("분석", "분석한 이미지 수", f"{n_images}장"),
        ("판정", "목표 CD", f"{P['target']:.2f} µm"),
        ("판정", "허용오차", f"± {P['tol']:.2f} µm"),
        ("판정", "정상 범위", f"{lo_:.2f} ~ {hi_:.2f} µm"),
        ("판정", "허용 한계 근처 폭", f"{P['band']:.2f} µm (경계로 표시, 수율에서는 정상으로 계산)"),
        ("측정", "경계 기준", f"까만 테두리 바깥 끝 (가장 어두운 곳에서 배경 쪽으로 {core.BOUNDARY_FRAC * 100:.0f}% 밝아지는 지점, 고정)"),
        ("측정", "지름 계산", "패드 중심에서 180방향으로 경계를 잡고, 마주 보는 두 방향을 더한 지름 90개의 평균"),
        ("제외", "측정 한계 제외(수율에서 뺌)", "이미지 가장자리에 잘림 / 가장자리 근처 / 스케일바·주석 근처"),
        ("제외", "불량으로 계산(수율 분모에 포함)",
         f"크기 이상(합쳐짐·이물) / 경계 검출 실패 / 이상치(중앙값 ±{P['outlier_pct']:g}%와 중앙 절대편차 3배 중 큰 값)"),
        ("제외", "패드가 아닌 작은 이물", "패드 크기의 절반 미만, 수율에 넣지 않음"),
        ("수율", "정의", "CD 규격 기준 치수 수율(누락된 패드는 세지 못함) = 허용범위 안(정상+경계) ÷ (사용한 패드 + 불량으로 계산한 패드)"),
        ("형상", "검토 기준", f"비대칭(최대/최소 지름) {P['ovality']:g}, 반지름 변동 {P['rough']:g}% - 표시만 하고 수율에는 반영하지 않음"),
    ]
    for m in core.MAGS:
        rows.append(("스케일", f"배율 ×{m} 기본 환산율", f"{P['umpx_table'][m]:.4f} µm/px ({core.UMPX_NOTE.get(m, '')})"))
    rows.append(("스케일", "스케일바가 없을 때", "배율 기본값으로 계산하고 '추정'으로 표시, 둘 다 없으면 측정 불가"))
    if P["design_pitch"] > 0:
        rows.append(("스케일", "설계 피치(검산용)", f"{P['design_pitch']:g} µm"))
    for key, value in P.get("process_conditions", {}).items():
        rows.append(("공정 조건", key, value))
    rows.append(("변경", "기본값에서 바꾼 설정", ", ".join(changed_list) if changed_list else "없음"))
    return pd.DataFrame(rows, columns=["구분", "항목", "값"])


def build_downloads(images, pads, groups, P, changed_list):
    out = {"pads": b""}                                       # 측정된 패드가 하나도 없어도 내려받기가 깨지지 않게 기본값을 둠
    cols = ["파일", "패턴", "배율", "스케일_상태", "실험 조건", "위치", "그룹 메모", "pad_no", "분류", "reason", "CD_um",
            "D_min_um", "D_max_um", "ovality", "rough_pct", "판정", "형상 검토", "cx", "cy"]
    if len(pads):
        pc = pads[[c for c in cols if c in pads.columns]].rename(columns={
            "스케일_상태": "스케일 상태", "reason": "사유", "CD_um": "CD(µm)", "D_min_um": "최소 지름(µm)", "D_max_um": "최대 지름(µm)",
            "ovality": "최대/최소 지름", "rough_pct": "반지름 변동(%)", "cx": "중심 x(px)", "cy": "중심 y(px)"})
        pc = _round_cols(pc, {"CD(µm)": 2, "최소 지름(µm)": 2, "최대 지름(µm)": 2, "최대/최소 지름": 2, "반지름 변동(%)": 1,
                              "중심 x(px)": 0, "중심 y(px)": 0})
        out["pads"] = pc.to_csv(index=False).encode("utf-8-sig")
    ic = images.drop(columns=[c for c in ("_stored",) if c in images.columns]).rename(
        columns={"스케일_상태": "스케일 상태", "스케일_출처": "스케일 출처"})
    ic = _round_cols(ic, {"µm/px": 4, "평균 CD(µm)": 2, "CD 중앙값(µm)": 2, "패드 SD(µm)": 2, "수율(%)": 0, "피치(µm)": 2})
    out["images"] = ic.to_csv(index=False).encode("utf-8-sig")
    gc = _round_cols(groups, {"평균 CD(µm)": 2, "패드 SD(µm)": 2, "이미지 간 편차(µm)": 2, "위치 간 편차(µm)": 2,
                              "목표와 차이(µm)": 2, "수율(%)": 0}) if len(groups) else groups
    out["groups"] = (gc.to_csv(index=False) if len(gc) else "").encode("utf-8-sig")
    out["settings"] = build_settings(P, changed_list, int((images["상태"] == "측정").sum()) if len(images) else 0
                                     ).to_csv(index=False).encode("utf-8-sig")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for k, name in (("pads", "pads.csv"), ("images", "images.csv"), ("groups", "summary.csv"), ("settings", "settings.csv")):
            zf.writestr(name, out[k])
        for p in ANALYZED.glob("*_analyzed.png"):
            zf.write(p, f"analyzed_images/{p.name}")
    out["zip"] = buf.getvalue()
    return out


P_now = dict(target=target_cd, tol=tolerance_um, band=band, outlier_pct=outlier_pct, ovality=ovality_lim, rough=rough_lim,
             design_pitch=design_pitch, show_all=show_all, umpx_table=umpx_table,
             process_conditions=process_conditions)

if manifest:
    if st.button("분석하기", type="primary"):
        for i, x in enumerate(manifest):
            x["mag"] = mag_from_label(edited_main.iloc[i]["배율"])
            x["bar_um"] = num_or_none(edited_main.iloc[i]["스케일바(µm)"])
            x["cond"] = txt(edited_extra.iloc[i]["실험 조건"])
            x["pos"] = txt(edited_extra.iloc[i]["위치"])
            x["pattern"] = txt(edited_extra.iloc[i]["패턴"])
            x["memo"] = txt(edited_extra.iloc[i]["그룹 메모"])
        write_json(MANIFEST, manifest)
        st.session_state["res"] = analyze_all(manifest, P_now)

# ------------------------------------------------------------ 결과
TONE = {"good": ("#E3F4EA", "#176B3A"), "warn": ("#FFF0D6", "#9A5B00"), "bad": ("#FCE4E4", "#A61B1B"),
        "neutral": ("#EEF1F5", "#2B2F36")}


def card(title, value, sub="", note="", tone="neutral"):
    bg, fg = TONE[tone]
    return (f"<div style='background:{bg};border-radius:12px;padding:14px 16px;height:100%'>"
            f"<div style='font-size:0.85rem;color:#4a4f57'>{html.escape(title)}</div>"
            f"<div style='font-size:2.1rem;font-weight:700;color:{fg};line-height:1.25'>{html.escape(value)}</div>"
            f"<div style='font-size:0.82rem;color:#3d4249'>{html.escape(sub)}</div>"
            f"<div style='font-size:0.72rem;color:#5b616a;margin-top:4px'>{html.escape(note)}</div></div>")


def tone_of(mean, P):
    j = core.judge(mean, P["target"], P["tol"], P["band"])
    return {"정상": "good", "경계": "warn"}.get(j, "bad")


def show_cards(g, P):
    sd = g["패드 SD(µm)"]
    sub_cd = f"패드 간 편차 ±{sd:.2f} µm" if pd.notna(sd) else ""
    if pd.notna(g["이미지 간 편차(µm)"]):
        sub_cd += f" · 이미지 간 ±{g['이미지 간 편차(µm)']:.2f}"
    if pd.notna(g["위치 간 편차(µm)"]):
        sub_cd += f" · 위치 간 {g['위치 간 편차(µm)']:.2f}"
    tone = tone_of(g["평균 CD(µm)"], P)
    diff = g["목표와 차이(µm)"]
    mark = {"good": "✓ ", "warn": "△ ", "bad": "✕ "}[tone]
    word = "목표와 같음" if abs(diff) < 0.005 else ("목표보다 큼" if diff > 0 else "목표보다 작음")
    cols = st.columns(4)
    cols[0].markdown(card("평균 CD", f"{g['평균 CD(µm)']:.2f} µm", sub_cd, "경계 기준: 까만 테두리 바깥 끝", tone), unsafe_allow_html=True)
    cols[1].markdown(card("목표와 차이", f"{mark}{diff:+.2f} µm", f"{word} · 허용 ±{P['tol']:g} µm", "", tone), unsafe_allow_html=True)
    y = g["수율(%)"]
    cols[2].markdown(card("수율", f"{y:.0f} %" if pd.notna(y) else "-", f"규격 {P['target'] - P['tol']:.2f}~{P['target'] + P['tol']:.2f} µm",
                          "CD 규격 기준 치수 수율(누락된 패드는 세지 못함)"), unsafe_allow_html=True)
    few = g["사용 패드"] < 7
    cols[3].markdown(card("측정 패드 수", f"{int(g['사용 패드'])} 개",
                          f"제외 {int(g['측정 한계 제외'] + g['불량으로 계산'])}개 (측정 한계 {int(g['측정 한계 제외'])} · 불량 {int(g['불량으로 계산'])})",
                          "패드가 적어 숫자를 믿기 어렵습니다" if few else "", "warn" if few else "neutral"), unsafe_allow_html=True)


def cond_axis(conds):
    """모든 조건이 '숫자+같은 단위'이면 숫자축, 아니면 글자축"""
    parsed = []
    for c in conds:
        m = re.match(r"^\s*(-?\d+(?:\.\d+)?)\s*(.*)$", str(c))
        if not m:
            return None
        parsed.append((float(m.group(1)), m.group(2).strip()))
    units = {u for _, u in parsed}
    return (dict(zip(conds, [p[0] for p in parsed])), units.pop()) if len(units) == 1 else None


def group_chart(imgs, groups, P):
    g = groups[groups["실험 조건"] != ""].copy()
    g["스케일_상태"] = g["스케일"]
    conds = list(dict.fromkeys(g["실험 조건"]))
    ax = cond_axis(conds)
    im = imgs[(imgs["상태"] == "측정") & (imgs["실험 조건"] != "")].copy()
    im["계열"] = im["패턴"].astype(str) + " " + im["배율"].astype(str)
    g["계열"] = g["패턴"].astype(str) + " " + g["배율"].astype(str)
    if ax:
        mp, unit = ax
        im["x"], g["x"] = im["실험 조건"].map(mp), g["실험 조건"].map(mp)
        allx = list(im["x"]) + list(g["x"])
        span = max(max(allx) - min(allx), 1e-9)
        pad = span * 0.08 if span > 1e-6 else 0.5
        xenc = alt.X("x:Q", title=f"실험 조건 ({unit})" if unit else "실험 조건",
                     scale=alt.Scale(domain=[min(allx) - pad, max(allx) + pad], nice=False),
                     axis=alt.Axis(values=sorted(set(allx))))
    else:
        im["x"], g["x"] = im["실험 조건"], g["실험 조건"]
        xenc = alt.X("x:N", sort=conds, title="실험 조건")
    ys = list(im["평균 CD(µm)"]) + list(g["평균 CD(µm)"]) + [P["target"] - P["tol"], P["target"] + P["tol"]]
    ydom = [min(ys) - 0.3, max(ys) + 0.3]
    yenc = alt.Y("평균 CD(µm):Q", title="CD (µm)", scale=alt.Scale(domain=ydom))
    shape = alt.Shape("스케일_상태:N", scale=alt.Scale(domain=["정확", "추정"], range=["circle", "diamond"]), title="스케일")
    color = alt.Color("계열:N", title="패턴·배율")
    band = alt.Chart(pd.DataFrame({"a": [P["target"] - P["tol"]], "b": [P["target"] + P["tol"]]})).mark_rect(
        opacity=0.13, color="#2e9e6b").encode(y=alt.Y("a:Q", scale=alt.Scale(domain=ydom)), y2="b:Q")
    pts = alt.Chart(im).mark_point(size=60, filled=True, opacity=0.5).encode(
        x=xenc, y=yenc, color=color, shape=shape, tooltip=["파일", "실험 조건", alt.Tooltip("평균 CD(µm):Q", format=".3f")])
    layers = [band, pts]
    g_acc, g_est = g[g["스케일_상태"] == "정확"], g[g["스케일_상태"] == "추정"]
    if len(g_acc):
        layers.append(alt.Chart(g_acc).mark_line(point=alt.OverlayMarkDef(size=130, filled=True)).encode(x=xenc, y=yenc, color=color))
    if len(g_est):                                              # 추정 평균은 선 없이 속이 빈 마름모로만 표시
        layers.append(alt.Chart(g_est).mark_point(size=150, shape="diamond", filled=False, strokeWidth=2.5).encode(
            x=xenc, y=yenc, color=color))
    return alt.layer(*layers).properties(height=320)


def hist_chart(cd, P):
    d = pd.DataFrame({"CD": cd})
    x0, x1 = min(d["CD"].min(), P["target"] - P["tol"]) - 0.3, max(d["CD"].max(), P["target"] + P["tol"]) + 0.3
    sc = alt.Scale(domain=[x0, x1])
    bars = alt.Chart(d).mark_bar(color="#5b8def").encode(
        x=alt.X("CD:Q", bin=alt.Bin(maxbins=20, extent=[x0, x1]), title="패드별 CD (µm)", scale=sc), y=alt.Y("count()", title="패드 수", axis=alt.Axis(tickMinStep=1)))
    band = alt.Chart(pd.DataFrame({"a": [P["target"] - P["tol"]], "b": [P["target"] + P["tol"]]})).mark_rect(
        opacity=0.15, color="#2e9e6b").encode(x=alt.X("a:Q", scale=sc), x2="b:Q")
    rule = alt.Chart(pd.DataFrame({"t": [P["target"]]})).mark_rule(color="#2e9e6b", strokeDash=[4, 3]).encode(x=alt.X("t:Q", scale=sc))
    return (band + bars + rule).properties(height=230)


def show_image_card(row, res, k):
    P, imgdata = res["params"], res["imgdata"]
    with st.container(border=True):
        if row["상태"] != "측정":
            st.markdown(f"**⚪ {row['파일']}**")
            (st.error if row["상태"] == "분석 오류" else st.warning)(f"{row['상태']}: {row['안내']}")
            return
        mean = row["평균 CD(µm)"]
        tone = tone_of(mean, P) if pd.notna(mean) else "bad"
        sig = {"good": "🟢", "warn": "🟡", "bad": "🔴"}[tone]
        tag = "📏 스케일 정확" if row["스케일_상태"] == "정확" else "⚠ 스케일 추정"
        a, b = st.columns([1, 3])
        with a:
            st.image(str(ANALYZED / row["분석 이미지"]), width=260)
        with b:
            st.markdown(f"**{sig} {row['파일']}** · {tag}")
            st.markdown(f"CD **{mean:.2f} µm** (목표와 {mean - P['target']:+.2f}) · 수율 **{row['수율(%)']:.0f}%** · 사용 {int(row['사용 패드'])}개")
            if row["스케일_상태"] == "추정" and row["안내"]:
                st.warning(row["안내"])
            if row["경고"]:
                st.caption("⚠ " + row["경고"])
            idata = imgdata.get(row["_stored"])
            nshape = int((idata["df"]["형상 검토"] != "").sum()) if idata else 0
            if nshape:
                st.caption(f"형상 검토 표시 패드 {nshape}개 (비대칭 또는 반지름 변동 큼) - 자세히 보기의 표에서 확인하세요")
        with st.expander("자세히 보기"):
            if not idata:
                return
            df = idata["df"]
            st.caption(core.LEGEND)
            if core.numbers_suppressed(df, P["show_all"]):
                st.caption("규격 밖 패드가 많아 그림에는 CD 숫자를 표시하지 않았습니다. 아래 표에서 확인하세요.")
            box = st.container()
            tbl = df[df["분류"].isin([core.C_USED, core.C_DEFECT])][["pad_no", "분류", "reason", "CD_um", "판정", "형상 검토"]].rename(
                columns={"분류": "구분", "reason": "사유", "CD_um": "CD(µm)", "pad_no": "번호"})
            sel_idx = None
            h1, h2 = st.columns([2, 1])
            with h2:
                used = df[df["분류"] == core.C_USED]["CD_um"]
                if len(used):
                    st.altair_chart(hist_chart(used, P))
                ex = row.get("무시한 작은 이물", 0)
                st.caption(f"제외: 측정 한계 {int(row['측정 한계 제외'])}개 · 불량으로 계산 {int(row['불량으로 계산'])}개"
                           + (f" · 패드가 아닌 작은 이물 {int(ex)}개(수율에 넣지 않음)" if ex else ""))
            with h1:
                st.caption("표에서 행을 누르면 그 패드가 그림에서 강조됩니다.")
                ev_ = st.dataframe(tbl, hide_index=True, on_select="rerun", selection_mode="single-row", key=f"sel_{k}",
                                   column_config={"CD(µm)": st.column_config.NumberColumn(format="%.2f")})
                rows = getattr(getattr(ev_, "selection", None), "rows", None) or []
                if rows:
                    sel_idx = int(tbl.index[rows[0]])
            with box:
                if sel_idx is not None:
                    img = core.load_rgb(ORIGINALS / row["_stored"])
                    bar = core.detect_scale_bar(img)
                    st.image(core.render_overlay(img, df, idata["pts"], bar, P["show_all"], highlight=sel_idx))
                else:
                    st.image(str(ANALYZED / row["분석 이미지"]))


def show_suggestion(groups, P):
    """큰 숫자(또는 조건별 표) 바로 아래에 제안 한 줄과 접힌 근거를 보여줍니다."""
    sg = core.suggest(groups, P["target"], P["tol"])
    if not sg["items"]:
        st.caption("💡 " + sg["guidance"])
        for n in sg["notes"]:
            st.caption("· " + n)
        return
    many = len(sg["items"]) > 1
    for k, it in enumerate(sg["items"]):
        prefix = f"[{it['label']}] " if many and it["label"] else ""
        if it["headline"]:
            st.info(f"💡 제안: {prefix}{it['headline']}")
        else:
            st.caption(f"💡 제안 없음: {prefix}{it['reason']}")
        st.caption(it["nearest"])
        with st.expander("근거 보기"):
            st.dataframe(it["table"], hide_index=True, column_config={
                "평균 CD(µm)": st.column_config.NumberColumn(format="%.2f"),
                "이미지 간 편차(µm)": st.column_config.NumberColumn(format="%.2f"),
                "수율(%)": st.column_config.NumberColumn(format="%.0f")})
            if it["trend"]:
                st.caption(it["trend"])
            for n in it["notes"] + (sg["notes"] if k == 0 else []):
                st.caption("· " + n)
            st.caption(core.SUGGEST_NOTE)



def build_rule_based_review(res):
    """측정 결과만으로 생성하는 규칙 기반 총평. 인과관계나 미측정 물성은 단정하지 않는다."""
    P, imgs, groups, pads = res["params"], res["images"], res["groups"], res["pads"]
    target, tol = float(P["target"]), float(P["tol"])
    lo, hi = target - tol, target + tol
    measured = imgs[imgs["상태"] == "측정"] if len(imgs) else pd.DataFrame()
    valid_groups = groups.copy() if len(groups) else pd.DataFrame()
    used = pads[pads["분류"] == core.C_USED] if len(pads) and "분류" in pads else pd.DataFrame()
    lines = []
    lines.append(f"분석 결과 {len(measured)}장의 이미지에서 측정이 완료되었으며, 목표 CD는 {target:.2f} µm, 허용 범위는 {lo:.2f}~{hi:.2f} µm입니다.")
    if len(used):
        mean_cd = float(used["CD_um"].mean())
        median_cd = float(used["CD_um"].median())
        sd_cd = float(used["CD_um"].std()) if len(used) > 1 else float("nan")
        delta = mean_cd - target
        state = "목표보다 큰 편" if delta > tol else "목표보다 작은 편" if delta < -tol else "허용 범위 안"
        lines.append(f"사용 패드 {len(used)}개의 전체 평균 CD는 {mean_cd:.2f} µm(중앙값 {median_cd:.2f} µm, 패드 간 표준편차 {sd_cd:.2f} µm)이며, 목표 대비 {delta:+.2f} µm로 {state}입니다.")
        if len(valid_groups):
            best = valid_groups.sort_values(["수율(%)", "이미지 간 편차(µm)"], ascending=[False, True], na_position="last").iloc[0]
            lines.append(f"조건별로는 '{best['조건']}'에서 CD 규격 수율 {best['수율(%)']:.1f}%가 관찰되었습니다. 이 조건이 최적이라고 확정하기보다는 반복 측정으로 재현성을 확인할 후보로 보는 것이 적절합니다.")
            if valid_groups["위치 간 편차(µm)"].notna().any():
                posrow = valid_groups.loc[valid_groups["위치 간 편차(µm)"].idxmax()]
                lines.append(f"위치 간 편차가 가장 큰 조건은 '{posrow['조건']}'이며 편차는 {posrow['위치 간 편차(µm)']:.2f} µm입니다. Center·Left-Upper·Right-Down의 위치별 평균을 확인해 웨이퍼/기판 내 균일도 문제인지 점검하세요.")
        else:
            lines.append("조건별 비교 데이터가 부족해 어떤 공정 조건이 더 우수한지 판단하기 어렵습니다.")
    else:
        lines.append("규격 판정에 사용할 측정 패드가 없습니다. 이미지 초점, 스케일바 보정, 패턴 검출 상태를 먼저 확인해야 합니다.")

    n_est = int((imgs["스케일_상태"] == "추정").sum()) if len(imgs) else 0
    n_bad = int((imgs["상태"] != "측정").sum()) if len(imgs) else 0
    if n_est:
        lines.append(f"측정 이미지 중 {n_est}장은 배율 기본값으로 스케일을 추정했습니다. 해당 결과는 정량 비교의 신뢰도가 낮을 수 있으므로 스케일바가 포함된 이미지로 재보정한 뒤 결론을 확정하세요.")
    if n_bad:
        lines.append(f"총 {n_bad}장의 이미지는 측정 완료 상태가 아닙니다. 제외 사유를 확인하고, 분석 가능한 이미지 수가 충분한지 검토하세요.")
    if len(used) < 30:
        lines.append("표본 수가 많지 않으므로 평균 CD와 수율은 예비 결과로 해석하고, 조건별 반복 측정을 추가하는 것이 좋습니다.")
    lines.append("주의: 광학 2D 이미지의 CD만으로 레지스트 막 두께, 측벽 형상, 언더컷 또는 공정 변수의 인과관계를 직접 판정할 수 없습니다.")

    pc = P.get("process_conditions", {})
    recs = []
    # 관찰된 CD 편차에 따라 우선 실험 변수를 제시하되, 레지스트 톤/마스크 형상 미확인 시 변화 방향은 가설로 한정한다.
    if len(used):
        mean_cd = float(used["CD_um"].mean())
        if mean_cd > hi:
            direction = "CD가 목표보다 큽니다. 노광량을 낮추는 방향을 우선 검토하되, Positive/Negative tone과 패턴의 밝고 어두운 영역에 따라 CD 반응 방향이 달라질 수 있으므로 현재 조건의 ±5~10% 범위에서 소규모 확인 실험을 권장합니다."
        elif mean_cd < lo:
            direction = "CD가 목표보다 작습니다. 노광량을 높이는 방향을 우선 검토하되, 레지스트 톤과 마스크 패턴에 따라 반응 방향이 달라질 수 있으므로 현재 조건의 ±5~10% 범위에서 소규모 확인 실험을 권장합니다."
        else:
            direction = "평균 CD는 허용 범위 안입니다. 평균을 더 움직이기보다 위치 간 편차와 패드 간 산포를 줄이는 방향으로 공정 안정성을 확인하는 것이 우선입니다."
        recs.append(("1순위 · Exposure", f"현재 {pc.get('exposure_time_s', '미입력')} s", direction + " 한 번에 노광 시간만 바꾸고 나머지 7개 변수는 고정하세요."))
    recs.append(("2순위 · Spin coating", f"현재 {pc.get('spin_rpm', '미입력')} rpm", "막 두께와 패턴 형상에 영향을 줄 수 있으므로 현재 회전속도 기준 ±5% 수준의 2~3점 비교를 권장합니다. CD 변화와 함께 막 두께 측정이 가능하면 병행하세요."))
    recs.append(("3순위 · Soft bake", f"현재 {pc.get('soft_bake_temp_c', '미입력')} °C / {pc.get('soft_bake_time_s', '미입력')} s", "용매 제거 및 레지스트 상태에 영향을 줄 수 있습니다. 온도 또는 시간 중 하나만 선택해 현재값 주변의 작은 간격(예: 온도 ±5 °C 또는 시간 ±10 s)으로 비교하세요."))
    recs.append(("4순위 · IRB", f"현재 {pc.get('irb_temp_c', '미입력')} °C / {pc.get('irb_time_s', '미입력')} s", "IRB의 정확한 공정 정의와 레지스트 재료 사양을 확인한 뒤 조절하세요. 우선 온도 또는 시간 한 변수만 현재값 주변(예: ±5 °C 또는 ±10 s)에서 비교합니다."))
    recs.append(("5순위 · Development", f"현재 {pc.get('development_time_s', '미입력')} s", "현상 부족/과다 여부를 현상 후 잔사와 패턴 손실로 함께 확인하세요. 현재 시간 기준 ±5~10 s 범위의 소규모 비교를 권장하며, CD만 보고 방향을 단정하지 마세요."))
    recs.append(("유지 권장 · DI rinse", f"현재 {pc.get('di_rinse_time_s', '미입력')} s", "DI 세척은 CD 조절용 변수로 먼저 사용하지 말고, 잔류 현상액 제거가 충분한지 확인한 뒤 현재 조건을 고정하는 편이 좋습니다."))
    return lines, recs


def show_rule_based_review(res):
    lines, recs = build_rule_based_review(res)
    st.divider()
    st.subheader("📋 데이터 기반 자동 총평")
    st.caption("외부 API·유료 AI 없이 측정값과 사전 정의된 분석 규칙으로 생성합니다.")
    for line in lines:
        st.write("• " + line)
    st.subheader("🧪 추천 공정변수 조절 방향")
    st.warning("아래 조절 폭은 보편적인 최적 조건이 아니라 다음 실험을 설계하기 위한 초기 탐색 범위입니다. 레지스트 종류·톤·장비·공정 사양을 확인하고, 안전한 장비 운전 범위 안에서 적용하세요.")
    for title, current, advice in recs:
        with st.expander(f"{title}  |  {current}", expanded=(title.startswith("1순위"))):
            st.write(advice)
    st.markdown("**권장 실험 순서:** 기준 조건 반복 측정 → Exposure만 변경 → 가장 유망한 노광 조건 고정 → Spin coating 또는 Soft bake 중 한 변수만 변경 → IRB/Development 순으로 검토. 각 단계에서 최소 3회 반복하고 평균 CD, 표준편차, 위치 간 편차, 수율을 함께 비교하세요.")


def show_results(res):
    P, imgs, groups = res["params"], res["images"], res["groups"]
    st.divider()
    if (P["target"], P["tol"]) != (target_cd, tolerance_um):
        st.info("목표 CD 또는 허용오차가 바뀌었습니다. [분석하기]를 다시 누르면 반영됩니다.")
    n_est = int((imgs["스케일_상태"] == "추정").sum()) if len(imgs) else 0
    n_no = int((imgs["상태"] == "측정 불가").sum()) if len(imgs) else 0
    n_un = int((imgs["상태"] == "지원 안 함").sum()) if len(imgs) else 0
    if n_est:
        st.warning(f"스케일을 추정한 이미지가 {n_est}장 있습니다. 배율 기본값으로 계산한 값이라 정확도가 낮을 수 있고, 요약에서 따로 표시했습니다.")
    if n_no:
        st.warning(f"스케일바와 배율이 모두 없어 측정하지 못한 이미지가 {n_no}장 있습니다.")
    if n_un:
        st.warning(f"원형 패드 배열이 아니어서 측정하지 않은 이미지가 {n_un}장 있습니다. (현재는 원형 패드 배열만 지원합니다)")
    if res["changed"]:
        st.caption("※ 기본값에서 바꾼 설정: " + ", ".join(res["changed"]))
    if not len(groups):
        st.info("측정된 이미지가 없습니다. 아래 이미지별 안내를 확인하세요.")
    elif len(groups) == 1:
        g = groups.iloc[0]
        if g["스케일"] == "추정":
            st.warning("이 결과는 배율 기본값으로 계산한 추정치입니다.")
        show_cards(g, P)
        st.caption(f"목표 {P['target']:g} µm ± {P['tol']:g} µm · 경계 기준: 까만 테두리 바깥 끝(25% 지점)")
        show_suggestion(groups, P)
    else:
        st.subheader("조건별 결과")
        t = groups[["신호", "조건", "스케일", "평균 CD(µm)", "목표와 차이(µm)", "수율(%)", "사용 패드", "이미지 간 편차(µm)"]].copy()
        t["스케일"] = t["스케일"].map({"정확": "정확", "추정": "⚠ 추정"})
        st.dataframe(t, hide_index=True, column_config={
            "평균 CD(µm)": st.column_config.NumberColumn(format="%.2f"), "목표와 차이(µm)": st.column_config.NumberColumn(format="%+.2f"),
            "수율(%)": st.column_config.NumberColumn(format="%.0f"), "이미지 간 편차(µm)": st.column_config.NumberColumn(format="%.2f")})
        if groups["패턴"].nunique() > 1:
            st.caption("목표 CD는 하나만 입력되므로, 패턴이 다르면 목표와의 차이와 판정은 해당 패턴에만 의미가 있습니다.")
        st.caption(f"목표 {P['target']:g} µm ± {P['tol']:g} µm · 수율은 CD 규격 기준 치수 수율(누락된 패드는 세지 못함)")
        show_suggestion(groups, P)
        if groups.loc[groups["실험 조건"] != "", "실험 조건"].nunique() >= 2:
            st.altair_chart(group_chart(imgs, groups, P))
            st.caption("점: 이미지 1장의 평균 CD · 선: 조건 평균 · 초록 띠: 허용 범위 · ◆: 스케일 추정치 · 세로축은 0에서 시작하지 않습니다.")

    show_rule_based_review(res)

    st.subheader("이미지별 결과")
    for k, (_, row) in enumerate(imgs.iterrows()):
        show_image_card(row, res, k)

    with st.popover("⬇ 내려받기"):
        st.download_button("결과 전체 (zip: CSV 3개 + settings.csv + 분석 이미지)", res["files"]["zip"], "pattern_cd_results.zip", "application/zip")
        st.download_button("패드별 CSV", res["files"].get("pads", b""), "pads.csv", "text/csv")
        st.download_button("이미지별 CSV", res["files"]["images"], "images.csv", "text/csv")
        st.download_button("조건별 요약 CSV", res["files"]["groups"], "summary.csv", "text/csv")


if "res" in st.session_state:
    show_results(st.session_state["res"])

st.caption("판정은 입력한 목표 CD ± 허용오차에 따른 선별이며 공정 불량의 확정이 아닙니다. 광학 이미지로는 단면(언더컷)을 볼 수 없고, "
           "CD는 경계 기준(까만 테두리 바깥 끝)에 따라 달라집니다. 현재는 원형 패드 배열만 지원합니다.")


