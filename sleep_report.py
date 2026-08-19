"""睡眠分析報告產生器 (Streamlit + Gemini)

支援兩種輸入：
1. 穿戴裝置的表格檔（CSV / Excel / JSON）——多天趨勢分析。
2. PSG 原始檔（EDF / EDF+ / BDF）——單晚睡眠結構分析；分期可來自檔內註記、
   另外上傳的 Hypnogram 檔，或用 YASA 自動分期。註記含呼吸事件時會一併算 AHI / ODI。

使用方式:
    pip install -r requirements.txt
    export GEMINI_API_KEY="your-key"      # 或填入側邊欄 / .streamlit/secrets.toml
    streamlit run sleep_report.py
"""

from __future__ import annotations

import io
import json
import os
import re
from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st

import edf_utils

MODEL_OPTIONS = ["gemini-3.5-flash", "gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.1-pro-preview"]
DEFAULT_MODEL = "gemini-3.5-flash"

# 欄位自動辨識用的關鍵字（依序比對，先中後英）
COLUMN_HINTS = {
    "date": ["日期", "date", "day", "start_time", "bedtime", "上床"],
    "total": ["總睡眠", "睡眠時間", "total_sleep", "sleep_duration", "duration", "asleep", "時數"],
    "deep": ["深睡", "深層", "deep"],
    "light": ["淺睡", "淺層", "light"],
    "rem": ["rem", "快速動眼"],
    "awake": ["清醒", "醒來", "awake", "wake"],
    "score": ["分數", "評分", "score", "quality"],
    "hr": ["心率", "heart", "hr", "bpm"],
}

st.set_page_config(page_title="睡眠分析報告", page_icon="🌙", layout="wide")


# --------------------------------------------------------------------------- #
# 資料處理
# --------------------------------------------------------------------------- #
def load_file(uploaded) -> pd.DataFrame:
    """讀取 CSV / Excel / JSON 上傳檔為 DataFrame。"""
    name = uploaded.name.lower()
    data = uploaded.getvalue()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(data))
    if name.endswith(".json"):
        parsed = json.loads(data.decode("utf-8"))
        if isinstance(parsed, dict):
            # 常見格式: {"records": [...]} 或 {"data": [...]}
            for key in ("records", "data", "sleep", "items"):
                if isinstance(parsed.get(key), list):
                    parsed = parsed[key]
                    break
        return pd.json_normalize(parsed)
    for enc in ("utf-8-sig", "utf-8", "big5", "cp950"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(io.BytesIO(data), encoding="utf-8", encoding_errors="replace")


def guess_column(df: pd.DataFrame, key: str) -> str | None:
    """依關鍵字猜測某個語意欄位對應的實際欄名。"""
    for hint in COLUMN_HINTS[key]:
        for col in df.columns:
            if hint in str(col).lower().replace(" ", ""):
                return col
    return None


def to_minutes(value) -> float | None:
    """把各種睡眠長度表示法統一轉成分鐘。

    支援 450 (分)、7.5 (小時)、"7h30m"、"7:30"、"7 小時 30 分"。
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        return v * 60 if v <= 24 else v  # <=24 視為小時
    text = str(value).strip()
    if not text:
        return None
    if ":" in text:  # 7:30
        parts = text.split(":")
        try:
            return int(parts[0]) * 60 + int(parts[1])
        except ValueError:
            return None
    hours = re.search(r"(\d+(?:\.\d+)?)\s*(?:h|hr|hours?|小時|時)", text, re.I)
    mins = re.search(r"(\d+(?:\.\d+)?)\s*(?:m|min|minutes?|分)", text, re.I)
    if hours or mins:
        return (float(hours.group(1)) * 60 if hours else 0) + (float(mins.group(1)) if mins else 0)
    try:
        return to_minutes(float(text))
    except ValueError:
        return None


def build_stats(df: pd.DataFrame, mapping: dict[str, str | None]) -> dict:
    """計算送進模型與顯示用的統計摘要。"""
    stats: dict = {"記錄天數": int(len(df))}

    date_col = mapping.get("date")
    if date_col:
        dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
        if not dates.empty:
            stats["資料期間"] = f"{dates.min():%Y-%m-%d} ~ {dates.max():%Y-%m-%d}"

    for key, label in [
        ("total", "總睡眠"),
        ("deep", "深睡"),
        ("light", "淺睡"),
        ("rem", "REM"),
        ("awake", "清醒"),
    ]:
        col = mapping.get(key)
        if not col:
            continue
        series = df[col].map(to_minutes).dropna()
        if series.empty:
            continue
        stats[f"{label}平均(分鐘)"] = round(series.mean(), 1)
        stats[f"{label}標準差(分鐘)"] = round(series.std(ddof=0), 1)
        stats[f"{label}最短(分鐘)"] = round(series.min(), 1)
        stats[f"{label}最長(分鐘)"] = round(series.max(), 1)

    total_col = mapping.get("total")
    if total_col:
        mins = df[total_col].map(to_minutes).dropna()
        if not mins.empty:
            stats["睡足7小時天數比例"] = f"{(mins >= 420).mean():.0%}"
            stats["少於6小時天數"] = int((mins < 360).sum())
        deep_col = mapping.get("deep")
        if deep_col and not mins.empty:
            deep = df[deep_col].map(to_minutes)
            ratio = (deep / df[total_col].map(to_minutes)).dropna()
            if not ratio.empty:
                stats["深睡佔比平均"] = f"{ratio.mean():.1%}"

    for key, label in [("score", "睡眠分數"), ("hr", "睡眠心率")]:
        col = mapping.get(key)
        if col:
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if not series.empty:
                stats[f"{label}平均"] = round(series.mean(), 1)
                stats[f"{label}範圍"] = f"{series.min():g} ~ {series.max():g}"

    return stats


def trend_chart(df: pd.DataFrame, date_col: str, total_col: str):
    plot = pd.DataFrame(
        {
            "日期": pd.to_datetime(df[date_col], errors="coerce"),
            "睡眠時數": df[total_col].map(to_minutes) / 60,
        }
    ).dropna()
    if plot.empty:
        return None
    base = alt.Chart(plot).encode(x=alt.X("日期:T", title=None))
    line = base.mark_line(point=True, color="#5B8FF9").encode(
        y=alt.Y("睡眠時數:Q", title="小時", scale=alt.Scale(zero=False)),
        tooltip=["日期:T", alt.Tooltip("睡眠時數:Q", format=".2f")],
    )
    target = alt.Chart(pd.DataFrame({"y": [7]})).mark_rule(
        strokeDash=[6, 4], color="#F2994A"
    ).encode(y="y:Q")
    return (line + target).properties(height=280)


# --------------------------------------------------------------------------- #
# EDF
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False, max_entries=3)
def cached_read_edf(data: bytes, filename: str) -> dict:
    return edf_utils.read_edf(data, filename)


@st.cache_data(show_spinner=False, max_entries=3)
def cached_read_annotations(data: bytes, filename: str) -> list[dict]:
    return edf_utils.read_annotation_file(data, filename)


@st.cache_data(show_spinner=False, max_entries=2)
def cached_auto_stage(data: bytes, filename: str, eeg: str, eog: str | None, emg: str | None,
                      age: int | None, male: bool | None) -> dict:
    return edf_utils.auto_stage(data, filename, eeg, eog, emg, age, male)


def hypnogram_chart(hypno: pd.DataFrame):
    """畫 hypnogram：由上而下為 清醒 / REM / N1 / N2 / N3。"""
    levels = {stage: i for i, stage in enumerate(edf_utils.STAGE_ORDER)}
    plot = hypno.dropna(subset=["stage"]).copy()
    plot["層級"] = plot["stage"].map(levels)
    plot["小時"] = plot["time_min"] / 60
    plot["階段"] = plot["stage"].map(edf_utils.STAGE_LABELS)
    label_expr = " : ".join(
        f"datum.value == {i} ? '{edf_utils.STAGE_LABELS[s]}'" for s, i in levels.items()
    ) + " : ''"
    return (
        alt.Chart(plot)
        .mark_line(interpolate="step-after", color="#5B8FF9", strokeWidth=1.4)
        .encode(
            x=alt.X("小時:Q", title="距記錄起點（小時）"),
            y=alt.Y(
                "層級:Q",
                title=None,
                scale=alt.Scale(domain=[len(levels) - 0.5, -0.5]),
                axis=alt.Axis(values=list(levels.values()), labelExpr=label_expr, grid=True),
            ),
            tooltip=[alt.Tooltip("小時:Q", format=".2f"), "階段:N"],
        )
        .properties(height=260)
    )


# --------------------------------------------------------------------------- #
# Gemini
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = """你是一位睡眠健康分析師，擅長把穿戴裝置的睡眠數據轉成一般人看得懂的報告。
請用繁體中文輸出 Markdown 報告，結構如下：

## 一、整體摘要
三到五句話總結這段期間的睡眠狀況。

## 二、關鍵指標
用表格列出重要數據，並標註是否落在建議範圍。

## 三、觀察到的模式
指出趨勢、波動、異常日，並說明可能的原因。

## 四、具體建議
三到五項可執行的建議，每項說明「做什麼」與「為什麼」。

## 五、注意事項
若數據顯示可能的睡眠障礙徵兆，提醒就醫評估。

規則：
- 只根據提供的數據推論，數據不足的部分明確說「資料不足以判斷」，不要編造。
- 用具體數字支持每個論點。
- 這是健康資訊參考，不是醫療診斷，請在結尾註明。"""


PSG_NOTE = """這份數據來自單晚的多導睡眠檢查（PSG / EDF 檔），不是多天的穿戴裝置紀錄，
因此請針對「這一晚的睡眠結構」分析，不要談跨日趨勢。分析時請涵蓋：
睡眠效率、入睡潛伏期、REM 潛伏期、WASO、各睡眠期比例是否落在成人常模
（N1 約 5%、N2 約 45-55%、N3 約 15-25%、REM 約 20-25%）、以及睡眠片段化程度。
注意記錄可能包含上床前與起床後的清醒時間，解讀睡眠效率時要一併說明。
若統計中有 AHI，請解釋其嚴重度分級（<5 正常、5-15 輕度、15-30 中度、≥30 重度）的意義，
並明確說明這是篩檢參考、必須由醫師判讀；有 ODI 或血氧數據時一併討論。
若「分期來源」寫的是 YASA 自動分期，請在報告中說明分期由演算法推估、與人工判讀可能有差異。"""


def build_prompt(stats: dict, data_block: str, note: str, psg: bool = False) -> str:
    parts = [
        "以下是使用者的睡眠數據，請產生分析報告。",
        "\n### 統計摘要\n" + "\n".join(f"- {k}: {v}" for k, v in stats.items()),
        data_block,
    ]
    if psg:
        parts.append("\n### 分析重點\n" + PSG_NOTE)
    if note.strip():
        parts.append(f"\n### 使用者補充說明\n{note.strip()}")
    return "\n".join(parts)


def stream_report(api_key: str, model: str, prompt: str, temperature: float):
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    stream = client.models.generate_content_stream(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=temperature,
        ),
    )
    for chunk in stream:
        if chunk.text:
            yield chunk.text


def render_report_section(stats: dict, data_block: str, psg: bool) -> None:
    """兩種資料來源共用的「產生報告」區塊。"""
    st.subheader("產生報告")
    note = st.text_area(
        "補充說明（選填）",
        placeholder="例如：最近工作壓力大、有喝咖啡的習慣、想改善半夜醒來的問題……",
        height=90,
    )

    if st.button("✨ 產生睡眠分析報告", type="primary", width="stretch"):
        if not api_key:
            st.error("請先在側邊欄填入 Gemini API Key。")
            st.stop()
        prompt = build_prompt(stats, data_block, note, psg=psg)
        try:
            with st.spinner(f"{model} 分析中……"):
                report = st.write_stream(stream_report(api_key, model, prompt, temperature))
        except Exception as exc:
            st.error(f"呼叫 Gemini 失敗：{exc}")
            st.stop()
        st.session_state["report"] = report

    if st.session_state.get("report"):
        st.download_button(
            "⬇️ 下載報告 (Markdown)",
            st.session_state["report"],
            file_name=f"sleep_report_{datetime.now():%Y%m%d_%H%M}.md",
            mime="text/markdown",
        )
        st.caption("本報告由 AI 產生，僅供健康參考，不能取代醫療診斷。")


def default_api_key() -> str:
    """依序從環境變數與 secrets.toml 取得金鑰（沒有 secrets 檔也不會出錯）。"""
    if os.environ.get("GEMINI_API_KEY"):
        return os.environ["GEMINI_API_KEY"]
    try:
        return st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        return ""


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
st.title("🌙 睡眠分析報告產生器")
st.caption("上傳睡眠追蹤數據（CSV / Excel / JSON）或 PSG 原始檔（EDF / EDF+ / BDF），由 Gemini 自動產生分析報告。")

with st.sidebar:
    st.header("設定")
    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        value=default_api_key(),
        help="也可設定環境變數 GEMINI_API_KEY 或寫進 .streamlit/secrets.toml",
    )
    model = st.selectbox("模型", MODEL_OPTIONS, index=MODEL_OPTIONS.index(DEFAULT_MODEL))
    temperature = st.slider("創意程度 (temperature)", 0.0, 1.0, 0.3, 0.1)
    st.divider()
    st.download_button(
        "下載範例 CSV",
        pd.DataFrame(
            {
                "日期": pd.date_range("2026-08-01", periods=7).strftime("%Y-%m-%d"),
                "總睡眠時間(分鐘)": [412, 455, 380, 470, 350, 505, 430],
                "深睡(分鐘)": [78, 92, 61, 95, 52, 110, 84],
                "淺睡(分鐘)": [250, 268, 240, 275, 228, 290, 256],
                "REM(分鐘)": [84, 95, 79, 100, 70, 105, 90],
                "清醒(分鐘)": [22, 15, 34, 12, 41, 10, 18],
                "睡眠分數": [78, 85, 66, 88, 60, 91, 80],
                "平均心率": [58, 56, 62, 55, 64, 54, 57],
            }
        ).to_csv(index=False).encode("utf-8-sig"),
        file_name="sleep_sample.csv",
        mime="text/csv",
        width="stretch",
    )

uploaded = st.file_uploader(
    "上傳睡眠數據", type=["csv", "xlsx", "xls", "json", "edf", "bdf"]
)

if not uploaded:
    st.info("請先上傳檔案（表格檔或 EDF），或從側邊欄下載範例 CSV 試用。")
    st.stop()

# --------------------------------------------------------------------------- #
# EDF / PSG 流程
# --------------------------------------------------------------------------- #
if uploaded.name.lower().endswith((".edf", ".bdf")):
    with st.spinner("讀取 EDF……"):
        try:
            parsed = cached_read_edf(uploaded.getvalue(), uploaded.name)
        except Exception as exc:
            st.error(f"讀取 EDF 失敗：{exc}")
            st.stop()

    info = parsed["info"]
    annotations = parsed["annotations"]
    st.success(
        f"已載入 {info['通道數']} 個通道、{info['記錄時長(小時)']} 小時"
        + (f"，開始於 {info['記錄開始時間']}" if info["記錄開始時間"] else "")
    )

    with st.expander("通道與檔案資訊", expanded=False):
        st.json({k: v for k, v in info.items() if k != "通道"})
        st.write("通道：", "、".join(info["通道"]))

    # --- 分期來源：上傳的分期檔 > 檔內註記 > YASA 自動分期 ---
    inline_stages = [a for a in annotations if edf_utils.normalize_stage(a["description"])]
    stage_annotations, stage_source, offset = inline_stages, "檔內註記", 0.0
    extra_annotations: list[dict] = []

    if not inline_stages:
        st.warning(
            "這個 EDF 檔沒有睡眠分期註記（Sleep-EDF 的 *-PSG.edf 就屬於這種）。"
            "請上傳對應的 Hypnogram 檔，或使用下方的自動分期。"
        )

    hypno_file = st.file_uploader(
        "睡眠分期檔（Hypnogram，選填）",
        type=["edf", "csv", "txt", "tsv"],
        help="Sleep-EDF 的 *-Hypnogram.edf，或含 onset / stage 欄位的 CSV",
        key="hypno_upload",
    )
    if hypno_file is not None:
        try:
            parsed_ann = cached_read_annotations(hypno_file.getvalue(), hypno_file.name)
            stage_annotations = parsed_ann["annotations"]
            extra_annotations = parsed_ann["annotations"]
            stage_source = f"人工分期檔 {hypno_file.name}"
            offset = edf_utils.alignment_offset(info["記錄開始時間"], parsed_ann["start_time"])
            msg = f"已載入分期檔：{len(stage_annotations)} 段註記。"
            if offset:
                msg += f"（與訊號檔起點相差 {offset / 60:.1f} 分鐘，已自動對齊）"
            st.success(msg)
        except Exception as exc:
            st.error(f"讀取分期檔失敗：{exc}")

    hypno = (
        edf_utils.build_hypnogram(stage_annotations, offset) if stage_annotations else pd.DataFrame()
    )

    # --- 沒有人工分期時，用 YASA 自動分期 ---
    auto_result = None
    if hypno.empty:
        st.subheader("自動睡眠分期")
        if not edf_utils.yasa_available():
            st.info("安裝 YASA 後即可自動分期：`pip install yasa`")
        else:
            eeg_guess, eog_guess, emg_guess = edf_utils.guess_staging_channels(info["通道"])
            ch_options = ["（無）"] + info["通道"]
            c1, c2, c3, c4, c5 = st.columns(5)
            eeg = c1.selectbox(
                "EEG 通道（必填）", info["通道"],
                index=info["通道"].index(eeg_guess) if eeg_guess in info["通道"] else 0,
            )
            eog = c2.selectbox(
                "EOG 通道", ch_options,
                index=ch_options.index(eog_guess) if eog_guess in ch_options else 0,
            )
            emg = c3.selectbox(
                "EMG 通道", ch_options,
                index=ch_options.index(emg_guess) if emg_guess in ch_options else 0,
            )
            age = c4.number_input("年齡（選填）", min_value=0, max_value=120, value=0)
            sex = c5.selectbox("性別（選填）", ["未填", "男", "女"])
            st.caption("中央導程（C3/C4）效果最好；EOG、EMG、年齡、性別可留白，但提供會提高準確度。")

            if st.button("🤖 執行自動分期", width="stretch"):
                with st.spinner("YASA 分期中，數小時的記錄約需 20–60 秒……"):
                    try:
                        st.session_state["auto_stage"] = {
                            "file": f"{uploaded.name}:{uploaded.size}",
                            "result": cached_auto_stage(
                                uploaded.getvalue(), uploaded.name, eeg,
                                None if eog == "（無）" else eog,
                                None if emg == "（無）" else emg,
                                int(age) or None,
                                None if sex == "未填" else (sex == "男"),
                            ),
                        }
                    except Exception as exc:
                        st.error(f"自動分期失敗：{exc}")

        cached = st.session_state.get("auto_stage")
        if cached and cached["file"] == f"{uploaded.name}:{uploaded.size}":
            auto_result = cached["result"]
            hypno = edf_utils.epochs_to_hypnogram(auto_result["stages"])
            stage_source = f"YASA 自動分期（平均信心 {auto_result['mean_confidence']:.0%}）"
            st.success(
                f"自動分期完成：{len(auto_result['stages'])} 個 epoch，"
                f"平均信心 {auto_result['mean_confidence']:.0%}。"
            )
            st.caption("⚠️ 自動分期為演算法推估，與人工判讀會有差異（文獻上一致率約 80%），不能當臨床判讀依據。")

    stats: dict = {
        "資料來源": f"PSG 原始檔 {uploaded.name}",
        "分期來源": stage_source if not hypno.empty else "無分期",
        "記錄開始時間": info["記錄開始時間"],
        "記錄時長(小時)": info["記錄時長(小時)"],
        "取樣率(Hz)": info["取樣率(Hz)"],
        "量測通道": "、".join(info["通道"]),
        **{k: v for k, v in info["訊號摘要"].items() if not k.startswith("_")},
    }

    hourly = pd.DataFrame()
    analysis = hypno

    if not hypno.empty:
        # --- 分析區間：預設抓主睡眠期，避免整段 24 小時記錄稀釋掉睡眠效率 ---
        total_hours = float(hypno["time_min"].iloc[-1]) / 60
        bounds = edf_utils.main_sleep_period(list(hypno["stage"]))
        if bounds and total_hours > 12:
            # 前後各留 15 分鐘，免得偵測邊界剛好切掉最外側的睡眠 epoch
            pad = 0.25
            default = (
                round(max(0.0, float(hypno["time_min"].iloc[bounds[0]]) / 60 - pad), 2),
                round(min(total_hours, float(hypno["time_min"].iloc[bounds[1]]) / 60 + pad), 2),
            )
        else:
            default = (0.0, round(total_hours, 2))

        window = st.slider(
            "分析區間（距記錄起點的小時數）",
            0.0, round(total_hours, 2), default, step=0.25,
            help="連續 24 小時的記錄含大量白天清醒時間，預設只分析偵測到的主睡眠期。",
        )
        start_clock = edf_utils._clock(info["記錄開始時間"], window[0] * 60)
        end_clock = edf_utils._clock(info["記錄開始時間"], window[1] * 60)
        if start_clock:
            st.caption(f"目前分析 {start_clock} – {end_clock}（共 {window[1] - window[0]:.2f} 小時）")

        mask = (hypno["time_min"] >= window[0] * 60) & (hypno["time_min"] <= window[1] * 60)
        analysis = hypno[mask]

    if not analysis.empty:
        stats.update(edf_utils.hypnogram_metrics(analysis, info["記錄開始時間"]))

        st.subheader("睡眠結構")
        cards = [
            ("總睡眠時間", "總睡眠時間TST(分鐘)", lambda v: f"{v / 60:.1f} 小時"),
            ("睡眠效率 (TST/SPT)", "睡眠維持效率(TST/SPT)", str),
            ("入睡時刻", "首次入睡時刻", str),
            ("最後醒來", "最後醒來時刻", str),
            ("WASO", "入睡後清醒WASO(分鐘)", lambda v: f"{v:.0f} 分"),
            ("覺醒次數", "覺醒次數", lambda v: f"{v} 次"),
        ]
        shown = [(label, key, fmt) for label, key, fmt in cards if key in stats]
        for col, (label, key, fmt) in zip(st.columns(len(shown)), shown):
            col.metric(label, fmt(stats[key]))

        st.caption(
            f"睡眠效率以睡眠期間（SPT，首次入睡到最後醒來）為分母；"
            f"分析區間共 {stats['分析區間長度(分鐘)'] / 60:.1f} 小時，"
            f"其中 TST 佔 {stats.get('TST佔分析區間', '—')}。"
        )
        st.altair_chart(hypnogram_chart(analysis), width="stretch")

        stage_minutes = pd.DataFrame(
            {
                "階段": [edf_utils.STAGE_LABELS[s] for s in edf_utils.STAGE_ORDER
                         if f"{s}時間(分鐘)" in stats],
                "分鐘": [stats[f"{s}時間(分鐘)"] for s in edf_utils.STAGE_ORDER
                         if f"{s}時間(分鐘)" in stats],
            }
        )
        if not stage_minutes.empty:
            st.altair_chart(
                alt.Chart(stage_minutes)
                .mark_bar(cornerRadius=3, color="#5B8FF9")
                .encode(
                    x=alt.X("分鐘:Q", title="分鐘"),
                    y=alt.Y("階段:N", sort=list(stage_minutes["階段"]), title=None),
                    tooltip=["階段:N", "分鐘:Q"],
                )
                .properties(height=200),
                width="stretch",
            )

        hourly = edf_utils.hourly_breakdown(analysis, info["記錄開始時間"])
        with st.expander("每小時睡眠階段分佈（分鐘）"):
            st.dataframe(hourly, width="stretch")

    # --- 呼吸事件 ---
    events = edf_utils.classify_events(annotations + extra_annotations)
    spo2_count = info["訊號摘要"].get("_血氧下降事件數")
    event_stats = edf_utils.event_metrics(
        events, stats.get("總睡眠時間TST(分鐘)"), spo2_count
    )
    if event_stats:
        stats.update(event_stats)
        st.subheader("呼吸事件")
        event_cards = [
            ("AHI", "AHI(次/小時)", lambda v: f"{v} 次/小時"),
            ("AHI 分級", "AHI分級(參考)", str),
            ("ODI", "ODI(次/小時)", lambda v: f"{v} 次/小時"),
            ("腦波覺醒指數", "腦波覺醒指數(次/小時)", lambda v: f"{v} 次/小時"),
            ("血氧最低", "血氧最低(%)", lambda v: f"{v}%"),
        ]
        shown = [(l, k, f) for l, k, f in event_cards if k in stats]
        if shown:
            for col, (label, key, fmt) in zip(st.columns(len(shown)), shown):
                col.metric(label, fmt(stats[key]))
        if not events.empty:
            with st.expander("事件明細"):
                st.dataframe(
                    events["類型"].value_counts().rename("次數").reset_index(), width="stretch"
                )
                st.dataframe(events, width="stretch", height=240)
        st.caption("AHI 分級僅為參考區間（<5 正常、5–15 輕度、15–30 中度、≥30 重度），不是診斷。")
    elif spo2_count is None and not events.empty:
        st.info("註記中有事件，但缺少睡眠分期，無法換算成 AHI。")

    with st.expander("統計摘要（送給模型的內容）"):
        st.json(stats)

    data_block = ""
    if not hourly.empty:
        data_block = f"\n### 每小時睡眠階段分佈（分鐘）\n```csv\n{hourly.to_csv(index=False)}\n```"
    elif not annotations:
        data_block = "\n### 備註\n此檔僅有訊號、沒有任何註記，可用資訊有限。"

    render_report_section(stats, data_block, psg=True)
    st.stop()

# --------------------------------------------------------------------------- #
# 表格檔流程
# --------------------------------------------------------------------------- #
try:
    df = load_file(uploaded)
except Exception as exc:  # 檔案格式千奇百怪，統一給使用者看得懂的訊息
    st.error(f"讀取檔案失敗：{exc}")
    st.stop()

if df.empty:
    st.error("檔案沒有任何資料列。")
    st.stop()

st.success(f"已載入 {len(df)} 筆記錄、{len(df.columns)} 個欄位。")

with st.expander("原始數據預覽", expanded=False):
    st.dataframe(df, width="stretch", height=280)

# --- 欄位對應 ---
st.subheader("欄位對應")
st.caption("系統已自動辨識，若有誤請手動調整。")
options = ["（無）"] + list(df.columns)
labels = {
    "date": "日期",
    "total": "總睡眠時間",
    "deep": "深睡",
    "light": "淺睡",
    "rem": "REM",
    "awake": "清醒",
    "score": "睡眠分數",
    "hr": "心率",
}
mapping: dict[str, str | None] = {}
cols = st.columns(4)
for i, (key, label) in enumerate(labels.items()):
    guess = guess_column(df, key)
    with cols[i % 4]:
        choice = st.selectbox(
            label, options, index=options.index(guess) if guess in options else 0, key=f"map_{key}"
        )
    mapping[key] = None if choice == "（無）" else choice

stats = build_stats(df, mapping)

# --- 指標與圖表 ---
st.subheader("數據概覽")
metric_keys = [k for k in ("總睡眠平均(分鐘)", "深睡佔比平均", "睡眠分數平均", "睡足7小時天數比例") if k in stats]
if metric_keys:
    mcols = st.columns(len(metric_keys))
    for col, key in zip(mcols, metric_keys):
        value = stats[key]
        if key.endswith("(分鐘)"):
            value = f"{value / 60:.1f} 小時"
        col.metric(key.replace("(分鐘)", ""), value)

if mapping["date"] and mapping["total"]:
    chart = trend_chart(df, mapping["date"], mapping["total"])
    if chart is not None:
        st.altair_chart(chart, width="stretch")

stage_cols = {labels[k]: mapping[k] for k in ("deep", "light", "rem", "awake") if mapping[k]}
if len(stage_cols) >= 2:
    stage_avg = pd.DataFrame(
        {"階段": list(stage_cols), "平均分鐘": [df[c].map(to_minutes).mean() for c in stage_cols.values()]}
    ).dropna()
    st.altair_chart(
        alt.Chart(stage_avg)
        .mark_arc(innerRadius=60)
        .encode(theta="平均分鐘:Q", color=alt.Color("階段:N", legend=alt.Legend(title="睡眠階段")),
                tooltip=["階段:N", alt.Tooltip("平均分鐘:Q", format=".0f")])
        .properties(height=260),
        width="stretch",
    )

with st.expander("統計摘要（送給模型的內容）"):
    st.json(stats)

render_report_section(stats, df.head(60).to_csv(index=False), psg=False)
