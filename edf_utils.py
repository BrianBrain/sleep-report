"""EDF / EDF+ 睡眠檔案解析與睡眠結構指標計算（供 sleep_report.py 使用）。

支援兩種常見情況：
1. 單一 EDF+ 檔內含分期註記（annotations）。
2. 訊號檔（例如 Sleep-EDF 的 *-PSG.edf）＋ 另一個 Hypnogram 檔。
"""

from __future__ import annotations

import importlib.util
import os
import tempfile
from datetime import datetime, timedelta

import mne
import numpy as np
import pandas as pd

EPOCH_SEC = 30

# 分期標籤正規化：涵蓋 Sleep-EDF、AASM、常見廠商匯出寫法
STAGE_ALIASES: dict[str, str] = {}
for _key in ("w", "wake", "awake", "sleep stage w", "stage w", "0"):
    STAGE_ALIASES[_key] = "W"
for _key in ("1", "n1", "s1", "sleep stage 1", "stage 1", "sleep stage n1"):
    STAGE_ALIASES[_key] = "N1"
for _key in ("2", "n2", "s2", "sleep stage 2", "stage 2", "sleep stage n2"):
    STAGE_ALIASES[_key] = "N2"
for _key in ("3", "4", "n3", "n4", "s3", "s4", "sws", "deep", "sleep stage 3", "sleep stage 4",
             "stage 3", "stage 4", "sleep stage n3", "sleep stage 3/4"):
    STAGE_ALIASES[_key] = "N3"
for _key in ("r", "rem", "5", "sleep stage r", "stage r", "sleep stage rem"):
    STAGE_ALIASES[_key] = "R"

SLEEP_STAGES = ("N1", "N2", "N3", "R")
STAGE_ORDER = ("W", "R", "N1", "N2", "N3")  # 畫 hypnogram 由上而下的順序
STAGE_LABELS = {"W": "清醒", "R": "REM", "N1": "N1 淺睡", "N2": "N2 淺睡", "N3": "N3 深睡"}

SPO2_HINTS = ("spo2", "sao2", "sat", "血氧")
HR_HINTS = ("heart", "hr", "pulse", "心率")


def normalize_stage(description: str) -> str | None:
    """把註記文字轉成 W / N1 / N2 / N3 / R；無法辨識回傳 None。"""
    text = str(description).strip().lower()
    if text in STAGE_ALIASES:
        return STAGE_ALIASES[text]
    # 容忍 "Sleep stage W (Wake)"、"EPOCH N2" 之類的變體
    for key, stage in STAGE_ALIASES.items():
        if len(key) > 1 and key in text:
            return stage
    return None


def _write_temp(data: bytes, suffix: str) -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)
    return path


def read_edf(data: bytes, filename: str) -> dict:
    """讀取 EDF 標頭、註記與（若有）血氧/心率通道摘要。"""
    suffix = os.path.splitext(filename)[1] or ".edf"
    path = _write_temp(data, suffix)
    try:
        raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
        info = {
            "檔名": filename,
            "通道數": len(raw.ch_names),
            "通道": list(raw.ch_names),
            "取樣率(Hz)": float(raw.info["sfreq"]),
            "記錄時長(小時)": round(raw.n_times / raw.info["sfreq"] / 3600, 2),
            "記錄開始時間": raw.info["meas_date"].strftime("%Y-%m-%d %H:%M:%S")
            if raw.info["meas_date"]
            else None,
        }
        annotations = [
            {"onset": float(o), "duration": float(d), "description": str(s)}
            for o, d, s in zip(raw.annotations.onset, raw.annotations.duration,
                               raw.annotations.description)
        ]
        info["訊號摘要"] = _signal_summary(raw)
        return {"info": info, "annotations": annotations}
    finally:
        os.unlink(path)


def read_annotation_file(data: bytes, filename: str) -> dict:
    """讀取獨立的 hypnogram / 註記檔（EDF+ 或 CSV/TXT）。

    回傳 {"annotations": [...], "start_time": 該檔的起始時間或 None}；
    起始時間用來對齊訊號檔（分期檔若晚於訊號檔開始，onset 需要平移）。
    """
    suffix = os.path.splitext(filename)[1].lower() or ".edf"
    if suffix in (".csv", ".txt", ".tsv"):
        sep = "\t" if suffix == ".tsv" else None
        df = pd.read_csv(pd.io.common.BytesIO(data), sep=sep, engine="python")
        cols = {c.lower().strip(): c for c in df.columns}
        onset_col = next((cols[c] for c in cols if c in ("onset", "start", "time", "秒", "開始")), None)
        desc_col = next((cols[c] for c in cols if c in ("description", "stage", "label", "分期", "annotation")), None)
        if onset_col is None or desc_col is None:
            raise ValueError("CSV 註記檔需要包含 onset/start 與 stage/description 欄位")
        dur_col = next((cols[c] for c in cols if c in ("duration", "length", "長度")), None)
        return {
            "annotations": [
                {
                    "onset": float(row[onset_col]),
                    "duration": float(row[dur_col]) if dur_col else float(EPOCH_SEC),
                    "description": str(row[desc_col]),
                }
                for _, row in df.iterrows()
            ],
            "start_time": None,
        }

    path = _write_temp(data, suffix)
    try:
        ann = mne.read_annotations(path)
        return {
            "annotations": [
                {"onset": float(o), "duration": float(d), "description": str(s)}
                for o, d, s in zip(ann.onset, ann.duration, ann.description)
            ],
            "start_time": ann.orig_time.strftime("%Y-%m-%d %H:%M:%S") if ann.orig_time else None,
        }
    finally:
        os.unlink(path)


def alignment_offset(signal_start: str | None, annotation_start: str | None) -> float:
    """分期檔比訊號檔晚開始多少秒（兩邊都有時間才算得出來）。"""
    if not signal_start or not annotation_start:
        return 0.0
    try:
        a = datetime.strptime(signal_start, "%Y-%m-%d %H:%M:%S")
        b = datetime.strptime(annotation_start, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return 0.0
    return (b - a).total_seconds()


def _signal_summary(raw) -> dict:
    """抓血氧 / 心率通道的基本統計（有才算，載入時只讀該通道）。"""
    summary: dict = {}
    for ch in raw.ch_names:
        low = ch.lower()
        if any(h in low for h in SPO2_HINTS):
            values = raw.get_data(picks=[ch]).ravel()
            clean = values[np.isfinite(values) & (values > 50) & (values <= 100)]
            if clean.size:
                summary["血氧平均(%)"] = round(float(clean.mean()), 1)
                summary["血氧最低(%)"] = round(float(clean.min()), 1)
                count, t90 = spo2_events(values, float(raw.info["sfreq"]))
                summary["血氧<90%時間(分鐘)"] = t90
                # 底線開頭的鍵不進報告：要有 TST 才能換算成 ODI
                summary["_血氧下降事件數"] = count
            break
    for ch in raw.ch_names:
        low = ch.lower()
        if any(h in low for h in HR_HINTS):
            values = raw.get_data(picks=[ch]).ravel()
            values = values[np.isfinite(values) & (values > 20) & (values < 220)]
            if values.size:
                summary["心率平均(bpm)"] = round(float(values.mean()), 1)
                summary["心率最低(bpm)"] = round(float(values.min()), 1)
                summary["心率最高(bpm)"] = round(float(values.max()), 1)
            break
    return summary


def build_hypnogram(annotations: list[dict], offset_sec: float = 0.0) -> pd.DataFrame:
    """把註記展開成每 30 秒一列的 hypnogram。

    回傳欄位: epoch(序號)、time_min(距記錄起點分鐘)、stage(W/N1/N2/N3/R 或 None)。
    """
    rows: list[tuple[int, str]] = []
    for ann in annotations:
        stage = normalize_stage(ann["description"])
        if stage is None:
            # 呼吸事件之類的註記不能覆蓋掉該 epoch 的分期；未評分的 epoch 之後會補 None
            continue
        start = ann["onset"] + offset_sec
        n_epochs = max(1, int(round(ann["duration"] / EPOCH_SEC)))
        first = int(round(start / EPOCH_SEC))
        rows.extend((first + i, stage) for i in range(n_epochs))
    if not rows:
        return pd.DataFrame(columns=["epoch", "time_min", "stage"])

    stages: dict[int, str] = {}
    for epoch, stage in rows:
        stages[epoch] = stage  # 後出現的註記覆蓋先前的（處理重疊）
    index = range(min(stages), max(stages) + 1)
    return pd.DataFrame(
        {
            "epoch": list(index),
            "time_min": [i * EPOCH_SEC / 60 for i in index],
            "stage": [stages.get(i) for i in index],
        }
    )


def _clock(start_time: str | None, minutes_from_start: float) -> str | None:
    if not start_time:
        return None
    try:
        base = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    return (base + timedelta(minutes=minutes_from_start)).strftime("%H:%M")


def hypnogram_metrics(hypno: pd.DataFrame, start_time: str | None = None) -> dict:
    """由 hypnogram 計算標準睡眠結構指標。

    注意：EDF 的記錄起點不等於「上床時間」（例如 Sleep-EDF 從下午就開始錄），
    所以睡眠效率同時給「佔記錄總長」與「佔睡眠期間 SPT」兩個版本，避免誤讀。
    """
    stages = list(hypno["stage"])
    if not stages:
        return {}
    per_epoch_min = EPOCH_SEC / 60
    # time_min 是距「記錄起點」的分鐘數；hypno 可能只是其中一段（分析區間）
    offset_min = float(hypno["time_min"].iloc[0]) if "time_min" in hypno else 0.0
    scored = [s for s in stages if s is not None]
    sleep_idx = [i for i, s in enumerate(stages) if s in SLEEP_STAGES]

    metrics: dict = {
        "分析區間長度(分鐘)": round(len(stages) * per_epoch_min, 1),
        "已分期epoch數": len(scored),
    }
    if not sleep_idx:
        metrics["備註"] = "整段記錄沒有偵測到睡眠期"
        return metrics

    first_sleep, last_sleep = sleep_idx[0], sleep_idx[-1]
    spt = stages[first_sleep : last_sleep + 1]  # sleep period time
    tst = sum(1 for s in spt if s in SLEEP_STAGES) * per_epoch_min
    waso = sum(1 for s in spt if s == "W") * per_epoch_min

    metrics["總睡眠時間TST(分鐘)"] = round(tst, 1)
    metrics["睡眠期間SPT(分鐘)"] = round(len(spt) * per_epoch_min, 1)
    metrics["睡眠維持效率(TST/SPT)"] = f"{tst / (len(spt) * per_epoch_min):.1%}"
    metrics["TST佔分析區間"] = f"{tst / (len(stages) * per_epoch_min):.1%}"
    metrics["區間起點到首次入睡(分鐘)"] = round(first_sleep * per_epoch_min, 1)
    metrics["入睡後清醒WASO(分鐘)"] = round(waso, 1)

    first_clock = _clock(start_time, offset_min + first_sleep * per_epoch_min)
    last_clock = _clock(start_time, offset_min + (last_sleep + 1) * per_epoch_min)
    if first_clock:
        metrics["首次入睡時刻"] = first_clock
        metrics["最後醒來時刻"] = last_clock

    rem_idx = [i for i, s in enumerate(stages) if s == "R"]
    if rem_idx:
        metrics["REM潛伏期(分鐘)"] = round((rem_idx[0] - first_sleep) * per_epoch_min, 1)

    # 各期時間與佔 TST 比例
    for stage in SLEEP_STAGES:
        minutes = sum(1 for s in spt if s == stage) * per_epoch_min
        metrics[f"{stage}時間(分鐘)"] = round(minutes, 1)
        metrics[f"{stage}佔TST"] = f"{minutes / tst:.1%}" if tst else "0%"

    # 覺醒次數：SPT 內連續清醒段落
    awakenings = sum(
        1 for i, s in enumerate(spt) if s == "W" and (i == 0 or spt[i - 1] != "W")
    )
    metrics["覺醒次數"] = awakenings
    metrics["覺醒指數(次/小時)"] = round(awakenings / (tst / 60), 1) if tst else 0

    transitions = sum(
        1 for a, b in zip(spt, spt[1:]) if a is not None and b is not None and a != b
    )
    metrics["階段轉換次數"] = transitions
    metrics["睡眠片段化指數(轉換/小時)"] = round(transitions / (tst / 60), 1) if tst else 0
    return metrics


def hourly_breakdown(hypno: pd.DataFrame, start_time: str | None = None) -> pd.DataFrame:
    """每小時各睡眠階段分鐘數，送給模型看趨勢用（比原始 epoch 精簡很多）。"""
    if hypno.empty:
        return pd.DataFrame()
    df = hypno.dropna(subset=["stage"]).copy()
    if df.empty:
        return pd.DataFrame()
    df["小時"] = (df["time_min"] // 60).astype(int)
    table = (
        df.pivot_table(index="小時", columns="stage", values="epoch", aggfunc="count")
        .reindex(columns=[s for s in STAGE_ORDER], fill_value=0)
        .fillna(0)
        * (EPOCH_SEC / 60)
    )
    table.columns = [STAGE_LABELS.get(c, c) for c in table.columns]
    table = table.round(1).reset_index()
    if start_time:
        clocks = [_clock(start_time, h * 60) for h in table["小時"]]
        if all(clocks):
            table.insert(1, "時刻", clocks)
    # 整段都清醒的小時（例如記錄從下午就開始）對報告沒有價值，只留睡眠時段前後一小時
    stage_cols = [c for c in table.columns if c in STAGE_LABELS.values() and c != STAGE_LABELS["W"]]
    slept = table.index[table[stage_cols].sum(axis=1) > 0] if stage_cols else []
    if len(slept):
        table = table.loc[max(0, slept.min() - 1) : slept.max() + 1]
    return table.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 自動睡眠分期（YASA）
# --------------------------------------------------------------------------- #
def yasa_available() -> bool:
    """YASA 是選用相依：沒安裝時 app 其他功能照常運作。"""
    return importlib.util.find_spec("yasa") is not None


def guess_staging_channels(ch_names: list[str]) -> tuple[str | None, str | None, str | None]:
    """猜 YASA 需要的 EEG / EOG / EMG 通道（中央導程優先）。"""

    def pick(*hint_groups: tuple[str, ...]) -> str | None:
        for hints in hint_groups:
            for ch in ch_names:
                low = ch.lower()
                if any(h in low for h in hints):
                    return ch
        return None

    eeg = pick(("c4", "c3"), ("central",), ("eeg",))
    eog = pick(("eog",), ("loc", "roc"))
    emg = pick(("emg",), ("chin",))
    return eeg, eog, emg


def auto_stage(
    data: bytes,
    filename: str,
    eeg: str,
    eog: str | None = None,
    emg: str | None = None,
    age: int | None = None,
    male: bool | None = None,
) -> dict:
    """用 YASA 的預訓練模型自動分期，回傳每個 30 秒 epoch 的分期與信心。"""
    import yasa

    suffix = os.path.splitext(filename)[1] or ".edf"
    path = _write_temp(data, suffix)
    try:
        picks = [ch for ch in (eeg, eog, emg) if ch]
        # 只載入分期需要的通道；YASA 會自行降採樣到 100 Hz 並轉 µV，不要先濾波或標準化
        raw = mne.io.read_raw_edf(path, include=picks, preload=True, verbose="ERROR")
        metadata = {}
        if age is not None:
            metadata["age"] = age
        if male is not None:
            metadata["male"] = bool(male)

        sls = yasa.SleepStaging(
            raw, eeg_name=eeg, eog_name=eog or None, emg_name=emg or None,
            metadata=metadata or None,
        )
        pred = sls.predict()
        proba = getattr(pred, "proba", None)
        if proba is None:
            proba = sls.predict_proba()

        # 對機率做 5 個 epoch 的三角平滑：單一 epoch 的誤判會被前後文拉回來，
        # 實測可把假的「醒來」次數從 88 降到 28，一致率也略升。
        smoothed = proba.rolling(window=5, center=True, min_periods=1, win_type="triang").mean()
        stages = [normalize_stage(s) for s in smoothed.idxmax(axis=1)]
        confidence = smoothed.max(axis=1).to_numpy()
        return {
            "stages": stages,
            "confidence": [round(float(c), 3) for c in confidence],
            "mean_confidence": round(float(np.mean(confidence)), 3),
            "channels": {"EEG": eeg, "EOG": eog, "EMG": emg},
        }
    finally:
        os.unlink(path)


def epochs_to_hypnogram(stages: list[str | None]) -> pd.DataFrame:
    """把逐 epoch 的分期標籤轉成與 build_hypnogram() 相同結構的 DataFrame。"""
    return pd.DataFrame(
        {
            "epoch": list(range(len(stages))),
            "time_min": [i * EPOCH_SEC / 60 for i in range(len(stages))],
            "stage": stages,
        }
    )


# --------------------------------------------------------------------------- #
# 呼吸事件
# --------------------------------------------------------------------------- #
# 順序有意義：先比對低通氣與各型呼吸中止，最後才落到通用的 apnea
EVENT_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("低通氣", ("hypopnea", "hypopnoea", "低通氣", "低換氣")),
    ("阻塞型呼吸中止", ("obstructive apnea", "obstructive apnoea", "obstructive_apnea", "阻塞型")),
    ("中樞型呼吸中止", ("central apnea", "central apnoea", "central_apnea", "中樞型")),
    ("混合型呼吸中止", ("mixed apnea", "mixed apnoea", "mixed_apnea", "混合型")),
    ("其他呼吸中止", ("apnea", "apnoea", "呼吸中止")),
    ("血氧下降", ("desaturation", "desat", "血氧下降")),
    ("腦波覺醒", ("arousal", "微覺醒")),
    ("打鼾", ("snore", "snoring", "打鼾")),
    ("肢體運動", ("limb movement", "periodic leg", "plm", "肢體")),
)
APNEA_TYPES = ("阻塞型呼吸中止", "中樞型呼吸中止", "混合型呼吸中止", "其他呼吸中止")
RESPIRATORY_TYPES = APNEA_TYPES + ("低通氣",)


def classify_event(description: str) -> str | None:
    text = str(description).strip().lower()
    for label, hints in EVENT_PATTERNS:
        if any(h in text for h in hints):
            return label
    return None


def classify_events(annotations: list[dict]) -> pd.DataFrame:
    """從註記中挑出呼吸/覺醒/肢體事件（分期註記會被略過）。"""
    rows = []
    for ann in annotations:
        if normalize_stage(ann["description"]):
            continue
        label = classify_event(ann["description"])
        if label:
            rows.append(
                {
                    "類型": label,
                    "onset": float(ann["onset"]),
                    "duration": float(ann["duration"]),
                    "原始標籤": str(ann["description"]),
                }
            )
    return pd.DataFrame(rows, columns=["類型", "onset", "duration", "原始標籤"])


def ahi_severity(ahi: float) -> str:
    """成人 AHI 分級（僅供參考，非診斷）。"""
    if ahi < 5:
        return "正常範圍"
    if ahi < 15:
        return "輕度"
    if ahi < 30:
        return "中度"
    return "重度"


def event_metrics(
    events: pd.DataFrame, tst_minutes: float | None, spo2_events_count: int | None = None
) -> dict:
    """把事件次數換算成臨床常用的指數（AHI / ODI / 覺醒指數）。"""
    metrics: dict = {}
    if events.empty and not spo2_events_count:
        return metrics

    counts = events["類型"].value_counts().to_dict() if not events.empty else {}
    for label, _ in EVENT_PATTERNS:
        if counts.get(label):
            metrics[f"{label}次數"] = int(counts[label])

    resp = events[events["類型"].isin(RESPIRATORY_TYPES)] if not events.empty else pd.DataFrame()
    if not resp.empty:
        metrics["呼吸事件總次數"] = int(len(resp))
        metrics["呼吸事件平均長度(秒)"] = round(float(resp["duration"].mean()), 1)
        metrics["呼吸事件最長(秒)"] = round(float(resp["duration"].max()), 1)

    tst_hours = (tst_minutes or 0) / 60
    if tst_hours <= 0:
        if not resp.empty:
            metrics["備註"] = "沒有 TST 可用，無法換算 AHI（需要睡眠分期）"
        return metrics

    if not resp.empty:
        ahi = len(resp) / tst_hours
        metrics["AHI(次/小時)"] = round(ahi, 1)
        metrics["AHI分級(參考)"] = ahi_severity(ahi)

    desat = int(counts.get("血氧下降", 0)) or int(spo2_events_count or 0)
    if desat:
        metrics["血氧下降次數(≥3%)"] = desat
        metrics["ODI(次/小時)"] = round(desat / tst_hours, 1)

    if counts.get("腦波覺醒"):
        metrics["腦波覺醒指數(次/小時)"] = round(counts["腦波覺醒"] / tst_hours, 1)
    return metrics


def spo2_events(values: np.ndarray, sfreq: float, drop: float = 3.0,
                min_sec: float = 10.0) -> tuple[int, float]:
    """沒有血氧註記時，直接從 SpO2 訊號數 ≥3% 的下降事件。

    做法：降採樣到 1 Hz → 以前 120 秒的滾動中位數當基線 →
    找出低於基線 drop% 且持續 min_sec 秒以上的段落。回傳 (事件數, T90 分鐘)。
    """
    step = max(1, int(round(sfreq)))
    series = pd.Series(values[::step]).astype(float)
    series = series.where((series > 50) & (series <= 100))
    if series.notna().sum() < 120:
        return 0, 0.0

    baseline = series.rolling(120, min_periods=30).median()
    below = ((series < baseline - drop) & series.notna()).to_numpy()

    count, run = 0, 0
    for flag in below:
        if flag:
            run += 1
        else:
            if run >= min_sec:
                count += 1
            run = 0
    if run >= min_sec:
        count += 1

    t90 = float((series < 90).sum()) / 60  # 1 Hz → 分鐘
    return count, round(t90, 1)


def main_sleep_period(
    stages: list[str | None], window_epochs: int = 60, threshold: float = 0.5
) -> tuple[int, int] | None:
    """找出「主睡眠期」的 epoch 起訖（最長的一段高睡眠密度區間）。

    連續 24 小時的 PSG（例如 Sleep-EDF）包含大量白天清醒時間，直接算睡眠效率會嚴重失真；
    自動分期在白天也容易把安靜清醒誤判成 N1/N2。做法是以 30 分鐘的滑動視窗計算睡眠比例，
    取比例 >= threshold 的最長連續區段。找不到就回傳 None。
    """
    if not stages:
        return None
    is_sleep = pd.Series([1.0 if s in SLEEP_STAGES else 0.0 for s in stages])
    density = is_sleep.rolling(window_epochs, center=True, min_periods=window_epochs // 2).mean()
    good = (density >= threshold).to_numpy()

    best_len, best = 0, None
    i = 0
    while i < len(good):
        if good[i]:
            j = i
            while j + 1 < len(good) and good[j + 1]:
                j += 1
            if j - i + 1 > best_len:
                best_len, best = j - i + 1, (i, j)
            i = j + 1
        else:
            i += 1
    return best
