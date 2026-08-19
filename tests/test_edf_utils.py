"""edf_utils 的單元測試。

Sleep-EDF 沒有呼吸事件也沒有血氧通道，所以那兩塊用合成資料驗證；
睡眠結構指標則以真實的 Sleep-EDF hypnogram 當回歸基準。
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import edf_utils  # noqa: E402

HYPNOGRAM_FILE = "/home/brian/w1/data/SC4002EC-Hypnogram.edf"


# --------------------------------------------------------------------------- #
# 分期標籤
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text,expected",
    [
        ("Sleep stage W", "W"),
        ("Sleep stage 1", "N1"),
        ("Sleep stage 4", "N3"),   # R&K 的 S4 併入 N3
        ("Sleep stage R", "R"),
        ("WAKE", "W"),             # YASA 0.7 的輸出標籤
        ("REM", "R"),
        ("N2", "N2"),
        ("Movement time", None),
        ("Sleep stage ?", None),
    ],
)
def test_normalize_stage(text, expected):
    assert edf_utils.normalize_stage(text) == expected


# --------------------------------------------------------------------------- #
# 呼吸事件分類
# --------------------------------------------------------------------------- #
def _sample_events():
    labels = [
        "Obstructive apnea", "Obstructive Apnea", "Central apnea", "Mixed apnea",
        "Hypopnea", "Hypopnea", "Hypopnea",
        "SpO2 desaturation", "SpO2 desaturation",
        "EEG arousal", "Snore", "Limb movement",
        "Sleep stage 2",  # 分期註記不應被當成事件
    ]
    return [{"onset": i * 60.0, "duration": 15.0, "description": lab} for i, lab in enumerate(labels)]


def test_classify_events_separates_hypopnea_from_apnea():
    events = edf_utils.classify_events(_sample_events())
    counts = events["類型"].value_counts().to_dict()
    assert counts["低通氣"] == 3
    assert counts["阻塞型呼吸中止"] == 2
    assert counts["中樞型呼吸中止"] == 1
    assert counts["混合型呼吸中止"] == 1
    assert "其他呼吸中止" not in counts       # 具體型別不該落到通用類別
    assert len(events) == 12                  # 分期註記被排除


def test_event_metrics_indices():
    events = edf_utils.classify_events(_sample_events())
    # 呼吸事件 7 次（4 apnea + 3 hypopnea）、TST 420 分鐘 = 7 小時 → AHI = 1.0
    metrics = edf_utils.event_metrics(events, tst_minutes=420)
    assert metrics["呼吸事件總次數"] == 7
    assert metrics["AHI(次/小時)"] == 1.0
    assert metrics["AHI分級(參考)"] == "正常範圍"
    assert metrics["血氧下降次數(≥3%)"] == 2
    assert metrics["ODI(次/小時)"] == round(2 / 7, 1)
    assert metrics["腦波覺醒指數(次/小時)"] == round(1 / 7, 1)


@pytest.mark.parametrize(
    "ahi,level", [(2, "正常範圍"), (7, "輕度"), (20, "中度"), (45, "重度")]
)
def test_ahi_severity(ahi, level):
    assert edf_utils.ahi_severity(ahi) == level


def test_event_metrics_without_tst():
    events = edf_utils.classify_events(_sample_events())
    metrics = edf_utils.event_metrics(events, tst_minutes=None)
    assert "AHI(次/小時)" not in metrics       # 沒有分期就不能算 AHI
    assert metrics["呼吸事件總次數"] == 7


# --------------------------------------------------------------------------- #
# 血氧下降偵測
# --------------------------------------------------------------------------- #
def test_spo2_events_counts_known_dips():
    sfreq = 4.0
    minutes = 40
    baseline = np.full(int(minutes * 60 * sfreq), 97.0)
    # 每 5 分鐘插入一次 20 秒、下降 6% 的凹陷，共 6 次
    n_dips = 6
    for i in range(n_dips):
        start = int((300 * (i + 1)) * sfreq)
        baseline[start : start + int(20 * sfreq)] = 91.0
    count, t90 = edf_utils.spo2_events(baseline, sfreq)
    assert count == n_dips
    assert t90 == 0.0  # 91% 沒有低於 90%


def test_spo2_events_ignores_short_dips():
    sfreq = 1.0
    values = np.full(1800, 97.0)
    for i in range(5):
        start = 300 * (i + 1)
        values[start : start + 4] = 90.0  # 只有 4 秒，短於 10 秒門檻
    count, _ = edf_utils.spo2_events(values, sfreq)
    assert count == 0


def test_spo2_events_t90():
    sfreq = 1.0
    values = np.full(1200, 97.0)
    values[600:720] = 85.0  # 120 秒 = 2 分鐘低於 90%
    count, t90 = edf_utils.spo2_events(values, sfreq)
    assert count == 1
    assert t90 == pytest.approx(2.0, abs=0.1)


# --------------------------------------------------------------------------- #
# 睡眠結構指標（真實檔案回歸基準）
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not os.path.exists(HYPNOGRAM_FILE), reason="需要 Sleep-EDF 樣本檔")
def test_hypnogram_metrics_regression():
    with open(HYPNOGRAM_FILE, "rb") as fh:
        parsed = edf_utils.read_annotation_file(fh.read(), os.path.basename(HYPNOGRAM_FILE))
    hypno = edf_utils.build_hypnogram(parsed["annotations"])
    metrics = edf_utils.hypnogram_metrics(hypno, "1989-04-25 14:50:00")

    assert metrics["總睡眠時間TST(分鐘)"] == 472.0
    assert metrics["睡眠維持效率(TST/SPT)"] == "93.7%"
    assert metrics["覺醒次數"] == 21
    assert metrics["首次入睡時刻"] == "22:04"
    assert metrics["最後醒來時刻"] == "06:28"
    # 各期時間加總應等於 TST
    stage_total = sum(metrics[f"{s}時間(分鐘)"] for s in edf_utils.SLEEP_STAGES)
    assert stage_total == pytest.approx(metrics["總睡眠時間TST(分鐘)"])


def test_epochs_to_hypnogram_matches_build_hypnogram_shape():
    stages = ["W", "W", "N1", "N2", "N2", "R"]
    hypno = edf_utils.epochs_to_hypnogram(stages)
    assert list(hypno.columns) == ["epoch", "time_min", "stage"]
    assert hypno["time_min"].tolist() == [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    metrics = edf_utils.hypnogram_metrics(hypno)
    assert metrics["總睡眠時間TST(分鐘)"] == 2.0


def test_alignment_offset():
    assert edf_utils.alignment_offset("2026-08-19 22:00:00", "2026-08-19 22:30:00") == 1800.0
    assert edf_utils.alignment_offset(None, "2026-08-19 22:30:00") == 0.0


def test_guess_staging_channels():
    eeg, eog, emg = edf_utils.guess_staging_channels(
        ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal", "Resp oro-nasal", "EMG submental"]
    )
    assert (eeg, eog, emg) == ("EEG Fpz-Cz", "EOG horizontal", "EMG submental")

    eeg, _, _ = edf_utils.guess_staging_channels(["F3-M2", "C4-M1", "O1-M2"])
    assert eeg == "C4-M1"  # 中央導程優先


# --------------------------------------------------------------------------- #
# 主睡眠期偵測與區間裁切
# --------------------------------------------------------------------------- #
def test_main_sleep_period_skips_daytime_wake():
    # 前 6 小時清醒（720 epoch）、接著 8 小時睡眠（960 epoch）、最後 1 小時清醒
    stages = ["W"] * 720 + ["N2"] * 960 + ["W"] * 120
    bounds = edf_utils.main_sleep_period(stages)
    assert bounds is not None
    start, end = bounds
    assert 690 <= start <= 750      # 30 分鐘滑動視窗會讓邊界稍微外擴
    assert 1650 <= end <= 1710


def test_main_sleep_period_none_when_no_sleep():
    assert edf_utils.main_sleep_period(["W"] * 200) is None


@pytest.mark.skipif(not os.path.exists(HYPNOGRAM_FILE), reason="需要 Sleep-EDF 樣本檔")
def test_cropped_hypnogram_keeps_absolute_clock_times():
    """裁切成分析區間後，時鐘時間與 TST 必須和整段記錄一致。"""
    with open(HYPNOGRAM_FILE, "rb") as fh:
        parsed = edf_utils.read_annotation_file(fh.read(), os.path.basename(HYPNOGRAM_FILE))
    hypno = edf_utils.build_hypnogram(parsed["annotations"])
    bounds = edf_utils.main_sleep_period(list(hypno["stage"]))
    assert bounds is not None

    cropped = hypno.iloc[bounds[0] : bounds[1] + 1]
    metrics = edf_utils.hypnogram_metrics(cropped, "1989-04-25 14:50:00")
    assert metrics["首次入睡時刻"] == "22:04"
    assert metrics["最後醒來時刻"] == "06:28"
    assert metrics["總睡眠時間TST(分鐘)"] == 472.0
    # 裁切掉整片白天清醒後，TST 佔比才有意義
    assert float(metrics["TST佔分析區間"].rstrip("%")) > 90


def test_build_hypnogram_ignores_non_stage_annotations():
    """呼吸事件與分期註記重疊時，不可以把該 epoch 的分期洗掉。"""
    annotations = [
        {"onset": 0.0, "duration": 300.0, "description": "Sleep stage 2"},
        {"onset": 60.0, "duration": 20.0, "description": "Obstructive apnea"},
        {"onset": 120.0, "duration": 15.0, "description": "Hypopnea"},
    ]
    hypno = edf_utils.build_hypnogram(annotations)
    assert hypno["stage"].tolist() == ["N2"] * 10   # 300 秒 = 10 個 epoch，全部維持 N2
