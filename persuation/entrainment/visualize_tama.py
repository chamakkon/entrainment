from entrainment_metrics import InterPausalUnit
from entrainment_metrics.tama import calculate_time_series, Frame, MissingFrame, calculate_sample_correlation
from typing import List, Union
import json
import math
import os
import matplotlib.pyplot as plt
from entrainment_metrics.continuous import plot_time_series


# フレームの生成
def create_frames(ipus: List[InterPausalUnit], frame_length: float, time_step: float) -> List[Union[Frame, MissingFrame]]:
    frames = []
    max_end_time = max(ipu.end for ipu in ipus)
    current_start = 0.0
    
    while current_start < max_end_time:
        current_end = current_start + frame_length
        frame_ipus = [ipu for ipu in ipus if ipu.start < current_end and ipu.end > current_start]
        
        if frame_ipus:
            frame = Frame(
                start=current_start,
                end=current_end,
                is_missing=False,
                interpausal_units=frame_ipus
            )
        else:
            frame = MissingFrame(start=current_start, end=current_end)
        
        frames.append(frame)
        current_start += time_step
    
    return frames

def tama_cross_correlation(session_number, feature):
    session_number=session_number
    raw_json = open("../corpus_with_time.json", "r")
    corpus = json.load(raw_json)
    session = corpus[str(session_number)]
    audio_file = "../"+session["speech_filename"]

    timestamps = [{'start':time[0]/1000 , 'end': time[1]/1000, 'speaker': session["spks"][i]} for i, time in enumerate(session["times"])]

    # InterPausalUnitオブジェクトのリストを作成
    ipus_speaker_ERICA: List[InterPausalUnit] = [InterPausalUnit(start=t['start'], end=t['end']) for t in timestamps if t['speaker'] == 'ERICA']
    ipus_speaker_User: List[InterPausalUnit] = [InterPausalUnit(start=t['start'], end=t['end']) for t in timestamps if t['speaker'] == 'User']

    for ipu in ipus_speaker_ERICA:
        ipu.calculate_features(
            audio_file=audio_file,
            pitch_gender="F",  # 音声の性別に応じて "M" または "F" を使用
            extractor="praat"
        )
    for ipu in ipus_speaker_User:
        ipu.calculate_features(
            audio_file=audio_file,
            pitch_gender="F",  # 音声の性別に応じて "M" または "F" を使用
            extractor="praat"
        )
        print(ipu.features_values)
    frame_length = 16.0
    time_step = 8.0

    # フレームのリストを作成
    frames_ERICA = create_frames(ipus_speaker_ERICA, frame_length, time_step)
    frames_User = create_frames(ipus_speaker_User, frame_length, time_step)

    time_series_ERICA: List[float] = calculate_time_series(
        feature=feature,
        frames=frames_ERICA
    )

    time_series_User: List[float] = calculate_time_series(
        feature=feature,
        frames=frames_User
    )
    for step in time_series_ERICA:
        if math.isnan(time_series_ERICA[0]):
            time_series_ERICA.pop(0)
            time_series_User.pop(0)
        else:
            break

    sample_cross_correlations: List[float] = calculate_sample_correlation(
        time_series_a=time_series_ERICA,
        time_series_b=time_series_User,
        lags=6,  # 適切なラグの数を指定
    )
    print(time_series_ERICA)
    print(time_series_User)
    print(sample_cross_correlations)


    # 時系列データのプロット
    
    plt.figure(figsize=(10, 6))
    plt.plot(time_series_ERICA, label='ERICA')
    plt.plot(time_series_User, label='User')
    plt.title(f'Time Series of {feature}')
    plt.xlabel('Time')
    plt.ylabel(feature)
    plt.legend()
    plt.savefig(f"tama_fig/timeseries_{feature}_{session_number}.png")

    # クロスコリレーションのプロット
    plt.figure(figsize=(10, 6))
    plt.plot(sample_cross_correlations)
    plt.title('Sample Cross-Correlation')
    plt.xlabel('Lag')
    plt.ylabel('Cross-Correlation')
    plt.savefig(f"tama_fig/ccr_{feature}_{session_number}")

#os.mkdir("tama_fig")
for i in range(58):
    try:
        tama_cross_correlation(i, "F0_MEAN")
    except ValueError: 
        print(f"F0_MEAN of session{i} skipped")