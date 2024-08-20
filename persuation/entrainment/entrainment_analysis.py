
from entrainment_metrics import InterPausalUnit, tama
from entrainment_metrics.tama import get_frames, Frame, MissingFrame
from entrainment_metrics.continuous import TimeSeries, calculate_metric
from typing import List, Union
import json
from typing import List

def calculate_entrainment(session_num, feature):
    raw_json = open("../corpus_with_time.json", "r")
    corpus = json.load(raw_json)
    session = corpus[str(session_num)]
    audio_file = "../"+session["speech_filename"]
    timestamps = [{'start':time[0]/1000 , 'end': time[1]/1000, 'speaker': session["spks"][i]} for i, time in enumerate(session["times"])]

    # InterPausalUnitオブジェクトのリストを作成
    ipus_speaker_ERICA: List[InterPausalUnit] = [InterPausalUnit(start=t['start'], end=t['end']) for t in timestamps if t['speaker'] == 'ERICA']
    ipus_speaker_User: List[InterPausalUnit] = [InterPausalUnit(start=t['start'], end=t['end']) for t in timestamps if t['speaker'] == 'User']

    # 各InterPausalUnitについて特徴量を計算
        
    if feature == "speech_rate":
        extractor = "speech-rate"
    else:
        extractor = "praat"
    try:
        for ipu in ipus_speaker_ERICA:
            ipu.calculate_features(
                audio_file=audio_file,
                pitch_gender="F",  # 音声の性別に応じて "M" または "F" を使用
                extractor=extractor
            )
        for ipu in ipus_speaker_User:
            ipu.calculate_features(
                audio_file=audio_file,
                pitch_gender="F",  # 音声の性別に応じて "M" または "F" を使用
                extractor=extractor
            )
    except ValueError:
        print(f"No Values for {feature} in session {session_num}")
        return None
    frame_length = 16.0
    time_step = 8.0

# フレームのリストを作成
    frames_speaker_ERICA = create_frames(ipus_speaker_ERICA, frame_length, time_step)
    frames_speaker_User = create_frames(ipus_speaker_User, frame_length, time_step)
    try:
    # 発話者AのTimeSeriesオブジェクトを作成
        time_series_speaker_a = TimeSeries(
            interpausal_units=ipus_speaker_ERICA,
            feature=feature,
            method='knn',  # 使用する補間方法
            k=8  # k-NNのパラメータ
        )

        # 発話者BのTimeSeriesオブジェクトを作成
        time_series_speaker_b = TimeSeries(
            interpausal_units=ipus_speaker_User,
            feature=feature,
            method='knn',
            k=8
        )
    except ValueError:
        print(f"No Values for {feature} in session {session_num}")
        return None
    print(f"Calculating {feature} entrainment")
    convergence_result = calculate_metric(
        metric="convergence",
        time_series_a=time_series_speaker_a,
        time_series_b=time_series_speaker_b
    )

    print("Convergence:", convergence_result)

    synchrony_result = calculate_metric(
    metric="synchrony",
        time_series_a=time_series_speaker_a,
        time_series_b=time_series_speaker_b
    )
    print("Synchrony:", synchrony_result)

    proximity_result = calculate_metric(
        metric="proximity",
        time_series_a=time_series_speaker_a,
        time_series_b=time_series_speaker_b
    )

    print("Proximity", proximity_result)
   
    return { "proximity" : proximity_result, "convergence": convergence_result, "synchrony" : synchrony_result}



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

f = open("entrainment.txt", "a")
values_dict = {}
for i in range(58):
    session_values = {}
    print(f"\n\nCurrently in session{str(i)}\n\n")
    session_values["F0_MAX"] = calculate_entrainment(i, "F0_MAX")
    session_values["F0_MEAN"] = calculate_entrainment(i, "F0_MEAN")
    session_values["F0_MAS"] = calculate_entrainment(i, "F0_MAS")
    session_values["ENG_MAX"] = calculate_entrainment(i, "ENG_MAX")
    session_values["ENG_MEAN"] = calculate_entrainment(i, "ENG_MEAN")
    session_values["speech_rate"] = calculate_entrainment(i, "speech_rate")
    values_dict[str(i)] = session_values

with open("entrainment_score.json", mode="wt", encoding="utf-8") as f:
	json.dump(values_dict, f, ensure_ascii=False, indent=2)
    