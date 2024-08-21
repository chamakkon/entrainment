from entrainment_metrics import InterPausalUnit, tama
from entrainment_metrics.tama import get_frames, Frame, MissingFrame
from entrainment_metrics.continuous import TimeSeries, calculate_metric
from typing import List, Union
import json
from typing import List
import os
import requests
import subprocess
import pandas as pd
def calculate_entrainment(dir, id, features):
    filename = os.path.split(dir.path)[1]
    audio_file = os.path.join(*[dir.path, "processed", f"{filename}.mp3"])
    session = pd.read_csv(os.path.join(*[dir.path, "transcription", "transcript_audiophile.csv"]))
    print(session)
    speaker_file = open(os.path.join(*[dir.path, "processed", "channel_map.json"]), "r")
    speakers = json.load(speaker_file)
    speakerA = speakers["L"]
    speakerB = speakers["R"]
    timestamps = [{'start':data[["start"]].item(), 'end': data[["stop"]].item(), 'speaker': data[["speaker"]].item()} for index, data in session[["speaker", "start", "stop"]].iterrows()]

    # InterPausalUnitオブジェクトのリストを作成
    ipus_speaker_A: List[InterPausalUnit] = [InterPausalUnit(start=t['start'], end=t['end']) for t in timestamps if t['speaker'] == speakerA]
    ipus_speaker_B: List[InterPausalUnit] = [InterPausalUnit(start=t['start'], end=t['end']) for t in timestamps if t['speaker'] == speakerB]
    data_dict ={}
    # 各InterPausalUnitについて特徴量を計算
    for feature in features:
        if feature == "speech_rate":
            extractor = "speech-rate"
        else:
            extractor = "praat"
        try:
            for ipu in ipus_speaker_A:
                ipu.calculate_features(
                    audio_file=audio_file,
                    pitch_gender="F",  # 音声の性別に応じて "M" または "F" を使用
                    extractor=extractor
                )
            for ipu in ipus_speaker_B:
                ipu.calculate_features(
                    audio_file=audio_file,
                    pitch_gender="F",  # 音声の性別に応じて "M" または "F" を使用
                    extractor=extractor
                )
        except ValueError:
            print(f"No Values for {feature} in session {id}")
            return None
        frame_length = 16.0
        time_step = 8.0

    # フレームのリストを作成
        frames_speaker_ERICA = create_frames(ipus_speaker_A, frame_length, time_step)
        frames_speaker_User = create_frames(ipus_speaker_B, frame_length, time_step)
        try:
        # 発話者AのTimeSeriesオブジェクトを作成
            time_series_speaker_a = TimeSeries(
                interpausal_units=ipus_speaker_A,
                feature=feature,
                method='knn',  # 使用する補間方法
                k=8  # k-NNのパラメータ
            )

            # 発話者BのTimeSeriesオブジェクトを作成
            time_series_speaker_b = TimeSeries(
                interpausal_units=ipus_speaker_B,
                feature=feature,
                method='knn',
                k=8
            )
        except ValueError:
            print(f"No Values for {feature} in session {id}")
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
   
        data_dict[f"{feature}-P"] = proximity_result
        data_dict[f"{feature}-C"] = convergence_result
        data_dict[f"{feature}-S"] = synchrony_result

    survey = extract_survey(dir, speakerA, speakerB)
    return dict(**data_dict, **survey)

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

def extract_survey(dir, speaker_A, speaker_B):
    csv_path = os.path.join([dir.path, "survey.csv"])
    data = pd.read_csv(csv_path)
    data_a = data["user_id"==speaker_A]
    data_b = data["user_id"==speaker_B]
    columns=data.columns
    using_columns = ["pre_affect", "pre_arousal", "affect", "arousal", "overall_affect", "overall_arousal", "end_affect", "end_arousal", "how_enjoyable", "i_like_you", "you_like_me", "in_common", "i_felt_close_to_my_partner", "i_would_like_to_become_friends", "my_partner_paid_attention_to_me" ]
    metrics_a = data_a[[using_columns]]
    metrics_b = data_b[[using_columns]]
    data_dict = {}
    for column in using_columns:
        data_dict[f"{column}-A"]=metrics_a[[column]]
        data_dict[f"{column}-B"]=metrics_b[[column]]
    return data_dict

def load_extract_remove(url, num=1):
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(num)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    filename=f'candor/{num}.zip'
    features = ['F0_MAX', 'F0_MEAN', 'F0_MIN', 'F0_MAS', 'ENG_MAX', 'ENG_MEAN', "ENG_MIN", "speech_rate"]
    using_columns = ["pre_affect", "pre_arousal", "affect", "arousal", "overall_affect", "overall_arousal", "end_affect", "end_arousal", "how_enjoyable", "i_like_you", "you_like_me", "in_common", "i_felt_close_to_my_partner", "i_would_like_to_become_friends", "my_partner_paid_attention_to_me"]
    columns = features+using_columns
    urlData = requests.get(url).content
    i=0
    df = pd.DataFrame(columns=columns)
    with open(filename ,mode='wb') as f: # wb でバイト型を書き込める
        f.write(urlData)
    subprocess.run(["unzip", f"candor/1", "-d", "candor/"])
    dirs = os.scandir("candor")
    for dir in dirs:
        print(dir.path)
        if dir.is_file():
            continue
        data = calculate_entrainment(dir,i, features)
        df_append = pd.DataFrame(data=data, columns=columns)
        df = pd.concat(df, df_append, ignore_index=True, axis=0)
    subprocess.run(["rm", "-rf", "candor"])
    subprocess.run(["mkdir", "candor"])
    return df
        
        

url='https://betterup-public-dataset-release.s3.us-west-2.amazonaws.com/v1.0/processed_media_part_001.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIASDTHSKXDMDBG7CG5%2F20240814%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20240814T234103Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=eca7f07c51916844d940ae6318f00e44bcaebe686caead62b56613b514d4eb96'
num = 1
filename=f'candor/{num}.zip'
features = ['F0_MAX', 'F0_MEAN', 'F0_MIN', 'F0_MAS', 'ENG_MAX', 'ENG_MEAN', "ENG_MIN", "speech_rate"]
using_columns = ["pre_affect", "pre_arousal", "affect", "arousal", "overall_affect", "overall_arousal", "end_affect", "end_arousal", "how_enjoyable", "i_like_you", "you_like_me", "in_common", "i_felt_close_to_my_partner", "i_would_like_to_become_friends", "my_partner_paid_attention_to_me"]
columns = features+using_columns
#urlData = requests.get(url).content
i=0
df = pd.DataFrame(columns=columns)
#with open(filename ,mode='wb') as f: # wb でバイト型を書き込める
#    f.write(urlData)
#subprocess.run(["unzip", f"candor/1", "-d", "candor/"])
dirs = os.scandir("candor")
for dir in dirs:
    print(dir.path)
    if dir.is_file():
        continue
    data = calculate_entrainment(dir,i, features)
    df_append = pd.DataFrame(data=data, columns=columns)
    print(df_append)
    df = pd.concat([df, df_append], ignore_index=True, axis=0)

print(df)
url_file = open("url_file.txt", "r")
urls = url_file.read().split("\n")
features = ['F0_MAX', 'F0_MEAN', 'F0_MIN', 'F0_MAS', 'ENG_MAX', 'ENG_MEAN', "ENG_MIN", "speech_rate"]
using_columns = ["pre_affect", "pre_arousal", "affect", "arousal", "overall_affect", "overall_arousal", "end_affect", "end_arousal", "how_enjoyable", "i_like_you", "you_like_me", "in_common", "i_felt_close_to_my_partner", "i_would_like_to_become_friends", "my_partner_paid_attention_to_me"]
columns = features+using_columns
df = pd.DataFrame(columns=columns)
for url in urls:
    df_append = load_extract_remove(url)
    df = pd.concat([df, df_append], ignore_index=True, axis=1)

df.to_csv("candor_analysis.csv")

