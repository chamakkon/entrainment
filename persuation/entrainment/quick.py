from entrainment_metrics import InterPausalUnit
ipu = InterPausalUnit(0.0, 16.0)
result = ipu.calculate_features("../speech_splitted/20210717_02_audio_mix/40.wav", pitch_gender="F", extractor="opensmile")
print(result)
