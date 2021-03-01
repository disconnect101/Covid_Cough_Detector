import os
import librosa
import librosa.display
import IPython.display as ipd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as preproc
import soundfile as sf
import json

import constants





def func():
    audio_file = "cough-heavy.wav"
    ipd.Audio(audio_file)

    scale, sr = librosa.load(audio_file)
    frame_size = 2048
    hop_size = 512

    stft_scale = librosa.stft(scale, n_fft=frame_size, hop_length=hop_size)

    stft_scale.shape

    amplitude_stft_scale = np.abs(stft_scale)**2
    decibal_stft_scale = librosa.power_to_db(amplitude_stft_scale)



    #plt.figure(figsize=(25, 10))
    #librosa.display.specshow(decibal_stft_scale, sr=sr, hop_length=hop_size, x_axis="time", y_axis="linear")
    #plt.colorbar(format="%+2.f")

    #plt.show()

    #print(scale[:int(frame_size/5)])
    enery_progression = librosa.feature.rms(y=scale, frame_length=frame_size, hop_length=hop_size)
    #print(enery_progression)
    enery_progression = np.array(enery_progression[0])
    normalized_energy_progression = preproc.normalize([enery_progression], norm='max')
    #plt.figure(figsize=(25, 10))
    #plt.plot(np.arange(len(normalized_energy_progression[0])), normalized_energy_progression[0])
    #plt.show()
    #print(normalized_energy_progression)


    cleaned_scales = data_clean(scale, sr, frame_size, hop_size, 0.4, 7500)
    #print(cleaned_scales)
    scales = np.array([])
    for i, scale in enumerate(cleaned_scales):
        save_scale_as_wav(scale, sr, ".\\", "cleaned-{}".format(i))
        scales = np.concatenate((scales, scale))
    enery_progression = librosa.feature.rms(y=scales, frame_length=frame_size, hop_length=hop_size)
    enery_progression = np.array(enery_progression[0])
    normalized_energy_progression = preproc.normalize([enery_progression], norm='max')
    plt.plot(np.arange(len(normalized_energy_progression[0])), normalized_energy_progression[0])
    plt.show()

    stft_scale = librosa.stft(scales, n_fft=frame_size, hop_length=hop_size)
    amplitude_stft_scale = np.abs(stft_scale)**2
    decibal_stft_scale = librosa.power_to_db(amplitude_stft_scale)
    #
    plt.figure(figsize=(15, 10))
    librosa.display.specshow(decibal_stft_scale, sr=sr, hop_length=hop_size, x_axis="time", y_axis="linear")
    #plt.colorbar(format="%+2.f")
    #plt.show()

    save_scale_as_wav(scales, sr, ".\\", "cleaned.wav")

    #for i, scale in enumerate(cleaned_scales):
     #   scale_to_wave(scale, sr, "cough"+str(i))


def data_clean(scale, sr, frame_length, hop_length, threshold, duration_filter):
    status = "OFF"
    start = None
    end = None
    cleaned_frame_data_intervals = []
    cleaned_scale_data_intervals = []
    cleaned_scales = []

    energy_progression = librosa.feature.rms(y=scale, frame_length=frame_length, hop_length=hop_length)
    energy_progression = np.array(energy_progression[0])
    normalized_energy_progression = preproc.normalize([energy_progression], norm='max')
    normalized_energy_progression = normalized_energy_progression[0]

    for i, energy in enumerate(normalized_energy_progression):
        if status=="OFF" and energy>=threshold:
            status = "ON"
            start = i
            continue

        if status=="ON" and energy<threshold:
            status = "OFF"
            end = i
            cleaned_frame_data_intervals.append((start, end))
            continue

    if status=="ON":
        cleaned_frame_data_intervals.append((start, len(normalized_energy_progression)-1))

    #### Data frames numbers to indexes
    for frame_interval in cleaned_frame_data_intervals:
        start_frame = frame_interval[0]
        end_frame = frame_interval[1]

        start_scale_index = (start_frame-1)*hop_length
        end_scale_index = (end_frame-2)*hop_length + frame_length - 1

        cleaned_scale_data_intervals.append((start_scale_index, end_scale_index))

    #print(cleaned_scale_data_intervals)
    #### Data indexes to cleaned scale
    for scale_interval in cleaned_scale_data_intervals:
        if scale_interval[1]-scale_interval[0] > duration_filter:
            cleaned_scales.append(scale[scale_interval[0]: scale_interval[1]])

    return cleaned_scales


def save_scale_as_wav(scale, sr, path, filename):
    if not os.path.exists(path):
        os.makedirs(path)
    sf.write(path+"{}.wav".format(filename), scale, sr)


def load_audio(path):
    return librosa.load(path)


def process(data_point, folder):
    print("processing: " + data_point + "...")
    path = ".\..\Coswara-Data\Extracted_data" + "\{}\{}\cough-heavy.wav".format(folder, data_point)
    scale, sr = load_audio(path)

    cleaned_scales = data_clean(scale, sr, constants.FRAME_SIZE, constants.HOP_SIZE, 0.01, 7500)
    for i, scale in enumerate(cleaned_scales):
        save_path = ".\..\Cleaned-data\\" + folder + "\\" + data_point + "\\"
        save_filename = "cleaned-cough-heavy-{}.wav".format(i)
        save_scale_as_wav(scale, sr, save_path, save_filename)

    return len(cleaned_scales)

def main():
    i = 0
    error_logs = {
        "data-points": [],
    }
    for root, dirs, files in os.walk(".\..\Coswara-Data\Extracted_data"):
        if "cough-heavy.wav" in files:
            arr = root.split("\\")
            data_point = arr[-1]
            folder = arr[-2]
            try:
                num_of_processedfiles = process(data_point, folder)
            except Exception as e:
                print(str(e))
                error_logs["data-points"].append( { "data_point": data_point, "error": str(e) } )
                continue

            print(str(i) + ": " + data_point + " cleaned", str(num_of_processedfiles) + " cleaned files generated")
            i = i + 1

    with open(".\errors.txt", 'w') as error_logs_file:
        json.dump(error_logs, error_logs_file)

if __name__=="__main__":
    #func()
    main()