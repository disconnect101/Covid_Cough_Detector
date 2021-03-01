import os
import librosa
import csv
import constants
import math
import json

class DataGenerator:
    data_map = {}
    data_csv_path = None
    data_audio_path = None
    final_dataset = {
        'MFCC': [],
        'result': [],
    }
    number_of_samples_in_a_segment = None
    expected_number_of_frames_in_mfcc = None


    def __init__(self, DATA_CSV_PATH, DATA_AUDIO_FOLDER_PATH):
        self.data_csv_path = DATA_CSV_PATH
        self.data_audio_path = DATA_AUDIO_FOLDER_PATH
        self.number_of_samples_in_a_segment = constants.SR * constants.SEGMENT_DURATION
        self.expected_number_of_frames_in_mfcc = int(math.ceil(self.number_of_samples_in_a_segment/constants.HOP_SIZE))

    def start(self):
        self.load_csv()
        self.generate_MFCCs()
        self.save_data()

    def load_csv(self):
        try:
            with open(constants.DATA_CSV_PATH, 'r') as data_csv:
                csvreader = csv.reader(data_csv)
                fields = next(csvreader)
                print(fields[12])

                print("Loading CSV...")
                for row in csvreader:
                    id = row[0]
                    self.data_map[id] = row[-4]
        except Exception as e:
            print("Unable to load CSV: ", str(e))

        return

    def generate_MFCCs(self):
        for root, dirs, files in os.walk(self.data_audio_path):
            result = None
            if root.split("\\")[-1] == 'neg':
                result = 'N'
            if root.split("\\")[-1] == 'pos':
                result = 'P'

            if result:
                for file_name in files:
                    path_to_file = root + "\\" + file_name
                    scale, sr = librosa.load(path_to_file, sr=constants.SR)


                    number_of_segments = int(math.ceil(len(scale)/self.number_of_samples_in_a_segment))

                    for i in range(number_of_segments):
                        start_sample_index = self.number_of_samples_in_a_segment*i
                        end_sample_index = start_sample_index + self.number_of_samples_in_a_segment - 1

                        if end_sample_index >= len(scale):
                            end_sample_index = len(scale) - 1
                            start_sample_index = end_sample_index - self.number_of_samples_in_a_segment + 1

                        mfcc = librosa.feature.mfcc(scale[start_sample_index: end_sample_index],
                                                    sr=sr,
                                                    n_fft=constants.FRAME_SIZE,
                                                    n_mfcc=constants.NUMBER_OF_COFF_MFCC,
                                                    hop_length=constants.HOP_SIZE)

                        mfcc = mfcc.T

                        if len(mfcc)==self.expected_number_of_frames_in_mfcc:
                            self.final_dataset['MFCC'].append(mfcc.tolist())
                            if result=='P':
                                result = 1
                            if result=='N':
                                result = 0
                            self.final_dataset['result'].append(result)

                            print("segment {} of {} completed.".format(i, path_to_file))


    def save_data(self):
        with open(constants.JSON_DATA_PATH, 'w') as fp:
            json.dump(self.final_dataset, fp, indent=4)



if __name__=="__main__":
    data_generator = DataGenerator(constants.DATA_CSV_PATH, constants.DATA_AUDIO_FOLDER_PATH)
    data_generator.start()


