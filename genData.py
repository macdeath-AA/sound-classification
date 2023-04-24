import librosa
import librosa.display
import os
from pydub import AudioSegment
from pydub.utils import make_chunks
import math
import re
from tqdm import tqdm
import json

#paths
main_path = r"/home/aditi/Desktop/engineSounds"
raw_path = r"{}/audios/".format(main_path)
code_path = r"{}/process".format(main_path)
data_path = r"{}/data".format(main_path)
output_path = r"{}/output".format(main_path)

#------------------------------------------------------------------
dict_examples = {
    "ferrari": {
        "file": r"{}ferrari/{}.wav".format(raw_path, "austrian"),
        "color": "red"
    },
    "mercedes": {
        "file": r"{}mercedes/{}.wav".format(raw_path, "britain"),
        "color": "silver"
    }
}
#------------------------------------------------------------------
#Generating testing data for the neural network
raw_files = {
    "ferrari": r"{}/ferrari".format(raw_path),
    "mercedes": r"{}/mercedes".format(raw_path)
}

for team in ["ferrari", "mercedes", "montage"]:
    wav_files = os.listdir("{}/{}".format(raw_path, team))

    for file in wav_files:

        if not file.startswith("."):
            file_name = "{}/{}/{}".format(raw_path, team, file)
            myaudio = AudioSegment.from_file(file_name, "wav")
            chunk_length_ms = 1000
            chunks = make_chunks(myaudio, chunk_length_ms)

            for i, chunk in enumerate(chunks):
                padding = 3 - len(str(i))
                number = padding*"0" + str(i)
                chunk_name = "{}_{}".format(re.split(".wav", file)[0], number)
                chunk.export("{}/{}/{}.wav".format(data_path,
                             team, chunk_name), format="wav")

#------------------------------------------------------------------
#Bulk processing
data = {
    "train": {"mfcc": [], "labels": [], "category": []},
    "test": {"mfcc": [], "labels": [], "category": []},
    "montage": {"mfcc": []}
}

test_tracks = ["italy", "france"]

SAMPLE_RATE = 22050
n_mfcc = 13
n_fft = 2048
hop_length = 512

expected_num_mfcc_vectors = math.ceil(SAMPLE_RATE / hop_length)


for i, (dirpath, dirnames, filenames) in enumerate(os.walk(data_path)):
    #ensure that we are not at the root level
    if dirpath is not data_path:
        #save the team information
        dirpath_components = dirpath.split("/")
        label = dirpath_components[-1]

        #looping over the wav files
        for f in tqdm(filenames):
            if not f.startswith("."):
                file_path = os.path.join(dirpath, f)

                # extract the mfcc from the sound snippet
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                mfcc = librosa.feature.mfcc(y=signal, sr=sr,
                                            n_fft=n_fft,
                                            n_mfcc=n_mfcc,
                                            hop_length=hop_length)
                mfcc = mfcc.T.tolist()

                # to ensure that all snippets have the same length
                if len(mfcc) == expected_num_mfcc_vectors:
                    if any([track in f for track in test_tracks]):
                        data["test"]["mfcc"].append(mfcc)
                        data["test"]["labels"].append(i-1)
                        data["test"]["category"].append(label)

                    elif ("montage" in f):
                        print(f)
                        data["montage"]["mfcc"].append(mfcc)

                    else:
                        data["train"]["mfcc"].append(mfcc)
                        data["train"]["labels"].append(i-1)
                        # saving json with the results
                        data["train"]["category"].append(label)

with open("{}/processed_data.json".format(data_path), "w") as fp:
    json.dump(data, fp, indent=4)
