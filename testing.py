import numpy as np
import matplotlib.pyplot as plt
import os
from pydub import AudioSegment
from pydub.utils import make_chunks
import math
import re
from tqdm import tqdm
import json
import copy
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
from sklearn.metrics import accuracy_score
import moviepy.editor as mp


model = keras.models.load_model('my_model.h5')

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

with open("{}/processed_data.json".format(data_path), "r") as fp:
    data = json.load(fp)

#Applying the model
test_inputs = np.array(data["test"]["mfcc"])
test_targets = np.array(data["test"]["labels"])
predictions = model.predict(test_inputs)
predictions = np.argmax(predictions, axis=1)
print(predictions)
acc = accuracy_score(test_targets, predictions)

quit()
#montage test
montage_inputs = np.array(data["montage"]["mfcc"])
predictions = model.predict_on_batch(montage_inputs)
list_pred = [x.tolist()[0] for x in predictions]
engine_prediction = ["Mercedes" if x == 1 else "Ferrari" for x in list_pred]

quit()
#creating a video
montage = mp.VideoFileClip("{}/video/montage_1.mp4".format(main_path))
clip_list = []

for second, engine in tqdm(enumerate(engine_prediction)):
    engine_text = (mp.TextClip(
        "Prediction at Position {}:\n{}".format(second, engine),
        fontsize=60, color='green',
        bg_color="black")
        .set_duration(1)
        .set_start(0))

    clip_list.append(engine_text)

final_clip = (mp.concatenate(clip_list, method="compose").set_position(("center", "center")))
montage = mp.CompositeVideoClip([montage, final_clip])
    
montage.write_videofile(
    "{}/video/prediction_montage.mp4".format(main_path),fps=24, codec="mpeg4")