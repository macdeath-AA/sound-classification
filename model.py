import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
from sklearn.metrics import accuracy_score
import moviepy.editor as mp

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
#building the Neural Network

# load and convert data
with open("{}/processed_data.json".format(data_path), "r") as fp:
    data = json.load(fp)
inputs = np.array(data["train"]["mfcc"])
targets = np.array(data["train"]["labels"])  

# turn data into train and testset
(inputs_train, inputs_test,
 target_train, target_test) = train_test_split(inputs, targets,
                                               test_size=0.2) 

# build the network architecture

model = keras.Sequential([
    # input layer
    keras.layers.Flatten(input_shape=(inputs.shape[1],
                                      inputs.shape[2])),  
    
    # 1st hidden layer
    keras.layers.Dense(512, activation="relu"),
    keras.layers.Dropout(0.3),
    # 2nd hidden layer
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dropout(0.3),
    # 3rd hidden layer
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dropout(0.3),
    # output layer
    keras.layers.Dense(1, activation="sigmoid")])  

# compiling the network
optimizer = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss="binary_crossentropy",
              metrics=["accuracy"])
model.summary()  

# train the network
history = model.fit(inputs_train, target_train,
                    validation_data=(inputs_test, target_test),
                    epochs=100,
                    batch_size=32)

model.save('my_model.h5')

#------------------------------------------------------------------
#Plotting the accuracy and error over the epochs

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(
    20, 10))  

# create accuracy subplot
axs[0].plot(history.history["accuracy"], label="train_accuracy")
axs[0].plot(history.history["val_accuracy"], label="test_accuracy")
axs[0].set_ylabel("Accuracy", fontsize=18)
axs[0].legend(loc="lower right", prop={"size": 16})
axs[0].set_title("Accuracy evaluation", fontsize=20)
axs[0].tick_params(axis="both", labelsize=16)  

# create error subplot
axs[1].plot(history.history["loss"], label="train error")
axs[1].plot(history.history["val_loss"], label="test error")
axs[1].set_ylabel("Error", fontsize=18)
axs[1].legend(loc="upper right", prop={"size": 16})
axs[1].set_title("Error evaluation", fontsize=20)
axs[1].tick_params(axis="both", labelsize=16)

fig.savefig("{}/accuracy_error.png".format(output_path),
            bbox_inches="tight")


