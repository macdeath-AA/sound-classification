import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os

#paths
main_path = r"/home/aditi/Desktop/engineSounds"
raw_path = r"{}/audios/".format(main_path)
code_path = r"{}/process".format(main_path)
data_path = r"{}/data".format(main_path)
output_path = r"{}/output".format(main_path)

#------------------------------------------------------------------
# Specifying plotting information
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
#comparing the amplitudes for both cars for random tracks
fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(20, 10))
SR = 22050
axs = axs.ravel()

for num, team in enumerate(dict_examples):

    signal, sr = librosa.load(dict_examples[team]["file"], sr=SR)
    dict_examples[team]["signal"] = signal

    librosa.display.waveshow(signal, sr=sr, ax=axs[num],
                             color=dict_examples[team]["color"])

    axs[num].set_title(team, {"fontsize": 18})
    axs[num].tick_params(axis="both", labelsize=16)
    axs[num].set_ylabel("Amplitude", fontsize=18)
    axs[num].set_xlabel("Time", fontsize=18)

fig.savefig("{}/waveplot.png".format(output_path),
            bbox_inches="tight")

#------------------------------------------------------------------
# plotting all the ferrari and mercedes waveforms
plt.figure(figsize=(15,10))

fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(15, 10))
ferrari_path = "audios/mercedes"
file_list = os.listdir(ferrari_path)

for filename in file_list:
    file_path = os.path.join(ferrari_path, filename)
    signal, sr = librosa.load(file_path, sr=SR)
    team = "mercedes"
    dict_examples[team]["signal"] = signal
    librosa.display.waveshow(signal, sr=sr, ax=axs,
                             color=dict_examples[team]["color"])

    #set axis titles
    filename, ext = os.path.splitext(filename)
    axs.set_title(f"{filename}-{team}", fontsize= 18)
    axs.tick_params(axis="both", labelsize=16)
    axs.set_ylabel("Amplitude", fontsize=18)
    axs.set_xlabel("Time", fontsize=18)

    #save the files
    fig.savefig(os.path.join(output_path, f"{team}_{filename}.png"), bbox_inches="tight")

#------------------------------------------------------------------
#generating the power spectrums
fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(20, 10))
axs = axs.ravel()

for num, team in enumerate(dict_examples):

    # Calculating the fourier transform
    signal = dict_examples[team]["signal"]
    fft = np.fft.fft(signal)
    magnitude = np.abs(fft)
    frequency = np.linspace(0, sr, len(magnitude))
    left_frequency = frequency[:int(len(frequency)/2)]
    left_magnitude = magnitude[:int(len(frequency)/2)]

    #plotting and labelling
    axs[num].plot(left_frequency, left_magnitude)
    axs[num].set_title(team, {"fontsize": 18})
    axs[num].tick_params(axis="both", labelsize=16)
    axs[num].set_ylabel("Magnitude", fontsize=18)
    axs[num].set_xlabel("Frequency", fontsize=18)
    axs[num].plot(left_frequency, left_magnitude,
                  color=dict_examples[team]["color"])

fig.savefig("{}/powerspectrum.png".format(output_path),
            bbox_inches="tight")

#------------------------------------------------------------------
#Short Time Fourier Transform
n_fft = 2048  # Window for single fourier transform
hop_length = 512  # Amount for shifting to the right

#plotting
fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(20, 10))
axs = axs.ravel()

for num, team in enumerate(dict_examples):
    signal, sr = librosa.load(dict_examples[team]["file"], sr=SR)
    dict_examples[team]["signal"] = signal

    stft = librosa.core.stft(signal, hop_length=hop_length, n_fft=n_fft)
    spectogram = np.abs(stft)
    log_spectogram = librosa.amplitude_to_db(spectogram)

    plot = librosa.display.specshow(log_spectogram, sr=sr, hop_length=hop_length,
                                    ax=axs[num], x_axis='time')

    axs[num].tick_params(axis="both", labelsize=16)
    axs[num].set_title(team, {"fontsize": 18})
    axs[num].set_ylabel("Frequency", fontsize=18)
    axs[num].set_xlabel("Time", fontsize=18)

#color bar settings
cb = fig.colorbar(plot)
ticks = cb.get_ticks()
cb.set_ticks(ticks)
cb.ax.set_yticklabels(["{} dB".format(int(label)) for label in ticks])
cb.ax.tick_params(labelsize=16)

fig.savefig(r"{}/short_fourier.png".format(output_path), bbox_inches="tight")

#------------------------------------------------------------------
#Mel Frequency Ceptral Coefficients (MFCCs)
fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(20, 10))
axs = axs.ravel()

for num, team in enumerate(dict_examples):
    signal = dict_examples[team]["signal"]
    MFCCs = librosa.feature.mfcc(
        y=signal, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)

    plot = librosa.display.specshow(
        MFCCs, sr=sr, hop_length=hop_length, ax=axs[num], x_axis='time')

    axs[num].tick_params(axis="both", labelsize=16)
    axs[num].set_title(team, {"fontsize": 18})
    axs[num].set_ylabel("Time", fontsize=18)
    axs[num].set_xlabel("Frequency", fontsize=18)

cb = fig.colorbar(plot)
ticks = cb.get_ticks()
cb.set_ticks(ticks)
cb.ax.set_yticklabels(["{} dB".format(int(label)) for label in ticks])
cb.ax.tick_params(labelsize=16)

fig.savefig(r"{}/mfccs.png".format(output_path), bbox_inches="tight")