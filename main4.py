# 10/20/2025
# Riley Mohr

print("Importing libraries")
import kagglehub
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
from IPython.display import Audio, display
import soundfile as sf

# 10/30/2025

import torch
#import os
# from torch import nn
# from torch.utils.data import DataLoader
# from torchvision import datasets,transforms
print("done importing libraries")


# 11/10/2025
# method to parse the spectrogram into windows, probably works with more than spectrograms as well
# parse the spectrogram spec into windows of width frames every interval frames
def parse_spectrogram(specT,window_width,interval):
    spec=specT.T

    #initialize expandable list
    windowed_spec=[]
    # append to expandable list
    # windowed_spec.append(1)

    # amount of horizontal length or time length of spectrogram
    total_spec_size=spec.shape[0]

    # total number of windows, minus 1 potentially by integer rounding down
    total_window_count=int(total_spec_size/interval)
    
    # be sure to account for last window not lining up; 
    # if there is leftover spectrogram space left then 
    # snap the remaining amount to the right end of the spectrogram, shifting it left to do so
    dealt_with_final_window=False

    for i in range(total_window_count):
        window_start_index=int(i*interval) #perform floor operation, it all should work
        window_end_index=int(i*interval+window_width) #perform floor operation
        if window_end_index>total_spec_size:
            # deal with final window here
            windowed_spec.append(spec[-interval:].T)

            dealt_with_final_window=True
            break
        # insert window into windowed_spec
        windowed_spec.append(spec[window_start_index:window_end_index].T)
    
    if dealt_with_final_window==False:
        # determine if a final window is even needed
        last_window_end_index=int(total_window_count*interval+window_width)
        if last_window_end_index<total_spec_size:
            # deal with final window here
            windowed_spec.append(spec[-int(interval):].T) #really weird that i need to add int() here but not above but ok
            # not needed because the variable isn't checked again
            # dealt_with_final_window=True

    # checks
    print(f"spectrogram:")
    print(spec)
    print(f"total spectrogram size: {total_spec_size}")
    print(f"width: {window_width}")
    print(f"interval: {interval}")
    print(f"windowed spectrogram list: ")
    print(windowed_spec)
    if windowed_spec:
        print(f"shape of first window: {windowed_spec[0].shape}")
        print(f"shape of second window: {windowed_spec[1].shape}")
        print(f"shape of second to last window: {windowed_spec[-2].shape}")
        print(f"shape of last window: {windowed_spec[-1].shape}")
    else:
        print("windowed_spec is empty :(")
    return windowed_spec




# Set device type
print(f"CUDA available? {torch.cuda.is_available()}")
print(f"CPU available? {torch.cpu.is_available()}")

deviceName=""
# set to false if no gpu, set to true if there is gpu
if(torch.cuda.is_available()):
    deviceName="gpu"
elif(torch.cpu.is_available()):
    deviceName="cpu"
else:
    print("No device available!!!")
    quit()
print(f"Using {deviceName}")


"""class HiMom(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatted = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10),

        )

    def forward(self, x):
        x=self.flatten(x)
        logits=self.linear_relu_stack(x)
        return logits


some_data=[[2,3,4],[3,4,5]]

#model = HiMom().to("cuda")
#model = HiMom().to("cpu")
model = HiMom().to(deviceName)
X = some_data
logits=model(X)
pred_probab=nn.Softmax(dim=1)(logits)
y_pred=pred_probab.argmax(1)
print(f"And my prediction is... {y_pred}")

"""


#Get all of the data



# Download latest version
internet = False #because apparently my code can't run without internet without this
if internet:
    print("Downloading dataset")
    path = kagglehub.dataset_download("uwrfkaggler/ravdess-emotional-speech-audio")
else:
    print("Dataset already downloaded")
    path="C:\\Users\\iwbmo\\.cache\\kagglehub\\datasets\\uwrfkaggler\\ravdess-emotional-speech-audio\\versions\\1"
print()
print()
print("Path to dataset files:", path)

printone=True


# https://stackoverflow.com/questions/44787437/how-to-convert-a-wav-file-to-a-spectrogram-in-python3
#import matplotlib.pyplot as plt
#from scipy import signal
#from scipy.io import wavfile

#sample_rate, samples = wavfile.read('path-to-mono-audio-file.wav')
file=(f"{path}\\Actor_01\\03-01-01-01-01-01-01.wav")
print(file)
#sample_rate, samples = wavfile.read(f'{path}\\Actor_01\\03-01-01-01-01-01-01.wav')
#sample_rate, samples = wavfile.read(file)
#frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

#plt.pcolormesh(times, frequencies, spectrogram)
#plt.imshow(spectrogram)
#plt.ylabel('Frequency [Hz]')
#plt.xlabel('Time [sec]')
#plt.show()


# 1. get window
window_width=1024
# 2. window step
window_step=512
window_max_pos=window_width+1

# 5. find top and bottom 5% energies from the distribution (???)
#top_5_percent_energy = 0.05 * np.percentile(rms, 95)
#bottom_5_percent_energy = 0.05 * np.percentile(rms, 5)

cmap_name="magma"

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def db_to_rgb(db_value, min_db=-80.0, max_db=0.0, cmap_name=cmap_name):
    # 1. Create the Normalizer
    # This maps the input range [min_db, max_db] to [0.0, 1.0]
    norm = mcolors.Normalize(vmin=min_db, vmax=max_db)
    
    # 2. Get the Colormap object from matplotlib
    cmap = plt.get_cmap(cmap_name)
    
    # 3. Calculate the color
    # norm(db_value) converts dB to 0-1 scale
    # cmap(...) takes that 0-1 value and returns (R, G, B, A) in 0.0-1.0 floats
    rgba_color = cmap(norm(db_value))
    
    # 4. Return only RGB (first 3 values)
    # If you need 0-255 integers, multiply these by 255 and cast to int
    return rgba_color #[:3] # removing alpha remover

method=2
# get the list of the three threshold types given the rms_waveform
def getThresholds(rms_waveform):
    # A: given method
    # 6A1. set threshold to 5th percentile, essentially always cutting out exactly 5% of the data
    threshold1=np.percentile(rms_waveform,5)
    # B: custom method
    # 6B1. set threshold to 0.05*95th percentile
    threshold2=0.05*np.percentile(rms_waveform,95)
    # C: poor thresholding method
    # 6C1. set threshold to 0.05*(max energy)
    threshold3=0.05*np.max(rms_waveform)
    return [threshold1,threshold2,threshold3]

# get the specified threshold given the threshold method selected
def getThreshold(rms_waveform,method:int):
    """
    # branch off into two methods:

    # A: given method
    # 6A1. set threshold to 5th percentile, essentially always cutting out exactly 5% of the data
    threshold1=np.percentile(rms_waveform,5)
    # B: custom method
    # 6B1. set threshold to 0.05*95th percentile
    threshold2=0.05*np.percentile(rms_waveform,95)
    # C: poor thresholding method
    # 6C1. set threshold to 0.05*(max energy)
    threshold3=0.05*np.max(rms_waveform)
    if method==1:
        threshold=threshold1
    elif method==2:
        threshold=threshold2
    elif method==3:
        threshold=threshold3"""

    return getThresholds(rms_waveform)[method-1]

                    
# method to get cut spectrogram data given file path to audio file
# returns 1 if file does not exist
# returns 2 if file path does not exist or there is an error with loading the audio file
def getCutAudio(file_path):
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found.")
        return 1,1 #failure
    # Load the file
    try:
        y,sr=librosa.load(file_path,sr=None)
    except Exception as e:
        # some files don't have opening permissions, don't know why
        print(f"An error occurred for file path {file_path}: {e}")
        return 2,2 #failure
    
    y, sr = librosa.load(file_path, sr=None) #sr=None keeps the original sampling rate
    D = librosa.stft(y)
    rms_waveform = librosa.feature.rms(y=y, frame_length=window_width, hop_length=window_step)[0]
    # show energy over time graph with different thresholds
    # Create time axis for RMS (one time value per frame)
    frames = np.arange(len(rms_waveform))
    #t_rms = frames * window_step / sr  # convert frame index → seconds
    # calculate the post-cut width
    # cut off data outside of thresholds from data
    thresholds=getThresholds(rms_waveform)
    #threshold=getThreshold(rms_waveform,method)
    #thresholds=[threshold1,threshold2,threshold3]
    cut_datas_indeces_waveform=np.where(rms_waveform<thresholds[method])[0]
    #cut_datas_indeces_y=0
    cut_datas_indeces_spectrogram=cut_datas_indeces_waveform
    # datas removed by filter
    #cut_out_datas_waveform=rms_waveform[cut_datas_indeces_waveform]
    
    # Define the spectrogram as S_db
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    #cut_out_datas_spectrogram=S_db[:,cut_datas_indeces_spectrogram]
    # datas kept after filter
    cut_datas_waveform=np.delete(rms_waveform,cut_datas_indeces_waveform)
    cut_datas_spectrogram=np.delete(S_db,cut_datas_indeces_spectrogram,axis=1)
    return cut_datas_waveform, cut_datas_spectrogram



#rms_waveform = librosa.feature.rms(y=y, frame_length=window_width, hop_length=window_step)[0]



# 1. Define the path to your RAVDESS audio file.
# Replace this with the actual path to a .wav file from your dataset.
# Example path structure: 'RAVDESS/Actor_01/03-01-01-01-01-01-01.wav'
# The file naming convention details the emotion and intensity.
# Example filename: 03-01-06-01-02-01-12.wav (audio-only, speech, fearful, normal intensity, statement 2, repetition 1, actor 12)
file_path_working = file#'RAVDESS/Actor_01/03-01-06-01-02-01-12.wav'


# find smallest file, use its window length as the default
print("Finding smallest size")
smallestsize=float('inf')
num_files_success=0
num_files_total=0
for folder in os.listdir(path):
    for filename in os.listdir(f"{path}\\{folder}"):
        file_path=f"{path}\\{folder}\\{filename}"
        num_files_total+=1
        # Check if file exists
        #if not os.path.exists(file_path):
        #    print(f"Error: The file '{file_path}' was not found.")
        #    continue
        
        """# Load the file
        try:
            y,sr=librosa.load(file_path,sr=None)
        except Exception as e:
            # some files don't have opening permissions, don't know why
            print(f"An error occurred for file path {file_path}: {e}")
            continue
        num_files_success+=1
        
        y, sr = librosa.load(file_path, sr=None) #sr=None keeps the original sampling rate
        D = librosa.stft(y)
        rms_waveform = librosa.feature.rms(y=y, frame_length=window_width, hop_length=window_step)[0]
        # show energy over time graph with different thresholds
        # Create time axis for RMS (one time value per frame)
        frames = np.arange(len(rms_waveform))
        t_rms = frames * window_step / sr  # convert frame index → seconds
        # calculate the post-cut width
        # cut off data outside of thresholds from data
        thresholds=getThresholds(rms_waveform)
        threshold=getThreshold(rms_waveform,method)
        #thresholds=[threshold1,threshold2,threshold3]
        cut_datas_indeces_waveform=np.where(rms_waveform<thresholds[method])[0]
        #cut_datas_indeces_y=0
        cut_datas_indeces_spectrogram=cut_datas_indeces_waveform
        # datas removed by filter
        cut_out_datas_waveform=rms_waveform[cut_datas_indeces_waveform]
        
        # Define the spectrogram as S_db
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        cut_out_datas_spectrogram=S_db[:,cut_datas_indeces_spectrogram]
        # datas kept after filter
        cut_datas_waveform=np.delete(rms_waveform,cut_datas_indeces_waveform)
        cut_datas_spectrogram=np.delete(S_db,cut_datas_indeces_spectrogram,axis=1)"""

        cut_datas_waveform, cut_datas_spectrogram=getCutAudio(file_path)
        # continue on errors
        if(type(cut_datas_waveform)==int):
            continue
        num_files_success+=1

        # Print its shape
        #print(f"cut_datas_spectrogram shape for {file_path}: {cut_datas_spectrogram.shape}")
        #print(f"shape of {file_path}: {y.shape[0]}")
        if cut_datas_spectrogram.shape[1]<smallestsize:
            smallestsize=cut_datas_spectrogram.shape[1]

print(f"Successfully opened {num_files_success}/{num_files_total} files")
print(f"smallest size: {smallestsize} cut spectrogram frames")


#Load and plot spectrograms for all .wav files
for folder in os.listdir(path):
    print(folder)
    for filename in os.listdir(f"{path}\\{folder}"):
        print(f"   {filename}")
        if filename.lower().endswith(".wav"):
            #print("asdas")
            file_path=f"{path}\\{folder}\\{filename}"
            # Check if the file exists
            if not os.path.exists(file_path):
                print(f"Error: The file '{file_path}' was not found.")
            else:
                #print("BASBDABV")
                # 2. Load the audio file
                # librosa.load returns the audio time series (y) and the sampling rate (sr)
                #try:
                    print(f"THE file path1: {file_path}")

                    # load the audio file
                    print(f"loading audio file {file_path}")
                    y, sr = librosa.load(file_path, sr=None) #sr=None keeps the original sampling rate
                    print(f"loaded audio file {file_path}")
                    
                    print(f"sr: {sr} Shape of y: {y.shape}")
                    #print(sr)
                    #print(f"Shape of sr:{sr.shape}")
                    #print("y:")
                    #print(y)
                    #print(f"Shape of y:{y.shape}")
                    
                    
                    # get rms energy list from data
                    # 3a. get energy and store it in a list
                    print(f"calculating rms_waveform for {file_path}")
                    rms_waveform = librosa.feature.rms(y=y, frame_length=window_width, hop_length=window_step)[0]
                    print(f"done calculating rms_waveform for {file_path}")

                    # 4. construct energy distribution
                    
                    
                    

                    # 5. find top and bottom 5% energies from the distribution (???)
                    #top_5_percent_energy = 0.05 * np.percentile(rms, 95)
                    #bottom_5_percent_energy = 0.05 * np.percentile(rms, 5)
                    
                    #method=2
                    # branch off into two methods:
                    threshold=getThreshold(rms_waveform,method)
                    # A: given method
                    # 6A1. set threshold to 5th percentile, essentially always cutting out exactly 5% of the data
                    #threshold1=np.percentile(rms_waveform,5)
                    # B: custom method
                    # 6B1. set threshold to 0.05*95th percentile
                    #threshold2=0.05*np.percentile(rms_waveform,95)
                    # C: poor thresholding method
                    # 6C1. set threshold to 0.05*(max energy)
                    #threshold3=0.05*np.max(rms_waveform)
                    #if method==1:
                    #    threshold=threshold1
                    #elif method==2:
                    #    threshold=threshold2
                    #elif method==3:
                    #    threshold=threshold3
                    thresholds=getThresholds(rms_waveform)
                    # 7. cut off data outside of threshold from data
                    cut_data_indices=np.where(rms_waveform<threshold)
                    cut_data=np.delete(rms_waveform,cut_data_indices)
                    # 8. convert to spectrogram
                    # 9. plot spectrogram of cut and kept and uncut audio data
                    
                    # 3. Compute the Short-Time Fourier Transform (STFT)
                    # This converts the audio signal into a spectrogram
                    # The result is complex, so we take the absolute value to get the magnitude
                    # https://librosa.org/doc/latest/generated/librosa.stft.html
                    # documentation says suggested nfft as 512
                    D = librosa.stft(y)
                    print(f"")
                    #print("D:")
                    #print(D)
                    print(f"Shape of D:{D.shape}")
                    # Define the spectrogram as S_db
                    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
                    #print("S_db:")
                    #print(S_db)
                    print(f"Shape of S_db:{S_db.shape}")
                    
                    plt.subplot(4,1,1)
                    print("Showing original:")
                    #plt.figure(figsize=(10, 5))
                    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz',cmap=cmap_name)
                    plt.colorbar(format='%+2.0f dB')
                    plt.title('Original RAVDESS audio file')
                    plt.xlabel("Time (s)") #redundant?
                    plt.ylabel("Frequency (Hz)") #redundant?
                    #plt.tight_layout()

                    
                    plt.subplot(4, 1, 2)
                    #print(rms_waveform)
                    # Create time axis in seconds
                    time = np.arange(len(y)) / sr
                    plt.plot(time, y, label="waveform", color='orange')
                    #plt.plot(np.linspace(0, len(energy), len(new_energy)), new_energy, label="Processed STE", color='blue')
                    #plt.axhline(y=threshold1, color='red', linestyle='--', label="Threshold1 (5th percentile)")
                    #plt.axhline(y=threshold2, color='orange', linestyle='--', label="Threshold2 (0.05*95th percentile)")
                    #plt.axhline(y=threshold3, color='yellow', linestyle='--', label="Threshold3 (0.05*max value)")
                    plt.title("Waveform")
                    plt.xlabel("Seconds")
                    plt.ylabel("Pressure")
                    plt.legend()

                    # show energy over time graph with different thresholds
                    # Create time axis for RMS (one time value per frame)
                    frames = np.arange(len(rms_waveform))
                    t_rms = frames * window_step / sr  # convert frame index → seconds
                    # (3) Short-term energy comparison
                    plt.subplot(4, 1, 3)
                    print(f"Shape of rms_waveform: {rms_waveform.shape}")
                    #print(rms_waveform)
                    plt.plot(t_rms, rms_waveform, label="RMS", color='orange')
                    #plt.plot(np.linspace(0, len(energy), len(new_energy)), new_energy, label="Processed STE", color='blue')
                    plt.axhline(y=thresholds[0], color='red', linestyle='--', label="Threshold1 (5th percentile)")
                    plt.axhline(y=thresholds[1], color='orange', linestyle='--', label="Threshold2 (0.05*95th percentile)")
                    plt.axhline(y=thresholds[2], color='yellow', linestyle='--', label="Threshold3 (0.05*max value)")
                    plt.title("Energy Graph")
                    plt.xlabel("Seconds")
                    plt.ylabel("Energy")
                    plt.legend()
                    #plt.show()

                    
                    # cut off data outside of thresholds from data
                    #thresholds=[threshold1,threshold2,threshold3]
                    cut_datas_indeces_waveform=np.where(rms_waveform<thresholds[method])[0]
                    #cut_datas_indeces_y=0
                    cut_datas_indeces_spectrogram=cut_datas_indeces_waveform
                    # datas removed by filter
                    cut_out_datas_waveform=rms_waveform[cut_datas_indeces_waveform]
                    cut_out_datas_spectrogram=S_db[:,cut_datas_indeces_spectrogram]
                    # datas kept after filter
                    cut_datas_waveform=np.delete(rms_waveform,cut_datas_indeces_waveform)
                    cut_datas_spectrogram=np.delete(S_db,cut_datas_indeces_spectrogram,axis=1)

                    print("Showing filtered:")
                    #plt.figure(figsize=(10, 5))
                    plt.subplot(5, 2, 9)
                    librosa.display.specshow(cut_datas_spectrogram, sr=sr, x_axis='time', y_axis='hz',cmap=cmap_name)
                    plt.colorbar(format='%+2.0f dB')
                    plt.title('Filtered RAVDESS audio file')
                    plt.xlabel("Time (s)") #redundant?
                    plt.ylabel("Frequency (Hz)") #redundant?
                    #plt.tight_layout()
                    
                    print("Showing filtered away:")
                    #plt.figure(figsize=(10, 5))
                    plt.subplot(5, 2, 10)
                    librosa.display.specshow(cut_out_datas_spectrogram, sr=sr, x_axis='time', y_axis='hz',cmap=cmap_name)
                    plt.colorbar(format='%+2.0f dB')
                    #print("deletion indices:")
                    #print(np.unique(deletion_indices))
                    #plt.axvline(x=np.unique(deletion_indices).any(), color='red', linestyle='--', label="Threshold")
                    plt.title('Removed parts from RAVDESS audio file')
                    plt.xlabel("Time (s)") #redundant?
                    plt.ylabel("Frequency (Hz)") #redundant?
                    #plt.tight_layout()


                    
                    plt.tight_layout()
                    #plt.show()




                    # chatgpt play back audio

                    # Convert frame indices (which are per-hop) to sample indices
                    hop_length = window_step  # you already defined this
                    frame_length = window_width
                    
                    # Build a mask of which samples to keep
                    mask = np.ones_like(y, dtype=bool)
                    
                    # Convert frame indices to sample ranges and mark as False where rms < threshold
                    for frame_idx in cut_datas_indeces_waveform:
                        start = frame_idx * hop_length
                        end = min(start + frame_length, len(y))
                        mask[start:end] = False
                    
                    # Separate kept and removed parts
                    y_kept = y[mask]
                    y_removed = y[~mask]
                    
                    print("Kept audio samples:", y_kept.shape)
                    print("Removed audio samples:", y_removed.shape)
                    
                    # Play them!
                    print("🎧 Kept (speech) audio:")
                    display(Audio(y_kept, rate=sr))
                    
                    print("🔇 Removed (silence/noise) audio:")
                    display(Audio(y_removed, rate=sr))

                    sf.write("kept.wav", y_kept, sr)
                    sf.write("removed.wav", y_removed, sr)
                    print("✅ Saved 'kept.wav' and 'removed.wav'")










                    # get the conversion constant from spectrogram frames to sample frames to preserve time
                    spec_frames_per_sample=1/window_step

                    # Parse the filtered spectrogram into windows of uniform length
                    # window size in seconds
                    window_size_seconds=0.02 # 960 frames
                    # window interval in seconds
                    window_interval_seconds=0.05 # 2400 frames
                    # window size in # of frames
                    window_size_frames=window_size_seconds*sr*spec_frames_per_sample # < 1 probably
                    # window interval in # of frames
                    window_interval_frames=window_interval_seconds*sr*spec_frames_per_sample
                    
                    parsed_spectrogram=parse_spectrogram(cut_datas_spectrogram,window_size_frames,window_interval_frames)

                    #cut_spectrogram_half_width=smallestsize*sr*spec_frames_per_sample/2
                    cut_spectrogram_half_width=smallestsize/2
                    left_bound=int(S_db.shape[0]/2-cut_spectrogram_half_width)
                    right_bound=int(S_db.shape[0]/2+cut_spectrogram_half_width)
                    print(f"window left bound: {left_bound}")
                    print(f"window right bound: {right_bound}")
                    print(f"window right bound-left bound: {right_bound-left_bound}")
                    #exact center doesn't matter; what matters is the same width
                    spectrogram_value_data=S_db[left_bound:right_bound]
                    # now the spectrogram is a bunch of numbers



                    # break down into an RGB bitmap of 3 colors
                    spectrogram_color_data=db_to_rgb(spectrogram_value_data)
                    print(f"shape of spectrogram_color_data: {spectrogram_color_data.shape}")
                    # first get the bitmap

                    """break








                    # 10. implement a way to hear the audio cut and kept audio data, 
                    #   implementing a visual of where on the spectrogram the current audio playback point is

                    
                    

                    print(S_db.shape)

                    # 4. Plot the spectrogram
                    #changed from 10,4 to 10,5 to match
                   # plt.figure(figsize=(10, 8)) #makes a new window
                   # plt.subplot(3, 1, 1)
                   # librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz',cmap=cmap_name)
                   # plt.colorbar(format='%+2.0f dB')
                   # plt.title('Spectrogram of a RAVDESS audio file')
                   # plt.xlabel("Time (s)") #redundant?
                   # plt.ylabel("Frequency (Hz)") #redundant?
                   # plt.tight_layout()
                    #plt.show()


                    # plot the spectrogram with silences cut out
                    #1. find silences
                    #1.a. get rms
                    #print("original:")
                    #print(S_db)
                    #print("zeroed:")
                    zeroed=S_db+80
                    #print(zeroed)
                    squared=np.square(zeroed)
                    #print("squared:")
                    #print(squared)
                    means=np.mean(squared,axis=0)
                    #print("means:")
                    #print(means)
                    rms=np.sqrt(means[:])
                    #print("rms:")
                    #print(rms)
                    #print("ah.")
                    #print(np.square(S_db[:]))
                    #print(np.rms(S_db[:]))
                    #1.b. set threshold
                    print("min:")
                    print(np.min(rms))
                    print("max:")
                    print(np.max(rms))
                    #2. remove silences
                    #3. plot spectrogram

                    threshold=0.5*np.max(rms)

                    # locations of sound
                    deletion_indices=np.where(rms>threshold)[0] #[0] because this is a tuple for some reason
                    
                    # Find the number of cuts:
                    #print(indices)
                    # do a funny trick i just thought up of to find the number of jumps
                    #print("shape:")
                    #print(indices.shape)
                    #print("size:")
                    #print(indices.size)
                    numbers=np.arange(1,deletion_indices.size+1)
                    #print("numbers: ")
                    #print(numbers)
                    # the subtraction makes it such that there are repeated numbers when there are consecutive integers, 
                    # and a different repeated number is repeated after every jump
                    #indices-numbers
                    # get number of jumps, that being how many sections were cut off
                    print("indices-numbers:")
                    print(deletion_indices-numbers)
                    print(f"Number of cuts: {len(np.unique(deletion_indices-numbers))}") # what an interesting method

                    print("indices:")
                    print(deletion_indices)

                    #print("Cut:")
                    S_db_filtered_away=np.delete(S_db,deletion_indices,axis=1)
                    S_db_filtered=S_db[:,deletion_indices]
                    #print(S_db_filtered)
                    print(f"Shape of original: {S_db.shape}")
                    print(f"Shape of cut: {S_db_filtered_away.shape}")
                    print(f"Shape of kept: {S_db_filtered.shape}")

                    # (3) Short-term energy comparison
                    plt.subplot(5, 1, 3)
                    plt.plot(rms, label="RMS", color='orange')
                    #plt.plot(np.linspace(0, len(energy), len(new_energy)), new_energy, label="Processed STE", color='blue')
                    plt.axhline(y=0.05*np.max(rms), color='red', linestyle='--', label="Threshold")
                    plt.title("Short-Term Energy")
                    plt.xlabel("Frame Index")
                    plt.ylabel("Energy")
                    plt.legend()
                    #plt.show()


                    ## (3) Short-term energy comparison
                    #plt.subplot(5, 1, 3)
                    #plt.plot(rms, label="RMS", color='orange')
                    ##plt.plot(np.linspace(0, len(energy), len(new_energy)), new_energy, label="Processed STE", color='blue')
                    #plt.axhline(y=0.05*np.max(rms), color='red', linestyle='--', label="Threshold")
                    #plt.title("Short-Term Energy (Filtered)")
                    #plt.xlabel("Frame Index")
                    #plt.ylabel("Energy")
                    #plt.legend()
                    ##plt.show()

                    print("Showing filtered:")
                    #plt.figure(figsize=(10, 5))
                    plt.subplot(5, 2, 7)
                    librosa.display.specshow(S_db_filtered, sr=sr, x_axis='time', y_axis='hz',cmap=cmap_name)
                    plt.colorbar(format='%+2.0f dB')
                    plt.title('Filtered RAVDESS audio file')
                    plt.xlabel("Time (s)") #redundant?
                    plt.ylabel("Frequency (Hz)") #redundant?
                    #plt.tight_layout()
                    
                    print("Showing filtered away:")
                    #plt.figure(figsize=(10, 5))
                    plt.subplot(5, 2, 8)
                    librosa.display.specshow(S_db_filtered_away, sr=sr, x_axis='time', y_axis='hz',cmap=cmap_name)
                    plt.colorbar(format='%+2.0f dB')
                    #print("deletion indices:")
                    #print(np.unique(deletion_indices))
                    #plt.axvline(x=np.unique(deletion_indices).any(), color='red', linestyle='--', label="Threshold")
                    plt.title('Removed parts from RAVDESS audio file')
                    plt.xlabel("Time (s)") #redundant?
                    plt.ylabel("Frequency (Hz)") #redundant?
                    plt.tight_layout()
                    
                    # show the window
                    #%matplotlib notebook
                    plt.tight_layout()
                    plt.show()"""

                    if printone: break

                #except Exception as e:
                    #print(f"An error occurred: {e}")
    if printone: break

