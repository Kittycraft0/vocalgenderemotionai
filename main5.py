# 10/20/2025
# Riley Mohr

print("Importing libraries")

print("importing librosa")
import librosa
print("importing librosa.display")
import librosa.display
print("importing matplotlib.pyplot")
import matplotlib.pyplot as plt
print("importing numpy")
import numpy as np
print("importing os")
import os
print("importing IPython.display Audio, display")
from IPython.display import Audio, display
print("importing soundfile")
import soundfile as sf
# progress bars!!!
from tqdm import tqdm
import time

# 10/30/2025

print("importing torch")
import torch
print("importing PIL from Image")
from PIL import Image

#import os
# from torch import nn
# from torch.utils.data import DataLoader
# from torchvision import datasets,transforms
print("done importing libraries")

# 11/26/2025
# debug logs
debug=False
logs=[]
def log(string):
    logs.append(string)
    if debug==True:
        print(string)

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
    log(f"spectrogram:")
    log(spec)
    log(f"total spectrogram size: {total_spec_size}")
    log(f"width: {window_width}")
    log(f"interval: {interval}")
    log(f"windowed spectrogram list: ")
    log(windowed_spec)
    if windowed_spec:
        log(f"shape of first window: {windowed_spec[0].shape}")
        log(f"shape of second window: {windowed_spec[1].shape}")
        log(f"shape of second to last window: {windowed_spec[-2].shape}")
        log(f"shape of last window: {windowed_spec[-1].shape}")
    else:
        log("windowed_spec is empty :(")
    return windowed_spec




# Set device type
print(f"CUDA available? {torch.cuda.is_available()}")
print(f"CPU available? {torch.cpu.is_available()}")

deviceName=""
# set to false if no gpu, set to true if there is gpu
if(torch.cuda.is_available()):
    deviceName="cuda"
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


def importDataset():
    print("importing kagglehub")
    import kagglehub # Why import if it's never going to be used? That's why I put it here--it takes a couple seonds to import, soo...
    # Download latest kagglehub ravdess dataset version
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
    return path

printone=False


# https://stackoverflow.com/questions/44787437/how-to-convert-a-wav-file-to-a-spectrogram-in-python3
#import matplotlib.pyplot as plt
#from scipy import signal
#from scipy.io import wavfile

#sample_rate, samples = wavfile.read('path-to-mono-audio-file.wav')
#file=(f"{path}\\Actor_01\\03-01-01-01-01-01-01.wav")
#print(file)
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

def db_to_rgba(db_value, min_db=-80.0, max_db=0.0, cmap_name=cmap_name):
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



def getSmallestSpectrogramSize(path,folders):
    # 1. Define the path to your RAVDESS audio file.
    # Replace this with the actual path to a .wav file from your dataset.
    # Example path structure: 'RAVDESS/Actor_01/03-01-01-01-01-01-01.wav'
    # The file naming convention details the emotion and intensity.
    # Example filename: 03-01-06-01-02-01-12.wav (audio-only, speech, fearful, normal intensity, statement 2, repetition 1, actor 12)
    #file_path_working = file#'RAVDESS/Actor_01/03-01-06-01-02-01-12.wav'


    # find smallest file, use its window length as the default
    #print("Finding smallest size")
    smallestsize=float('inf')
    num_files_success=0
    num_files_total=0
    #for folder in os.listdir(path):
    #    for filename in os.listdir(f"{path}\\{folder}"):


    # OUTER LOOP: Iterates through folders
    # desc="Processing Folders" sets the text label for the bar
    for folder in tqdm(folders, desc="Finding smallest size"):
        folder_path = os.path.join(path, folder)
        # Get list of files
        files = os.listdir(folder_path)

        log(folder)
        # INNER LOOP: Iterates through files in the current folder
        # leave=False ensures this bar disappears after the folder is finished
        # so you don't end up with hundreds of finished bars in your terminal
        for filename in tqdm(files, desc=f"  {folder}", leave=False):
            file_path=f"{path}\\{folder}\\{filename}"
            num_files_total+=1

            
            # make sure the file is actually a .wav file and not, say, a folder
            if not filename.lower().endswith('.wav'):
                continue

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

    return smallestsize






# imports, converts, and stores model data
def convertAndStoreData(foldername):
    # if the "bitmap_training_data_incomplete" folder exists then take the route of converting new data
    import shutil
    if(os.path.exists(foldername+"_incomplete")):
        print(f"Found folder '{foldername}_incomplete'; removing")
        shutil.rmtree(foldername+"_incomplete")
    if(os.path.exists(foldername)):
        print(f"Found folder '{foldername}'; removing")
        shutil.rmtree(foldername) # just to be safe perhaps?

    # Creates folder "bitmap_training_data" if it does not exist
    # exist_ok=True creates the directory if it doesn't exist, 
    # and does nothing (without error) if it does exist.
    os.makedirs("bitmap_training_data_incomplete", exist_ok=True)

    #Load and plot spectrograms for all .wav files
    #for folder in os.listdir(path):

    
    # 11/26/2025
    # Stored data
    data_names=[]
    data_data=[]
    #def addDatumToTotalData(data,name):
    #    data_names.append(name)
    #    data_data.append(data)


    # Import the dataset
    path=importDataset()

    # Get the list of folders first so tqdm knows the total count
    # sorted for hash checking
    folders = sorted([f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))])

    # Get the smallest spectrogram size:
    smallestsize=getSmallestSpectrogramSize(path,folders)

    # OUTER LOOP: Iterates through folders
    # desc="Processing Folders" sets the text label for the bar
    for folder in tqdm(folders, desc="Converting wavs to bitmaps"):
        folder_path = os.path.join(path, folder)
        # Get list of files
        # sorted for hash checking
        files = sorted(os.listdir(folder_path))

        log(folder)
        # INNER LOOP: Iterates through files in the current folder
        # leave=False ensures this bar disappears after the folder is finished
        # so you don't end up with hundreds of finished bars in your terminal
        for filename in tqdm(files, desc=f"  {folder}", leave=False):
        #for filename in os.listdir(f"{path}\\{folder}"):
            log(f"   {filename}")
            if filename.lower().endswith(".wav"):
                #log("asdas")
                file_path=f"{path}\\{folder}\\{filename}"
                # Check if the file exists
                if not os.path.exists(file_path):
                    print(f"Error: The file '{file_path}' was not found.")
                else:
                    #log("BASBDABV")
                    # 2. Load the audio file
                    # librosa.load returns the audio time series (y) and the sampling rate (sr)
                    #try:
                        log(f"THE file path1: {file_path}")

                        # load the audio file
                        log(f"loading audio file {file_path}")
                        y, sr = librosa.load(file_path, sr=None) #sr=None keeps the original sampling rate
                        log(f"loaded audio file {file_path}")

                        log(f"sr: {sr} Shape of y: {y.shape}")
                        #log(sr)
                        #log(f"Shape of sr:{sr.shape}")
                        #log("y:")
                        #log(y)
                        #log(f"Shape of y:{y.shape}")


                        # get rms energy list from data
                        # 3a. get energy and store it in a list
                        log(f"calculating rms_waveform for {file_path}")
                        rms_waveform = librosa.feature.rms(y=y, frame_length=window_width, hop_length=window_step)[0]
                        log(f"done calculating rms_waveform for {file_path}")

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
                        log(f"")
                        #log("D:")
                        #log(D)
                        log(f"Shape of D:{D.shape}")
                        # Define the spectrogram as S_db
                        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
                        #log("S_db:")
                        #log(S_db)
                        log(f"Shape of S_db:{S_db.shape}")


                        # show energy over time graph with different thresholds
                        # Create time axis for RMS (one time value per frame)
                        frames = np.arange(len(rms_waveform))
                        t_rms = frames * window_step / sr  # convert frame index → seconds

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
                        left_bound=int(S_db.shape[1]/2-cut_spectrogram_half_width)
                        right_bound=int(S_db.shape[1]/2+cut_spectrogram_half_width)
                        log(f"window left bound: {left_bound}")
                        log(f"window right bound: {right_bound}")
                        log(f"window right bound-left bound: {right_bound-left_bound}")
                        log(f"shape of S_db: {S_db.shape}")
                        #exact center doesn't matter; what matters is the same width
                        spectrogram_value_data=S_db[:,left_bound:right_bound]
                        # now the spectrogram is a bunch of numbers
                        log(f"shape of spectrogram_value_data: {spectrogram_value_data.shape}")

                        # saves spectrogram as bitmap file
                        # returns 0 if success
                        #def saveSpectrogramDataAsBitmap(S_db,filename):
                        #    # transposed because it needs that for some reason
                        #    spectrogram_color_data=db_to_rgba(S_db)
                        #    spectrogram_bitmap_array = (spectrogram_color_data[:, :, :3] * 255).astype(np.uint8)
                        #    Image.fromarray(spectrogram_bitmap_array).save(filename)
                        #    return 0

                        # break down into an RGB bitmap of 3 colors
                        #spectrogram_color_data=db_to_rgba(spectrogram_value_data)
                        
                        # save as a bitmap
                        # Converts (Height, Width, 4) floats -> (Height, Width, 3) integers
                        #spectrogram_bitmap_array = (spectrogram_color_data[:, :, :3] * 255).astype(np.uint8)
                        #Image.fromarray(spectrogram_bitmap_array).save("cut_data_spectrogram_bitmap.bmp")

                        # Returns a (Height, Width, 3) matrix of integers
                        #loaded_bitmap_data = np.array(Image.open("cut_data_spectrogram_bitmap.bmp"))

                        # for testing
                        #saveSpectrogramDataAsBitmap(S_db,"uncut_data_spectrogram_bitmap.bmp")

                        log(f"Filename: {filename}")
                        filenamedata=os.path.splitext(filename)[0]
                        # This automatically adds the correct slash (\ for Windows, / for Mac/Linux)
                        #bitmap_save_path="bitmap_training_data"+filenamedata+".bmp"
                        bitmap_save_path = os.path.join("bitmap_training_data_incomplete", filenamedata + ".bmp")
                        
                        # save the spectrogram as a bitmap for future reference
                        #saveSpectrogramDataAsBitmap(spectrogram_value_data,bitmap_save_path)
                        
                        # save the name data
                        data_names.append(filenamedata)
                        
                        # convert to what is imported if it were form the spectrogram
                        spectrogram_color_data=db_to_rgba(spectrogram_value_data)
                        spectrogram_bitmap_array = (spectrogram_color_data[:, :, :3] * 255).astype(np.uint8)
                        data_data.append(spectrogram_bitmap_array)
                        Image.fromarray(spectrogram_bitmap_array).save(bitmap_save_path)


                        log(f"shape of spectrogram_color_data: {spectrogram_color_data.shape}")
                        log("spectrogram color data:")
                        log(spectrogram_color_data)


                        



                        # lol one liner
                        # it's like stoicheometry, but... different!!!!!!!!
                        # actually honestly kinda similar--you could say these are unit conversions hahahahaha 11/20/2025
                        # Image.fromarray((db_to_rgba(spectrogram_value_data)[:, :, :3] * 255).astype(np.uint8)).save("my_spectrogram.bmp")



                        if printone: break

                    #except Exception as e:
                        #print(f"An error occurred: {e}")
        if printone: break

    os.rename(foldername+"_incomplete", foldername)
    
    # --- THE FIX: Sort the data to match importData's alphabetical order ---
    print("Sorting converted data to ensure hash consistency...")
    
    # 1. Zip the names and data together so they stay paired
    combined = list(zip(data_names, data_data))
    
    # 2. Sort by the name (index 0)
    combined.sort(key=lambda x: x[0]) 
    
    # 3. Unzip them back into separate lists
    data_names, data_data = zip(*combined)
    
    # 4. Convert back to standard lists (zip returns tuples)
    return list(data_names), list(data_data)


# imports data 
def importData(path):
    folder_path = path
    # Get list of files
    files = sorted(os.listdir(folder_path)) #sorted to stop hash discrepancy
    
    # 11/26/2025
    # Stored data
    data_names=[]
    data_data=[]
    #def addDatumToTotalData(data,name):
    #    data_names.append(name)
    #    data_data.append(data)

    # INNER LOOP: Iterates through files in the current folder
    # leave=False ensures this bar disappears after the folder is finished
    # so you don't end up with hundreds of finished bars in your terminal
    for file in tqdm(files, desc=f"  {path}", leave=False):

        # get the file path
        file_path=os.path.join(path, file)
        # get the data
        imagedata=np.array(Image.open(file_path))
        
        # get the name
        name=os.path.splitext(file)[0]
        
        # add the data to the lists
        data_names.append(name)
        data_data.append(imagedata)
        
    return data_names, data_data


def getData(reimportdata=0):
    # whether or not to re convert data;
    # 0 means determine decisionautomatically
    # 1 means always re convert it
    # 2 means always skip conversion
    #reimportdata="0"
    #print(f"reinmportdata: {reimportdata}")
    #print(f"is reimportdata equal to 1: {reimportdata==1}")
    #print("decision:") # had to remove quotes okay
    #print(reimportdata==1 or (reimportdata==0 and (os.path.exists("bitmap_training_data_incomplete") or not os.path.exists("bitmap_training_data"))))
    if reimportdata==1 or (reimportdata==0 and (os.path.exists("bitmap_training_data_incomplete") or not os.path.exists("bitmap_training_data"))):
        print("Converting and storing data")
        data_names, data_data = convertAndStoreData("bitmap_training_data")
    # otherwise import the data from bitmap_training_data
    else: #assuming reimportdata==2 because 0 and 1 was taken care of
        print("Importing data")
        data_names, data_data = importData("bitmap_training_data")

    return data_names, data_data





# check conversion and import both yield the same data, from AI

def verifyConversionAndImportingYieldsSameData():
    import hashlib

    def generate_integrity_hashes(names_list, data_list):
        # 1. Hash the Names (List of strings)
        # We convert the list to a string representation and encode it
        names_hash = hashlib.md5(str(names_list).encode('utf-8')).hexdigest()

        # 2. Hash the Data (List of Numpy Arrays)
        data_hash_obj = hashlib.md5()

        # We iterate through every array, convert it to bytes, and update the hash
        print("Computing data hash (this might take a moment)...")
        for arr in tqdm(data_list, desc="Hashing Data"):
            # ensure array is contiguous in memory before getting bytes
            if not arr.flags['C_CONTIGUOUS']:
                arr = np.ascontiguousarray(arr)
            data_hash_obj.update(arr.tobytes())

        final_data_hash = data_hash_obj.hexdigest()

        return names_hash, final_data_hash

    # --- USE IT LIKE THIS AT THE END OF YOUR CODE ---

    data_names_converted, data_data_converted = getData(1)
    data_names_imported, data_data_imported = getData(2)

    def integrityCheck(names, data, checkname):
        print(f"\n--- INTEGRITY CHECK FOR {checkname.upper()}---")
        print(f"Total items: {len(names)}")
        print(f"Data type: {type(data[0])}")
        if len(data) > 0:
            print(f"Data1 shape (first item): {data[0].shape}")

        current_names_hash, current_data_hash = generate_integrity_hashes(names, data)
        
        print(f"NAMES HASH {checkname.upper()}: {current_names_hash}")
        print(f"DATA HASH {checkname.upper()}: {current_data_hash}")
        
        return current_names_hash, current_data_hash

    names_converted_hash, data_converted_hash = integrityCheck(data_names_converted, data_data_converted, "converted")
    names_imported_hash, data_imported_hash = integrityCheck(data_names_imported, data_data_imported, "imported")

    print(f"Names hashes equal: {names_converted_hash==names_imported_hash}")
    print(f"Datas hashes equal: {data_converted_hash==data_imported_hash}")
    
    # If you want to compare against a previous run manually, you can just print them.
    # If you are running this in a loop where you do both generation and import:
    # if calculated_hash == imported_hash:
    #     print("SUCCESS: Data is identical.")



verifyConversionAndImportingYieldsSameData()
