# 10/20/2025
# Riley Mohr


import kagglehub
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
from IPython.display import Audio, display
import soundfile as sf

# Download latest version
path = kagglehub.dataset_download("uwrfkaggler/ravdess-emotional-speech-audio")
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



# 1. Define the path to your RAVDESS audio file.
# Replace this with the actual path to a .wav file from your dataset.
# Example path structure: 'RAVDESS/Actor_01/03-01-01-01-01-01-01.wav'
# The file naming convention details the emotion and intensity.
# Example filename: 03-01-06-01-02-01-12.wav (audio-only, speech, fearful, normal intensity, statement 2, repetition 1, actor 12)
file_path_working = file#'RAVDESS/Actor_01/03-01-06-01-02-01-12.wav'

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
                    print("loading audio file")
                    y, sr = librosa.load(file_path, sr=None) #sr=None keeps the original sampling rate
                    print("loaded audio file")
                    
                    print("sr:")
                    print(sr)
                    #print(f"Shape of sr:{sr.shape}")
                    print("y:")
                    print(y)
                    print(f"Shape of y:{y.shape}")
                    
                    # 1. get window
                    window_width=1024
                    # 2. window step
                    window_step=512
                    window_max_pos=window_width+1
                    
                    # get rms energy list from data
                    # 3a. get energy and store it in a list
                    print("calculating rms_waveform")
                    rms_waveform = librosa.feature.rms(y=y, frame_length=window_width, hop_length=window_step)[0]
                    print("done calculating rms_waveform")

                    # 4. construct energy distribution
                    
                    
                    

                    # 5. find top and bottom 5% energies from the distribution (???)
                    #top_5_percent_energy = 0.05 * np.percentile(rms, 95)
                    #bottom_5_percent_energy = 0.05 * np.percentile(rms, 5)
                    
                    method=1
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
                        threshold=threshold3
                    # 7. cut off data outside of threshold from data
                    cut_data_indices=np.where(rms_waveform<threshold)
                    cut_data=np.delete(rms_waveform,cut_data_indices)
                    # 8. convert to spectrogram
                    # 9. plot spectrogram of cut and kept and uncut audio data
                    
                    # 3. Compute the Short-Time Fourier Transform (STFT)
                    # This converts the audio signal into a spectrogram
                    # The result is complex, so we take the absolute value to get the magnitude
                    D = librosa.stft(y)
                    #print("D:")
                    #print(D)
                    print(f"Shape of D:{D.shape}")
                    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
                    #print("S_db:")
                    #print(S_db)
                    print(f"Shape of S_db:{S_db.shape}")
                    
                    plt.subplot(4,1,1)
                    print("Showing original:")
                    #plt.figure(figsize=(10, 5))
                    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz',cmap="magma")
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
                    plt.axhline(y=threshold1, color='red', linestyle='--', label="Threshold1 (5th percentile)")
                    plt.axhline(y=threshold2, color='orange', linestyle='--', label="Threshold2 (0.05*95th percentile)")
                    plt.axhline(y=threshold3, color='yellow', linestyle='--', label="Threshold3 (0.05*max value)")
                    plt.title("Energy Graph")
                    plt.xlabel("Seconds")
                    plt.ylabel("Energy")
                    plt.legend()
                    #plt.show()

                    
                    # cut off data outside of thresholds from data
                    thresholds=[threshold1,threshold2,threshold3]
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
                    librosa.display.specshow(cut_datas_spectrogram, sr=sr, x_axis='time', y_axis='hz',cmap="magma")
                    plt.colorbar(format='%+2.0f dB')
                    plt.title('Filtered RAVDESS audio file')
                    plt.xlabel("Time (s)") #redundant?
                    plt.ylabel("Frequency (Hz)") #redundant?
                    #plt.tight_layout()
                    
                    print("Showing filtered away:")
                    #plt.figure(figsize=(10, 5))
                    plt.subplot(5, 2, 10)
                    librosa.display.specshow(cut_out_datas_spectrogram, sr=sr, x_axis='time', y_axis='hz',cmap="magma")
                    plt.colorbar(format='%+2.0f dB')
                    #print("deletion indices:")
                    #print(np.unique(deletion_indices))
                    #plt.axvline(x=np.unique(deletion_indices).any(), color='red', linestyle='--', label="Threshold")
                    plt.title('Removed parts from RAVDESS audio file')
                    plt.xlabel("Time (s)") #redundant?
                    plt.ylabel("Frequency (Hz)") #redundant?
                    #plt.tight_layout()


                    
                    plt.tight_layout()
                    plt.show()




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

                    break








                    # 10. implement a way to hear the audio cut and kept audio data, 
                    #   implementing a visual of where on the spectrogram the current audio playback point is

                    
                    

                    print(S_db.shape)

                    # 4. Plot the spectrogram
                    #changed from 10,4 to 10,5 to match
                   # plt.figure(figsize=(10, 8)) #makes a new window
                   # plt.subplot(3, 1, 1)
                   # librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz',cmap="magma")
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
                    librosa.display.specshow(S_db_filtered, sr=sr, x_axis='time', y_axis='hz',cmap="magma")
                    plt.colorbar(format='%+2.0f dB')
                    plt.title('Filtered RAVDESS audio file')
                    plt.xlabel("Time (s)") #redundant?
                    plt.ylabel("Frequency (Hz)") #redundant?
                    #plt.tight_layout()
                    
                    print("Showing filtered away:")
                    #plt.figure(figsize=(10, 5))
                    plt.subplot(5, 2, 8)
                    librosa.display.specshow(S_db_filtered_away, sr=sr, x_axis='time', y_axis='hz',cmap="magma")
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
                    plt.show()

                    if printone: break

                #except Exception as e:
                    #print(f"An error occurred: {e}")
    if printone: break