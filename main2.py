# 10/20/2025
# Riley Mohr


import kagglehub
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

# Download latest version
path = kagglehub.dataset_download("uwrfkaggler/ravdess-emotional-speech-audio")
print()
print()
print("Path to dataset files:", path)



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
                try:
                    
                    # 1. get window
                    # 2. window step
                    # 3. iterate over audio waveform
                    # 3a. get energy and store it in a list
                    # 4. construct energy distribution
                    # 5. find top and bottom 5% energies from the distribution (???)
                    # branch off into two methods:
                    # A: given method
                    # 6A1. 
                    # B: custom method
                    # 6B1. 
                    # C: poor threshholding method
                    # 6C1. set threshhold to 0.05*(max energy)
                    # 7. cut off data outside of threshhold from data
                    # 8. convert to spectrogram
                    # 9. plot spectrogram of cut and kept and uncut audio data
                    # 10. implement a way to hear the audio cut and kept audio data, 
                    #   implementing a visual of where on the spectrogram the current audio point is
                    

                    print(f"THE file path1: {file_path}")

                    # load the audio file
                    y, sr = librosa.load(file_path, sr=None) #sr=None keeps the original sampling rate

                    # 3. Compute the Short-Time Fourier Transform (STFT)
                    # This converts the audio signal into a spectrogram
                    # The result is complex, so we take the absolute value to get the magnitude
                    D = librosa.stft(y)
                    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

                    print(S_db.shape)

                    # 4. Plot the spectrogram
                    #changed from 10,4 to 10,5 to match
                    plt.figure(figsize=(10, 8)) #makes a new window
                    plt.subplot(3, 1, 1)
                    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz',cmap="magma")
                    plt.colorbar(format='%+2.0f dB')
                    plt.title('Spectrogram of a RAVDESS audio file')
                    plt.xlabel("Time (s)") #redundant?
                    plt.ylabel("Frequency (Hz)") #redundant?
                    plt.tight_layout()
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
                    #1.b. set threshhold
                    print("min:")
                    print(np.min(rms))
                    print("max:")
                    print(np.max(rms))
                    #2. remove silences
                    #3. plot spectrogram

                    threshhold=0.5*np.max(rms)

                    # locations of sound
                    deletion_indices=np.where(rms>threshhold)[0] #[0] because this is a tuple for some reason
                    
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
                    plt.tight_layout()
                    
                    print("Showing filtered away:")
                    #plt.figure(figsize=(10, 5))
                    plt.subplot(5, 2, 8)
                    librosa.display.specshow(S_db_filtered_away, sr=sr, x_axis='time', y_axis='hz',cmap="magma")
                    plt.colorbar(format='%+2.0f dB')
                    #print("deletion indeces:")
                    #print(np.unique(deletion_indices))
                    #plt.axvline(x=np.unique(deletion_indices).any(), color='red', linestyle='--', label="Threshold")
                    plt.title('Removed parts from RAVDESS audio file')
                    plt.xlabel("Time (s)") #redundant?
                    plt.ylabel("Frequency (Hz)") #redundant?
                    plt.tight_layout()
                    
                    # show the window
                    #%matplotlib notebook
                    plt.show()



                except Exception as e:
                    print(f"An error occurred: {e}")