# 12/22/2025
# 4/7/2026
# Riley Mohr

# pitch <--
# formants <--
# vocal weight <--
# intonation




# Signal processing and feature extraction
# Optimal data structures for vocal signal analysis
# Python libraries to aid in tasks
# Pitch extraction
# Formant extraction (F1, F2, F3, F4, etc.)
# Vocal mode classification and extraction (M0,M1,M2,M3, false chord usage, whistle register usage, etc.)
# Compensating for background noise

# Quantification of subjective qualities
# Vocal weight/"tone" quantification
# "Breathiness" quantification
# "Brightness" quantification
# "Fullness" quantification
# "Richness" quantification
# "Dryness" quantification
# "Nasality"/hyponasality/hypernasality/"twang" quantification

# Machine learning and data analysis
# Utterance length optimization
# Dialect quantification/classification by traits such as quantified intonation and pronunciation differences by different heard sounds compared to an expected sound
# Comparing accuracies of various artificial intelligence methods for various vocal traits
# Generating gender and emotion classification from parameterized voice vector instead of neural network on spectrogram data (ex. applying many of the algorithms learned in machine learning to the set of numbers instead of just neural networks)  
# Utterance reconstruction from data
# Analyzing relationships between different variables, i.e. vocal weight and pitch range, audible dryness over time since the speaker last drank water, dryness and richness, etc.
# Explore machine learning methods for identifying additional relationships between variables



# input audio stream


# 4/7/2026
# Riley Mohr


import librosa
import numpy as np

def get_audio_data(audio_buffer,time,sample_rate):
    #print(len(audio_buffer),time,sample_rate)
    if(int(sample_rate*time)!=int(len(audio_buffer))):
        print("The input data does not match up! Something might be wrong! int(sample_rate*time)!=int(len(audio_buffer))")
    data={
        "pitch":-1,
        "formants":[],
        "vocalweight":-1
    }
    data["pitch"]=get_pitch(audio_buffer,sample_rate)

    if data["pitch"] > 0:
        # ONLY pass the most recent 100 milliseconds of audio to Praat!
        recent_audio = audio_buffer[-int(sample_rate * 0.100):]
        data["formants"] = get_formants_praat(recent_audio, sample_rate)
    else:
        # If silence, don't make Praat do any math at all
        data["formants"] = [0.0, 0.0, 0.0, 0.0, 0.0]
    
    
    
    data["vocalweight"]=get_vocal_weight(audio_buffer, sample_rate)

    return(data)


def get_pitch(audio_buffer,sample_rate):
    # ai allows me to not have to read the documentation and spent 3 hours producing something otherwise produced in 3 minutes
    """
    Analyzes an audio buffer and returns the median F0 (pitch) in Hertz.
    """
    # 1. Run the pYIN algorithm
    # fmin and fmax bound the human vocal range to prevent false tracking from room noise.
    # 75Hz to 600Hz captures almost all human speech.
    #print(f"length of used data: {len(audio_buffer[-int(sample_rate*0.050):])}")
    f0 = librosa.yin( # algorithm that gets f0, use yin instead of pyin because probabilistic is slow
        y=audio_buffer[-int(sample_rate*0.050):], # most recent 50ms of audio
        fmin=75, # lowest human speech
        fmax=600, # highest human speech f0 plus some because i go higher sometimes to 1000-2000, was 600 now 6000 not anymore neither is it 3515
        sr=sample_rate 
        #fill_na=np.nan # Unvoiced frames will return 'NaN' (Not a Number) #only used for pyin
    )
    
    # 2. Filter out the unvoiced/silent frames (the NaNs)
    valid_pitch_frames = f0[~np.isnan(f0)]

    # 3. Calculate the average pitch for this specific buffer
    if len(valid_pitch_frames) > 0: #gotta make sure it's not empty!!!
        # We use median instead of mean to ignore sudden mic pops or glitches
        current_pitch = np.median(valid_pitch_frames)
        return current_pitch
    else:
        # Returns 0 if you are whispering or totally silent
        return 0.0


    #return -1


def get_formants(audio_buffer):
    return []

import parselmouth

def get_formants_praat(audio_buffer, sample_rate):
    """
    Extracts F1, F2, F3 using the industry-standard Praat algorithm.
    """
    # Create a Parselmouth Sound object from the numpy array
    sound = parselmouth.Sound(audio_buffer, sampling_frequency=sample_rate)
    
    # Get the formant object (max 5 formants, max frequency 5500Hz for adult male/female)
    # Change 5500 to 5000 for standard male voices, or leave at 5500 for universal.
    formants = sound.to_formant_burg(max_number_of_formants=5, maximum_formant=5500.0)
    
    # Get the time at the exact middle of the audio buffer snippet
    mid_time = sound.get_total_duration() / 2.0
    
    # Extract the values of F1, F2, and F3 at that specific time
    f1 = formants.get_value_at_time(1, mid_time)
    f2 = formants.get_value_at_time(2, mid_time)
    f3 = formants.get_value_at_time(3, mid_time)
    f4 = formants.get_value_at_time(4, mid_time)
    f5 = formants.get_value_at_time(5, mid_time)
    
    # Handle 'NaN' if no formants were found (e.g., during silence)
    if np.isnan(f1): f1 = 0
    if np.isnan(f2): f2 = 0
    if np.isnan(f3): f3 = 0
    if np.isnan(f4): f4 = 0
    if np.isnan(f5): f5 = 0
    
    return [f1, f2, f3, f4, f5]

import warnings # Make sure this is at the very top of your file!

def get_vocal_weight(audio_buffer, sample_rate):
    # 1. Match Meyda's exact 256 sample buffer size
    buffer_size = 256
    recent_audio = audio_buffer[-buffer_size:]

    if np.max(np.abs(recent_audio)) < 0.001:
        return 0.0
    
    # A. Meyda applies a Hanning window
    window = np.hanning(buffer_size)
    windowed_audio = recent_audio * window
    
    # B. Meyda gets the Amplitude Spectrum and squares it to get Power
    complex_spec = np.fft.rfft(windowed_audio)
    power_spec = np.abs(complex_spec) ** 2
    
    # C. Meyda applies a Mel Filterbank (Using the HTK formula)
    # --- THIS IS THE FIX: Mute the Empty Filter warning! ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mel_basis = librosa.filters.mel(sr=sample_rate, n_fft=buffer_size, n_mels=100, htk=True)
        
    mel_energies = np.dot(mel_basis, power_spec)
    
    # D. Meyda uses Natural Log
    mel_log = np.log(mel_energies + 1.0)
    
    # E. Meyda uses an un-normalized Type-II DCT
    N = len(mel_log)
    n = np.arange(N)
    k = np.arange(N).reshape(-1, 1)
    meyda_dct_matrix = 2 * np.cos(np.pi * k * (2 * n + 1) / (2 * N))
    
    mfccs = np.dot(meyda_dct_matrix, mel_log)

    # ---------------------------------------------------------
    # EXACT LINE-BY-LINE PORT OF THE HTML LOGIC
    # ---------------------------------------------------------
    NumBins = 100
    RangeLimit = 100.0
    IntensityThreshold = -4.0 
    
    peaks = 0
    potentialpeaks=0
    mels = mfccs 
    
    max_range = int((RangeLimit / 100.0) * (len(mels) - 1))
    
    print(max_range)
    potentialslist=[]
    for i in range(1, max_range):
        #print((mels[i] < IntensityThreshold)) #false
        #print((mels[i] < mels[i-1])) #false
        #print((mels[i] < mels[i+1])) #true
        #if (mels[i] < mels[i+1]):
        #    if (mels[i] < mels[i-1]):
        #        print(mels[i]) #see how low it is below the intensity threshold!?
        #        if (mels[i] < IntensityThreshold):
        #            print("(mels[i] < IntensityThreshold) and (mels[i] < mels[i-1]) and (mels[i] < mels[i+1])")
        #        else:
        #            pass #print("(mels[i] < mels[i-1]) and (mels[i] < mels[i+1])")
        #    else:
        #        pass #print("(mels[i] < mels[i+1])")
        #else:
        #    pass
        if (mels[i] < mels[i-1]) and (mels[i] < mels[i+1]):
            potentialpeaks+=1
            potentialslist.append(mels[i])
        if (mels[i] < IntensityThreshold) and (mels[i] < mels[i-1]) and (mels[i] < mels[i+1]):
            peaks += 1
    potentialslist.sort(reverse=True)
    roundedpotentialslist=[]
    for value in potentialslist:
        roundedpotentialslist.append("{:.3g}".format(value))
    #print(f"potential peaks: {potentialpeaks}")
    #print(f"potentials list: {roundedpotentialslist}")
    return potentialslist[0]
            
    print(peaks)
    thickness = min(100.0, (100.0 * peaks) / (RangeLimit * NumBins / 300.0))
    return thickness