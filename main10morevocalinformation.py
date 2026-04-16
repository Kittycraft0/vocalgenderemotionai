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
    
    
    
    data["vocalweight"]=get_vocal_weight(audio_buffer)

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

def get_vocal_weight(audio_buffer):
    return -1