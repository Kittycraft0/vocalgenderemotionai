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

def get_audio_data(audio_buffer,time,sample_rate,noise_power_profile):
    #print(len(audio_buffer),time,sample_rate)
    if(int(sample_rate*time)!=int(len(audio_buffer))):
        print("The input data does not match up! Something might be wrong! int(sample_rate*time)!=int(len(audio_buffer))")
    data={
        "pitch":-1,
        "formants":[],
        "vocalweight":-1,
        "harmonic_series":{"freqs":[],"mags":[]},
        "harmonic_rolloff": -1, # NEW
        "noise_rolloff": -1     # NEW
    }
    data["pitch"]=get_pitch(audio_buffer,sample_rate)
    
    # ONLY pass the most recent 100 milliseconds of audio to Praat!
    recent_audio = audio_buffer[-int(sample_rate * 0.100):]
    if data["pitch"] > 0:
        data["formants"] = get_formants_praat(recent_audio, sample_rate)
    else:
        # If silence, don't make Praat do any math at all
        data["formants"] = [0.0, 0.0, 0.0, 0.0, 0.0]
    
    data["vocalweight"]=get_vocal_weight(audio_buffer, sample_rate)
    
    #harmonic_freqs, harmonic_mags=get_harmonic_series(recent_audio, sample_rate, data["pitch"])
    #data["harmonic_series"]["freqs"]=harmonic_freqs
    #data["harmonic_series"]["mags"]=harmonic_mags
    harmonic_freqs, harmonic_mags = get_harmonic_series_denoised(
        recent_audio, sample_rate, data["pitch"], noise_power_profile
    )
    data["harmonic_series"]["freqs"]=harmonic_freqs
    data["harmonic_series"]["mags"]=harmonic_mags

    # Calculate both rolloffs instantly using the pitch we just found
    h_roll, n_roll = get_dual_rolloffs(audio_buffer, sample_rate, data["pitch"])
    data["harmonic_rolloff"] = h_roll
    data["noise_rolloff"] = n_roll

    return(data)


def get_pitch(audio_buffer,sample_rate):
    # ai allows me to not have to read the documentation and spent 3 hours producing something otherwise produced in 3 minutes
    """
    Analyzes an audio buffer and returns the median F0 (pitch) in Hertz.
    """
    
    # most recent 50ms of audio
    recent_audio = audio_buffer[-int(sample_rate*0.050):]
    # --- 1. VOICE ACTIVITY DETECTION (VAD) VIA LIBROSA ---
    # Check if the audio is loud enough (RMS Energy)
    rms_energy = np.mean(librosa.feature.rms(y=recent_audio))
    if rms_energy < 0.001: # Tweak this threshold based on your mic's noise floor
        return -1
        
    # Check if it's unvoiced/hissy noise (Zero-Crossing Rate)
    # Voiced speech (vowels) has a low ZCR. Static and "S" sounds have high ZCR.
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=recent_audio))
    if zcr > 0.15: # Tweak this. Usually, > 0.15 to 0.20 means it's unvoiced.
        return -1
    
    # 1. Run the pYIN algorithm
    # fmin and fmax bound the human vocal range to prevent false tracking from room noise.
    # 75Hz to 600Hz captures almost all human speech.
    #print(f"length of used data: {len(audio_buffer[-int(sample_rate*0.050):])}")
    f0 = librosa.yin( # algorithm that gets f0, use yin instead of pyin because probabilistic is slow
        y=recent_audio, # most recent 50ms of audio
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
        return -1


#def get_formants(audio_buffer):
#    return []

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

import warnings

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
    NumBins = 100#200 #100
    RangeLimit = 100#200 #100.0
    IntensityThreshold = -0.0001 #-4.0 
    
    mels = mfccs 
    
    max_range = int((RangeLimit / 100.0) * (len(mels) - 1))

    # for loop allergy code to get rid of the need to use the for loop commented out below
    # 1. Slice the array to only include the range we care about
    # (+1 because we need the right-side neighbor for the very last check)
    mels_sliced = mels[:max_range + 1]

    # 2. Create the overlapping shifted arrays
    left = mels_sliced[:-2]   # Everything from index 0 up to the second-to-last
    center = mels_sliced[1:-1] # Everything from index 1 up to the last
    right = mels_sliced[2:]    # Everything from index 2 to the end

    # 3. Vectorized Math: Apply your exact boolean logic to all items instantly
    is_valley = (center < IntensityThreshold) & (center < left) & (center < right)

    # 4. np.sum() counts all the 'True' values in the boolean array
    peaks = np.sum(is_valley)

    #peaks = 0
    #potentialpeaks=0
    ##potentialslist=[]
    #for i in range(1, max_range):
    ##    if (mels[i] < mels[i-1]) and (mels[i] < mels[i+1]):
    ##        potentialpeaks+=1
    ##        potentialslist.append(mels[i])
    #    if (mels[i] < IntensityThreshold) and (mels[i] < mels[i-1]) and (mels[i] < mels[i+1]):
    #        peaks += 1
    #potentialslist.sort(reverse=True)
    #roundedpotentialslist=[]
    #for value in potentialslist:
    #    roundedpotentialslist.append("{:.3g}".format(value))
    #print(f"potential peaks: {potentialpeaks}")
    #print(f"potentials list: {roundedpotentialslist}")
    #return potentialslist[0]
    #return min(100.0, (100.0 * potentialpeaks) / (RangeLimit * NumBins / 300.0))
            
    #print(peaks)
    thickness = min(100.0, (100.0 * peaks) / (RangeLimit * NumBins / 300.0))#*3.5 #added *10 multiplier
    #print(thickness)
    return thickness

# idea: voice spectrum or spectral rolloff visual


def get_harmonic_series(recent_audio, sample_rate, pitch_hz, max_hz=4500):
    """
    Calculates the FFT, but only returns the frequencies and decibel 
    levels of the fundamental pitch and its exact integer multiples.
    """
    if pitch_hz <= 0 or np.max(np.abs(recent_audio)) < 0.001:
        return [], []

    # 1. Standard FFT math
    window = np.hanning(len(recent_audio))
    windowed_audio = recent_audio * window
    spectrum_complex = np.fft.rfft(windowed_audio)
    spectrum_mag = np.abs(spectrum_complex)
    spectrum_db = 20 * np.log10(np.clip(spectrum_mag, 1e-10, None))

    # Normalize peak to 0 dB
    max_db = np.max(spectrum_db)
    if max_db > -80:
        spectrum_db = spectrum_db - max_db

    # Get the X-axis frequencies
    freqs = np.fft.rfftfreq(len(recent_audio), d=1.0/sample_rate)

    # 2. Hunt for the Harmonics
    harmonic_freqs = []
    harmonic_mags = []

    n = 1 # Start at n=1 (F0), then n=2 (H2), n=3 (H3)...
    while (n * pitch_hz) <= max_hz:
        target_hz = n * pitch_hz
        
        # Find the FFT bin that is closest to our target frequency
        closest_bin = np.argmin(np.abs(freqs - target_hz))
        
        # F0 naturally wavers slightly, so we search a tiny 3-bin window 
        # around the target to grab the absolute peak of that specific harmonic
        start_bin = max(0, closest_bin - 1)
        end_bin = min(len(spectrum_db) - 1, closest_bin + 1)
        
        local_peak_mag = np.max(spectrum_db[start_bin:end_bin+1])

        harmonic_freqs.append(target_hz)
        harmonic_mags.append(local_peak_mag)
        
        n += 1

    return harmonic_freqs, harmonic_mags


def get_harmonic_rolloff(h_freqs, h_mags, roll_percent=0.85):
    if len(h_freqs) == 0:
        return -1.0
        
    # 1. Convert dB back to linear power/energy
    # formula: Power = 10^(dB/10)
    linear_energy = 10 ** (np.array(h_mags) / 10.0)
    
    # 2. Find total energy and the target threshold
    total_energy = np.sum(linear_energy)
    target_energy = total_energy * roll_percent
    
    # 3. Calculate cumulative energy across the harmonics
    cumulative_energy = np.cumsum(linear_energy)
    
    # 4. Find the index of the first harmonic that crosses the 85% threshold
    # np.where returns an array of indices where the condition is true
    threshold_indices = np.where(cumulative_energy >= target_energy)[0]
    
    if len(threshold_indices) > 0:
        rolloff_idx = threshold_indices[0]
        return h_freqs[rolloff_idx]
    else:
        return h_freqs[-1] # Fallback to the highest harmonic


def get_harmonic_series_denoised(recent_audio, sample_rate, pitch_hz, noise_power_profile, max_hz=4500):
    if pitch_hz <= 0 or np.max(np.abs(recent_audio)) < 0.001:
        return [], []

    # 1. Standard FFT math for the live audio
    window = np.hanning(len(recent_audio))
    windowed_audio = recent_audio * window
    live_complex = np.fft.rfft(windowed_audio)
    live_power = np.abs(live_complex) ** 2
    
    # 2. THE MAGIC: Spectral Subtraction
    # Subtract the AC hum power from the live power. 
    # Use np.maximum to prevent negative power values.
    # Note: We scale down the noise profile slightly (e.g., * 0.8) to prevent 
    # "musical noise" artifacts (weird underwater bubbling sounds).
    clean_power = np.maximum(live_power - (noise_power_profile * 0.8), 1e-10)
    
    # Now convert our clean power to Decibels
    spectrum_db = 10 * np.log10(clean_power)

    # Normalize peak to 0 dB
    max_db = np.max(spectrum_db)
    if max_db > -80:
        spectrum_db = spectrum_db - max_db

    # Get the X-axis frequencies
    freqs = np.fft.rfftfreq(len(recent_audio), d=1.0/sample_rate)

    # 2. Hunt for the Harmonics
    harmonic_freqs = []
    harmonic_mags = []

    n = 1 # Start at n=1 (F0), then n=2 (H2), n=3 (H3)...
    while (n * pitch_hz) <= max_hz:
        target_hz = n * pitch_hz
        
        # Find the FFT bin that is closest to our target frequency
        closest_bin = np.argmin(np.abs(freqs - target_hz))
        
        # F0 naturally wavers slightly, so we search a tiny 3-bin window 
        # around the target to grab the absolute peak of that specific harmonic
        start_bin = max(0, closest_bin - 1)
        end_bin = min(len(spectrum_db) - 1, closest_bin + 1)
        
        local_peak_mag = np.max(spectrum_db[start_bin:end_bin+1])

        harmonic_freqs.append(target_hz)
        harmonic_mags.append(local_peak_mag)
        
        n += 1

    return harmonic_freqs, harmonic_mags




def get_dual_rolloffs(audio_buffer, sample_rate, pitch_hz):
    """
    Computes a single FFT and splits it into Harmonics and Noise to find 
    both spectral rolloffs simultaneously for maximum efficiency.
    """
    # 100ms buffer is optimal for FFT frequency resolution vs time resolution
    recent_audio = audio_buffer[-int(sample_rate * 0.100):]
    
    if np.max(np.abs(recent_audio)) < 0.001 or pitch_hz <= 0:
        return -1.0, -1.0

    # 1. Single FFT calculation (Only done once!)
    window = np.hanning(len(recent_audio))
    complex_spec = np.fft.rfft(recent_audio * window)
    power_spec = np.abs(complex_spec) ** 2
    freqs = np.fft.rfftfreq(len(recent_audio), d=1.0/sample_rate)

    # 2. Create a boolean mask to isolate harmonics
    harmonic_mask = np.zeros(len(power_spec), dtype=bool)
    bin_width = sample_rate / len(recent_audio) # Usually ~10Hz per bin
    
    # Identify where the harmonics live and mark those bins as True
    max_n = int(4500 / pitch_hz)
    for n in range(1, max_n + 1):
        center_bin = int((n * pitch_hz) / bin_width)
        # Mask a 3-bin window to catch the peak without grabbing surrounding noise
        start = max(0, center_bin - 1)
        end = min(len(power_spec), center_bin + 2)
        harmonic_mask[start:end] = True

    # 3. Vectorized Split: Separate the power instantly
    harmonic_power = np.where(harmonic_mask, power_spec, 0)
    noise_power = np.where(~harmonic_mask, power_spec, 0)

    # 4. Helper function to find the 85% Rolloff point
    def calc_85_percent_rolloff(power_arr):
        total = np.sum(power_arr)
        if total <= 0: return -1.0
        # searchsorted instantly finds the index where cumulative sum crosses the threshold
        idx = np.searchsorted(np.cumsum(power_arr), total * 0.98)
        return freqs[min(idx, len(freqs)-1)]

    # Return both!
    return calc_85_percent_rolloff(harmonic_power), calc_85_percent_rolloff(noise_power)