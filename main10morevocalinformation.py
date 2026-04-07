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


def get_audio_data(audio_buffer,rf,time):
    print(len(audio_buffer),rf,time)
    if(int(rf*time)!=int(len(audio_buffer))):
        print("The input data does not match up! Something might be wrong! int(rf*time)!=int(len(audio_buffer))")
    data={
        "pitch":-1,
        "formants":[],
        "vocalweight":[]
    }
    
    return(data)