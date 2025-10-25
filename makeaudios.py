# 10//23/2025
# just for curiosity's sake
import kagglehub
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
from IPython.display import Audio, display
import soundfile as sf
import numpy as np
import timeit
import random
import math

# movie quality sampling rate! lol!
sr=48000
arr1 = 1 - 2 * (np.arange(5*sr) % 2)  # -1 and 1, dtype=int64
#arr=random.random()*2-1
#arr1 = np.tile([1, -1], 5*sr)
#arr1 = np.tile([1, -1], 50000)
arr2 = np.random.random(5 * sr)*2-1  # shape (240000,)


# convert to float32
arr1 = arr1.astype(np.float32)
arr2 = arr2.astype(np.float32)

# Separate kept and removed parts

print("Audio length:", arr1.shape)
print("Audio length:", arr2.shape)

# Play them!
print("🎧 Kept (speech) audio:")
display(Audio(arr1, rate=sr))
sf.write("sounds/loud.wav", arr1, sr)

display(Audio(arr2, rate=sr))
sf.write("sounds/noise.wav", arr2, sr)
print("✅ Saved 'loud.wav'")


length3 = 5 * sr
arr3 = np.random.random(length3)  # random values 0..1

indices = np.arange(length3)  # 0, 1, 2, ..., length-1
arr3 = arr3 * ((length3 - indices) / length3) #+ (indices / length3) * (indices % 2)
#multipliers = (length3 - indices) + indices * (indices % 2)

#arr3 = arr3 * multipliers  # element-wise multiplication
print(arr3)
sf.write("sounds/fadednoise.wav", arr3, sr)


length4 = 5 * sr
arr4 = np.random.random(length4)  # random values 0..1

indices = np.arange(length4)  # 0, 1, 2, ..., length-1
arr4 = arr4 * ((length4 - indices) / length4) + (indices / length4) * (np.sin(indices))
#multipliers = (length3 - indices) + indices * (indices % 2)

#arr3 = arr3 * multipliers  # element-wise multiplication
print(arr4)
sf.write("sounds/ofadedtobereplacednoiseow.wav", arr4, sr)

length4 = 5 * sr
arr4 = np.random.random(length4)  # random values 0..1

indices = np.arange(length4)  # 0, 1, 2, ..., length-1
arr4 = arr4 * ((length4 - indices) / length4) + (indices / length4) * (np.sin(0.005*indices))
#multipliers = (length3 - indices) + indices * (indices % 2)

#arr3 = arr3 * multipliers  # element-wise multiplication
print(arr4)
sf.write("sounds/ofadedtobereplacednoise.wav", arr4, sr)


length=5*sr
indices=np.arange(length)
arr=np.sin(0.05*indices)*np.sin(0.005*indices)
sf.write("sounds/audio007.wav",arr,sr)

length=5*sr
indices=np.arange(length)
arr=np.sin(0.000005*indices)*np.sin(0.005*indices)+np.sin(0.0005*indices)
sf.write("sounds/audio008.wav",arr,sr)

length=5*sr
indices=np.arange(length)
arr=np.sin(0.00005*indices)*np.sin(0.005*indices)+np.sin(0.0005*indices)
sf.write("sounds/audio009.wav",arr,sr)

length=5*sr
indices=np.arange(length)
arr=np.sin(0.00005*indices*indices)*np.sin(0.005*indices)+np.sin(0.0005*indices)
sf.write("sounds/audio010.wav",arr,sr)
print(np.max(arr))


length=5*sr
indices=np.arange(length)
arr=np.sin(0.0000025*indices*indices)*np.sin(0.005*indices)+np.sin(0.0005*indices)
sf.write("sounds/audio011.wav",arr,sr)
print(np.max(arr))


length=5*sr
indices=np.arange(length)

harmonics=np.arange(80)+1
# Fundamental frequency in Hz
f0 = 440  # A4
# Compute time in seconds for each sample
t = indices / sr  # shape (length,)
# Compute the sum of harmonics
arr = np.sum(np.sin(2 * np.pi * f0 * harmonics[:, np.newaxis] * t), axis=0)
#print(arr)
#print(harmonics[:,np.newaxis])
# Optional: normalize to avoid clipping
arr /= np.max(np.abs(arr))
#arr=np.sin(0.0000025*indices*indices)*np.sin(0.005*indices)+np.sin(0.0005*indices)
sf.write("sounds/audio012.wav",arr,sr)


length=5*sr
indices=np.arange(length)

harmonics=np.arange(80)+1
# Fundamental frequency in Hz
f0 = 440  # A4
# Compute time in seconds for each sample
t = indices / sr  # shape (length,)
# Compute the sum of harmonics
arr = np.sum(np.sin(2 * np.pi * (f0*np.sin(indices/length)) * harmonics[:, np.newaxis] * t), axis=0)
#print(arr)
#print(harmonics[:,np.newaxis])
# Optional: normalize to avoid clipping
arr /= np.max(np.abs(arr))
#arr=np.sin(0.0000025*indices*indices)*np.sin(0.005*indices)+np.sin(0.0005*indices)
sf.write("sounds/audio013.wav",arr,sr)


length=15*sr
indices=np.arange(length)

harmonics=np.arange(80)+1
# Fundamental frequency in Hz
f0 = 440  # A4
# Compute time in seconds for each sample
t = indices / sr  # shape (length,)
# Compute the sum of harmonics
arr = np.sum(np.sin(2 * np.pi * (f0*np.sin(indices/length)) * harmonics[:, np.newaxis] * t), axis=0)
#print(arr)
#print(harmonics[:,np.newaxis])
# Optional: normalize to avoid clipping
arr /= np.max(np.abs(arr))
#arr=np.sin(0.0000025*indices*indices)*np.sin(0.005*indices)+np.sin(0.0005*indices)
sf.write("sounds/audio014.wav",arr,sr)


length=15*sr
indices=np.arange(length)
notes_individual=np.array([5,1,3,5,4,6,2,4,6,7,4,5,7,6,9,4,5,6])
notes=np.repeat(notes_individual,length//notes_individual.size)

harmonics=np.arange(80)+1
# Fundamental frequency in Hz
f0 = 440  # A4
# Compute time in seconds for each sample
t = indices / sr  # shape (length,)
# Compute the sum of harmonics
arr = np.sum(np.sin(2 * np.pi * (f0*notes) * harmonics[:, np.newaxis] * t), axis=0)
#print(arr)
#print(harmonics[:,np.newaxis])
# Optional: normalize to avoid clipping
arr /= np.max(np.abs(arr))
#arr=np.sin(0.0000025*indices*indices)*np.sin(0.005*indices)+np.sin(0.0005*indices)
sf.write("sounds/audio015.wav",arr,sr)


length=15*sr
indices=np.arange(length)
notes_individual=np.array([5,1,3,5,4,6,2,4,6,7,4,5,7,6,9,4,5,6])
notes=np.repeat(notes_individual,length//notes_individual.size)
notes=notes+10
notes=notes/2

harmonics=np.arange(80)+1
# Fundamental frequency in Hz
f0 = 440  # A4
# Compute time in seconds for each sample
t = indices / sr  # shape (length,)
# Compute the sum of harmonics
arr = np.sum(np.sin(2 * np.pi * (f0*notes) * harmonics[:, np.newaxis] * t), axis=0)
#print(arr)
#print(harmonics[:,np.newaxis])
# Optional: normalize to avoid clipping
arr /= np.max(np.abs(arr))
#arr=np.sin(0.0000025*indices*indices)*np.sin(0.005*indices)+np.sin(0.0005*indices)
sf.write("sounds/audio016.wav",arr,sr)



#print(timeit.timeit("1 - 2 * (np.arange(100000) % 2)", globals=globals(), number=10000))
#print(timeit.timeit("np.fromfunction(lambda i: (-1)**i, (100000,), dtype=int)", globals=globals(), number=1000))
#print(timeit.timeit("np.tile([1, -1], 50000)", globals=globals(), number=10000))


#from pydub import AudioSegment
#
## convert float array [-1,1] to 16-bit PCM
#arr_int16 = (arr * 32767).astype(np.int16)
#sf.write("temp.wav", arr_int16, sr)
#sound = AudioSegment.from_wav("temp.wav")
#sound.export("loud.mp3", format="mp3")