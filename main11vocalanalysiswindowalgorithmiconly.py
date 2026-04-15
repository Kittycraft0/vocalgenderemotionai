#4/14/2026
# so i don't need to wait for imports every single time i want to test something
# holy unoptomized no wonder the thing is so slow

import matplotlib.pyplot as plt
import sounddevice as sd
import matplotlib.animation as animation
from main10morevocalinformation import get_audio_data
import numpy as np
import librosa
import matplotlib.colors as mcolors


fig = plt.figure(figsize=(12, 9), facecolor='#886688') # Change 'darkcyan' to 'cyan' if you want it bright!
fig.suptitle('Live Audio Information', fontsize=18, color='white', fontweight='bold', y=0.96)

# --- CONFIGURATION ---
#SAMPLE_RATE = 22050     
WINDOW_SECONDS = 2.0    
UPDATE_INTERVAL = 30    # 30ms = ~33 FPS (Smoother) #changed to 250 cuz laggy on non gpu device
DEVICE_INDEX = None

# 1. get vertical resolution
n_fft=1024
# 2. get window
#window_width=512
# 3. window step
window_step=256
#window_max_pos=window_width+1
#parse_width=64
#parse_interval=32
final_image_height=64
# 5. find top and bottom 5% energies from the distribution (???)
#top_5_percent_energy = 0.05 * np.percentile(rms, 95)
#bottom_5_percent_energy = 0.05 * np.percentile(rms, 5)
cmap_name="magma"


# Ask the computer for the stats of this specific microphone
device_info = sd.query_devices(DEVICE_INDEX, 'input')
# Pull out the sample rate and convert it to an integer
SAMPLE_RATE = int(device_info['default_samplerate'])
print(f"Microphone detected! Running at {SAMPLE_RATE} Hz")

# --- AUDIO BUFFER STATE ---
buffer_size = int(SAMPLE_RATE * WINDOW_SECONDS)
audio_buffer = np.zeros(buffer_size, dtype=np.float32)
def audio_callback(indata, frames, time, status):
    global audio_buffer
    if status: print(status)
    new_data = indata.flatten()
    audio_buffer = np.roll(audio_buffer, -len(new_data))
    audio_buffer[-len(new_data):] = new_data




# set ratios and stuffs
gs = fig.add_gridspec(4, 4, height_ratios=[1, 1, 1, 5], width_ratios=[2,2,2,5])


# what is this i don't get it
# FIX: Set height back to 64, but keep width at 256
# 1. Spectrogram
# size of spectrogram is set here, changed first : from 0 such that now it takes up the full window
ax_spec = fig.add_subplot(gs[:, :])
ax_spec.set_title("Spectrogram", fontsize=14)
ax_spec.axis('off')
dummy_img = np.zeros((64, 256, 3), dtype=np.uint8) 
specrogram_display = ax_spec.imshow(dummy_img, aspect='auto', origin='upper', animated=True)



# idea:
# have spectrogram stored as a bitmap image ready for rendering
# LIFO new data in from the right out to the left, e.g. all of the data is moved down and then new data is put on top of it or something
# the new data that is put in is the second half of putting a snipped that is twice as long as the hole created, converted into mel spectrogram,
# and then converted to bitmap.

def process_live_audio(y, sr, min_db=-80.0, max_db=0.0, cmap_name=cmap_name):
    """
    Takes raw audio data (y) and sample rate (sr) directly from memory.
    Returns the formatted RGB bitmap array ready for the neural network.
    """
    # 1. Compute Mel Spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=window_step, n_mels=final_image_height)
    
    # 2. Convert to dB
    S_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # 3. RMS Energy calculation (for consistency with training, though silence removal on live 
    #    chunks is tricky. We might skip rigorous thresholding to prevent crashing on silence,
    #    or just ensure we have data).
    
    # For live audio, we usually skip complex silence removal (cutting) because it messes up 
    # the timing of the live stream. We just process the window as is.
    
    # 1. Create the Normalizer
    # This maps the input range [min_db, max_db] to [0.0, 1.0]
    min_db=-80.0
    max_db=0.0
    norm = mcolors.Normalize(vmin=min_db, vmax=max_db)
    
    # 2. Get the Colormap object from matplotlib
    cmap_name=cmap_name
    cmap = plt.get_cmap(cmap_name)
    
    # 3. Calculate the color
    # norm(db_value) converts dB to 0-1 scale
    # cmap(...) takes that 0-1 value and returns (R, G, B, A) in 0.0-1.0 floats
    
    # 4. Return only RGB (first 3 values)
    # If you need 0-255 integers, multiply these by 255 and cast to int
    #return rgba_color #[:3] # removing alpha remover

    # 4. Color conversion (Must match training EXACTLY)
    # The training data used your db_to_rgba function
    spectrogram_color_data = cmap(norm(S_db))
    
    # Vertical flip (matched from your convertAndStoreData function)
    spectrogram_color_data = np.flipud(spectrogram_color_data)
    
    # Convert to 0-255 uint8 RGB
    spectrogram_bitmap_array = (spectrogram_color_data[:, :, :3] * 255).astype(np.uint8)
    
    return spectrogram_bitmap_array


def update_dashboard(frame):
    # 1. Bring in all our globals properly!
    #global emotion_smooth, gender_smooth, frame_counter
    #global target_g_probs, target_e_probs, current_color
    
    #frame_counter += 1  # Add 1 every frame
    
    # 1. Get Audio
    current_audio = audio_buffer.copy()
    
    # --- SILENCE GATE ---
    #volume = np.sqrt(np.mean(current_audio**2))
    

    # run the math to get audio data
    audio_data=get_audio_data(audio_buffer,WINDOW_SECONDS,SAMPLE_RATE)
    #print(f"Audio data: {audio_data}")
    print(f"Audio pitch: {audio_data["pitch"]}")
    #print(librosa.hz_to_mel(audio_data["pitch"]))


    bitmap = process_live_audio(current_audio, SAMPLE_RATE)
    #bitmap[bitmap.shape[0],np.floor(63-librosa.hz_to_mel(audio_data["pitch"]))]=0
    #target_w = 256
    #if bitmap.shape[1] >= target_w:
    # 
    #    start = (bitmap.shape[1] - target_w) // 2
    #    crop = bitmap[:, start:start+target_w, :]
    #else:
    #    crop = np.zeros((bitmap.shape[0], target_w, 3), dtype=np.uint8) 
    #    crop[:, :bitmap.shape[1], :] = bitmap
    #specrogram_display.set_data(crop)
    specrogram_display.set_data(bitmap)


    visual_update_list = [specrogram_display]

    # return whatever you want updated
    return visual_update_list


def run():
    stream = sd.InputStream(
        device=DEVICE_INDEX,
        channels=1,
        samplerate=SAMPLE_RATE,
        callback=audio_callback,
        blocksize=int(SAMPLE_RATE * 0.1) 
    )
    
    with stream:
        print("Microphone Active. Starting Dashboard...")
        # blit=True is the key to high performance
        # but blit=true makes lag so i'm switching it to false -4/7/2026
        ani = animation.FuncAnimation(fig, update_dashboard, interval=UPDATE_INTERVAL, blit=True) 
        plt.show()

if __name__ == "__main__":
    run()