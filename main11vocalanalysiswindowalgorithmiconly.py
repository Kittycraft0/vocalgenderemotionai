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


#fig = plt.figure(figsize=(12, 9), facecolor='#886688') # Change 'darkcyan' to 'cyan' if you want it bright!
#fig.suptitle('Live Audio Information', fontsize=18, color='white', fontweight='bold', y=0.96)

# ai made pyqtgraph setup
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
# --- PYQTGRAPH UI SETUP ---
# Create the application
app = pg.mkQApp("Live Audio Dashboard")

# Create the main window
win = pg.GraphicsLayoutWidget(show=True, title="Live Audio Information")
win.resize(1000, 600)
win.setBackground('#886688')

# Add a plot for the spectrogram
p1 = win.addPlot(title="Spectrogram")
p1.hideAxis('bottom')
p1.hideAxis('left')

# Create the ImageItem and add it to the plot
img = pg.ImageItem()
p1.addItem(img)

# Set up the Colormap (Magma)
colormap = pg.colormap.get('magma')
img.setLookupTable(colormap.getLookupTable())
img.setLevels([-80, 0]) # Maps -80dB to black, 0dB to bright/white



# --- CONFIGURATION ---
#SAMPLE_RATE = 22050
TOTAL_WINDOW_SECONDS=10.0
BUFFER_SECONDS = 2.0    
UPDATE_INTERVAL_MS = 30    # 30ms = ~33 FPS (Smoother) #changed to 250 cuz laggy on non gpu device
DEVICE_INDEX = None



# Ask the computer for the stats of this specific microphone
device_info = sd.query_devices(DEVICE_INDEX, 'input')
# Pull out the sample rate and convert it to an integer
SAMPLE_RATE = int(device_info['default_samplerate'])
print(f"Microphone detected! Running at {SAMPLE_RATE} Hz")


# 1. get vertical resolution
n_fft=1024
# 2. get window
#window_width=512
# 3. window step
resolutionfactor=8
#window_step=int(65536/resolutionfactor) #was 256 # i don't understand why this is the magic number that makes each pixel a square
window_step=int(4096*4/resolutionfactor) #was 256 # i don't understand why this is the magic number that makes each pixel a square
#window_step=int(SAMPLE_RATE/resolutionfactor) #was 256
#window_max_pos=window_width+1
#parse_width=64
#parse_interval=32
spectrogram_pixel_height=16*resolutionfactor
# 5. find top and bottom 5% energies from the distribution (???)
#top_5_percent_energy = 0.05 * np.percentile(rms, 95)
#bottom_5_percent_energy = 0.05 * np.percentile(rms, 5)
cmap_name="magma"



# --- AUDIO BUFFER STATE ---
buffer_size = int(SAMPLE_RATE * BUFFER_SECONDS)
audio_buffer = np.zeros(buffer_size, dtype=np.float32)
def audio_callback(indata, frames, time, status):
    global audio_buffer
    if status: print(status)
    new_data = indata.flatten()
    audio_buffer = np.roll(audio_buffer, -len(new_data))
    audio_buffer[-len(new_data):] = new_data




# set ratios and stuffs
#gs = fig.add_gridspec(4, 4, height_ratios=[1, 1, 1, 5], width_ratios=[2,2,2,5])


# what is this i don't get it
# FIX: Set height back to 64, but keep width at 256
# 1. Spectrogram
# size of spectrogram is set here, changed first : from 0 such that now it takes up the full window
#ax_spec = fig.add_subplot(gs[:, :])
#ax_spec.set_title("Spectrogram", fontsize=14)
#ax_spec.axis('off')
#dummy_img = np.zeros((64, 256, 3), dtype=np.uint8) 
#specrogram_display = ax_spec.imshow(dummy_img, aspect='auto', origin='upper', animated=True)



# idea:
# have spectrogram stored as a bitmap image ready for rendering
# LIFO new data in from the right out to the left, e.g. all of the data is moved down and then new data is put on top of it or something
# the new data that is put in is the second half of putting a snipped that is twice as long as the hole created, converted into mel spectrogram,
# and then converted to bitmap.
#prev_array=np.array((spectrogram_pixel_height,int(2*SAMPLE_RATE/(window_step)+1),3))#(spectrogram_color_data[:, :, :3] * 255).astype(np.uint8)
#full_spectrogram_bitmap_array=np.zeros((spectrogram_pixel_height,int(2*SAMPLE_RATE/(window_step)+1)*10,3)).astype(np.uint8)
n=0
slice_seconds=1
import time
start_time = time.time()
#time.time-start_time
#width=int(SAMPLE_RATE*slice_seconds/(window_step)+1)
b=1
last_time=start_time

# 1. Create the Normalizer
# This maps the input range [min_db, max_db] to [0.0, 1.0]
min_db=-80.0
max_db=0.0
norm = mcolors.Normalize(vmin=min_db, vmax=max_db)

# 2. Get the Colormap object from matplotlib
cmap_name=cmap_name
cmap = plt.get_cmap(cmap_name)

# 1. Pre-calculate the exact maximum width of our visual buffer
# e.g., 10 seconds of history to display on screen
MAX_COLUMNS = int(TOTAL_WINDOW_SECONDS * SAMPLE_RATE / window_step)
spectrogram_data = np.full((MAX_COLUMNS, spectrogram_pixel_height), -80.0, dtype=np.float32)
#full_spectrogram_bitmap_array = np.zeros((spectrogram_pixel_height, MAX_COLUMNS, 3), dtype=np.uint8)

# 2. Keep track of exact time so we don't drift
last_processed_time = time.time()
start_time=last_processed_time
bbb=0
def process_live_audio(y, sr, min_db=-80.0, max_db=0.0, cmap_name=cmap_name):
    """
    Takes raw audio data (y) and sample rate (sr) directly from memory.
    Returns the formatted RGB bitmap array ready for the neural network.
    """
    global last_time, bbb
    current_time=time.time()
    elapsed_seconds=current_time-last_time #useful
    #last_time=current_time
    print(current_time-start_time)
    # ai helped me here
    # 1. Figure out how many raw audio samples represent the elapsed time
    ideal_samples = int(elapsed_seconds * sr)
    # 2. CRITICAL MATH: Round down to a perfect multiple of 'window_step' (hop_length).
    # If we don't do this, the image columns jump around and stitch poorly.
    new_columns = ideal_samples // window_step
    # If not enough time has passed to make at least 1 column of pixels, just wait.
    #if new_columns < 1:
    #    return spectrogram_data #full_spectrogram_bitmap_array
    # 3. Librosa STFT Math (The Secret Sauce)
    # To get exactly 'new_columns' of output without Librosa injecting silence at the edges,
    # we need this exact number of historical samples from the audio buffer:
    print(f"bbb: {bbb}")
    bbb+=1
    samples_to_pull = (new_columns - 1) * window_step + n_fft
    # Check if the buffer even has enough data yet (prevents crashing on startup)
    print(samples_to_pull)
    print(len(y))
    if samples_to_pull > len(y):
        samples_to_pull=len(y)-1
        #return spectrogram_data #full_spectrogram_bitmap_array
    
    
    #global width
    #global b
    #print(time.time()-start_time) #time since initialization
    # get slice_seconds slice width
    #global slice_seconds
    #y_slice_width=int(SAMPLE_RATE*elapsed_seconds)
    new_window_width=int(SAMPLE_RATE*elapsed_seconds/(window_step)+1)
    #print(new_window_width)
    #print(samples_to_pull)
    # get slice of y
    print(f"bbb: {bbb}")
    bbb+=1
    if len(y)<=samples_to_pull:
        print(f"AUDIO BUFFER NOT BIG ENOUGH!!! GOT f{int(y.shape[0])} NEEDS AT LEAST f{int(samples_to_pull)}")
        return
    # Extract exactly what we need from the very end of the rolling audio buffer
    y_slice=y[-samples_to_pull:]
    # 4. Compute Spectrogram 
    # center=False is MANDATORY here. It stops Librosa from adding silence padding!
    mel_spec = librosa.feature.melspectrogram(y=y_slice, sr=sr, n_fft=n_fft, hop_length=window_step, n_mels=spectrogram_pixel_height,center=False)
    #print(y, sr, n_fft, window_step, spectrogram_pixel_height)
    #print(mel_spec.shape)
    #print(mel_spec.shape[1], int(2*sr/(window_step)+1)) #they are the same

    # 5. Color formatting
    S_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # PyQtGraph expects (X, Y). Librosa outputs (Y, X). 
    # .T transposes it so it faces the right way!
    S_db = S_db.T
    
    # 3. RMS Energy calculation (for consistency with training, though silence removal on live 
    #    chunks is tricky. We might skip rigorous thresholding to prevent crashing on silence,
    #    or just ensure we have data).
    
    # For live audio, we usually skip complex silence removal (cutting) because it messes up 
    # the timing of the live stream. We just process the window as is.
    
    
    
    # 3. Calculate the color
    # norm(db_value) converts dB to 0-1 scale
    # cmap(...) takes that 0-1 value and returns (R, G, B, A) in 0.0-1.0 floats
    
    # 4. Return only RGB (first 3 values)
    # If you need 0-255 integers, multiply these by 255 and cast to int
    #return rgba_color #[:3] # removing alpha remover

    # 4. Color conversion (Must match training EXACTLY)
    # The training data used your db_to_rgba function
    #spectrogram_color_data = cmap(norm(S_db))
    
    # Vertical flip (matched from your convertAndStoreData function)
    #spectrogram_color_data = np.flipud(spectrogram_color_data)
    
    # Convert to 0-255 uint8 RGB
    #new_bitmap = (spectrogram_color_data[:, :, :3] * 255).astype(np.uint8)

    # Double check actual output width just to be completely safe
    actual_new_cols = S_db.shape[0]
    #actual_new_cols = new_bitmap.shape[1]
    
    # 6. SHIFT the main image array left by the exact number of new columns
    spectrogram_data[:-actual_new_cols,:] = spectrogram_data[actual_new_cols:,:]
    # 7. PASTE the new data onto the right edge
    spectrogram_data[-actual_new_cols:,:] = S_db #new_bitmap
    
    ## roll the image to the left by window length/2
    ## roll the second half or so of received array visual to the image 
    ##array=np.tile(np.arange(1,9),8).reshape(8,8)
    ##zeros=int(np.zeros((8,3)))
    ##int_arr=zeros.floor()
    ##print(int_arr)
    ##array=np.concat([array[:,0:5],zeros],axis=1)
    #global full_spectrogram_bitmap_array
    #global n
    #n+=1
    #print(spectrogram_bitmap_array.shape[1])
    #print(int(2*SAMPLE_RATE/(window_step)+1))
    ##if n%5==0:
    #
    ## shift currently displayed image to the left by width*b pixels
    #full_spectrogram_bitmap_array[:,:-int(new_window_width*b),:]=full_spectrogram_bitmap_array[:,int(new_window_width*b):,:]
    ## append new image to right
    #full_spectrogram_bitmap_array[:,-int(new_window_width*b):,:]=spectrogram_bitmap_array[:,-int(new_window_width*b):,:]
    ##np.roll(full_spectrogram_bitmap_array,-100,axis=1)
    ##full_spectrogram_bitmap_array[:,:-int(2*SAMPLE_RATE/(window_step)+1),:]=spectrogram_bitmap_array
    ##full_spectrogram_bitmap_array=np.concat(
    ##    [full_spectrogram_bitmap_array[:,:-int(spectrogram_bitmap_array.shape[1]/2),:],
    ##     spectrogram_bitmap_array[:,:-int(spectrogram_bitmap_array.shape[1]/2),:]],axis=1)
    ##else:
    ##    full_spectrogram_bitmap_array=np.concat(
    ##        [full_spectrogram_bitmap_array[:,:-int(spectrogram_bitmap_array.shape[2]/2),:],
    ##         int(np.zeros(
    ##             (
    ##             spectrogram_pixel_height,
    ##             int((2*SAMPLE_RATE/(window_step)+1)/2)
    ##             ,3
    ##             )
    ##             ))],axis=1)

    # 8. Update timer. 
    # We only advance the timer by the EXACT amount of audio we processed. 
    # This completely prevents timing jitter and drift over long periods.
    #print(spectrogram_data[0][0])
    global last_processed_time
    last_processed_time += (actual_new_cols * window_step / sr)
    last_time=current_time

    #global bbb
    print(f"bbb: {bbb}")
    bbb+=1
    
    return spectrogram_data

def update_dashboard():
    # 1. Bring in all our globals properly!
    #global emotion_smooth, gender_smooth, frame_counter
    #global target_g_probs, target_e_probs, current_color
    
    #frame_counter += 1  # Add 1 every frame
    
    # 1. Get Audio
    current_audio = audio_buffer.copy()
    

    # --- SILENCE GATE ---
    #volume = np.sqrt(np.mean(current_audio**2))
    

    # run the math to get audio data
    #audio_data=get_audio_data(audio_buffer,BUFFER_SECONDS,SAMPLE_RATE)
    #print(f"Audio data: {audio_data}")
    #print(f"Audio pitch: {audio_data["pitch"]}")
    #print(librosa.hz_to_mel(audio_data["pitch"]))


    bitmap = process_live_audio(current_audio, SAMPLE_RATE)
    
    # --- 3. RENDER ---
    # Give the raw numbers to the GPU and let it handle the colors
    img.setImage(bitmap, autoLevels=False)

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
    #specrogram_display.set_data(bitmap)


    #visual_update_list = [specrogram_display]

    # return whatever you want updated
    #return visual_update_list


# --- START THE LOOP ---
# PyQtGraph uses QTimer instead of Matplotlib's FuncAnimation
timer = QtCore.QTimer()
timer.timeout.connect(update_dashboard)
timer.start(UPDATE_INTERVAL_MS)


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
        #ani = animation.FuncAnimation(fig, update_dashboard, interval=UPDATE_INTERVAL_MS, blit=True) 
        #plt.show()
        # Start the audio stream and the GUI event loop
        pg.exec() # Keeps the application running


if __name__ == "__main__":
    run()
