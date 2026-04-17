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

# --- NEW: Setup the Pitch Plot ---
win.nextRow() # This tells PyQtGraph to go to the line below the spectrogram
p2 = win.addPlot(title="Pitch Tracker (Hz)")
p2.setYRange(0, 1000) # Locks the Y-axis to standard human voice range
p2.showGrid(x=True, y=True, alpha=0.3)

# Create a green line graph to hold our data
pitch_curve = p2.plot(pen=pg.mkPen('g', width=2))

# Tell the UI to drop down to the next row before drawing the graphs!
win.nextRow()
# --- NEW: Create the Text Readout ---
# size='20pt' makes it nice and big, color='w' makes it white
readout_label = win.addLabel(text="Pitch: -- Hz | Note: --", size='20pt', bold=True, color='w')

# --- NEW: Setup the Formant Plot ---
win.nextRow() # Drop down to a new row
p3 = win.addPlot(title="Formant Tracker (Hz)")
p3.setYRange(0, 5500) # Praat searches up to 5500Hz by default
p3.showGrid(x=True, y=True, alpha=0.3)

# --- NEW: Setup the Thickness / Weight Plot ---
win.nextRow() # Drop down to a new row
p4 = win.addPlot(title="Thickness / Weight (%)")
p4.setYRange(0, 100) # Match the HTML 0-100% scale
p4.showGrid(x=True, y=True, alpha=0.3)

# Create 3 differently colored curves to match the HTML color bands!
weight_green_curve = p4.plot(pen=pg.mkPen(color=(0, 255, 0), width=2), connect='finite')
weight_red_curve = p4.plot(pen=pg.mkPen(color=(255, 0, 0), width=2), connect='finite')
weight_blue_curve = p4.plot(pen=pg.mkPen(color=(0, 127, 255), width=2), connect='finite')

# Create 5 differently colored lines for F1 through F5
f1_curve = p3.plot(pen=pg.mkPen(color=(0, 255, 0), width=2))
f2_curve = p3.plot(pen=pg.mkPen(color=(0, 255, 127), width=2))
f3_curve = p3.plot(pen=pg.mkPen(color=(255, 0, 0), width=2))
f4_curve = p3.plot(pen=pg.mkPen(color=(255, 0, 255), width=2))
f5_curve = p3.plot(pen=pg.mkPen(color=(255, 255, 255), width=2))

# Set up the Colormap (Magma)
#colormap = pg.colormap.get('magma')
#img.setLookupTable(colormap.getLookupTable())
#img.setLevels([-80, 0]) # Maps -80dB to black, 0dB to bright/white



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
#n_fft=1024
n_fft=4096
# 2. get window
#window_width=512
# 3. window step
resolutionfactor=16
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
# NEW: Keep a running tally of exactly how much audio the mic has collected
total_samples_received = 0 
samples_processed = 0
def audio_callback(indata, frames, time, status):
    global audio_buffer, total_samples_received
    if status: print(status)
    new_data = indata.flatten()
    audio_buffer = np.roll(audio_buffer, -len(new_data))
    audio_buffer[-len(new_data):] = new_data
    total_samples_received += len(new_data)




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
#spectrogram_data = np.full((MAX_COLUMNS, spectrogram_pixel_height), -80.0, dtype=np.float32)
#full_spectrogram_bitmap_array = np.zeros((spectrogram_pixel_height, MAX_COLUMNS, 3), dtype=np.uint8)
#spectrogram_data = np.zeros((spectrogram_pixel_height, MAX_COLUMNS, 3), dtype=np.uint8)
spectrogram_data = np.zeros((MAX_COLUMNS, spectrogram_pixel_height, 3), dtype=np.uint8)

# Create an array of zeros to hold our pitch history.
# Making it MAX_COLUMNS long means it perfectly matches the 2-second width of the spectrogram!
#pitch_history = np.zeros(MAX_COLUMNS, dtype=np.float32)
# A 2D array: 5 rows (for F1-F5) and MAX_COLUMNS wide
#formant_history = np.zeros((5, MAX_COLUMNS), dtype=np.float32)
# (Change your zeros to np.full with np.nan so the lines don't draw at 0 before they fill up!)
pitch_history = np.full(MAX_COLUMNS, np.nan, dtype=np.float32)
formant_history = np.full((5, MAX_COLUMNS), np.nan, dtype=np.float32)
weight_history = np.full(MAX_COLUMNS, np.nan, dtype=np.float32)


# --- NEW: Weight History ---
weight_history = np.full(MAX_COLUMNS, np.nan, dtype=np.float32)

# 2. Keep track of exact time so we don't drift
last_processed_time = time.time()
start_time=last_processed_time
bbb=0
def process_live_audio(y, sr, min_db=-80.0, max_db=0.0, cmap_name=cmap_name):
    """
    Takes raw audio data (y) and sample rate (sr) directly from memory.
    Returns the formatted RGB bitmap array ready for the neural network.
    """
    global bbb, last_processed_time, samples_processed
    # 1. Check exactly how much fresh audio is sitting in the buffer waiting for us
    unprocessed_samples = total_samples_received - samples_processed
    # 2. How many columns can we draw with this fresh audio?
    new_columns = unprocessed_samples // window_step
    # If we don't have enough fresh audio to draw a column, ABORT and wait.
    # This prevents the duplicate-drawing stutter!
    if new_columns < 1:
        return spectrogram_data
    
    
    #global last_time, bbb
    current_time=time.time()
    #elapsed_seconds=current_time-last_time #useful
    elapsed_seconds=current_time-last_processed_time #useful

    # --- THE DEADLOCK FIX ---
    # What is the absolute maximum number of columns our 2-second buffer can hold?
    max_possible_columns = (len(y) - n_fft) // window_step + 1
    
    # If the UI lagged (like during startup) and the mic collected more audio 
    # than the buffer can hold, we MUST drop the oldest data to catch up!
    if new_columns > max_possible_columns:
        print(f"Skipping {new_columns - max_possible_columns} dropped frames to catch up!")
        # Advance the "processed" counter to skip the lost data
        samples_processed += (new_columns - max_possible_columns) * window_step
        new_columns = max_possible_columns

    #last_time=current_time
    #last_time=current_time
    #print(current_time-start_time)
    # ai helped me here
    # 1. Figure out how many raw audio samples represent the elapsed time
    #ideal_samples = int(elapsed_seconds * sr)
    # 2. CRITICAL MATH: Round down to a perfect multiple of 'window_step' (hop_length).
    # If we don't do this, the image columns jump around and stitch poorly.
    #new_columns = ideal_samples // window_step
    # If not enough time has passed to make at least 1 column of pixels, just wait.
    #if new_columns < 1:
    #    return spectrogram_data #full_spectrogram_bitmap_array
    # 3. Librosa STFT Math (The Secret Sauce)
    # To get exactly 'new_columns' of output without Librosa injecting silence at the edges,
    # we need this exact number of historical samples from the audio buffer:
    #print(f"bbb: {bbb}")
    bbb+=1
    samples_to_pull = (new_columns - 1) * window_step + n_fft
    # Check if the buffer even has enough data yet (prevents crashing on startup)
    #print(samples_to_pull)
    #print(len(y))
    # this triggers if the samples to pull is larger than the buffer as well
    if samples_to_pull > len(y):
        #samples_to_pull=len(y)-1
        return spectrogram_data #full_spectrogram_bitmap_array
    
    
    #global width
    #global b
    #print(time.time()-start_time) #time since initialization
    # get slice_seconds slice width
    #global slice_seconds
    #y_slice_width=int(SAMPLE_RATE*elapsed_seconds)
    #new_window_width=int(SAMPLE_RATE*elapsed_seconds/(window_step)+1)
    #print(new_window_width)
    #print(samples_to_pull)
    # get slice of y
    #print(f"bbb: {bbb}")
    bbb+=1
    if len(y)<samples_to_pull:
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
    #for i in range(1, mel_spec.shape[0]):
    #    # If the absolute maximum energy in this frequency band is 0, it's a dead band.
    #    if np.max(mel_spec[i, :]) == 0.0:
    #        # Copy the raw energy from the frequency band directly below it
    #        mel_spec[i, :] = mel_spec[i-1, :]
    
    # 5. Color formatting
    #S_db = librosa.power_to_db(mel_spec, ref=np.max)
    S_db = librosa.power_to_db(mel_spec, ref=1.0)
    
    # PyQtGraph expects (X, Y). Librosa outputs (Y, X). 
    # .T transposes it so it faces the right way!
    S_db = S_db.T

    # Loop through every frequency band (starting at 1 so we can look backwards at 0)
    #for i in range(1, S_db.shape[1]):
    #    # If the loudest sound in this entire frequency band is -80 (meaning it's empty)
    #    if np.max(S_db[:, i]) <= -80.0:
    #        # Copy all the visual data from the band directly below it!
    #        S_db[:, i] = S_db[:, i-1]
    
    actual_new_cols = S_db.shape[0]
    
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
    spectrogram_color_data = cmap(norm(S_db))
    
    # Vertical flip (matched from your convertAndStoreData function)
    #spectrogram_color_data = np.flipud(spectrogram_color_data)
    
    # Convert to 0-255 uint8 RGB
    new_bitmap = (spectrogram_color_data[:, :, :3] * 255).astype(np.uint8)

    # Double check actual output width just to be completely safe
    actual_new_cols = S_db.shape[0]
    #actual_new_cols = new_bitmap.shape[1]
    
    # 6. SHIFT the main image array left by the exact number of new columns
    spectrogram_data[:-actual_new_cols,:,:] = spectrogram_data[actual_new_cols:,:]
    # 7. PASTE the new data onto the right edge
    spectrogram_data[-actual_new_cols:,:,:] = new_bitmap #S_db #new_bitmap
    
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

    # 7. MARK THESE SAMPLES AS PROCESSED!
    # This completely locks the UI framerate to the microphone's hardware speed.
    samples_processed += (actual_new_cols * window_step)
    
    # 8. Update timer. 
    # We only advance the timer by the EXACT amount of audio we processed. 
    # This completely prevents timing jitter and drift over long periods.
    #print(spectrogram_data[0][0])
    #global last_processed_time
    last_processed_time += (actual_new_cols * window_step / sr)
    

    #global bbb
    #print(f"bbb: {bbb}")
    bbb+=1
    
    return spectrogram_data

def update_dashboard():
    global pitch_history, formant_history

    # 1. Bring in all our globals properly!
    #global emotion_smooth, gender_smooth, frame_counter
    #global target_g_probs, target_e_probs, current_color
    
    #frame_counter += 1  # Add 1 every frame
    
    # 1. Get Audio
    current_audio = audio_buffer.copy()
    

    # --- SILENCE GATE ---
    #volume = np.sqrt(np.mean(current_audio**2))
    
    bitmap = process_live_audio(current_audio, SAMPLE_RATE)

    # run the math to get audio data
    audio_data=get_audio_data(audio_buffer,BUFFER_SECONDS,SAMPLE_RATE)
    #print(f"Audio data: {audio_data}")
    #print(f"Audio pitch: {pitch_hz}")
    #print(librosa.hz_to_mel(pitch_hz))
    pitch_hz=audio_data["pitch"]
    if pitch_hz > 0: # If a pitch is actually detected
        # Let Librosa calculate the closest musical note!
        closest_note = librosa.hz_to_note(pitch_hz)
        max_mel = librosa.hz_to_mel(SAMPLE_RATE / 2)
        pitch_mel=int(librosa.hz_to_mel(pitch_hz))
        pitch_y = int((pitch_mel / max_mel) * spectrogram_pixel_height)
        pitch_y = np.clip(pitch_y, 2, spectrogram_pixel_height - 2) # Numpy clamp!
        bitmap[-5:, pitch_y-2:pitch_y+2] = [0, 255, 255]
        # Format the text so it limits decimals to 1 spot (e.g., 440.5 Hz)
        readout_label.setText(f"Pitch: {pitch_hz:.1f} Hz  |  Note: {closest_note}")
    else:
        # If silence, clear the readout
        readout_label.setText("Pitch: -- Hz  |  Note: --")


    
    #pitch_y = int((pitch_mel / max_mel) * spectrogram_pixel_height)
    #pitch_y = max(2, min(pitch_y, spectrogram_pixel_height - 2))
    #bitmap[-5:,pitch_y-2:pitch_y+2]=[0,255,255]
    
    # 1. Convert to pure Numpy array
    formants_arr = np.array(audio_data["formants"])
    # 2. Vectorized Math: Convert all 5 to Mel, scale them, and cast to int instantly
    formants_mel = librosa.hz_to_mel(formants_arr)
    formants_y = ((formants_mel / max_mel) * spectrogram_pixel_height).astype(int)
    # 3. Vectorized Clamp: Keep all 5 safely inside the screen bounds
    formants_y = np.clip(formants_y, 2, spectrogram_pixel_height - 2)

    formant_colors = [[0, 255, 0], [0, 255, 127], [255, 0, 0], [255, 0, 255], [255, 255, 255]]
    
    # 4. Inject colors (Unrolled to avoid loops. We check >0 so silence doesn't draw at the bottom)
    if formants_arr[0] > 0: bitmap[-5:, formants_y[0]:formants_y[0]+2] = formant_colors[0]
    if formants_arr[1] > 0: bitmap[-5:, formants_y[1]:formants_y[1]+2] = formant_colors[1]
    if formants_arr[2] > 0: bitmap[-5:, formants_y[2]:formants_y[2]+2] = formant_colors[2]
    if formants_arr[3] > 0: bitmap[-5:, formants_y[3]:formants_y[3]+2] = formant_colors[3]
    if formants_arr[4] > 0: bitmap[-5:, formants_y[4]:formants_y[4]+2] = formant_colors[4]


    # render the formants and graph them out onto a separate graph
    #formants_y=librosa.hz_to_mel(audio_data["formants"]).astype(int)
    #formants_y[:] = max(2, min(formants_y[:], spectrogram_pixel_height - 2))
    bitmap[-5:,formants_y[0]:formants_y[0]+2]=formant_colors[0]
    bitmap[-5:,formants_y[1]:formants_y[1]+2]=formant_colors[1]
    bitmap[-5:,formants_y[2]:formants_y[2]+2]=formant_colors[2]
    bitmap[-5:,formants_y[3]:formants_y[3]+2]=formant_colors[3]
    #[0,255,0]
    #[0,255,127]
    #[255,0,0]
    #[255,0,255]

    
    # --- 3. RENDER ---
    # Give the raw numbers to the GPU and let it handle the colors
    img.setImage(bitmap, autoLevels=False)


    # --- 3. NEW: UPDATE PITCH GRAPH ---
    # Shift the history array to the left by 1
    pitch_history = np.roll(pitch_history, -1)
    
    # Put the newest pitch on the far right edge
    pitch_history[-1] = pitch_hz

    # np.where is an inline vectorized if/else statement
    pitch_history[-1] = np.where(pitch_hz > 0, pitch_hz, np.nan)
    
    # Give the array to the curve to draw it!
    pitch_curve.setData(pitch_history)
    
    # --- 5. UPDATE FORMANT GRAPH (VECTORIZED) ---
    global formant_history
    formant_history = np.roll(formant_history, -1, axis=1)
    
    # Vectorized assignment: If formant > 0, use the value, otherwise insert NaN
    formant_history[:, -1] = np.where(formants_arr > 0, formants_arr, np.nan)

    f1_curve.setData(formant_history[0])
    f2_curve.setData(formant_history[1])
    f3_curve.setData(formant_history[2])
    f4_curve.setData(formant_history[3])
    f5_curve.setData(formant_history[4])

    
    #bitmap[bitmap.shape[0],np.floor(63-librosa.hz_to_mel(pitch_hz))]=0
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
    # 4. Update the Text Readout
    
    # --- 4. UPDATE TEXT READOUT ---
    weight_percent = audio_data.get("vocalweight", 0.0)
    
    if pitch_hz > 0:
        closest_note = librosa.hz_to_note(pitch_hz)
        readout_label.setText(f"Pitch: {pitch_hz:.1f} Hz  |  Note: {closest_note}  |  Weight: {weight_percent:.1f}%")
    else:
        # The HTML app considers unvoiced sounds (S, Sh, F) as thick/heavy
        readout_label.setText(f"Pitch: -- Hz  |  Note: --  |  Weight: {weight_percent:.1f}% (Unvoiced)")
    
    # ... (existing Formant Graph update code) ...

    # --- 10. UPDATE WEIGHT GRAPH (Vectorized Color Bridging) ---
    global weight_history
    weight_history = np.roll(weight_history, -1)
    
    # Check for absolute silence using the raw audio buffer to hide the line
    if np.max(np.abs(current_audio)) < 0.001:
        weight_history[-1] = np.nan
    else:
        weight_history[-1] = weight_percent

    # 1. Create true/false masks for the 3 color thresholds from the HTML file
    green_mask = (weight_history < 16.5)
    red_mask = (weight_history >= 16.5) & (weight_history < 27.5)
    blue_mask = (weight_history >= 27.5)

    # 2. To prevent visual gaps where colors change, we use a NumPy shift trick 
    # to stretch the masks forward by 1 point so the lines perfectly bridge together!
    green_mask = green_mask | np.pad(green_mask[:-1], (1, 0), constant_values=False)
    red_mask = red_mask | np.pad(red_mask[:-1], (1, 0), constant_values=False)
    blue_mask = blue_mask | np.pad(blue_mask[:-1], (1, 0), constant_values=False)

    # 3. Apply the masks to create 3 separate fragmented lines and render them
    weight_green_curve.setData(np.where(green_mask, weight_history, np.nan))
    weight_red_curve.setData(np.where(red_mask, weight_history, np.nan))
    weight_blue_curve.setData(np.where(blue_mask, weight_history, np.nan))


# --- START THE LOOP ---
# PyQtGraph uses QTimer instead of Matplotlib's FuncAnimation
timer = QtCore.QTimer()
timer.timeout.connect(update_dashboard)
timer.start(UPDATE_INTERVAL_MS)

# direct hardware microphone data?
directmicrophonedatatoggle=True
def run():
    if not directmicrophonedatatoggle:
        stream = sd.InputStream(
            device=DEVICE_INDEX,
            channels=1,
            samplerate=SAMPLE_RATE,
            callback=audio_callback,
            blocksize=int(SAMPLE_RATE * 0.03) 
        )

        with stream:
            print("Microphone Active. Starting Dashboard...")
            #ani = animation.FuncAnimation(fig, update_dashboard, interval=UPDATE_INTERVAL_MS, blit=True) 
            #plt.show()
            # Start the audio stream and the GUI event loop
            pg.exec() # Keeps the application running
    else:

        # 1. Look for WASAPI (The Windows Audio API that allows raw hardware access)
        wasapi_info = sd.query_hostapis()
        wasapi_index = None
        for i, api in enumerate(wasapi_info):
            if 'WASAPI' in api['name']:
                wasapi_index = i
                break

        # 2. Setup the stream parameters
        stream_kwargs = {
            'channels': 1,
            'samplerate': SAMPLE_RATE,
            'callback': audio_callback,
            'blocksize': int(SAMPLE_RATE * 0.03) 
        }

        # 3. If WASAPI is found, apply the Exclusive Mode bypass trick!
        if wasapi_index is not None:
            print("WASAPI detected. Engaging Exclusive Mode to bypass Windows noise gates...")

            # We have to find the specific device index for the WASAPI version of your mic
            # (Since device indices change depending on the API you use)
            devices = sd.query_devices()
            for i, dev in enumerate(devices):
                if dev['hostapi'] == wasapi_index and dev['max_input_channels'] > 0:
                    # Grab the first available WASAPI input device
                    stream_kwargs['device'] = i
                    break

            # Apply the magic flag that blocks Windows from modifying the audio
            stream_kwargs['extra_settings'] = sd.WasapiSettings(exclusive=True)

        else:
            # Fallback for Mac/Linux users (Mac uses CoreAudio which doesn't have aggressive AGC by default)
            print("Standard audio API detected...")
            stream_kwargs['device'] = DEVICE_INDEX

        # 4. Start the Stream!
        try:
            stream = sd.InputStream(**stream_kwargs)
            with stream:
                print("Microphone Active. Starting Dashboard...")
                pg.exec() # Keeps the application running
        except Exception as e:
            print(f"\nCRITICAL AUDIO ERROR: {e}")
            print("Note: Exclusive Mode requires your SAMPLE_RATE to perfectly match your hardware's native rate.")


if __name__ == "__main__":
    run()
