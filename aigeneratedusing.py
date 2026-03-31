import pyaudio
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import queue
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
import librosa
from collections import deque

# Import your CNN architecture
from main8neuralnetwork import ConvolutionalNeuralNetwork, set_up_device

# --- CRITICAL CONFIGURATION MATH ---
CHUNK = 2048            
FORMAT = pyaudio.paFloat32
CHANNELS = 1
# MUST be 48000 to match RAVDESS. Otherwise, formants shift and genders flip!
RATE = 48000

# Keep a full 1-second buffer to prevent STFT edge artifacts
BUFFER_SAMPLES = RATE  
audio_buffer = np.zeros(BUFFER_SAMPLES, dtype=np.float32)

audio_queue = queue.Queue()

# --- MODEL SETUP ---
device = set_up_device()
model = ConvolutionalNeuralNetwork(num_outputs=2).to(device)
if os.path.exists("best_gender_model.pth"):
    model.load_state_dict(torch.load("best_gender_model.pth", map_location=device))
    print("Loaded best_gender_model.pth")
else:
    print("Error: Could not find best_gender_model.pth.")
    sys.exit()
model.eval()

# --- HELPER FUNCTIONS ---
def db_to_rgba(db_value, min_db=-80.0, max_db=0.0, cmap_name="magma"):
    """Replicates your exact color mapping from main6.py"""
    norm = mcolors.Normalize(vmin=min_db, vmax=max_db)
    cmap = plt.get_cmap(cmap_name)
    return cmap(norm(db_value))

def process_live_audio(audio_data):
    """
    Takes the sliding audio buffer and mathematically perfectly replicates
    the main6.py spectrogram extraction.
    """
    # 1. Replicate exact STFT geometry
    n_fft = 2048
    window_step = 256
    
    # Calculate STFT over the full 1-second buffer
    D = librosa.stft(audio_data, n_fft=n_fft, hop_length=window_step)
    mag = np.abs(D)
    
    # 2. FIX THE BLACK SMEAR:
    # librosa.stft pads the final n_fft//2 samples with zeros. 
    # 1024 samples / 256 hop = exactly 4 corrupted frames at the end.
    # We slice out 64 frames completely avoiding the corrupted padding.
    valid_mag = mag[:, -68:-4]
    
    # 3. FIX THE AUTO-GAIN:
    # If the buffer is silent, np.max() blows the noise floor up to 0 dB.
    # We clamp the reference so pure silence stays dark.
    # (Typical speech hits ~0.3 to 1.0 amplitude. 0.05 is a safe noise floor).
    ref_val = max(np.max(valid_mag), 0.05)
    
    # Convert to Decibels
    S_db = librosa.amplitude_to_db(valid_mag, ref=ref_val)
    
    # 4. Convert to RGB Image using your exact main6.py math
    spectrogram_color_data = db_to_rgba(S_db)
    
    # Vertical flip to fix rendering orientation (from main6.py)
    spectrogram_color_data = np.flipud(spectrogram_color_data)
    
    # Convert to 0-255 uint8 format, stripping the alpha channel
    img_array = (spectrogram_color_data[:, :, :3] * 255).astype(np.uint8)
    
    # 5. Convert to PyTorch Tensor
    img = Image.fromarray(img_array)
    transform = transforms.Compose([transforms.ToTensor()])
    tensor = transform(img).unsqueeze(0)
    
    return tensor, img_array

def audio_callback(in_data, frame_count, time_info, status):
    """Puts live mic data into the queue without blocking the audio stream"""
    audio_data = np.frombuffer(in_data, dtype=np.float32)
    audio_queue.put(audio_data)
    return (in_data, pyaudio.paContinue)

# --- GRAPH SETUP ---
history_length = 50
prob_history = deque([0.5] * history_length, maxlen=history_length)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [1, 1]})
fig.canvas.manager.set_window_title("Live Biofeedback")

# Top Subplot: The Spectrogram
dummy_img = np.zeros((1025, 64, 3), dtype=np.uint8)
# aspect='auto' stretches the 64-width image to fit the UI screen.
img_plot = ax1.imshow(dummy_img, aspect='auto', origin='upper', interpolation='bilinear')
ax1.set_title("What the AI Sees (Clean 64-Frame Window)", fontsize=12)
ax1.axis('off')

# Bottom Subplot: The Probability Line
line, = ax2.plot(prob_history, color='#b19cd9', linewidth=3)
ax2.set_ylim(-0.05, 1.05)
ax2.set_xlim(0, history_length)
ax2.set_title("Live Resonance Probability", fontsize=12)
ax2.set_ylabel("1.0 = Female | 0.0 = Male", fontsize=10)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
ax2.set_xticks([])

plt.tight_layout()

def update_graph(frame):
    global audio_buffer
    
    updated = False
    
    # Pull ALL new audio from the queue and slide the buffer
    while not audio_queue.empty():
        new_audio = audio_queue.get()
        audio_buffer = np.roll(audio_buffer, -len(new_audio))
        audio_buffer[-len(new_audio):] = new_audio
        updated = True
        
    if updated:
        img_tensor, rgb_array = process_live_audio(audio_buffer)
        
        # Update Spectrogram UI
        img_plot.set_data(rgb_array)
        
        # Run Model & Update Probability
        img_tensor = img_tensor.to(device)
        with torch.no_grad():
            logits = model(img_tensor)
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            female_prob = probabilities[0][1].item()
            
        prob_history.append(female_prob)
        line.set_ydata(prob_history)
        
        # Color coding
        if female_prob > 0.6:
            line.set_color('#ff69b4') # Pink
        elif female_prob < 0.4:
            line.set_color('#4169e1') # Blue
        else:
            line.set_color('#ffd700') # Yellow
            
    return img_plot, line

def main():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                    input=True, frames_per_buffer=CHUNK, stream_callback=audio_callback)

    print("\n--- Starting Live Biofeedback ---")
    stream.start_stream()

    ani = FuncAnimation(fig, update_graph, interval=30, blit=True, cache_frame_data=False)
    plt.show() 

    stream.stop_stream()
    stream.close()
    p.terminate()
    print("Stream stopped.")

if __name__ == "__main__":
    main()