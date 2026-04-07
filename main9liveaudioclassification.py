# 12/17/2025
# Riley Mohr
from main8neuralnetwork import Classifier
deviceName=Classifier()


# i need something that takes live audio input and get it as waveform data
# sooo...
# append to list new audio
# when new audio is appended, take a window of the most recent audio data
# pipe that audio though main6 somehow to get out a singular spectrogram
# plug said spectrogram into the classifier
# link the classifier to the display
# :)


#ai moment
# main9live.py
# Riley Mohr - Real-time Audio Classifier Dashboard
# 12/17/2025

import numpy as np
import torch
import sounddevice as sd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
from torchvision import transforms
from main6 import process_live_audio
from main8neuralnetwork import load_models_only_no_dataset, classify_with_model

# --- CONFIGURATION ---
SAMPLE_RATE = 22050     
WINDOW_SECONDS = 2.0    
UPDATE_INTERVAL = 30    # 30ms = ~33 FPS (Smoother) #changed to 250 cuz laggy on non gpu device
DEVICE_INDEX = None     

# Labels
EMOTIONS = ["Neutral", "Calm", "Happy", "Sad", "Angry", "Fearful", "Disgust", "Surprised"]
GENDERS = ["Male", "Female"]

# --- SETUP MODELS ---
print("Loading Neural Networks...")
device, gender_model, emotion_model_M, emotion_model_F = load_models_only_no_dataset()
transform = transforms.Compose([transforms.ToTensor()])

# --- AUDIO BUFFER STATE ---
buffer_size = int(SAMPLE_RATE * WINDOW_SECONDS)
audio_buffer = np.zeros(buffer_size, dtype=np.float32)

def audio_callback(indata, frames, time, status):
    global audio_buffer
    if status: print(status)
    new_data = indata.flatten()
    audio_buffer = np.roll(audio_buffer, -len(new_data))
    audio_buffer[-len(new_data):] = new_data

# --- VISUALIZATION SETUP ---
# Fixed size, manual layout
#plt.style.use('dark_background')
fig = plt.figure(figsize=(12, 9), facecolor='darkcyan') # Change 'darkcyan' to 'cyan' if you want it bright!
fig.suptitle('Live SER Classifier (RAVDESS)', fontsize=18, color='white', fontweight='bold', y=0.96)

# Manually reserve space: 
# bottom=0.25 gives huge room for labels
# hspace=0.3 gives room between spectrogram and bars
fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.25, wspace=0.2, hspace=0.3)

gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])

# 1. Spectrogram
ax_spec = fig.add_subplot(gs[0, :])
ax_spec.set_title("Live Mel-Spectrogram Input", fontsize=14)
ax_spec.axis('off')

# FIX: Set height back to 64, but keep width at 256
dummy_img = np.zeros((64, 256, 3), dtype=np.uint8) 
im_display = ax_spec.imshow(dummy_img, aspect='auto', origin='upper', animated=True)

# 2. Gender Bar
ax_gender = fig.add_subplot(gs[1, 0])
ax_gender.set_title("Gender", fontsize=14)
ax_gender.set_ylim(0, 1.1) 
gender_bars = ax_gender.bar(GENDERS, [0.01, 0.01], color=['cyan', 'magenta']) # Start near 0
ax_gender.set_xticks(range(len(GENDERS)))
ax_gender.set_xticklabels(GENDERS, fontsize=12, color='white', fontweight='bold')

# 3. Emotion Bar
ax_emotion = fig.add_subplot(gs[1, 1])
ax_emotion.set_title("Emotion", fontsize=14)
ax_emotion.set_ylim(0, 1.1) 
emotion_bars = ax_emotion.bar(EMOTIONS, [0.01]*8, color='orange')
ax_emotion.set_xticks(range(len(EMOTIONS)))
ax_emotion.set_xticklabels(EMOTIONS, rotation=45, ha='right', fontsize=12, color='white')

# Clean up axes
for ax in [ax_gender, ax_emotion]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

# --- TEXT LABELS ---
# Pre-initialize text objects so we don't create new ones every frame (causes lag)
gender_texts = []
for bar in gender_bars:
    text = ax_gender.text(
        bar.get_x() + bar.get_width() / 2, 
        0.05, 
        '', 
        ha='center', va='bottom', color='white', fontsize=12, fontweight='bold', animated=True
    )
    gender_texts.append(text)

emotion_texts = []
for bar in emotion_bars:
    text = ax_emotion.text(
        bar.get_x() + bar.get_width() / 2, 
        0.05, 
        '', 
        ha='center', va='bottom', color='white', fontsize=10, fontweight='bold', animated=True
    )
    emotion_texts.append(text)

# Smoothing
emotion_smooth = np.zeros(8)
gender_smooth = np.zeros(2)
ALPHA = 0.3 # Slightly faster response

# --- SILENCE CONFIGURATION ---
SILENCE_THRESHOLD = 0.01  # Amplitude threshold (0.0 to 1.0). Tweak this if it cuts off your voice.

def update_dashboard(frame):
    global emotion_smooth, gender_smooth
    
    # 1. Get Audiotomorrow
    current_audio = audio_buffer.copy()
    
    # --- SILENCE GATE ---
    # Calculate volume (Root Mean Square)
    volume = np.sqrt(np.mean(current_audio**2))

    print(f"Mic Volume: {volume:.5f}")
    
    # If too quiet, fade bars to zero and skip the heavy AI processing
    if volume < SILENCE_THRESHOLD:
        # Fade out smoothly
        gender_smooth = (gender_smooth * (1 - ALPHA)) + (0 * ALPHA)
        emotion_smooth = (emotion_smooth * (1 - ALPHA)) + (0 * ALPHA)
        
        # Still update the visual bars to show they are dropping
        for bar, height, text in zip(gender_bars, gender_smooth, gender_texts):
            bar.set_height(height)
            text.set_text("") # Hide text when silent
            text.set_y(height + 0.02)
            
        for bar, height, text in zip(emotion_bars, emotion_smooth, emotion_texts):
            bar.set_height(height)
            text.set_text("")
            text.set_y(height + 0.02)
            
        # Optional: You can still show the spectrogram if you want to see the "silence"
        # But we won't run the neural network
        try:
            bitmap = process_live_audio(current_audio, SAMPLE_RATE)
            # Crop logic...
            target_w = 256 
            if bitmap.shape[1] >= target_w:
                start = (bitmap.shape[1] - target_w) // 2
                crop = bitmap[:, start:start+target_w, :]
            else:
                # FIX: Use bitmap.shape[0] (which is 64) for the height so it fits perfectly
                crop = np.zeros((bitmap.shape[0], target_w, 3), dtype=np.uint8) 
                crop[:, :bitmap.shape[1], :] = bitmap
            im_display.set_data(crop)
            return [im_display] + list(gender_bars) + list(emotion_bars)
        # Fix the first one (inside the silence gate)
        except Exception as e:
            print(f"Spectrogram error: {e}")
            return [im_display] + list(gender_bars) + list(emotion_bars) + gender_texts + emotion_texts

    # --- BELOW IS THE STANDARD AI PROCESSING (Only runs if volume > threshold) ---
    
    try:
        bitmap = process_live_audio(current_audio, SAMPLE_RATE)
    # Fix the second one (in the standard AI processing)
    except Exception as e:
        print(f"Spectrogram error: {e}")
        return [im_display] + list(gender_bars) + list(emotion_bars) + gender_texts + emotion_texts

    # Crop
    target_w = 256 
    if bitmap.shape[1] >= target_w:
        start = (bitmap.shape[1] - target_w) // 2
        crop = bitmap[:, start:start+target_w, :]
    else:
        # FIX: Use bitmap.shape[0] so the heights match!
        crop = np.zeros((bitmap.shape[0], target_w, 3), dtype=np.uint8) 
        crop[:, :bitmap.shape[1], :] = bitmap

    im_display.set_data(crop)
    im_display.set_data(crop)
    
    # 2. Inference
    img_pil = Image.fromarray(crop)
    
    # FIX: Stretch the image to 256x256 right before the neural network gets it
    img_pil = img_pil.resize((256, 256)) 
    
    img_tensor = transform(img_pil).unsqueeze(0).to(device)
    
    with torch.no_grad():
        g_logits = gender_model(img_tensor)
        g_probs = torch.nn.functional.softmax(g_logits, dim=1).cpu().numpy()[0]
        
        predicted_gender = np.argmax(g_probs)
        
        if predicted_gender == 0: 
            e_logits = emotion_model_M(img_tensor)
            model_color = 'cyan' 
        else: 
            e_logits = emotion_model_F(img_tensor)
            model_color = 'magenta'

        e_probs = torch.nn.functional.softmax(e_logits, dim=1).cpu().numpy()[0]

    # 3. Smoothing
    gender_smooth = (gender_smooth * (1 - ALPHA)) + (g_probs * ALPHA)
    emotion_smooth = (emotion_smooth * (1 - ALPHA)) + (e_probs * ALPHA)

    # 4. Update Artists
    artists = [im_display]
    
    # Gender Updates
    for bar, height, text in zip(gender_bars, gender_smooth, gender_texts):
        bar.set_height(height)
        text.set_text(f"{height:.0%}")
        text_y = min(max(height + 0.02, 0.05), 1.0)
        text.set_y(text_y)
        artists.append(bar)
        artists.append(text)
    
    # Emotion Updates
    for bar, height, text in zip(emotion_bars, emotion_smooth, emotion_texts):
        bar.set_height(height)
        bar.set_color(model_color)
        text.set_text(f"{height:.0%}")
        text_y = min(max(height + 0.02, 0.05), 1.0)
        text.set_y(text_y)
        artists.append(bar)
        artists.append(text)

    return artists
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