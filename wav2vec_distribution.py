import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# Load Wav2Vec2 Model
MODEL_NAME = "facebook/wav2vec2-large-960h"
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)

# Load Audio File (Ensure it's 16kHz mono)
AUDIO_PATH = "data/yes/c120e80e_nohash_4.wav"  # Replace with your file
waveform, sample_rate = librosa.load(AUDIO_PATH, sr=16000)  # Force 16kHz for Wav2Vec2
print("waveform: ", waveform.shape)
# waveform = waveform[12010:]
print("waveform: ", waveform.shape)
# Convert to tensor
input_values = processor(waveform, sampling_rate=sample_rate, return_tensors="pt", padding=True).input_values

# Transcription & Logits
with torch.no_grad():
    logits = model(input_values).logits

print("logits shape:",logits.shape)
print("logits",logits)


# Get character-wise probabilities
probs = torch.nn.functional.softmax(logits, dim=-1)
min_value = torch.min(probs)
max_value = torch.max(probs)

print("Min Value:", min_value.item())
print("Max Value:", max_value.item())

print("probs shape",probs.shape)
print("probabilities",probs)


# Get predicted tokens
pred_ids = torch.argmax(logits, dim=-1)[0].numpy()  # Convert tensor to NumPy array
print("pred_ids shape:",pred_ids.shape)
print('pred_ids:')
print(pred_ids)

# Decode using Wav2Vec2 processor (CTC decoding)
transcription = processor.batch_decode([pred_ids])[0]
print("Transcription: ", transcription)

# Token-to-character mapping (for alignment)
tokens = processor.tokenizer.convert_ids_to_tokens(pred_ids)
tokens_replaced = processor.tokenizer.convert_ids_to_tokens(pred_ids)

print("Type of tokens:",type(tokens))
print('Actual tokens:')
print(tokens)
# tokens_replaced = []
for i,n in enumerate(tokens):
    if n == '<pad>' or n == '|':
        tokens_replaced[i] = '0'
    else:
        tokens_replaced[i] = tokens[i]

print("Replaced tokens:")
print(tokens_replaced)

# Compute window size
num_windows = probs.shape[1]  # Number of frames predicted by Wav2Vec2
time_per_window = waveform.shape[0] / sample_rate / num_windows  # Time step per frame
timestamps = np.array([i * time_per_window for i in range(num_windows)])

# Convert timestamps to sample numbers
sample_positions = (timestamps * sample_rate).astype(int)

# Filter out blank tokens and align characters
char_data = []
for i, token in enumerate(tokens):
    if token != "|":  # Ignore blank tokens (CTC uses "|" for silence)
        char = token.replace("▁", "")  # Remove Wav2Vec2 subword marker
        start_sample = sample_positions[i]  # Start sample number
        end_sample = sample_positions[i + 1] if i + 1 < len(sample_positions) else waveform.shape[0]  # End sample
        char_data.append((char, start_sample, end_sample))

# Print sample number and corresponding character
print(f"\nSample Rate: {sample_rate} Hz")
print("\nCharacter Sample Number Report:")
for char, start, end in char_data:
    if char != "<pad>":
        print(f"Character '{char}': Start Sample = {start}, End Sample = {end}")

# Plot waveform
fig, ax1 = plt.subplots(figsize=(12, 4))

# Time axis (bottom)
time_axis = np.linspace(0, waveform.shape[0] / sample_rate, num=len(waveform))
ax1.plot(time_axis, waveform, alpha=0.7, label="Waveform")

ax1.set_xlabel("Time (seconds)")
ax1.set_ylabel("Amplitude")
ax1.set_title(f"Wav2Vec2 Transcription with Sample Positions: {transcription}")

# Sample number axis (top)
ax2 = ax1.twiny()
ax2.set_xlim(0, waveform.shape[0])
ax2.set_xlabel("Sample Number")

# Annotate detected character positions
for char, start, end in char_data:
    if char != "<pad>":
        start_time = start / sample_rate  # Convert sample number to time
        ax1.text(start_time, max(waveform) * 0.8, char, fontsize=12, ha="center", color="red")

plt.legend()
plt.show()