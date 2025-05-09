import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torchaudio
import torch
import scipy.io.wavfile as wav
import soundfile as sf

# Load Pretrained Wav2Vec2 model and processor
model_name = "facebook/wav2vec2-large-960h"  # Pretrained ASR model
model = Wav2Vec2ForCTC.from_pretrained(model_name)
processor = Wav2Vec2Processor.from_pretrained(model_name)

# Load Audio File
# audio_file = "data/right/1eddce1d_nohash_2.wav"
# audio_file = "data/no/28ce0c58_nohash_7.wav"
audio_file = "untargeted_epoch_84_20250506_190930.wav"
speech_array, sampling_rate = torchaudio.load(audio_file)
speech_np = speech_array[0].cpu().numpy()  # Convert to NumPy array

# speech_np = np.clip(speech_np, -2.0, 2.0)  # Clip the values to be within -1.0 and 1.0
# print("Adv_audio_max, Adv_audio_min", max(speech_array[0]), min(speech_array[0]))
# wav.write("generated_audio_clipped_epoch_66.wav", 16000, speech_np)  # Save as 16-bit PCM

# speech_array = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)(speech_array)
# speech_array = speech_array.squeeze().numpy()
#
# # Process Input Features
input_values = processor(speech_array, sampling_rate=16000, return_tensors="pt", padding=True).input_values
input_values = input_values.reshape(1, -1)

# Perform Inference
with torch.no_grad():
    logits = model(input_values).logits
print(logits.shape)

# Decode Transcription and Probabilities
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)[0]
print("Transcription:", transcription)