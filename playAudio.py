import sounddevice as sd
import time
from scipy.io import wavfile

def play_wav(file):
    sample_rate, data = wavfile.read(file)  # Read WAV file
    sd.play(data, sample_rate)  # Play audio
    sd.wait()  # Wait until playback is finished

# Original Audio
original_audio_path = "data/yes/c120e80e_nohash_4.wav"
# original_audio_path = "untargeted_epoch_68_20250506_180344.wav"
print("Start Original Audio")
play_wav(original_audio_path)
print("Original Audio ended")

time.sleep(2)

# Adversarial Audio
adversarial_audio_path = "untargeted_epoch_84_20250506_190930.wav"
print("Start Adversarial Audio")
play_wav(adversarial_audio_path)
print("Adversarial Audio ended")