from audiocraft.models import musicgen
from audiocraft.data.audio import audio_write
import os

# Ask the user for input
prompt = input("🧠 Enter your music prompt:\n> ")
filename = input("💾 Enter output filename (without extension):\n> ")

# Load the model
model = musicgen.MusicGen.get_pretrained('facebook/musicgen-medium', device='cpu')

# Set generation parameters
model.set_generation_params(duration=30)  # 3 minutes of audio

# Generate music
print("\n🎧 Generating music...")
wav = model.generate([prompt])

# Save the file
audio_write(filename, wav[0].cpu(), model.sample_rate, format="wav")

print(f"File size: {os.path.getsize(f'{filename}.wav')} bytes")
print(f"✅ Done! Saved as {filename}.wav")
