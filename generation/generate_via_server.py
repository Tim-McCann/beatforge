import sys
import os
from audiocraft.audiocraft.models import musicgen
from audiocraft.audiocraft.data.audio import audio_write

if len(sys.argv) < 3:
    print("Usage: python generate_via_server.py <prompt> <filename>")
    sys.exit(1)

prompt = sys.argv[1]
filename = sys.argv[2]

# ✅ NAS mount path inside the container
nas_path = "/mnt/files"
full_path = os.path.join(nas_path, filename)

print(f"Full path: {full_path}")

# Load model
model = musicgen.MusicGen.get_pretrained('facebook/musicgen-small', device='cpu')
model.set_generation_params(duration=180)

print("\n🎧 Generating music...")
wav = model.generate([prompt])

# ✅ Write file
audio_write(full_path, wav[0].cpu(), model.sample_rate, format="wav")

if os.path.exists(full_path):
    print(f"✅ File saved successfully at {full_path}")
else:
    print(f"❌ Failed to save the file at {full_path}")
