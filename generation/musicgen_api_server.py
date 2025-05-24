import json
import os
import sys
import time
from flask import Flask, request, jsonify
import torch
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

app = Flask(__name__)

# Configure paths
NAS_PATH = "/mnt/files"
os.makedirs(NAS_PATH, exist_ok=True)

# Global model variable
model = None

def load_model():
    """Load the MusicGen model if not already loaded"""
    global model
    if model is None:
        print("🔄 Loading MusicGen model...", file=sys.stderr)
        start_time = time.time()
        model = MusicGen.get_pretrained('facebook/musicgen-small', device='cuda' if torch.cuda.is_available() else 'cpu')
        print(f"✅ Model loaded successfully in {time.time() - start_time:.2f} seconds", file=sys.stderr)
        print(f"🔋 Using device: {next(model.parameters()).device}", file=sys.stderr)
    return model

@app.route("/")
def index():
    """Health check endpoint"""
    return jsonify({"status": "ok", "message": "MusicGen API server is running"})

@app.route("/generate", methods=["POST"])
def generate():
    """Generate music based on the prompt"""
    try:
        # Parse request data
        data = request.json
        prompt = data.get("prompt", "")
        outpath = data.get("outpath", "output.wav")
        duration = int(data.get("duration", 30))
        
        # Validate input
        if not prompt:
            return jsonify({"error": "Missing prompt parameter"}), 400
        
        # Cap duration for safety
        duration = min(duration, 180)  # Max 3 minutes
        
        # Full path for output file
        full_output_path = os.path.join(NAS_PATH, outpath)
        
        # Load model
        model = load_model()
        
        # Set generation parameters
        print(f"🎼 Setting generation parameters: duration={duration}s", file=sys.stderr)
        model.set_generation_params(duration=duration)
        
        # Generate music
        print(f"🎵 Generating music for prompt: '{prompt}'", file=sys.stderr)
        start_time = time.time()
        wav = model.generate([prompt])
        generation_time = time.time() - start_time
        print(f"✅ Generation completed in {generation_time:.2f} seconds", file=sys.stderr)
        
        # Save audio
        print(f"💾 Saving audio to {full_output_path}", file=sys.stderr)
        audio_write(full_output_path, wav[0].cpu(), model.sample_rate, format="wav")
        
        # Verify file was saved
        if os.path.exists(full_output_path):
            print(f"✅ File saved successfully", file=sys.stderr)
            return jsonify({
                "status": "success", 
                "message": "Music generated successfully",
                "file_path": full_output_path,
                "duration": duration,
                "generation_time": generation_time
            })
        else:
            print(f"❌ Failed to save file", file=sys.stderr)
            return jsonify({"error": "Failed to save generated audio"}), 500
    
    except Exception as e:
        print(f"❌ Error: {str(e)}", file=sys.stderr)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Pre-load the model at startup
    try:
        load_model()
    except Exception as e:
        print(f"⚠️ Warning: Failed to pre-load model: {e}", file=sys.stderr)
        print("Will attempt to load model at first request", file=sys.stderr)
    
    # Run the Flask app
    app.run(host="0.0.0.0", port=8001, debug=False)