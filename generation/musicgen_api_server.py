import json
import os
import sys
import time
from flask import Flask, request, jsonify
import torch
from audiocraft.models import musicgen
from audiocraft.data.audio import audio_write

app = Flask(__name__)

# Configure paths
NAS_PATH = "/mnt/files"
os.makedirs(NAS_PATH, exist_ok=True)

# Global model variable
model = None

def get_device():
    """Get the best available device, forcing CPU if specified"""
    # Check environment variables for CPU forcing
    if os.getenv('USE_CPU', '0') == '1' or os.getenv('CUDA_VISIBLE_DEVICES', '') == '':
        print("🔋 Forcing CPU usage due to environment configuration", file=sys.stderr)
        return 'cpu'
    
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

def get_model_device(model):
    """Get the device that the MusicGen model is using"""
    try:
        # Try different ways to access the model device
        if hasattr(model, 'device'):
            return model.device
        elif hasattr(model, 'compression_model') and hasattr(model.compression_model, 'device'):
            return model.compression_model.device
        elif hasattr(model, 'generation_model'):
            # Try to get device from generation model parameters
            try:
                return next(model.generation_model.parameters()).device
            except:
                pass
        
        # Fallback: try to access any model component
        for attr_name in ['compression_model', 'generation_model', 'lm']:
            if hasattr(model, attr_name):
                attr = getattr(model, attr_name)
                if hasattr(attr, 'parameters'):
                    try:
                        return next(attr.parameters()).device
                    except:
                        continue
        
        # Final fallback
        return torch.device('cpu')
    except Exception as e:
        print(f"⚠️ Could not determine model device: {e}", file=sys.stderr)
        return torch.device('cpu')

def load_model():
    """Load the MusicGen model if not already loaded"""
    global model
    if model is None:
        print("🔄 Loading MusicGen model...", file=sys.stderr)
        start_time = time.time()
        
        device = get_device()
        print(f"🔋 Using device: {device}", file=sys.stderr)
        
        try:
            # For CPU usage, we might want to use a smaller model
            model_name = 'facebook/musicgen-small'  
            if device == 'cpu':
                print("💡 Using small model for CPU optimization", file=sys.stderr)
            
            # Try loading with explicit device specification
            model = musicgen.MusicGen.get_pretrained(model_name, device=device)
            
        except Exception as e:
            print(f"⚠️ Failed to load with device={device}, trying auto device selection: {e}", file=sys.stderr)
            # Fallback: let the model choose the device automatically
            model = musicgen.MusicGen.get_pretrained('facebook/musicgen-small')
        
        print(f"✅ Model loaded successfully in {time.time() - start_time:.2f} seconds", file=sys.stderr)
        
        # Print actual device being used
        try:
            actual_device = get_model_device(model)
            print(f"🔋 Model is using device: {actual_device}", file=sys.stderr)
        except Exception as e:
            print(f"⚠️ Could not determine model device: {e}", file=sys.stderr)
    
    return model

@app.route("/")
def index():
    """Health check endpoint"""
    device_info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "pytorch_version": torch.__version__,
        "forced_cpu": os.getenv('USE_CPU', '0') == '1',
        "cuda_visible_devices": os.getenv('CUDA_VISIBLE_DEVICES', 'not_set')
    }
    
    if hasattr(torch.backends, 'mps'):
        device_info["mps_available"] = torch.backends.mps.is_available()
    
    return jsonify({
        "status": "ok", 
        "message": "MusicGen API server is running",
        "device_info": device_info
    })

@app.route("/generate", methods=["POST"])
def generate():
    """Generate music based on the prompt"""
    try:
        # Parse request data
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        prompt = data.get("prompt", "")
        outpath = data.get("outpath", "output.wav")
        duration = int(data.get("duration", 30))
        
        # Validate input
        if not prompt:
            return jsonify({"error": "Missing prompt parameter"}), 400
        
        # Cap duration for safety (especially important for CPU)
        duration = min(max(duration, 5), 120)  # Min 5s, Max 2 minutes for CPU
        
        # Ensure outpath has .wav extension
        if not outpath.endswith('.wav'):
            outpath += '.wav'
        
        # Full path for output file
        full_output_path = os.path.join(NAS_PATH, outpath)

        print(f"📁 Output path: {full_output_path}", file=sys.stderr)
        
        # Load model
        print(f"🔄 Loading model if needed...", file=sys.stderr)
        model = load_model()
        
        # Set generation parameters (CPU-optimized)
        print(f"🎼 Setting generation parameters: duration={duration}s", file=sys.stderr)
        
        # Check if model is on CPU and apply optimizations
        model_device = get_model_device(model)
        if model_device.type == 'cpu':
            print("🔧 Applying CPU optimizations...", file=sys.stderr)
            model.set_generation_params(
                duration=duration,
                use_sampling=True,
                top_k=250,  # Reduced from default
                top_p=0.0   # Disabled for speed
            )
        else:
            model.set_generation_params(duration=duration)
        
        # Generate music
        print(f"🎵 Generating music for prompt: '{prompt}'", file=sys.stderr)
        start_time = time.time()
        
        with torch.no_grad():  # Save memory during generation
            # For CPU inference, we can also reduce batch size if needed
            wav = model.generate([prompt])
        
        generation_time = time.time() - start_time
        print(f"✅ Generation completed in {generation_time:.2f} seconds", file=sys.stderr)
        
        # Normalize the audio to prevent silent output
        print("🔊 Normalizing audio...", file=sys.stderr)
        wav_normalized = wav / (wav.abs().max() + 1e-8) * 0.9  # Add small epsilon to prevent division by zero
        
        # Save audio
        print(f"💾 Saving audio to {full_output_path}", file=sys.stderr)
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(full_output_path), exist_ok=True)
        
        # Convert tensor to CPU if it's on GPU
        wav_cpu = wav_normalized[0].cpu()
        
        # Debug audio tensor
        print(f"🔍 Audio tensor shape: {wav_cpu.shape}", file=sys.stderr)
        print(f"🔍 Audio min/max values: {wav_cpu.min().item():.6f} / {wav_cpu.max().item():.6f}", file=sys.stderr)
        print(f"🔍 Audio mean: {wav_cpu.mean().item():.6f}", file=sys.stderr)
        
        # Save the audio file
        try:
            audio_write(
                full_output_path.replace('.wav', ''),  # audio_write adds .wav automatically
                wav_cpu, 
                model.sample_rate, 
                format="wav"
            )
        except Exception as save_error:
            print(f"⚠️ Error with audio_write, trying alternative: {save_error}", file=sys.stderr)
            # Alternative saving method
            import torchaudio
            torchaudio.save(full_output_path, wav_cpu.unsqueeze(0) if wav_cpu.dim() == 1 else wav_cpu, model.sample_rate)
        
        # Check for the actual saved file (audio_write might add .wav)
        possible_paths = [
            full_output_path,
            full_output_path.replace('.wav', '') + '.wav'
        ]
        
        saved_path = None
        for path in possible_paths:
            if os.path.exists(path):
                saved_path = path
                break
        
        if saved_path and os.path.exists(saved_path):
            file_size = os.path.getsize(saved_path)
            print(f"✅ File saved successfully: {saved_path} ({file_size} bytes)", file=sys.stderr)
            return jsonify({
                "status": "success",
                "message": "Music generated successfully",
                "file_path": saved_path,
                "duration": duration,
                "generation_time": generation_time,
                "file_size": file_size,
                "device_used": str(model_device),
                "audio_stats": {
                    "min": wav_cpu.min().item(),
                    "max": wav_cpu.max().item(),
                    "mean": wav_cpu.mean().item()
                }
            })
        else:
            print(f"❌ Failed to save file to any of: {possible_paths}", file=sys.stderr)
            return jsonify({"error": "Failed to save generated audio"}), 500
            
    except Exception as e:
        print(f"❌ Error: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    """Detailed health check"""
    try:
        # Check if model is loaded
        model_loaded = model is not None
        
        # Check disk space
        import shutil
        disk_usage = shutil.disk_usage(NAS_PATH)
        free_space_gb = disk_usage.free / (1024**3)
        
        # Memory info
        memory_info = {}
        if torch.cuda.is_available():
            memory_info["cuda_memory_allocated"] = torch.cuda.memory_allocated()
            memory_info["cuda_memory_reserved"] = torch.cuda.memory_reserved()
        
        # Model device info
        model_device_info = None
        if model_loaded:
            try:
                model_device_info = str(get_model_device(model))
            except:
                model_device_info = "unknown"
        
        return jsonify({
            "status": "healthy",
            "model_loaded": model_loaded,
            "model_device": model_device_info,
            "free_space_gb": round(free_space_gb, 2),
            "nas_path": NAS_PATH,
            "pytorch_version": torch.__version__,
            "memory_info": memory_info,
            "cpu_forced": os.getenv('USE_CPU', '0') == '1'
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500

if __name__ == "__main__":
    print(f"🚀 Starting MusicGen Flask Server", file=sys.stderr)
    print(f"📁 Output directory: {NAS_PATH}", file=sys.stderr)
    print(f"🔥 PyTorch version: {torch.__version__}", file=sys.stderr)
    print(f"🔋 CUDA available: {torch.cuda.is_available()}", file=sys.stderr)
    print(f"💻 CPU forced: {os.getenv('USE_CPU', '0') == '1'}", file=sys.stderr)
    print(f"🎯 CUDA_VISIBLE_DEVICES: {os.getenv('CUDA_VISIBLE_DEVICES', 'not_set')}", file=sys.stderr)
    
    # Pre-load the model at startup
    try:
        print("🔄 Pre-loading model...", file=sys.stderr)
        load_model()
        print("✅ Model pre-loaded successfully", file=sys.stderr)
    except Exception as e:
        print(f"⚠️ Warning: Failed to pre-load model: {e}", file=sys.stderr)
        print("Will attempt to load model at first request", file=sys.stderr)
    
    # Run the Flask app
    print("🌐 Starting Flask server on http://0.0.0.0:8001", file=sys.stderr)
    app.run(host="0.0.0.0", port=8001, debug=False, threaded=True)