# Fix the import statement
from audiocraft.audiocraft.models import MusicGen

def preload():
    # Initialize the model
    model = MusicGen.get_pretrained('facebook/musicgen-small')
    return model

if __name__ == "__main__":
    preload()