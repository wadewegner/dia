import os
import gradio as gr
import time
from pathlib import Path
import soundfile as sf
from dia.model import Dia

# Create output directory if it doesn't exist
os.makedirs("audio_outputs", exist_ok=True)

# Initialize model
print("Loading Dia model...")
model = Dia.from_pretrained("nari-labs/Dia-1.6B", device="cuda", compute_dtype="float16")
print("Model loaded successfully!")

def generate_speech(text, audio_prompt=None, cfg_scale=3.0, temperature=1.3, top_p=0.95, seed=None):
    """Generate speech and save it to a file"""
    if not text:
        return None, "Please enter text to generate speech."
    
    # Format the text if it doesn't have speaker tags
    if not text.strip().startswith("[S1]") and not text.strip().startswith("[S2]"):
        text = f"[S1] {text}"
    
    try:
        # Generate timestamp for unique filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"audio_outputs/speech_{timestamp}.wav"
        
        # Set seed if provided
        if seed is not None:
            try:
                seed = int(seed)
                import torch
                import random
                import numpy as np
                torch.manual_seed(seed)
                random.seed(seed)
                np.random.seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
            except ValueError:
                return None, "Seed must be an integer."
        
        # Generate audio
        print(f"Generating speech for: {text[:50]}...")
        audio = model.generate(
            text=text,
            audio_prompt=audio_prompt,
            cfg_scale=cfg_scale,
            temperature=temperature,
            top_p=top_p,
            use_torch_compile=True,
            verbose=True
        )
        
        # Save audio
        sf.write(filename, audio, 44100)
        print(f"Audio saved to {filename}")
        
        return filename, f"Speech generated and saved to {filename}"
    except Exception as e:
        return None, f"Error generating speech: {str(e)}"

def list_audio_files():
    """List all audio files in the audio_outputs directory"""
    files = list(Path("audio_outputs").glob("*.wav"))
    files.extend(list(Path("audio_outputs").glob("*.mp3")))
    files = sorted(files, key=os.path.getmtime, reverse=True)  # Sort by modification time (newest first)
    return [str(f) for f in files]

def refresh_audio_list():
    """Refresh the audio file list"""
    return gr.Dropdown(choices=list_audio_files(), value=None)

def select_audio_file(file_path):
    """Select an audio file to play"""
    if file_path:
        return file_path
    return None

# Build the Gradio interface
with gr.Blocks(title="Dia Speech Generator") as app:
    gr.Markdown("# Dia Speech Generator and Player")
    gr.Markdown("Generate speech using the Dia model and play audio files.")
    
    with gr.Tab("Generate Speech"):
        with gr.Row():
            with gr.Column(scale=3):
                text_input = gr.TextArea(
                    label="Text to generate speech from", 
                    placeholder="[S1] Hello, this is a test of the Dia text-to-speech model.",
                    lines=5
                )
                with gr.Row():
                    generate_btn = gr.Button("Generate Speech", variant="primary")
                    clear_btn = gr.Button("Clear")
                
                with gr.Accordion("Advanced Options", open=False):
                    with gr.Row():
                        cfg_scale = gr.Slider(minimum=1.0, maximum=10.0, value=3.0, step=0.1, label="CFG Scale")
                        temperature = gr.Slider(minimum=0.1, maximum=2.0, value=1.3, step=0.1, label="Temperature")
                    with gr.Row():
                        top_p = gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top P")
                        seed = gr.Textbox(label="Seed (optional)", placeholder="Leave empty for random")
                    
                    audio_prompt = gr.Audio(label="Audio Prompt (optional, for voice cloning)", type="filepath")
                
            with gr.Column(scale=2):
                output_audio = gr.Audio(label="Generated Speech", interactive=False)
                output_message = gr.Textbox(label="Status")
    
    with gr.Tab("Audio Player"):
        with gr.Row():
            refresh_btn = gr.Button("Refresh Audio List")
            audio_files = gr.Dropdown(choices=list_audio_files(), label="Select Audio File")
        
        selected_audio = gr.Audio(label="Selected Audio")
    
    # Events
    generate_btn.click(
        generate_speech, 
        inputs=[text_input, audio_prompt, cfg_scale, temperature, top_p, seed], 
        outputs=[output_audio, output_message]
    )
    
    clear_btn.click(
        lambda: (None, None, ""), 
        inputs=[], 
        outputs=[text_input, output_audio, output_message]
    )
    
    refresh_btn.click(
        refresh_audio_list,
        inputs=[],
        outputs=[audio_files]
    )
    
    audio_files.change(
        select_audio_file,
        inputs=[audio_files],
        outputs=[selected_audio]
    )

# Launch the app with a public URL
app.launch(server_name="0.0.0.0", share=True) 