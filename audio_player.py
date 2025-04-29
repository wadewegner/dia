import os
import gradio as gr
import time
import re
import random
import numpy as np
import torch
import soundfile as sf
from pathlib import Path
from typing import List, Optional, Tuple
from dia.model import Dia

# Create output directory if it doesn't exist
os.makedirs("audio_outputs", exist_ok=True)
os.makedirs("tmp_audio_chunks", exist_ok=True)

# Initialize model
print("Loading Dia model...")
# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available! Using GPU.")
    device = "cuda"
else:
    print("CUDA is not available. Using CPU instead.")
    device = "cpu"
    
model = Dia.from_pretrained("nari-labs/Dia-1.6B", device=device, compute_dtype="float16")
print("Model loaded successfully!")

def set_seed(seed: int):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def chunk_by_speakers(text: str, max_words_per_chunk: int = 150) -> List[str]:
    """
    Chunk text by speaker pairs [S1][S2] to maintain voice consistency.
    
    Args:
        text: Input text with [S1] and [S2] tags
        max_words_per_chunk: Maximum words per chunk
        
    Returns:
        List of text chunks
    """
    # First, ensure the text starts with [S1] or [S2]
    if not re.match(r'^\s*\[(S1|S2)\]', text):
        text = f"[S1] {text}"
    
    # Split text into lines
    lines = text.split('\n')
    
    chunks = []
    current_chunk = []
    current_word_count = 0
    speaker_pairs = []
    current_pair = []
    intro_included = False
    
    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:
            current_chunk.append(line)
            continue
            
        # Check for speaker tags
        speaker_match = re.match(r'^\s*\[(S[12])\](.*)', stripped_line)
        
        if speaker_match:
            speaker = speaker_match.group(1)
            content = speaker_match.group(2).strip()
            
            # Add to speaker pair tracking
            current_pair.append(speaker)
            if len(current_pair) == 2 and current_pair[0] != current_pair[1]:
                # We have a full pair of different speakers
                speaker_pairs.append(tuple(current_pair))
                current_pair = []
            elif len(current_pair) == 2:
                # Same speaker twice, keep only the latest
                current_pair = [current_pair[1]]
            
            # Count words in this line
            word_count = len(re.findall(r'\S+', content))
            
            # Check if adding this line would exceed the word limit
            if current_word_count + word_count > max_words_per_chunk and current_chunk:
                if current_chunk:
                    chunks.append('\n'.join(current_chunk))
                current_chunk = []
                current_word_count = 0
                
                # For new chunks, ensure they don't get intro music
                if not intro_included:
                    intro_included = True
                else:
                    # Add an inaudible command to avoid intro music in subsequent chunks
                    # This is a special marker that will be processed later
                    current_chunk.append("__NO_INTRO_MUSIC__")
            
            # Add line to current chunk
            current_chunk.append(line)
            current_word_count += word_count
        else:
            # Non-speaker line, just add it
            current_chunk.append(line)
    
    # Add the last chunk if not empty
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    
    return chunks

def adjust_audio_speed(audio: np.ndarray, speed_factor: float, verbose: bool = False) -> np.ndarray:
    """
    Adjust the speed of audio by resampling.
    
    Args:
        audio: Input audio array
        speed_factor: Speed factor (lower is slower, e.g., 0.9 = 10% slower)
        verbose: Print verbose output
        
    Returns:
        Speed-adjusted audio array
    """
    # Ensure speed_factor is positive and not excessively small/large to avoid issues
    speed_factor = max(0.1, min(speed_factor, 5.0))
    original_len = len(audio)
    # Target length based on speed_factor - lower value = longer audio = slower speech
    target_len = int(original_len / speed_factor)
    
    # Only interpolate if length changes and is valid
    if target_len != original_len and target_len > 0:
        x_original = np.arange(original_len)
        x_resampled = np.linspace(0, original_len - 1, target_len)
        adjusted_audio = np.interp(x_resampled, x_original, audio).astype(np.float32)
        
        if verbose:
            print(f"Applied speed factor {speed_factor}: audio length changed from {original_len} to {target_len} samples ({100*(1-speed_factor):.1f}% slower)")
        
        return adjusted_audio
    
    return audio

def combine_audio_files(files: List[str], output_file: str, sample_rate: int = 44100, add_silence: float = 0.1):
    """
    Combine multiple audio files into one, adding silence between them.
    
    Args:
        files: List of audio file paths
        output_file: Path to save the combined audio
        sample_rate: Sample rate of the audio files
        add_silence: Seconds of silence to add between chunks
    """
    combined_audio = np.array([])
    silence = np.zeros(int(add_silence * sample_rate))
    
    for file in files:
        audio, sr = sf.read(file)
        assert sr == sample_rate, f"Sample rate mismatch: {sr} != {sample_rate}"
        
        # Add silence between chunks
        if len(combined_audio) > 0:
            combined_audio = np.concatenate([combined_audio, silence, audio])
        else:
            combined_audio = audio
    
    # Save combined audio
    sf.write(output_file, combined_audio, sample_rate)
    return output_file

def process_chunk(text: str, audio_prompt=None, cfg_scale=3.0, temperature=1.3, top_p=0.95, 
                 max_tokens=None, speed_factor=0.9, cfg_filter_top_k=50, verbose=True) -> np.ndarray:
    """Process a single text chunk and generate audio."""
    # Remove the special no intro marker if present
    skip_intro = "__NO_INTRO_MUSIC__" in text
    text = text.replace("__NO_INTRO_MUSIC__", "")
    
    # Ensure text has proper speaker tags
    if not text.strip().startswith("[S1]") and not text.strip().startswith("[S2]"):
        text = f"[S1] {text}"
    
    # Generate audio
    if verbose:
        print(f"Generating chunk: {text[:50]}...")
    
    output_audio = model.generate(
        text=text,
        audio_prompt=audio_prompt,
        max_tokens=max_tokens,
        cfg_scale=cfg_scale,
        temperature=temperature,
        top_p=top_p,
        cfg_filter_top_k=cfg_filter_top_k,
        use_torch_compile=torch.cuda.is_available(),
        verbose=verbose
    )
    
    # Apply speed adjustment if needed
    if speed_factor != 1.0:
        output_audio = adjust_audio_speed(output_audio, speed_factor, verbose)
    
    return output_audio

def generate_speech(text, audio_prompt=None, cfg_scale=3.0, temperature=1.3, top_p=0.95, seed=None, 
                   longform=True, max_words=80, speed_factor=0.9, cfg_filter_top_k=50, add_silence=0.1):
    """Generate speech and save it to a file"""
    if not text:
        return None, "Please enter text to generate speech."
    
    try:
        # Generate timestamp for unique filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"audio_outputs/speech_{timestamp}.wav"
        
        # Set seed if provided
        if seed is not None and seed != "":
            try:
                seed = int(seed)
                set_seed(seed)
                print(f"Using seed: {seed}")
            except ValueError:
                return None, "Seed must be an integer."
        
        # For shorter text or if longform is disabled, use the standard Dia generation
        if not longform or len(text.split()) < 50:
            print(f"Generating short-form speech for: {text[:50]}...")
            
            # Format the text if it doesn't have speaker tags
            if not text.strip().startswith("[S1]") and not text.strip().startswith("[S2]"):
                text = f"[S1] {text}"
            
            audio = model.generate(
                text=text,
                audio_prompt=audio_prompt,
                cfg_scale=cfg_scale,
                temperature=temperature,
                top_p=top_p,
                cfg_filter_top_k=cfg_filter_top_k,
                use_torch_compile=torch.cuda.is_available(),
                verbose=True
            )
            
            # Apply speed adjustment if needed
            if speed_factor != 1.0:
                audio = adjust_audio_speed(audio, speed_factor, True)
            
            # Save audio
            sf.write(filename, audio, 44100)
            return filename, f"Short-form speech generated and saved to {filename}"
        
        # For longer text, use chunking approach
        print(f"Generating long-form speech using chunking...")
        
        # Chunk the text
        chunks = chunk_by_speakers(text, max_words_per_chunk=max_words)
        print(f"Split text into {len(chunks)} chunks")
        
        # Process each chunk
        chunk_files = []
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}...")
            
            # Create temporary file for this chunk
            chunk_path = f"tmp_audio_chunks/chunk_{timestamp}_{i:03d}.wav"
            
            # Process chunk
            try:
                output_audio = process_chunk(
                    text=chunk,
                    audio_prompt=audio_prompt,
                    cfg_scale=cfg_scale,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=None,
                    speed_factor=speed_factor,
                    cfg_filter_top_k=cfg_filter_top_k,
                    verbose=True
                )
                
                # Save audio to file
                sf.write(chunk_path, output_audio, 44100)
                chunk_files.append(chunk_path)
                print(f"Successfully generated chunk {i+1}")
            except Exception as e:
                print(f"Error processing chunk {i+1}: {str(e)}")
                # Continue with other chunks even if one fails
        
        if not chunk_files:
            return None, "Failed to generate any audio chunks."
        
        # Combine chunks
        print(f"Combining {len(chunk_files)} audio chunks...")
        combine_audio_files(
            files=chunk_files,
            output_file=filename,
            add_silence=add_silence
        )
        
        # Clean up temporary files
        for file in chunk_files:
            try:
                os.remove(file)
            except:
                pass
        
        print(f"Long-form audio saved to {filename}")
        return filename, f"Long-form speech ({len(chunks)} chunks) generated and saved to {filename}"
    
    except Exception as e:
        import traceback
        traceback.print_exc()
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
                    lines=10  # Increased for longer text
                )
                with gr.Row():
                    generate_btn = gr.Button("Generate Speech", variant="primary")
                    clear_btn = gr.Button("Clear")
                
                with gr.Accordion("Advanced Options", open=False):
                    with gr.Row():
                        longform = gr.Checkbox(label="Enable Long-form Generation", value=True, 
                                             info="Split long text into chunks for better audio quality")
                    
                    with gr.Row():
                        cfg_scale = gr.Slider(minimum=1.0, maximum=10.0, value=3.5, step=0.1, 
                                            label="CFG Scale", info="Higher values: more adherence to text")
                        temperature = gr.Slider(minimum=0.1, maximum=2.0, value=1.3, step=0.1, 
                                              label="Temperature", info="Higher values: more variation")
                    
                    with gr.Row():
                        top_p = gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, 
                                        label="Top P", info="Nucleus sampling probability")
                        cfg_filter_top_k = gr.Slider(minimum=15, maximum=50, value=50, step=1, 
                                                   label="CFG Filter Top K", info="Higher values: better script adherence")
                    
                    with gr.Row():
                        speed_factor = gr.Slider(minimum=0.7, maximum=1.2, value=0.9, step=0.05, 
                                               label="Speed Factor", info="Lower values: slower speech (0.9 = 10% slower)")
                        max_words = gr.Slider(minimum=50, maximum=200, value=80, step=10, 
                                            label="Max Words Per Chunk", info="For long-form generation")
                    
                    with gr.Row():
                        add_silence = gr.Slider(minimum=0.0, maximum=1.0, value=0.1, step=0.05, 
                                              label="Silence Between Chunks (seconds)", info="For long-form generation")
                        seed = gr.Textbox(label="Seed (optional)", placeholder="Leave empty for random", 
                                         info="Integer for reproducible results")
                    
                    audio_prompt = gr.Audio(label="Audio Prompt (optional, for voice cloning)", type="filepath")
                
            with gr.Column(scale=2):
                output_audio = gr.Audio(label="Generated Speech", interactive=False, type="filepath", autoplay=True)
                output_message = gr.Textbox(label="Status")
    
    with gr.Tab("Audio Player"):
        with gr.Row():
            refresh_btn = gr.Button("Refresh Audio List")
            audio_files = gr.Dropdown(choices=list_audio_files(), label="Select Audio File")
        
        selected_audio = gr.Audio(label="Selected Audio", type="filepath", autoplay=True)
        gr.Markdown("#### Troubleshooting")
        gr.Markdown("- If you can't hear audio, try clicking the download button above the player")
        gr.Markdown("- You can also try refreshing your browser or using a different browser")
        gr.Markdown("- Make sure your volume is turned up")
    
    # Events
    generate_btn.click(
        generate_speech, 
        inputs=[
            text_input, audio_prompt, cfg_scale, temperature, top_p, seed, 
            longform, max_words, speed_factor, cfg_filter_top_k, add_silence
        ], 
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

# Display a message about the server
print("\nStarting Gradio server...")
print("The interface will be available at:")
print("  - Local URL: http://127.0.0.1:7860")
print("  - Public URL will be displayed below (if enabled)")
print("\nIf you're running this on a remote server, make sure to use SSH port forwarding:")
print("  ssh -L 7860:localhost:7860 your-server-username@your-server-ip\n")

# Launch the app with a public URL
app.launch(server_name="0.0.0.0", share=True) 