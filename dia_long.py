#!/usr/bin/env python3
import argparse
import os
import re
import time
import random
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio

from dia.model import Dia


def set_seed(seed: int):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for cuDNN (if used)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def detect_device():
    """Detect the best available device for inference"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


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


def process_chunk(
    model: Dia,
    text: str,
    output_path: str,
    audio_prompt: Optional[str] = None,
    cfg_scale: float = 3.0,
    temperature: float = 1.3,
    top_p: float = 0.95,
    max_tokens: Optional[int] = None,
    speed_factor: float = 0.94,  # Default to slightly slower for more natural speech
    cfg_filter_top_k: int = 30,  # Add cfg_filter_top_k parameter with default from app.py
    retry_count: int = 3,
    verbose: bool = False
) -> str:
    """
    Process a single text chunk and generate audio.
    
    Args:
        model: Dia model instance
        text: Text chunk to process
        output_path: Path to save the output audio
        audio_prompt: Path to audio prompt file (for voice cloning)
        cfg_scale: Classifier-free guidance scale
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        max_tokens: Maximum tokens per generation
        speed_factor: Speed factor (lower is slower)
        cfg_filter_top_k: Top k filter for CFG guidance
        retry_count: Number of retries on failure
        verbose: Print verbose output
        
    Returns:
        Path to the generated audio file
    """
    # Remove the special no intro marker if present
    skip_intro = "__NO_INTRO_MUSIC__" in text
    text = text.replace("__NO_INTRO_MUSIC__", "")
    
    # Ensure text has proper speaker tags (following app.py example)
    if not text.strip().startswith("[S1]") and not text.strip().startswith("[S2]"):
        text = f"[S1] {text}"
    
    for attempt in range(retry_count):
        try:
            # Generate audio
            output_audio = model.generate(
                text=text,
                audio_prompt=audio_prompt,
                max_tokens=max_tokens,
                cfg_scale=cfg_scale,
                temperature=temperature,
                top_p=top_p,
                cfg_filter_top_k=cfg_filter_top_k,  # Pass cfg_filter_top_k to model
                verbose=verbose
            )
            
            # Apply speed adjustment if needed
            if speed_factor != 1.0:
                output_audio = adjust_audio_speed(output_audio, speed_factor, verbose)
            
            # Save audio to file
            sf.write(output_path, output_audio, 44100)
            if verbose:
                print(f"Successfully generated audio: {output_path}")
            return output_path
        except Exception as e:
            if "Clamping" in str(e) or "out of bounds" in str(e):
                if verbose:
                    print(f"Retry {attempt+1}/{retry_count} due to clamping error: {e}")
                continue
            else:
                print(f"Error generating audio for chunk: {e}")
                raise
    
    raise RuntimeError(f"Failed to generate audio after {retry_count} attempts")


def main():
    parser = argparse.ArgumentParser(description="Generate long-form audio using the Dia model with chunking")
    
    # Input/output arguments
    parser.add_argument("--text", type=str, help="Input text for speech generation")
    parser.add_argument("--text-file", type=str, help="Path to input text file (alternative to --text)")
    parser.add_argument("--output", type=str, required=True, help="Path to save the final audio file")
    
    # Model loading arguments
    parser.add_argument("--repo-id", type=str, default="nari-labs/Dia-1.6B", help="Hugging Face repo ID")
    parser.add_argument("--device", type=str, default=None, help="Device for inference (cuda, mps, cpu)")
    parser.add_argument("--compute-dtype", type=str, default="float16", help="Compute datatype")
    
    # Chunking parameters
    parser.add_argument("--max-words", type=int, default=150, help="Maximum words per chunk")
    parser.add_argument("--add-silence", type=float, default=0.1, help="Silence between chunks (seconds)")
    parser.add_argument("--tmp-dir", type=str, default="./tmp_audio_chunks", help="Directory for temporary files")
    parser.add_argument("--keep-tmp", action="store_true", help="Keep temporary audio files")
    
    # Generation parameters
    parser.add_argument("--audio-prompt", type=str, default=None, help="Audio prompt for voice cloning")
    parser.add_argument("--max-tokens", type=int, default=None, help="Maximum tokens per generation")
    parser.add_argument("--cfg-scale", type=float, default=3.0, help="Classifier-free guidance scale")
    parser.add_argument("--temperature", type=float, default=1.3, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling parameter")
    parser.add_argument("--speed-factor", type=float, default=0.94, 
                       help="Speech speed factor (lower is slower, e.g., 0.9 for 10%% slower)")
    parser.add_argument("--cfg-filter-top-k", type=int, default=30, 
                       help="Top k filter for CFG guidance (controls generation quality)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--retry-count", type=int, default=3, help="Number of retries on generation failure")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    
    args = parser.parse_args()
    
    # Validate input
    if not args.text and not args.text_file:
        parser.error("Either --text or --text-file must be provided")
    
    if args.text and args.text_file:
        parser.error("Only one of --text or --text-file should be provided")
    
    # Set seed if provided
    if args.seed is not None:
        set_seed(args.seed)
        if args.verbose:
            print(f"Using random seed: {args.seed}")
    
    # Determine device
    device = detect_device() if args.device is None else torch.device(args.device)
    if args.verbose:
        print(f"Using device: {device}")
    
    # Create temporary directory
    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(exist_ok=True, parents=True)
    
    # Get input text
    if args.text_file:
        with open(args.text_file, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        text = args.text
    
    # Chunk the text
    if args.verbose:
        print(f"Chunking input text...")
    
    chunks = chunk_by_speakers(text, max_words_per_chunk=args.max_words)
    if args.verbose:
        print(f"Split text into {len(chunks)} chunks")
    
    # Load model
    if args.verbose:
        print(f"Loading Dia model from {args.repo_id}...")
    
    start_time = time.time()
    model = Dia.from_pretrained(
        args.repo_id,
        compute_dtype=args.compute_dtype,
        device=device
    )
    
    if args.verbose:
        print(f"Model loaded in {time.time() - start_time:.2f} seconds")
    
    # Process chunks
    chunk_files = []
    for i, chunk in enumerate(chunks):
        if args.verbose:
            print(f"Processing chunk {i+1}/{len(chunks)}...")
        
        # Create unique output path for this chunk
        chunk_path = tmp_dir / f"chunk_{i:03d}.wav"
        
        # Generate audio for this chunk
        process_chunk(
            model=model,
            text=chunk,
            output_path=str(chunk_path),
            audio_prompt=args.audio_prompt,
            cfg_scale=args.cfg_scale,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            speed_factor=args.speed_factor,
            cfg_filter_top_k=args.cfg_filter_top_k,
            retry_count=args.retry_count,
            verbose=args.verbose
        )
        
        chunk_files.append(str(chunk_path))
    
    # Combine audio files
    if args.verbose:
        print(f"Combining {len(chunk_files)} audio files...")
    
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    combine_audio_files(
        files=chunk_files,
        output_file=str(output_path),
        add_silence=args.add_silence
    )
    
    if args.verbose:
        print(f"Audio successfully saved to {args.output}")
    
    # Clean up temporary files if not keeping them
    if not args.keep_tmp:
        if args.verbose:
            print("Cleaning up temporary files...")
        
        for file in chunk_files:
            Path(file).unlink(missing_ok=True)


if __name__ == "__main__":
    main() 