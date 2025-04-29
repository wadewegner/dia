# Dia Long-Form Audio Generator

This utility allows you to generate long-form audio (beyond the ~25 second limitation) using the Dia model by intelligently chunking text and maintaining speaker consistency across chunks.

## Overview

The standard Dia model has a limitation where it can only generate approximately 25 seconds of audio in a single pass. For longer content, it tends to speed up speech to fit within this limit or cut off the audio. This tool addresses this limitation by:

1. Breaking input text into chunks based on speaker pairs
2. Processing each chunk separately
3. Combining the chunks into a seamless output

## Features

- Maintains speaker voice consistency across chunks
- Prevents duplicate intro music in subsequent chunks
- Adds configurable silence between chunks for natural transitions
- Supports voice cloning via audio prompts
- Optimized for GPU processing

## Installation

Ensure you have the Dia model installed:

```bash
pip install git+https://github.com/nari-labs/dia.git
```

## Usage

Basic usage:

```bash
python dia_long.py --text "[S1] This is a long dialogue. [S2] It contains multiple speakers. [S1] And continues beyond the 25-second limit." --output output.wav
```

Using a text file as input:

```bash
python dia_long.py --text-file dialogue.txt --output output.wav
```

With voice cloning:

```bash
python dia_long.py --text-file dialogue.txt --output output.wav --audio-prompt voice_sample.mp3
```

Full GPU acceleration:

```bash
python dia_long.py --text-file dialogue.txt --output output.wav --device cuda --compute-dtype float16
```

## Command Line Arguments

### Input/Output Arguments
- `--text`: Direct text input (alternative to --text-file)
- `--text-file`: Path to a text file containing the dialogue
- `--output`: Path to save the final audio file (required)

### Model Loading Arguments
- `--repo-id`: Hugging Face repo ID (default: "nari-labs/Dia-1.6B")
- `--device`: Device for inference ("cuda", "mps", "cpu", default: auto-detect)
- `--compute-dtype`: Compute datatype (default: "float16")

### Chunking Parameters
- `--max-words`: Maximum words per chunk (default: 150)
- `--add-silence`: Silence between chunks in seconds (default: 0.1)
- `--tmp-dir`: Directory for temporary files (default: "./tmp_audio_chunks")
- `--keep-tmp`: Keep temporary audio files

### Generation Parameters
- `--audio-prompt`: Audio prompt file for voice cloning
- `--max-tokens`: Maximum tokens per generation
- `--cfg-scale`: Classifier-free guidance scale (default: 3.0)
- `--temperature`: Sampling temperature (default: 1.3)
- `--top-p`: Top-p sampling parameter (default: 0.95)
- `--seed`: Random seed for reproducibility
- `--retry-count`: Number of retries on generation failure (default: 3)
- `--verbose`: Print verbose output

## Example Format for Text Files

```
[S1] Hello, this is speaker one talking about something important.
[S2] And I'm speaker two responding to that point with my perspective.
[S1] That's interesting, let me follow up with a question about what you just said?
[S2] I'd be happy to elaborate on my earlier point with a detailed explanation.
```

## Tips for Best Results

1. **Speaker Alternation**: Always alternate between `[S1]` and `[S2]` tags for best voice consistency
2. **Chunk Size**: Adjust `--max-words` based on your content (lower for more frequent speaker changes)
3. **Voice Cloning**: For best results, use 5-10 seconds of high-quality audio with the same speaker pattern
4. **Seed Value**: Set a seed value for reproducible results
5. **Retry Count**: Increase `--retry-count` if you encounter generation failures

## Troubleshooting

- **Inconsistent Voices**: Try using an audio prompt file with the `--audio-prompt` option
- **Audio Cutoffs**: Decrease the `--max-words` value to create smaller chunks
- **Generation Errors**: Increase the `--retry-count` value
- **Memory Issues**: If you encounter GPU out-of-memory errors, try reducing `--max-tokens`

## Credits

This tool is based on the excellent [Dia model by nari-labs](https://github.com/nari-labs/dia) and inspired by community solutions in [GitHub issue #35](https://github.com/nari-labs/dia/issues/35). 