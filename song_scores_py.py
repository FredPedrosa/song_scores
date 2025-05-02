

#@title Setup (Run this cell first)
# Install necessary packages
!pip install -q transformers torch librosa soundfile sentencepiece accelerate ffmpeg-python datasets torchinfo yt-dlp ipywidgets

import torch
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from transformers import AutoProcessor, AutoModel, pipeline
from IPython.display import display, Audio, clear_output, Markdown # For Colab display
import gc # Garbage collector
import yt_dlp
import os
import re # For basic URL validation
import gc # Garbage collector

# Suppress specific warnings
warnings.filterwarnings('ignore', message='.*Maximum duration.*')
warnings.filterwarnings('ignore', message='.*Using PipelineChunkIterator.*')

#@title Device Setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(f"GPU detected. Using device: {torch.cuda.get_device_name(0)}")
    # Display GPU memory info (useful for debugging)
    !nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader,nounits
else:
    device = torch.device("cpu")
    print("GPU not detected. Using CPU (processing will be slower).")

#@title Functions

# --- Helper function to load audio safely ---
def load_audio_segment(audio_path, target_sr, offset=None, duration=None):
    """Loads an audio segment with error handling."""
    try:
        audio_array, sr = librosa.load(audio_path, sr=target_sr, mono=True, offset=offset, duration=duration)
        if audio_array.size == 0:
            print(f"Warning: Audio segment resulted in empty array (offset={offset}, duration={duration}).")
            return None, None
        return audio_array, sr
    except Exception as e:
        print(f"Error loading audio {audio_path}: {e}")
        return None, None

# --- Main Analysis Function ---
def song_scores_py(
    audio_path: str,
    audio_classes: list,
    text_classes: list = None,
    lyrics: str = None,
    transcribe_audio: bool = False,
    start_sec: float = None,
    end_sec: float = None,
    clap_model_id: str = "laion/clap-htsat-unfused",
    # --- ALTERAÇÃO AQUI ---
    nli_model_id: str = "joeddav/xlm-roberta-large-xnli", # Default changed!
    # ----------------------
    asr_model_id: str = "openai/whisper-base",
    asr_language: str = None,
    verbose: bool = True
    ):
    """
    Performs multimodal emotion analysis on a song (audio + lyrics).

    Args:
        # ... (outros args) ...
        nli_model_id: Hugging Face ID for the NLI/zero-shot text model.
                      Defaults to 'joeddav/xlm-roberta-large-xnli' (multilingual). # Updated docstring
        # ... (resto dos args e docstring) ...
    """
    # --- ATUALIZAR O VALOR PADRÃO NO results TAMBÉM ---
    results = {
        "audio_scores": None,
        "text_scores": None,
        "transcribed_text": None,
        "text_source": "none",
        "audio_segment_duration_s": None,
        "clap_model": clap_model_id,
        # Use o valor do parâmetro aqui (que agora tem o novo default)
        "nli_model": nli_model_id if text_classes else None,
        "asr_model": asr_model_id if (transcribe_audio and lyrics is None) else None # Correção aqui também
    }

    # --- Calculate offset and duration ---
    offset = start_sec if start_sec is not None else None
    duration = (end_sec - start_sec) if start_sec is not None and end_sec is not None else None
    if start_sec is not None and end_sec is not None and end_sec <= start_sec:
         print("Error: 'end_sec' must be greater than 'start_sec'.")
         return results
    results["audio_segment_duration_s"] = duration

    segment_description = f"segment from {start_sec:.2f}s to {end_sec:.2f}s (duration: {duration:.2f}s)" if duration is not None else "the entire audio file"
    if verbose: print(f"\n--- Analysis requested for {segment_description} ---")

    # --- 1. Audio Analysis (CLAP) ---
    clap_processor = None
    clap_model = None
    try:
        if verbose: print(f"\nInitiating AUDIO analysis with CLAP: {clap_model_id}...")
        if verbose: print("Loading CLAP model and processor...")
        clap_processor = AutoProcessor.from_pretrained(clap_model_id)
        clap_model = AutoModel.from_pretrained(clap_model_id).to(device)
        clap_model.eval() # Set model to evaluation mode

        target_sr_clap = clap_processor.feature_extractor.sampling_rate if hasattr(clap_processor.feature_extractor, 'sampling_rate') else 48000
        if verbose: print(f"Loading audio for CLAP (resampling to {target_sr_clap} Hz)...")
        audio_array_clap, _ = load_audio_segment(audio_path, target_sr_clap, offset, duration)

        if audio_array_clap is not None:
            if verbose: print("Processing audio and text labels for CLAP...")
            # Ensure audio_array_clap is a numpy array before passing
            inputs = clap_processor(
                text=audio_classes,
                audios=[np.array(audio_array_clap)], # Pass as list containing numpy array
                return_tensors="pt",
                padding=True,
                sampling_rate=target_sr_clap
            )
            inputs = {k: v.to(device) for k, v in inputs.items()} # Move inputs to device

            if verbose: print("Running CLAP inference...")
            with torch.no_grad():
                outputs = clap_model(**inputs)
                # Shape is [audio_batch_size, text_batch_size]
                logits_per_audio = outputs.logits_per_audio[0] # Get scores for the single audio input
                probs = torch.softmax(logits_per_audio, dim=0).cpu().numpy()

            results["audio_scores"] = dict(zip(audio_classes, probs))
            if verbose: print("Audio analysis completed.")
            # Clean up GPU memory
            del inputs, outputs, logits_per_audio, probs
        else:
             results["audio_scores"] = {c: np.nan for c in audio_classes}
             print("Audio analysis skipped due to loading error or empty segment.")

    except Exception as e:
        print(f"Error during CLAP analysis: {e}")
        results["audio_scores"] = {c: np.nan for c in audio_classes} # Indicate failure
    finally:
        # Explicitly delete model and processor, then clear cache
        del clap_processor, clap_model, audio_array_clap
        gc.collect()
        if device.type == 'cuda': torch.cuda.empty_cache()
        if verbose: print("CLAP resources released.")

    # --- 2. Text Preparation / Transcription (ASR - Whisper) ---
    text_to_analyze = None
    perform_text_analysis = text_classes is not None
    asr_pipeline = None # Initialize outside try block

    if perform_text_analysis:
        # CORRECTED BLOCK START
        if lyrics is not None:
            # --- Lines below are now correctly indented ---
            if verbose: print("\nUsing provided lyrics ('lyrics').")
            # Handle if lyrics is a list or a single string
            if isinstance(lyrics, list):
                text_to_analyze = "\n".join(lyrics) # Join list elements with newline
            elif isinstance(lyrics, str):
                text_to_analyze = lyrics # Use the string directly
            else:
                # Handle unexpected type if necessary, or assume string/list
                print("Warning: 'lyrics' provided is neither a string nor a list. Attempting to use as string.")
                text_to_analyze = str(lyrics)

            results["text_source"] = "provided_lyrics"
            # --- End of indented block for 'if lyrics is not None:' ---

        # This elif is now correctly positioned after the 'if' block
        elif transcribe_audio:
            try:
                if verbose: print(f"\nInitiating audio TRANSCRIPTION with ASR (Whisper): {asr_model_id}... (This may take time!)")
                if verbose: print("Loading ASR pipeline...")
                # Load pipeline
                asr_pipeline = pipeline(
                    "automatic-speech-recognition",
                    model=asr_model_id,
                    device=device
                )

                target_sr_asr = asr_pipeline.feature_extractor.sampling_rate
                if verbose: print(f"Loading audio for ASR (resampling to {target_sr_asr} Hz)...")
                audio_array_asr, _ = load_audio_segment(audio_path, target_sr_asr, offset, duration)

                if audio_array_asr is not None:
                    # Pass return_timestamps=True for long-form generation
                    generate_kwargs = {"return_timestamps": True}
                    if asr_language:
                        generate_kwargs['language'] = asr_language

                    if verbose: print("Running ASR inference (with timestamps enabled for long audio)...")
                    with torch.no_grad():
                        # Pass numpy array directly
                        transcription_result = asr_pipeline(np.array(audio_array_asr), generate_kwargs=generate_kwargs)

                    # Extract text from result
                    results["transcribed_text"] = transcription_result["text"].strip() if transcription_result and "text" in transcription_result else ""

                    if results["transcribed_text"]:
                        if verbose: print("Transcription completed.")
                        if verbose: print(f"Transcribed text: {results['transcribed_text']}...")
                        text_to_analyze = results["transcribed_text"]
                        results["text_source"] = "transcribed"
                    else:
                        if verbose: print("Transcription completed but resulted in empty text.")
                        results["text_source"] = "transcribed_empty"
                    # Clean up intermediate ASR variables early if possible
                    del transcription_result, audio_array_asr
                    gc.collect() # Optional: trigger garbage collection

                else:
                    results["text_source"] = "transcribed_failed"
                    print("Transcription skipped due to audio loading error or empty segment.")

            except Exception as e:
                print(f"Error during ASR (Whisper) transcription: {e}")
                results["transcribed_text"] = None
                results["text_source"] = "transcribed_failed"
            finally:
                # Clean up ASR pipeline and clear cache (ensure it's defined)
                if 'asr_pipeline' in locals() and asr_pipeline is not None:
                     del asr_pipeline
                     gc.collect()
                     if device.type == 'cuda': torch.cuda.empty_cache()
                     if verbose: print("ASR resources released.")

        else:
            # This 'else' corresponds to 'if lyrics is not None:' and 'elif transcribe_audio:'
            if verbose: print("\nNo lyrics provided ('lyrics'=NULL) and transcription not requested ('transcribe_audio'=FALSE). Skipping text analysis.")
            results["text_source"] = "none"
    else:
        # This 'else' corresponds to 'if perform_text_analysis:'
        if verbose: print("\nText analysis not requested ('text_classes'=NULL).")
        results["text_source"] = "none"


    # --- 3. Text Analysis (NLI / Zero-Shot) ---
    nli_pipeline = None
    if text_to_analyze and perform_text_analysis:
        try:
            if verbose: print(f"\nInitiating TEXT analysis with Zero-Shot model: {nli_model_id}...")
            if verbose: print("Loading Zero-Shot pipeline...")
            # Use the zero-shot classification pipeline
            nli_pipeline = pipeline("zero-shot-classification", model=nli_model_id, device=device)

            if verbose: print("Running Zero-Shot inference...")
            # The pipeline handles tokenization etc.
            with torch.no_grad():
                # Pass the single text block and candidate labels
                nli_output = nli_pipeline(text_to_analyze, candidate_labels=text_classes)

            # Extract scores in the correct order
            results["text_scores"] = dict(zip(nli_output['labels'], nli_output['scores']))
            if verbose: print("Text analysis completed.")
            del nli_output # Clean up

        except Exception as e:
            print(f"Error during Zero-Shot text analysis: {e}")
            results["text_scores"] = None # Indicate failure
        finally:
            del nli_pipeline
            gc.collect()
            if device.type == 'cuda': torch.cuda.empty_cache()
            if verbose: print("Zero-Shot resources released.")

    elif perform_text_analysis and results["text_source"] in ["none", "transcribed_empty", "transcribed_failed"]:
        if verbose: print("\nText analysis skipped due to lack of textual content.")


    if verbose: print("\n--- Processing completed ---")
    return results

# --- Plotting Function ---
def plot_scores_py(results: dict):
    """
    Generates a bar plot comparing audio and text emotion scores.

    Args:
        results: A dictionary returned by song_scores_py.
    """
    audio_scores = results.get("audio_scores")
    text_scores = results.get("text_scores")

    plot_data = []
    all_classes = set()

    if isinstance(audio_scores, dict):
        for cls, score in audio_scores.items():
            if not np.isnan(score): # Exclude potential NaNs from failed analysis
                 plot_data.append({"class": cls, "score": score, "modality": "Audio"})
                 all_classes.add(cls)

    if isinstance(text_scores, dict):
        for cls, score in text_scores.items():
             if not np.isnan(score): # Exclude potential NaNs from failed analysis
                 plot_data.append({"class": cls, "score": score, "modality": "Text"})
                 all_classes.add(cls)

    if not plot_data:
        print("No valid score data found to plot.")
        return None

    df = pd.DataFrame(plot_data)
    # Ensure consistent class ordering
    ordered_classes = sorted(list(all_classes))
    df['class'] = pd.Categorical(df['class'], categories=ordered_classes, ordered=True)
    df = df.sort_values('class') # Sort dataframe by class for consistent plotting

    fig, ax = plt.subplots(figsize=(10, 6)) # Adjust figure size as needed

    # Create the bar plot using Pandas plotting directly or Matplotlib
    # Using pivot for easier plotting with Pandas/Matplotlib
    try:
        pivot_df = df.pivot(index='class', columns='modality', values='score')
        pivot_df.plot(kind='bar', ax=ax, color={"Audio": "#4c9a9e", "Text": "#b3bef2"}, edgecolor='black')
    except Exception as plot_err:
        # Fallback if pivot fails (e.g., only one modality has data)
        print(f"Plotting warning: Could not pivot data ({plot_err}). Using direct bar plot.")
        bar_width = 0.35
        x = np.arange(len(ordered_classes))
        offset = 0
        colors = {"Audio": "#4c9a9e", "Text": "#b3bef2"}
        for modality in df['modality'].unique():
            modality_data = df[df['modality'] == modality]
            # Create mapping from class to index
            class_to_idx = {cls: i for i, cls in enumerate(ordered_classes)}
            modality_indices = [class_to_idx[cls] for cls in modality_data['class']]
            ax.bar(np.array(modality_indices) + offset, modality_data['score'], bar_width, label=modality, color=colors[modality], edgecolor='black')
            offset += bar_width if len(df['modality'].unique()) > 1 else 0 # Adjust offset only if dodging

        ax.set_xticks(x + bar_width / (2 if len(df['modality'].unique()) > 1 else 1))
        ax.set_xticklabels(ordered_classes)


    ax.set_title("Emotion Scores by Modality")
    ax.set_xlabel("Emotion Class")
    ax.set_ylabel("Score / Probability")
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title="Modality")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y))) # Format y-axis as percentage
    ax.set_ylim(0, 1) # Ensure y-axis goes from 0 to 1 (or 100%)
    plt.tight_layout() # Adjust layout
    plt.show()

    return fig # Return the figure object if needed


 # --- YouTube Download Helper Function ---
def download_youtube_audio(url, output_dir='/content/youtube_audio', preferred_codec='mp3'):
    """
    Downloads the best audio from a YouTube URL using yt-dlp.

    Args:
        url (str): The YouTube video URL.
        output_dir (str): Directory to save the downloaded audio.
        preferred_codec (str): Preferred audio codec (e.g., 'mp3', 'wav', 'm4a').

    Returns:
        str: The full path to the downloaded audio file, or None if download failed.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Basic URL validation
    if not re.match(r'(https?://)?(www\.)?(youtube\.com|youtu\.?be)/.+', url):
        print(f"Error: Invalid YouTube URL provided: {url}")
        return None

    output_template = os.path.join(output_dir, '%(id)s.%(ext)s')

    ydl_opts = {
        'format': 'bestaudio/best', # Select best audio quality
        'outtmpl': output_template, # Output path template (using video ID)
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': preferred_codec, # Convert to specified codec
            #'preferredquality': '192', # Optional: specify quality
        }],
        'quiet': False, # Show yt-dlp output
        'no_warnings': True,
        'noprogress': False,
    }

    downloaded_file_path = None
    try:
        print(f"Attempting to download audio from: {url}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=False) # Get info first
            video_id = info_dict.get('id', 'downloaded_audio')
            expected_filename = f"{video_id}.{preferred_codec}"
            expected_filepath = os.path.join(output_dir, expected_filename)

            # Download the audio
            ydl.download([url])

            # Verify download
            if os.path.exists(expected_filepath):
                downloaded_file_path = expected_filepath
                print(f"Successfully downloaded and converted audio to: {downloaded_file_path}")
            else:
                # Sometimes yt-dlp might save with a different extension initially
                # Search for the file based on ID in the output directory
                found = False
                for f in os.listdir(output_dir):
                     if f.startswith(video_id) and f.endswith(f".{preferred_codec}"):
                          downloaded_file_path = os.path.join(output_dir, f)
                          print(f"Successfully downloaded and converted audio to: {downloaded_file_path}")
                          found = True
                          break
                if not found:
                     print(f"Error: Download finished, but expected file not found: {expected_filepath}")


    except yt_dlp.utils.DownloadError as e:
        print(f"Error downloading video: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during download: {e}")

    return downloaded_file_path

#@title Interactive YouTube Analysis Interface (with Custom Classes & ASR Model)
import ipywidgets as widgets

# --- Constants ---
UPLOAD_DIR = '/content/uploaded_audio' # Directory to save uploaded files

# --- Widgets ---

# --- Source Selection ---
source_choice = widgets.RadioButtons(
    options=[
        ('YouTube URL', 'youtube'),
        ('Upload Audio File', 'upload')
    ],
    value='youtube', # Default source
    description='Audio Source:',
    disabled=False,
    layout={'width': 'max-content'}
)

# --- YouTube URL Input (Visibility controlled) ---
url_input = widgets.Text(
    value='',
    placeholder='Enter YouTube URL here (if source is YouTube)',
    description='YouTube URL:',
    disabled=False, # Enabled by default, controlled by handler
    layout=widgets.Layout(width='80%', visibility='visible') # Visible by default
)

# --- File Upload Widget (Visibility controlled) ---
file_upload_widget = widgets.FileUpload(
    accept='.mp3,.wav,.flac,.m4a,.ogg',  # Specify acceptable audio formats
    multiple=False, # Allow only one file upload
    description="Upload Audio:",
    disabled=True, # Disabled by default
    layout=widgets.Layout(width='80%', visibility='hidden') # Hidden by default
)

# --- Other Widgets (Time, Classes, Text Method, ASR Model) ---
start_input = widgets.FloatText( value=0, description='Start (sec):', disabled=False, step=1.0)
end_input = widgets.FloatText( value=30, description='End (sec):', disabled=False, step=1.0)
analyze_whole_checkbox = widgets.Checkbox( value=False, description='Analyze Full Song?', disabled=False)
classes_input = widgets.Textarea(
    value='joy, sadness, anger, calmness, nostalgia, neutral',
    placeholder='Enter emotion classes, separated by commas or newlines',
    description='Emotion Classes:', disabled=False, layout=widgets.Layout(width='80%', height='80px')
)
text_analysis_choice = widgets.RadioButtons(
    options=[('Transcribe Audio (Whisper)', 'transcribe'), ('Enter Lyrics Manually', 'manual'), ('No Text Analysis', 'none')],
    value='transcribe', description='Text Analysis:', disabled=False, layout={'width': 'max-content'}
)
manual_lyrics_input = widgets.Textarea(
    value='', placeholder='Paste or type song lyrics here...', description='Manual Lyrics:',
    disabled=True, layout=widgets.Layout(width='80%', height='150px', visibility='hidden')
)
asr_model_dropdown = widgets.Dropdown(
    options=[('Whisper Base', 'openai/whisper-base'), ('Whisper Large v3', 'openai/whisper-large-v3')],
    value='openai/whisper-base', description='ASR Model:', disabled=False, layout=widgets.Layout(width='80%', visibility='visible')
)

# --- Function to update UI based on SOURCE Choice ---
def handle_source_choice_change(change):
    choice = change['new']
    if choice == 'youtube':
        url_input.disabled = False
        url_input.layout.visibility = 'visible'
        file_upload_widget.disabled = True
        file_upload_widget.layout.visibility = 'hidden'
    elif choice == 'upload':
        url_input.disabled = True
        url_input.layout.visibility = 'hidden'
        file_upload_widget.disabled = False
        file_upload_widget.layout.visibility = 'visible'

# Link the handler function to the source radio button
source_choice.observe(handle_source_choice_change, names='value')
# Call handler once to set initial UI state
handle_source_choice_change({'new': source_choice.value})


# --- Function to update UI based on TEXT Choice (Same as before) ---
def handle_text_choice_change(change):
    choice = change['new']
    # (This function remains the same as in the previous version, controlling
    #  manual_lyrics_input and asr_model_dropdown based on text_analysis_choice)
    if choice == 'transcribe':
        manual_lyrics_input.disabled = True
        manual_lyrics_input.layout.visibility = 'hidden'
        asr_model_dropdown.disabled = False
        asr_model_dropdown.layout.visibility = 'visible'
    elif choice == 'manual':
        manual_lyrics_input.disabled = False
        manual_lyrics_input.layout.visibility = 'visible'
        asr_model_dropdown.disabled = True
        asr_model_dropdown.layout.visibility = 'hidden'
    elif choice == 'none':
        manual_lyrics_input.disabled = True
        manual_lyrics_input.layout.visibility = 'hidden'
        asr_model_dropdown.disabled = True
        asr_model_dropdown.layout.visibility = 'hidden'

# Link the handler function
text_analysis_choice.observe(handle_text_choice_change, names='value')
# Call handler once to set initial state
handle_text_choice_change({'new': text_analysis_choice.value})


analyze_button = widgets.Button(
    description='Analyze Song', disabled=False, button_style='success',
    tooltip='Download/Use audio and run emotion analysis', icon='music'
)

output_widget = widgets.Output()

# --- Button Click Logic (Handles Both Sources - CORRECTED FILE UPLOAD ACCESS) ---
def on_analyze_button_clicked(b):
    with output_widget:
        clear_output(wait=True)
        display(Markdown("--- Starting Analysis ---"))

        # --- Determine Audio Source and Get Path ---
        selected_source = source_choice.value
        audio_filepath = None

        if selected_source == 'youtube':
            # ... (lógica do youtube - sem alterações) ...
            youtube_url = url_input.value
            if not youtube_url:
                print("Error: Please enter a YouTube URL.")
                return
            display(Markdown(f"**Source:** YouTube URL (`{youtube_url}`)"))
            audio_filepath = download_youtube_audio(youtube_url, preferred_codec='mp3')

        elif selected_source == 'upload':
            uploaded_data = file_upload_widget.value # This is a DICT
            if not uploaded_data:
                print("Error: Please upload an audio file.")
                return
            # Check if more than one file was uploaded (shouldn't happen with multiple=False, but good practice)
            if len(uploaded_data) > 1:
                 print("Error: Please upload only one audio file.")
                 file_upload_widget.value = () # Reset upload
                 return

            try:
                filename, uploaded_file_info = list(uploaded_data.items())[0]
                content = uploaded_file_info['content'] # Get content from the info dict
            except Exception as e:
                print(f"Error accessing uploaded file data: {e}")
                file_upload_widget.value = () # Reset upload
                return

            display(Markdown(f"**Source:** Uploaded File (`{filename}`)"))

            # Ensure upload directory exists
            if not os.path.exists(UPLOAD_DIR):
                os.makedirs(UPLOAD_DIR)

            # Save the uploaded file to disk
            audio_filepath = os.path.join(UPLOAD_DIR, filename)
            try:
                with open(audio_filepath, 'wb') as f:
                    f.write(content)
                print(f"Uploaded file saved to: {audio_filepath}")
                try:
                     display(Audio(audio_filepath))
                except Exception as audio_err:
                     print(f"(Could not display audio player: {audio_err})")
            except Exception as e:
                print(f"Error saving uploaded file: {e}")
                audio_filepath = None
                file_upload_widget.value = ()
                return

        # --- Check if we have a valid audio path ---
        if not audio_filepath or not os.path.exists(audio_filepath):
             print("Error: Could not obtain a valid audio file path. Analysis aborted.")
             if selected_source == 'upload': file_upload_widget.value = ()
             return

        # --- Get common parameters ---
        analyze_full = analyze_whole_checkbox.value
        start_time = start_input.value if not analyze_full else None
        end_time = end_input.value if not analyze_full else None
        text_choice = text_analysis_choice.value
        lyrics_to_use = None
        transcribe_flag = False
        text_classes_to_use = None
        chosen_asr_model = None

        # Parse Emotion Classes
        classes_raw = classes_input.value
        emotion_labels = [ c.strip() for line in classes_raw.split('\n') for c in line.split(',') if c.strip() ]
        if not emotion_labels:
            print("Error: Please enter at least one emotion class.")
            if selected_source == 'upload': file_upload_widget.value = () # Reset upload
            return

        # Configure text analysis based on choice
        if text_choice == 'transcribe':
            transcribe_flag = True
            text_classes_to_use = emotion_labels
            chosen_asr_model = asr_model_dropdown.value
            display(Markdown(f"**Text Method:** Automatic Transcription (`{chosen_asr_model}`)"))
        elif text_choice == 'manual':
            lyrics_to_use = manual_lyrics_input.value
            if not lyrics_to_use:
                print("Error: 'Enter Lyrics Manually' selected, but no lyrics provided.")
                if selected_source == 'upload': file_upload_widget.value = () # Reset upload
                return
            text_classes_to_use = emotion_labels
            display(Markdown("**Text Method:** Manual Lyrics Provided"))
        elif text_choice == 'none':
            text_classes_to_use = None
            display(Markdown("**Text Method:** No Text Analysis"))


        # --- Run Analysis ---
        display(Markdown(f"**Audio File for Analysis:** `{os.path.basename(audio_filepath)}`"))
        display(Markdown(f"**Chosen Emotion Classes:** {', '.join(emotion_labels)}"))
        display(Markdown("Running `song_scores_py`... This might take a while."))

        results = song_scores_py(
            audio_path=audio_filepath,
            audio_classes=emotion_labels,
            text_classes=text_classes_to_use,
            lyrics=lyrics_to_use,
            transcribe_audio=transcribe_flag,
            start_sec=start_time,
            end_sec=end_time,
            asr_model_id=chosen_asr_model,
            asr_language=None,
            verbose=True
        )

        # --- Display Results ---
        display(Markdown("--- Results Summary ---"))
        # (The existing result display logic should work fine here)
        if results:
             print(f"Text Source: {results.get('text_source', 'N/A')}")
             if results.get('text_source') == 'transcribed' and results.get('transcribed_text'):
                  display(Markdown(f"**Transcription :**\n```\n{results['transcribed_text']}...\n```"))
             elif results.get('text_source') == 'provided_lyrics':
                  display(Markdown(f"**Manual Lyrics Used :**\n```\n{lyrics_to_use}...\n```"))
             elif results.get('text_source') == 'transcribed_failed':
                  print("Transcription failed.")
             elif text_choice == 'none':
                  print("Text analysis was not requested.")
             else:
                 print("Transcription not available or applicable.")

             print("\nAudio Scores:")
             if results.get('audio_scores'):
                 audio_scores_df = pd.DataFrame(list(results['audio_scores'].items()), columns=['Class', 'Score']).round(3)
                 display(audio_scores_df)
             else:
                 print("Audio scores not available.")

             print("\nText Scores:")
             if results.get('text_scores'):
                text_scores_df = pd.DataFrame(list(results['text_scores'].items()), columns=['Class', 'Score']).round(3)
                display(text_scores_df)
             elif text_choice == 'none':
                 print("Text analysis was not requested.")
             elif results.get('text_source') in ['transcribed_empty', 'transcribed_failed']:
                 print("Text analysis could not be performed (transcription issue).")
             elif text_choice == 'manual' and not lyrics_to_use:
                 print("Text analysis could not be performed (manual lyrics were empty).")
             else:
                  print("Text scores not available or analysis failed.")

             # Plot Results
             display(Markdown("--- Plotting Scores ---"))
             plot_fig = plot_scores_py(results)

        else:
            print("Analysis did not return results.")

        # --- Clear Upload Widget After Processing ---
        if selected_source == 'upload':
             try:
                 file_upload_widget.value.clear()
             except:
                 file_upload_widget.value = () # Fallback
             print("\nUpload field cleared.")


# --- Link Button and Display Widgets (Final Layout with Source Choice) ---
analyze_button.on_click(on_analyze_button_clicked)

# Arrange widgets vertically
ui = widgets.VBox([
    source_choice,       # Choose source first
    url_input,           # YouTube URL (visibility controlled)
    file_upload_widget,  # File Upload (visibility controlled)
    widgets.HTML("<hr>"), # Separator
    widgets.HBox([start_input, end_input, analyze_whole_checkbox]),
    classes_input,
    widgets.HTML("<hr>"), # Separator
    text_analysis_choice,
    manual_lyrics_input,
    asr_model_dropdown,
    widgets.HTML("<hr>"), # Separator
    analyze_button,
    output_widget
    ])
display(ui)