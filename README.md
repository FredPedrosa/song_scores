# Song Emotion Analysis (Audio & Lyrics) with R

This repository demonstrates multimodal emotion song analysis using R. It utilizes custom functions `song_scores()` and `plot_scores()` to analyze emotions derived from both the audio signal (using a CLAP model) and the song lyrics, using a Natural Language Inference model via the `transforEmotion` package (Tomasevic , 2024). The analysis includes an option for automatic speech recognition (ASR) to transcribe lyrics using OpenAI's Whisper model.

The primary example within this repository analyzes the song **"Gostava Tanto de Você"** by **Tim Maia**, showcasing the combined audio and text emotion scores.

## Features

*   **`song_scores()` function:**
    *   Performs zero-shot emotion classification on audio segments using a CLAP model (e.g., `laion/clap-htsat-unfused`).
    *   Optionally performs zero-shot emotion classification on song lyrics using an NLI model (e.g., `joeddav/xlm-roberta-large-xnli`) via `transforEmotion::transformer_scores`.
    *   Optionally transcribes audio segments to text using an ASR model (e.g., `openai/whisper-base`, `openai/whisper-large-v3`).
    *   Allows analysis of specific time segments (`start_sec`, `end_sec`).
    *   Accepts pre-written lyrics as input, skipping transcription.
*   **`plot_scores()` function:**
    *   Visualizes the emotion scores obtained from `song_scores()`.
    *   Generates a grouped bar plot comparing audio-based and text-based emotion scores side-by-side using `ggplot2`.

## Prerequisites

To run the analysis presented in the R Markdown file (`song_scores.Rmd`), you need the following set up:

1.  **R:** A recent version of R. RStudio is recommended but optional.
2.  **R Packages:**
    *   `transforEmotion`: Install from CRAN (`install.packages("transforEmotion")`). This is the core package providing text analysis capabilities and environment setup.
    *   `reticulate`: Handles the R-Python interface (`install.packages("reticulate")`).
    *   `ggplot2`: Used for plotting (`install.packages("ggplot2")`).
3.  **Python Environment (via Miniconda):** The analysis relies heavily on Python libraries. The `transforEmotion` package helps manage this using Miniconda.
    *   **Run Setup:** After installing `transforEmotion` in R, run `transforEmotion::setup_miniconda()`. This will:
        *   Install Miniconda specifically for R (if not already present) in a user-library location (e.g., `~/Library/r-miniconda` on macOS, `~\AppData\Local\r-miniconda` on Windows).
        *   Create a dedicated Conda environment named `transforEmotion`.
        *   Install core Python dependencies required by `transforEmotion` (including `transformers`, `torch`, `tensorflow`, `pandas`, `nltk`, `opencv-python`, etc.) into this environment using `pip`.
4.  **Additional Python Packages for Custom Functions:** The `song_scores` function requires a few more packages within the *same* `transforEmotion` Conda environment. Open your system Terminal or Anaconda Prompt, activate the environment, and install them:
    ```bash
    # Activate the environment created by setup_miniconda()
    conda activate transforEmotion

    # Install necessary libraries for audio processing and efficient model execution
    pip install librosa soundfile accelerate

    # Deactivate when done (optional)
    # conda deactivate
    ```
    *   `librosa` & `soundfile`: For robust audio loading and processing.
    *   `accelerate`: Often recommended for efficient execution of Hugging Face pipelines (like Whisper).
5.  **FFmpeg (Highly Recommended):** For `librosa` to load a wide variety of audio formats (especially `.mp3`), FFmpeg is often required as a backend decoder. Install it within the Conda environment:
    ```bash
    # Activate the environment (if not already active)
    conda activate transforEmotion

    # Install ffmpeg from the conda-forge channel
    conda install -c conda-forge ffmpeg
    ```

## Usage

1.  **Set up** the R and Python prerequisites as described above.
2.  **Open the R Markdown file** (`song_scores.Rmd`) in RStudio or your preferred R environment.
3.  **Ensure `reticulate` targets the correct environment:** At the beginning of your R session within the Rmd file, make sure `reticulate` knows to use the `transforEmotion` environment:
    ```R
    library(reticulate)
    # Point reticulate to the environment created by transforEmotion
    use_condaenv("transforEmotion", required = TRUE)
    ```
4.  **Load Functions:** The `.Rmd` file should contain or `source()` the code for the `song_scores()` and `plot_scores()` functions.
5.  **Run the Code:** Execute the chunks in the R Markdown file. This will:
    *   Define the song path and emotion classes.
    *   Call `song_scores()` to perform audio analysis and potentially transcribe/analyze the lyrics (this may take time, especially the first time models are downloaded or if using large ASR models).
    *   Call `plot_scores()` to visualize the results.

## Example: "Gostava Tanto de Você" - Tim Maia

The `.Rmd` file provides a concrete example using Tim Maia's famous song. It demonstrates how to:

*   Call `song_scores` requesting both audio analysis and automatic transcription (`transcribe_audio = TRUE`).
*   Specify relevant emotion classes for both modalities.
*   Use `plot_scores` to compare the emotional profile detected in the *sound* of the music versus the *content* of the transcribed lyrics.

## Notes

*   **Model Downloads:** The first time you run `song_scores` (or any function using a specific Hugging Face model like CLAP, Whisper, or NLI), the required model files will be downloaded to a local cache (`~/.cache/huggingface/hub` by default). Subsequent runs will load the models from this cache and will be much faster.
*   **Computational Resources:** Automatic Speech Recognition (especially with larger Whisper models like `large-v3`) can be computationally intensive and require significant RAM and time. Audio analysis with CLAP is generally faster.
*   **Transcription Accuracy:** ASR accuracy depends on audio quality, background noise, singing style, and the chosen Whisper model size and language setting.
