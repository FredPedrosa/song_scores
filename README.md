# Song Emotion Analysis (Audio & Lyrics) with R and Python (beta)

This repository provides tools for multimodal emotion song analysis, primarily demonstrated using R but also including a beta implementation in Python available as both a script and an interactive notebook.

**R Implementation:** It utilizes custom functions `song_scores()` and `plot_scores()` to analyze emotions derived from both the audio signal (using a CLAP model) and the song lyrics, using a Natural Language Inference model via the `transforEmotion` package (Tomasevic , 2024). The analysis includes an option for automatic speech recognition (ASR) to transcribe lyrics using OpenAI's Whisper model.

**Python Implementation (Beta):** A standalone Python script (`song_scores_py.py`) and an original Google Colab notebook (`song_scores_py.ipynb`) offer similar core analysis functionality using libraries like Transformers, Librosa, and yt-dlp. See the dedicated Python section below for details.

The primary R example within this repository analyzes the song **"Gostava Tanto de Você"** by **Tim Maia**, showcasing the combined audio and text emotion scores using the R functions. The Python files include their own examples.

---

# R

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

---

# Python - BETA

This repository also includes a **beta version** of the core analysis functionality implemented purely in Python. This is available in two formats:

1.  **`song_scores_py.py`:** A standalone Python script containing the core analysis function (`song_scores_py()`) and helper functions. This script is suitable for importing into other Python projects or running directly from the command line (it includes a basic example execution block).
2.  **`song_scores_py.ipynb`:** The original Google Colaboratory notebook from which the `.py` script was derived. This notebook includes the same core functions but also contains the interactive UI elements (`ipywidgets`) for easier experimentation directly within Colab or a compatible Jupyter environment.

**Status:** This Python implementation is currently considered **beta**. While functional, it may undergo changes and further refinement. It does not yet have the same level of integration testing as the R version. The notebook (`.ipynb`) might be slightly out of sync with the `.py` script regarding recent code refinements (like the example block).

### Python Prerequisites

To use either the Python script or the notebook, you need Python 3.8+ and the following packages installed, preferably in a dedicated virtual environment (like Conda or venv):

1.  **Create Environment (Example with Conda):**
    ```bash
    conda create -n song_py python=3.9 -y
    conda activate song_py
    ```
2.  **Install Packages:**
    ```bash
    # Core ML/Audio libraries
    pip install torch transformers accelerate librosa soundfile pandas numpy

    # Plotting
    pip install matplotlib

    # YouTube download capability
    pip install yt-dlp

    # For the Notebook's interactive UI and audio display:
    pip install ipywidgets ffmpeg-python ipython

    # Highly Recommended: FFmpeg for broad audio format support (can be installed via conda or system)
    conda install -c conda-forge ffmpeg # If using conda environment
    ```
    *   Ensure `torch` is installed correctly for your CPU/GPU setup (check PyTorch official website if needed).
    *   Note: `ffmpeg-python` is needed by the *notebook* specifically for its UI interactions if saving uploads; the core `librosa` functionality generally relies on the command-line `ffmpeg` installed via Conda or the system.

### Python Usage

*   **Using the Script (`song_scores_py.py`):**
    *   Import the functions into your own Python code as shown in the example below.
    *   Run the script directly (`python song_scores_py.py`) to see the built-in example execute.

    ```python
    # Example of importing and using the script's function
    import song_scores_py
    import matplotlib.pyplot as plt
    import os

    audio_file = "path/to/your/song.mp3" # Replace with your audio file
    emo_classes = ["happy", "sad", "angry", "calm"]

    if os.path.exists(audio_file):
        results = song_scores_py.song_scores_py(
            audio_path=audio_file,
            audio_classes=emo_classes,
            text_classes=emo_classes, # Also analyze text
            transcribe_audio=True,    # Request transcription
            start_sec=10.0,           # Analyze segment from 10s to 40s
            end_sec=40.0,
            asr_model_id="openai/whisper-base", # Choose ASR model
            verbose=True
        )

        print("\n--- Analysis Results ---")
        # Print scores using pandas for nice formatting
        import pandas as pd
        print("Audio Scores:")
        print(pd.Series(results.get('audio_scores')).to_string())
        print("\nText Scores:")
        print(pd.Series(results.get('text_scores')).to_string())
        print(f"\nTranscription Preview: {results.get('transcribed_text', '')[:100]}...")

        # Plot the results
        fig = song_scores_py.plot_scores_py(results, title="My Song Analysis (10-40s)")
        if fig:
            plt.show() # Display the plot
    else:
        print(f"Error: Audio file not found at {audio_file}")
    ```

*   **Using the Notebook (`song_scores_py.ipynb`):**
    *   Open the notebook in Google Colab ("File" -> "Open notebook" -> "GitHub" tab, paste the repository URL).
    *   Alternatively, run it in a local Jupyter environment that has the prerequisites installed.
    *   Execute the cells sequentially. The notebook provides an interactive interface using widgets to select options and run the analysis.

---

## How to Cite

### Citing this Function/Code:
Pedrosa, F. G. (2025). *song_scores: Functions to implement multimodal emotion song analysis in R and Python.*. [Software]. Retrieved from https://github.com/FredPedrosa/song_scores/

## Author

*   **Prof. Dr. Frederico G. Pedrosa**

### Reference

 Tomasevic A, Golino H, Christensen A (2024). “Decoding emotion dynamics in videos using dynamic Exploratory Graph Analysis and zero-shot image classification: A simulation and tutorial using the transforEmotion R package.” _PsyArXiv_. doi:10.31234/osf.io/hf3g7 <https://doi.org/10.31234/osf.io/hf3g7>, <https://osf.io/preprints/psyarxiv/hf3g7>.

## License

This project is licensed under a modified version of the GNU General Public License v3.0. 
Commercial use is not permitted without explicit written permission from the author.
