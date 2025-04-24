#conda activate transforEmotion
#pip install transformers torch librosa soundfile sentencepiece accelerate
#conda install -c conda-forge ffmpeg



# ---- PRÉ-REQUISITOS ----
# 1. Ambiente Conda 'transforEmotion' configurado (transforEmotion::setup_miniconda()).
# 2. Pacotes Python instalados:
#    conda activate transforEmotion
#    pip install transformers torch librosa soundfile sentencepiece accelerate
#    (Verifique se 'torch' foi instalado corretamente para sua CPU/GPU)
# 3. Pacote R 'transforEmotion' instalado e carregável.

#' Análise de Emoção em Música (Áudio e Letra)
#'
#' Combina análise de emoção baseada em áudio (usando CLAP) e em texto
#' (usando um modelo NLI via transforEmotion::transformer_scores).
#' Pode transcrever automaticamente a letra do áudio usando Whisper ou aceitar
#' letras pré-fornecidas.
#'
#' @param audio_path Caminho para o arquivo de áudio (ex: .wav, .mp3).
#' @param audio_classes Vetor de caracteres com rótulos de emoção para análise de ÁUDIO (CLAP).
#' @param text_classes Vetor de caracteres com rótulos de emoção para análise de TEXTO (NLI).
#'   Se NULL, a análise de texto é pulada.
#' @param lyrics Texto da letra da música (opcional). Se fornecido (não NULL),
#'   a transcrição automática será ignorada e este texto será usado para
#'   a análise de emoção textual. Se NULL, a função pode tentar transcrever.
#' @param transcribe_audio Logical. Se TRUE e `lyrics` for NULL, tenta transcrever
#'   o áudio (ou segmento) usando o modelo ASR especificado. Padrão: FALSE.
#'   A transcrição pode ser demorada.
#' @param start_sec Tempo inicial (segundos) do segmento para análise de ÁUDIO
#'   e, se `transcribe_audio=TRUE`, para a TRANSCRIÇÃO. Se NULL, começa do início.
#' @param end_sec Tempo final (segundos) do segmento. Se NULL, vai até o fim.
#' @param clap_model_id Identificador do modelo CLAP (Hugging Face).
#'   Padrão: "laion/clap-htsat-unfused".
#' @param nli_model_id Identificador do modelo NLI para texto (Hugging Face),
#'   usado por `transformer_scores`. Padrão: "joeddav/xlm-roberta-large-xnli".
#' @param asr_model_id Identificador do modelo ASR (Whisper) para transcrição
#'   (Hugging Face). Padrão: "openai/whisper-base". Modelos maiores
#'   (ex: "openai/whisper-large-v3") são mais precisos, mas mais lentos/pesados.
#' @param asr_language Código do idioma para o Whisper (ex: "portuguese", "english").
#'   Ajuda na precisão da transcrição. Se NULL (padrão), Whisper tenta detectar.
#' @param verbose Logical. Imprime mensagens de progresso. Padrão: TRUE.
#'
#' @return Uma lista contendo:
#'   \item{audio_scores}{Scores de emoção do CLAP para o áudio/segmento (vetor nomeado ou NA em caso de falha).}
#'   \item{text_scores}{Scores de emoção do NLI para a letra (resultado de `transformer_scores` ou NULL se não aplicável/falhou).}
#'   \item{transcribed_text}{O texto transcrito por Whisper (string ou NULL se não transcrito/falhou).}
#'   \item{text_source}{Indica a origem do texto analisado ("provided_lyrics", "transcribed", "transcribed_empty", "none").}
#' @export
#'
#' @examples
#' \dontrun{
#' # --- Pré-requisitos ---
#' # ... (instruções de instalação como acima) ...
#'
#' # --- Exemplo de uso ---
#' my_song <- "caminho/para/sua/musica.mp3" # Substitua!
#' emo_classes <- c("alegria", "tristeza", "raiva", "calma", "neutro")
#'
#' # 1. Analisar áudio e usar letra fornecida
#' letra_conhecida <- "Alegria, alegria, o sol está a brilhar..." # Substitua
#' results1 <- song_scores(
#'   audio_path = my_song,
#'   audio_classes = emo_classes,
#'   text_classes = emo_classes,
#'   lyrics = letra_conhecida,
#'   start_sec = 30, # Analisar áudio dos 30s aos 60s
#'   end_sec = 60
#' )
#' print("Resultados (Letra Fornecida):")
#' print(results1)
#'
#' # 2. Analisar áudio (segmento 0-30s) e TRANSCREVER a letra DESSE segmento
#' results2 <- song_scores(
#'   audio_path = my_song,
#'   audio_classes = emo_classes,
#'   text_classes = emo_classes,
#'   transcribe_audio = TRUE, # PEDIR TRANSCRIÇÃO
#'   start_sec = 0,
#'   end_sec = 30,
#'   asr_language = "portuguese", # Especificar idioma ajuda
#'   asr_model_id = "openai/whisper-large-v3" # Exemplo com modelo maior
#' )
#' print("Resultados (Transcrição Automática 0-30s):")
#' print(results2)
#'
#' # 3. Analisar apenas o áudio (inteiro), sem análise de texto
#' results3 <- song_scores(
#'   audio_path = my_song,
#'   audio_classes = emo_classes,
#'   text_classes = NULL # Não fazer análise de texto
#' )
#' print("Resultados (Apenas Áudio):")
#' print(results3)
#' }
song_scores <- function(audio_path,
                        audio_classes,
                        text_classes = NULL, # Permitir não analisar texto
                        lyrics = NULL,
                        transcribe_audio = FALSE,
                        start_sec = NULL,
                        end_sec = NULL,
                        clap_model_id = "laion/clap-htsat-unfused",
                        nli_model_id = "joeddav/xlm-roberta-large-xnli",
                        asr_model_id = "openai/whisper-base",
                        asr_language = NULL,
                        verbose = TRUE) {
  
  # --- Verificações Iniciais ---
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("Pacote 'reticulate' é necessário.", call. = FALSE)
  }
  # Verificar transforEmotion apenas se text_classes for solicitado
  if (!is.null(text_classes) && !requireNamespace("transforEmotion", quietly = TRUE)) {
    stop("Pacote 'transforEmotion' é necessário para análise de texto.", call. = FALSE)
  }
  
  if (!file.exists(audio_path)) {
    stop("Arquivo de áudio não encontrado: ", audio_path, call. = FALSE)
  }
  if (!is.character(audio_classes) || length(audio_classes) == 0) {
    stop("'audio_classes' deve ser um vetor de caracteres não vazio.", call. = FALSE)
  }
  if (!is.null(text_classes) && (!is.character(text_classes) || length(text_classes) == 0)) {
    stop("'text_classes', se fornecido, deve ser um vetor de caracteres não vazio.", call. = FALSE)
  }
  if (!is.null(lyrics) && !is.character(lyrics)) {
    stop("'lyrics', se fornecido, deve ser um vetor de caracteres.", call. = FALSE)
  }
  
  
  # --- Validar start_sec e end_sec ---
  offset_py <- reticulate::py_none()
  duration_py <- reticulate::py_none()
  segment_msg <- "o arquivo de áudio inteiro"
  
  if (!is.null(start_sec) && !is.null(end_sec)) {
    if (!is.numeric(start_sec) || start_sec < 0) stop("'start_sec' deve ser numérico >= 0.", call. = FALSE)
    if (!is.numeric(end_sec) || end_sec <= start_sec) stop("'end_sec' deve ser numérico > 'start_sec'.", call. = FALSE)
    offset_py <- reticulate::r_to_py(as.numeric(start_sec))
    duration_val <- as.numeric(end_sec - start_sec)
    duration_py <- reticulate::r_to_py(duration_val)
    segment_msg <- sprintf("o segmento de %.2f s a %.2f s (duração: %.2f s)", start_sec, end_sec, duration_val)
  } else if (!is.null(start_sec) || !is.null(end_sec)) {
    stop("Ambos 'start_sec' e 'end_sec' devem ser fornecidos ou ambos NULL.", call. = FALSE)
  }
  
  if (verbose) message(paste("Análise solicitada para", segment_msg))
  
  # --- Verificar Módulos Python ---
  required_py_modules <- c("transformers", "torch", "librosa", "accelerate")
  modules_available <- sapply(required_py_modules, reticulate::py_module_available)
  
  if (!all(modules_available)) {
    missing_modules <- names(modules_available[!modules_available])
    stop(
      "Módulos Python necessários não encontrados: ", paste(missing_modules, collapse = ", "), ".\n",
      "Instale-os no ambiente 'transforEmotion': 'pip install ", paste(missing_modules, collapse = " "), "'",
      call. = FALSE
    )
  }
  
  # --- Importar Módulos Python ---
  if (verbose) message("Importando módulos Python (transformers, torch, librosa)...")
  transformers <- reticulate::import("transformers", delay_load = TRUE)
  torch <- reticulate::import("torch", delay_load = TRUE)
  librosa <- reticulate::import("librosa", delay_load = TRUE)
  
  
  # --- Inicializar Lista de Resultados ---
  results <- list(
    audio_scores = setNames(rep(NA_real_, length(audio_classes)), audio_classes), # Inicializa com NA
    text_scores = NULL,
    transcribed_text = NULL,
    text_source = "none" # none, provided_lyrics, transcribed, transcribed_empty
  )
  
  # --- Código Python para CLAP (Adaptado da função anterior) ---
  reticulate::py_run_string("
import torch
import librosa
from transformers import AutoProcessor, AutoModel # Usando AutoModel como na versão anterior
import numpy as np

def get_clap_scores_py(audio_fpath, text_classes, model_ident, offset=None, duration=None):
    try:
        processor = AutoProcessor.from_pretrained(model_ident)
        model = AutoModel.from_pretrained(model_ident)

        target_sr = processor.feature_extractor.sampling_rate if hasattr(processor.feature_extractor, 'sampling_rate') else 48000
        audio_array, sr = librosa.load(audio_fpath, sr=target_sr, mono=True, offset=offset, duration=duration)

        if audio_array.size == 0:
             return {'error': 'Segmento de áudio CLAP vazio.', 'scores': []}

        audio_array = np.array(audio_array)
        inputs = processor(text=text_classes, audios=[audio_array], return_tensors=\"pt\", padding=True, sampling_rate=target_sr)

        with torch.no_grad():
            outputs = model(**inputs)
        logits_per_audio = outputs.logits_per_audio[0] # Assume 1 áudio processado
        probs = torch.softmax(logits_per_audio, dim=0).numpy()

        return {'error': None, 'scores': probs.tolist()}

    except Exception as e:
        return {'error': f'Erro no CLAP: {str(e)}', 'scores': []}
  ")
  
  # --- Código Python para ASR (Whisper) ---
  reticulate::py_run_string("
import torch
import librosa
from transformers import pipeline
import numpy as np
import warnings

# Desativar warnings específicos do pipeline que podem poluir o console
warnings.filterwarnings('ignore', message='.*Maximum duration.*')
warnings.filterwarnings('ignore', message='.*Using PipelineChunkIterator.*')

def transcribe_audio_segment_py(audio_fpath, model_ident, offset=None, duration=None, language=None):
    try:
        target_sr = 16000 # Whisper SR
        audio_array, sr = librosa.load(audio_fpath, sr=target_sr, mono=True, offset=offset, duration=duration)

        if audio_array.size == 0:
             return {'error': 'Segmento de áudio ASR vazio.', 'text': ''}

        audio_array = np.array(audio_array)
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        # Usar try-except para carregar o pipeline, pois pode falhar por memória, etc.
        try:
            pipe = pipeline(
                'automatic-speech-recognition',
                model=model_ident,
                device=device,
                chunk_length_s=30,
                stride_length_s=5
            )
        except Exception as pipe_e:
             return {'error': f'Erro ao carregar pipeline ASR ({model_ident}): {str(pipe_e)}', 'text': ''}


        generate_kwargs = {}
        if language:
          generate_kwargs['language'] = language

        # Usar try-except para a transcrição em si
        try:
            transcription_result = pipe(audio_array.copy(), generate_kwargs=generate_kwargs)
            transcribed_text = transcription_result['text'].strip() if transcription_result else ''
        except Exception as trans_e:
             return {'error': f'Erro durante a transcrição ASR: {str(trans_e)}', 'text': ''}


        return {'error': None, 'text': transcribed_text}

    except Exception as e:
        # Captura erros gerais como falha no librosa.load
        return {'error': f'Erro geral no ASR (Whisper): {str(e)}', 'text': ''}
  ")
  
  
  # --- 1. Análise de Áudio (CLAP) ---
  if (verbose) message(paste("Iniciando análise de ÁUDIO com CLAP:", clap_model_id, "..."))
  clap_results_py <- tryCatch({
    reticulate::py$get_clap_scores_py(
      audio_fpath = audio_path,
      text_classes = reticulate::r_to_py(audio_classes),
      model_ident = clap_model_id,
      offset = offset_py,
      duration = duration_py
    )
  }, error = function(e) {
    list(error = paste("Erro R ao chamar CLAP:", e$message), scores = list())
  })
  
  # Processar resultado CLAP
  if (!is.null(clap_results_py$error)) {
    warning("Falha na análise de áudio (CLAP): ", clap_results_py$error, call. = FALSE)
    # results$audio_scores já está inicializado com NA
  } else {
    scores_r <- reticulate::py_to_r(clap_results_py$scores)
    if (length(scores_r) == length(audio_classes)) {
      results$audio_scores <- setNames(scores_r, audio_classes) # Sobrescreve NA com scores válidos
      if (verbose) message("Análise de áudio concluída.")
    } else {
      warning("Número de scores CLAP (", length(scores_r), ") diferente do número de classes (", length(audio_classes), ").", call. = FALSE)
      # Mantém os NAs em results$audio_scores
    }
  }
  
  
  # --- 2. Preparação do Texto para Análise ---
  text_to_analyze <- NULL
  perform_text_analysis <- !is.null(text_classes) # Só analisa texto se text_classes for fornecido
  
  if (perform_text_analysis) {
    if (!is.null(lyrics)) {
      if (verbose) message("Usando letra fornecida ('lyrics').")
      # Se lyrics for um vetor com múltiplas strings, concatenar? Ou analisar separadamente?
      # Assumindo que é uma única string ou deve ser concatenado.
      text_to_analyze <- paste(lyrics, collapse = "\n")
      results$text_source <- "provided_lyrics"
    } else if (transcribe_audio) {
      if (verbose) message(paste("Iniciando TRANSCRIÇÃO de áudio com ASR (Whisper):", asr_model_id, "... (Pode demorar!)"))
      
      asr_results_py <- tryCatch({
        reticulate::py$transcribe_audio_segment_py(
          audio_fpath = audio_path,
          model_ident = asr_model_id,
          offset = offset_py,
          duration = duration_py,
          language = if (!is.null(asr_language)) asr_language else reticulate::py_none()
        )
      }, error = function(e) {
        list(error = paste("Erro R ao chamar ASR:", e$message), text = '')
      })
      
      # Processar resultado ASR
      if (!is.null(asr_results_py$error)) {
        warning("Falha na transcrição de áudio (ASR): ", asr_results_py$error, call. = FALSE)
        results$transcribed_text <- NULL # Falhou
        results$text_source <- "transcribed_failed" # Novo estado
      } else {
        results$transcribed_text <- reticulate::py_to_r(asr_results_py$text)
        if (!is.null(results$transcribed_text) && nchar(results$transcribed_text) > 0) {
          if (verbose) message("Transcrição concluída.")
          if(verbose) message(paste("Texto transcrito:", substr(results$transcribed_text, 1, 100), "..."))
          text_to_analyze <- results$transcribed_text
          results$text_source <- "transcribed"
        } else {
          if (verbose) message("Transcrição concluída, mas resultou em texto vazio ou NULL.")
          results$transcribed_text <- "" # Garantir que é string vazia, não NULL
          results$text_source <- "transcribed_empty"
        }
      }
    } else {
      if (verbose) message("Nenhuma letra fornecida ('lyrics'=NULL) e transcrição não solicitada ('transcribe_audio'=FALSE). Análise de texto será pulada.")
      results$text_source <- "none"
    }
  } else {
    if (verbose) message("Análise de texto não solicitada ('text_classes'=NULL).")
    results$text_source <- "none"
  }
  
  
  # --- 3. Análise de Texto (NLI) ---
  # Executar somente se tivermos texto E classes de texto foram fornecidas
  if (!is.null(text_to_analyze) && nchar(text_to_analyze) > 0 && perform_text_analysis) {
    if (verbose) message(paste("Iniciando análise de TEXTO com NLI:", nli_model_id, "..."))
    
    # Chamar transformer_scores dentro de tryCatch
    text_analysis_result <- tryCatch({
      # transformer_scores espera um vetor de textos. Passar a letra como um único elemento.
      transforEmotion::transformer_scores(
        text = text_to_analyze,
        classes = text_classes,
        transformer = nli_model_id
      )
    }, error = function(e) {
      warning("Falha na análise de texto (transformer_scores): ", e$message, call. = FALSE)
      return(NULL) # Retorna NULL em caso de erro na análise de texto
    })
    
    # Armazenar o resultado (pode ser NULL se o tryCatch falhou)
    results$text_scores <- text_analysis_result
    if (!is.null(text_analysis_result) && verbose) {
      message("Análise de texto concluída.")
    }
    
  } else if (perform_text_analysis && results$text_source %in% c("none", "transcribed_empty", "transcribed_failed")) {
    # Se a análise de texto foi solicitada mas não houve texto para analisar
    if (verbose) message("Análise de texto pulada por falta de conteúdo textual (letra/transcrição).")
  }
  
  
  # --- Retorno Final ---
  if (verbose) message("Processamento concluído.")
  return(results)
}


library(ggplot2)

plot_scores <- function(results) {
  # Extrair as pontuações e classes do áudio
  audio_classes <- names(results$audio_scores)
  audio_percentages <- results$audio_scores
  
  # Verificar se há pontuações de texto
  if (!is.null(results$text_scores)) {
    text_classes <- names(results$text_scores[[1]])
    text_percentages <- results$text_scores[[1]]
    
    # Criar um dataframe para o ggplot
    plot_data <- data.frame(
      class = c(audio_classes, text_classes),
      percentage = c(audio_percentages, text_percentages),
      legenda = rep(c("Audio", "Text"), each = length(audio_classes))
    )
  } else {
    # Criar um dataframe apenas com as pontuações de áudio
    plot_data <- data.frame(
      class = audio_classes,
      percentage = audio_percentages,
      legenda = rep("Audio", length(audio_classes))
    )
  }
  
  # Gerar o barplot
  ggplot(plot_data, aes(x = class, y = percentage, fill = legenda)) +
    geom_bar(stat = "identity", position = "dodge", color = "black") +
    scale_fill_manual(values = c("Audio" = "#4c9a9e", "Text" = "#b3bef2")) +
    labs(title = "", x = "Classes", y = "Percentage", fill = "Legend") +
    theme_classic()
}


