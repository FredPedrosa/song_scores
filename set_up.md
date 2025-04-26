**Instruções para Configurar e Usar `song_scores()` em R (Windows)**

1.  **Instalar o Pacote R `transforEmotion`:**
    *   Abra o R ou RStudio.
    *   Execute no console:
        ```R
        install.packages("transforEmotion")
        ```

2.  **Executar o Setup Inicial do Ambiente Python (Apenas Uma Vez):**
    *   Ainda no R/RStudio, carregue o pacote e rode o setup:
        ```R
        library(transforEmotion)
        transforEmotion::setup_miniconda()
        ```
    *   **Aguarde** este processo terminar completamente. Ele instalará um Miniconda isolado e o ambiente Python `transforEmotion` com as dependências *básicas* do pacote. Verifique se não há erros fatais no final.

3.  **Instalar Dependências Adicionais para `song_scores`:**
    *   **Feche** o R/RStudio para garantir que nenhum processo esteja bloqueando arquivos.
    *   **Abra o Prompt de Comando ou Anaconda Prompt** como Administrador (clique direito -> Executar como administrador) pode ajudar a evitar warnings de permissão.
    *   **Navegue até a pasta `condabin` do `r-miniconda`:**
        ```bash
        cd C:\Users\SEU_USUARIO\AppData\Local\r-miniconda\condabin
        ```
        *(Substitua `SEU_USUARIO` pelo seu nome de usuário)*.
    *   **Ative o ambiente `transforEmotion` específico:**
        ```bash
        .\conda activate transforEmotion
        ```
        *(Verifique se `(transforEmotion)` aparece no prompt)*.
    *   **Instale `librosa` e `soundfile` usando pip:** (Essenciais para carregar áudio na nossa função)
        ```bash
        pip install librosa soundfile
        ```
    *   **(Recomendado) Instale `ffmpeg` usando conda:** (Essencial para carregar MP3 e outros formatos via `librosa`)
        ```bash
        conda install -c conda-forge ffmpeg -y
        ```
    *   **(Opcional) Feche o Prompt de Comando.**

4.  **Usar a Função `song_scores` no R:**
    *   Abra o R/RStudio **NOVAMENTE**.
    *   **Carregue as bibliotecas necessárias** (o `reticulate` será configurado automaticamente pelo `transforEmotion`):
        ```R
        library(transforEmotion)
        # library(reticulate) # Só se precisar de funções py_... explicitamente
        # library(ggplot2)    # Se for usar plot_scores
        # library(tictoc)     # Se sua função usa os timers
        ```
    *   **Defina ou Carregue suas Funções:** Certifique-se de que as definições das funções `song_scores` e `plot_scores` (se for usar) estejam carregadas na sessão R:
        ```R
        # Se estiverem em arquivos .R:
        # source("caminho/para/song_scores.R")
        # source("caminho/para/plot_scores.R")

        # OU cole as definições completas das funções aqui e execute
        ```
    *   **Execute sua Análise:** Chame a função `song_scores` passando os argumentos:
        ```R
        my_song <- "C:/Users/[SEU USUÁRIO]/[PASTA]/[CANÇÃO].mp3" # Seu caminho
        emo_classes <- c("alegre", "triste", "neutro") # Suas classes

        if (!file.exists(my_song)) stop("Arquivo não encontrado!")

        results <- song_scores(
          audio_path = my_song,
          audio_classes = emo_classes,
          text_classes = emo_classes, # Ou NULL se não quiser análise de texto
          transcribe_audio = TRUE,   # Ou FALSE se for fornecer lyrics ou nenhuma análise
          asr_language = "pt",       # Opcional
          asr_model_id = "openai/whisper-base", # Recomendo base para começar
          # lyrics = "Sua letra aqui", # Se transcribe_audio = FALSE
          # start_sec = 10, end_sec = 40, # Se quiser segmento
          verbose = TRUE
        )

        # Visualize os resultados
        print(results)
        # plot_scores(results) # Se a função plot_scores estiver definida
        ```

**Resumo:** Instalar `transforEmotion` -> Rodar `setup_miniconda()` uma vez -> Instalar `librosa`, `soundfile`, `ffmpeg` no ambiente `transforEmotion` via terminal -> Reiniciar R -> Carregar `library(transforEmotion)` e sua função -> Chamar a função. Não use `use_condaenv` manualmente, deixe o pacote gerenciar.
