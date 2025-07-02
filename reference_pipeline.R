# ----------------------------- LOAD PACKAGES -------------------------------- #
if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")
pacman::p_load(
  tidyverse,
  furrr,
  future,
  progressr,
  lme4,
  lmerTest,
  broom.mixed,
  patchwork,
  svglite
)

source("scripts/utilities.R")

# ----------------------------- PARAMETERS ------------------------------------ #
LANGUAGE   <- "en"
MIN_FREQ   <- 1e-7            # minimal word frequency
LENGTHS    <- 4:11            # word lengths
LEFT_DROPS  <- seq(0.1, 0.9, 0.05)   # range for left drop
RIGHT_DROPS <- seq(0.1, 0.9, 0.05)   # range for right drop


# ----------------------- ENTROPY CALCULATION FUNCTIONS ----------------------- #

compute_letter_acuity_lr <- function(fix_position, word_length, drop_left, drop_right) {
  # For each letter position, apply a separate linear drop for the left vs. right side.
  # No normalization is done. If you want to keep average acuity constant, add extra logic.
  positions <- seq_len(word_length)
  dist <- positions - fix_position
  # Acuity to the left:
  acuity_left  <- pmax(1 - drop_left  * abs(dist), 0)
  # Acuity to the right:
  acuity_right <- pmax(1 - drop_right * abs(dist), 0)
  # If dist < 0 => left side, else right side:
  ifelse(dist < 0, acuity_left, acuity_right)
}

letter_masks <- function(len){
  # Return all positional masking combinations for a given string length.
  # Each combination is a logical vector of length 'len' indicating which letters are 'seen'.
  binary_strings <- R.utils::intToBin(0:(2^len - 1))
  masks <- map(str_split(binary_strings, pattern=""), 
               ~as.logical(as.numeric(.x)))
  return(masks)
}

letter_mask_probabilities_lr <- function(drop_left, drop_right, max_length) {
  # For each word length, each fixation position, and each mask,
  # compute probability = product(over letters)[
  #    if letter is 'seen' then acuity else (1 - acuity)
  # ]
  # using separate drop_left, drop_right.
  map(1:max_length, function(len) {
    masks <- letter_masks(len)
    map(seq_len(len), function(fix) {
      map(masks, function(mask) {
        acuity_vals <- compute_letter_acuity_lr(fix, len, drop_left, drop_right)
        prod(map2_dbl(mask, acuity_vals, function(seen, a) if_else(seen, a, 1 - a)))
      })
    })
  })
}

get_mask_probabilities_lr <- function(drop_left, drop_right, max_length) {
  message(sprintf("Calculating probabilities for drop_left=%.3f, drop_right=%.3f", 
                  drop_left, drop_right))
  letter_mask_probabilities_lr(drop_left, drop_right, max_length)
}

ILP_entropy <- function(target, word_data_by_length, drop_left, drop_right, 
                        masks_by_len, mask_probs) {
  # Compute ILP entropy for a single target word, given precomputed mask probabilities
  # and separate left/right drop parameters.
  len <- nchar(target)
  corpus_subset <- word_data_by_length[[as.character(len)]]
  if (is.null(corpus_subset) || nrow(corpus_subset) == 0) {
    # Edge case: no words with this length?
    return(as.data.frame(t(rep(NA_real_, len))))
  }
  
  target_letters <- str_split(target, "", simplify=TRUE)
  masks <- masks_by_len[[len]]  # all possible letter-masking combos for 'len'
  
  # Build regex for each mask
  regex_masks <- map(masks, ~ str_c("^", 
                                    str_flatten(ifelse(.x, target_letters, ".")),
                                    "$"))
  
  # For each mask, gather candidate words' frequencies
  freq_by_mask <- map(regex_masks, 
                      ~ corpus_subset$freq[str_detect(corpus_subset$word, .x)])
  # Convert to relative freq, then compute entropy = -∑ p log2 p
  # (Guard against empty sums)
  relfreq_by_mask <- map(freq_by_mask, function(ff) {
    s <- sum(ff)
    if (s > 0) ff/s else numeric(0)
  })
  entropy_by_mask <- map_dbl(relfreq_by_mask, function(pvec) {
    if (length(pvec) == 0) return(0)
    -sum(pvec * log2(pvec + 1e-15))
  })
  
  # Weighted sum of entropies across all masks, per fixation position
  probs_for_len <- mask_probs[[len]]
  mean_entropy_by_pos <- map_dbl(seq_len(len), function(pos) {
    sum(unlist(probs_for_len[[pos]]) * entropy_by_mask)
  })
  
  as.data.frame(t(mean_entropy_by_pos))
}

# ------------------------ MAIN WRAPPER: ENTROPY COMPUTATION ----------------- #

compute_entropy_lr <- function(language, 
                               left_drops, right_drops, 
                               min_freq, lengths, 
                               word_data, meco_words) {
  
  progressr::handlers(progressr::handler_progress(
    format = ":spin :current/:total (:message) [:bar] :percent",
    show_after = 0
  ))
  
  progressr::with_progress({
    n_workers <- future::nbrOfWorkers()
    message(sprintf("\nUsing %d workers for parallel processing", n_workers))
    
    # Group word data by length
    word_data_by_length <- word_data %>%
      dplyr::group_by(length) %>%
      dplyr::group_split() %>%
      stats::setNames(as.character(lengths))
    rm(word_data)
    
    # Generate letter masks for each length
    p_gen_masks <- progressr::progressor(1)
    p_gen_masks("Step 1/3: Generating letter masks")
    masks_by_length <- purrr::map(1:max(lengths), letter_masks)
    
    # Create parameter grid with separate left/right drops
    param_grid <- expand.grid(drop_left = left_drops, drop_right = right_drops)
    total_params <- nrow(param_grid)
    
    # Precompute mask probabilities for each (drop_left, drop_right) pair
    p_mask_probs <- progressr::progressor(total_params)
    message("\nStep 2/3: Computing mask probabilities")
    mask_prob_list <- furrr::future_map(1:total_params, function(i) {
      dl <- param_grid$drop_left[i]
      dr <- param_grid$drop_right[i]
      res <- get_mask_probabilities_lr(dl, dr, max(lengths))
      p_mask_probs(sprintf("Param %d/%d", i, total_params))
      res
    }, .options = furrr::furrr_options(seed = TRUE))
    
    # Create output directory
    timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
    base_dir  <- file.path("entropy", language)
    run_dir   <- file.path(base_dir)  # you can append timestamp if you want
    dir.create(run_dir, showWarnings = FALSE, recursive = TRUE)
    
    # Save metadata
    metadata <- list(
      language     = language,
      timestamp    = timestamp,
      min_freq     = min_freq,
      word_lengths = lengths,
      n_words      = length(meco_words),
      parameters   = param_grid
    )
    saveRDS(metadata, file.path(run_dir, "metadata.rds"))
    
    # Chunk up the MECO words for parallel processing
    chunk_size  <- ceiling(length(meco_words) / (n_workers * 2))
    word_chunks <- split(meco_words, ceiling(seq_along(meco_words)/chunk_size))
    full_task_grid <- expand.grid(
      param_idx = 1:nrow(param_grid),
      chunk_idx = seq_along(word_chunks)
    )
    
    # Process tasks
    p_tasks <- progressr::progressor(nrow(full_task_grid))
    message("\nStep 3/3: Processing words")
    future_results <- furrr::future_map(1:nrow(full_task_grid), function(task_idx) {
      param_idx <- full_task_grid$param_idx[task_idx]
      chunk_idx <- full_task_grid$chunk_idx[task_idx]
      
      dl <- param_grid$drop_left[param_idx]
      dr <- param_grid$drop_right[param_idx]
      chunk_words <- word_chunks[[chunk_idx]]
      mask_probs <- mask_prob_list[[param_idx]]
      
      chunk_result <- tibble::tibble(word = chunk_words) %>%
        dplyr::mutate(entropy = purrr::map(word, ~ ILP_entropy(
          .x, word_data_by_length, dl, dr, masks_by_length, mask_probs
        ))) %>%
        dplyr::rowwise() %>%
        dplyr::mutate(entropy = list(as.vector(na.omit(as.numeric(entropy))))) %>%
        dplyr::ungroup()
      
      # Save chunk results
      param_dir <- sprintf("dropL_%.3f_dropR_%.3f", dl, dr)
      output_dir <- file.path(run_dir, param_dir)
      dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
      output_file <- file.path(output_dir, sprintf("entropy_chunk_%04d.rds", chunk_idx))
      saveRDS(chunk_result, output_file)
      
      p_tasks(sprintf("Task %d/%d", task_idx, nrow(full_task_grid)))
      output_file
    }, .options = furrr::furrr_options(seed = TRUE, scheduling = 2))
    
    # Combine chunk results for each param set
    p_combine <- progressr::progressor(1)
    p_combine("Combining chunk results")
    param_dirs <- unique(sprintf("dropL_%.3f_dropR_%.3f", param_grid$drop_left, param_grid$drop_right))
    param_dirs <- file.path(run_dir, param_dirs)
    
    purrr::walk(param_dirs, function(dir_) {
      chunk_files <- list.files(dir_, pattern = "entropy_chunk_.*\\.rds", full.names = TRUE)
      if (length(chunk_files) > 0) {
        combined <- purrr::map_dfr(chunk_files, readRDS)
        saveRDS(combined, file.path(dir_, "entropy.rds"))
        file.remove(chunk_files)
      }
    })
    
    message(sprintf("\nILP Entropy calculation completed! Results in: %s", run_dir))
    return(run_dir)
  })
}


# --------------------------- MAIN: DATA + RUN -------------------------------- #

# 1) Load MECO words
load("data/meco_L2_trimmed.rda")
MECO_WORDS <- joint.data %>%
  rename(word = ia) %>%
  select(word) %>%
  mutate(word = str_to_lower(word)) %>%
  filter(str_detect(word, glob2rx("*'s"), negate = TRUE)) %>%
  mutate(word = str_replace_all(word, sprintf("[^%s]", str_flatten(get_alphabet(LANGUAGE))), "")) %>%
  distinct(word) %>%
  filter(nchar(word) %in% LENGTHS) %>%
  pull(word)
rm(joint.data)

# 2) Load OpenSubtitles word data
WORD_DATA <- read_tsv(sprintf("./data/opensubtitles_unigrams_%s.tsv", LANGUAGE),
                      locale = locale(encoding = "UTF-8")) %>%
  rename(word = unigram,
         count = unigram_freq) %>%
  mutate(freq = count / get_corpus_size(LANGUAGE),
         length = nchar(word)) %>%
  filter(freq >= MIN_FREQ,
         length %in% LENGTHS,
         str_detect(word, sprintf("[^%s]", str_flatten(get_alphabet(LANGUAGE))), negate = TRUE)) %>%
  select(-count)

# 3) Actually compute ILP entropy for the given range of left/right drops
plan(multisession)
compute_entropy_lr(
  language   = LANGUAGE,
  left_drops = LEFT_DROPS,
  right_drops= RIGHT_DROPS,
  min_freq   = MIN_FREQ,
  lengths    = LENGTHS,
  word_data  = WORD_DATA,
  meco_words = MECO_WORDS
)
plan(sequential)

message("\nEntropy computation done!\n")