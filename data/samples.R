set.seed(123) 
# Define proportions (between 0 and 1, exclusive)
proportions_all = sort(unique(finetuning_ids$proportion_background))

proportions = proportions_all[sort(proportions_all) %nin% c(0, 1)]

# Store sampled indices
sampled_indices <- list()

# Initialize previous samples per doc_id
doc_ids <- unique(into_reading_sequentially_arranged$doc_id)
previous_samples <- setNames(vector("list", length(doc_ids)), doc_ids)

# Loop through proportions
for (p in proportions) {
  current_sample <- into_reading_sequentially_arranged %>%
    group_by(doc_id) %>%
    group_map(~{
      indices <- .x$word_index_in_program
      prev <- previous_samples[[.y$doc_id]]
      n_to_sample <- ceiling(length(indices) * p)
      if (length(prev) >= n_to_sample) {
        sampled <- prev[1:n_to_sample]
      } else {
        additional <- setdiff(indices, prev)
        sampled <- c(prev, sample(additional, n_to_sample - length(prev)))
      }
      previous_samples[[.y$doc_id]] <<- sampled
      return(sampled)
    }) %>%
    unlist() %>%
    sort()
  
  sampled_indices[[paste0("p", p)]] <- current_sample
}



# Examine number of indices in each sample
sapply(sampled_indices, length)

for (e in names(sampled_indices)){
  
  print(length(sampled_indices[[e]])/max_index) # max_index set in create_finetuning_sets.Rmd
  
  
}



samples = tibble(p0 = rep(FALSE, max_index),
       p1 = rep(TRUE, max_index))

samples$word_index_in_program = into_reading_sequentially_arranged$word_index_in_program
samples$doc_id = into_reading_sequentially_arranged$doc_id

for (sample_ in names(sampled_indices)){
  
  targets = sampled_indices[[sample_]]
  samples[, sample_] = rep(FALSE, max_index)
  samples[[sample_]][targets] = TRUE 
  
  
}


samples_ = samples %>% 
  pivot_longer(cols = -c(word_index_in_program, doc_id), names_to = "proportion_background", values_to = "sampled_index") %>% 
  mutate(proportion_background = as.numeric(str_replace(proportion_background, "p", "")))



# TESTS
## test that all lower values of proportion_background contain indices that are in the next larger value for proportion_background
iters = seq(length(proportions_all))

for (i in iters){
  
  if (i < 11){
  
  cat("######################################################\n")
  cat(paste("Processing target proportion:", proportions_all[i]), "\n")
  tmp = samples_ %>% 
    filter(proportion_background == proportions_all[i])
  
  current_sample_indices = tmp %>% 
    filter(sampled_index) %>% 
    pull(word_index_in_program)
  
  observed_proportion_current_sample = length(current_sample_indices)/max_index
  
  cat(paste("Observed proportion of current sample (should ~target proportion):", observed_proportion_current_sample), "\n")
  stopifnot(proportions_all[i] == round(observed_proportion_current_sample, digits = 1))
  
  for (j in iters[iters > i]){
    
    next_sample_indices = samples_ %>% 
      filter(proportion_background == proportions_all[j]) %>% 
      filter(sampled_index) %>%
      pull(word_index_in_program)
    
    indices_missing_in_next_sample = current_sample_indices[current_sample_indices %nin% next_sample_indices]
    observed_proportion_next_sample = length(next_sample_indices)/max_index
    
    
    stopifnot(length(indices_missing_in_next_sample)==0)
    stopifnot(proportions_all[j] == round(observed_proportion_next_sample, digits = 1))

    cat(paste("  --Comparing with proportion background:", proportions_all[j]), "\n")
    cat(paste("  --Observed proportion of comparison sample (should ~value above):", observed_proportion_next_sample), "\n")
    cat(paste("  --Are all the sampled indices in the next sample:", all(current_sample_indices %in% next_sample_indices)), "\n")
    cat(paste("  --Number indices missing in next larger sample:", length(indices_missing_in_next_sample)), "\n")
    cat("  ---------------------------------------------------\n")
    
  }
  cat("Proportion", proportions_all[i], "...tests passed\n")
  }} 


# all cells to the right should be TRUE if they are included in the next larger sample for background vocabulary
check_true_right <- function(df) {
  apply(df, 1, function(row) {
    true_indices <- which(row == TRUE)
    if (length(true_indices) == 0) return(TRUE)
    all(row[seq(min(true_indices), length(row))] == TRUE)
  })
}

# This will return a logical vector indicating which rows satisfy the condition

samples = samples %>% 
  select(word_index_in_program, doc_id, p0, p0.1, p0.2, p0.3, p0.4, p0.5, p0.6, p0.6, p0.7, p0.8, p0.9, p1)

check_true_right_ <- check_true_right(samples)

stopifnot(all(check_true_right_))

# To see which rows fail the condition
cat("  How many sampled inidices fail the test:\n")
cat("  .....", length(which(!check_true_right_)), "\n")


stopifnot(nrow(samples) == nrow(into_reading_sequentially_arranged))
stopifnot(all.equal(samples$word_index_in_program, into_reading_sequentially_arranged$word_index_in_program))
cat("<<<<<< ALL PASSED >>>>>>\n")

cat(" Writing samples payload for documentation...\n")


# remove potentially problematic variables
rm(sampled_indices, doc_ids, previous_samples, e, samples_, iters, 
   i, tmp, current_sample_indices, observed_proportion_current_sample, j, next_sample_indices, indices_missing_in_next_sample, observed_proportion_next_sample, 
   check_true_right_, check_true_right, proportions, proportions_all)
