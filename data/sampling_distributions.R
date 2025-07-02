

ewfg_all = read_xlsx("raw/ewfg.xlsx") %>% 
  select(-c(sfi, d, u, f)) %>% 
  pivot_longer(cols = -word, names_to = "level", values_to = "frequency") %>% 
  filter(level %in% c("gr1", "gr2")) %>% 
  filter(!is.na(frequency)) %>% 
  mutate(frequency = frequency + 1) %>% 
  group_by(word) %>% 
  summarise(frequency = sum(frequency)) %>% 
  ungroup() %>% 
  mutate(length = str_length(word))

tmp_lengths = ewfg_all %>% 
  group_by(length) %>% 
  summarise(length_count = n()) %>% 
  ungroup() %>% 
  mutate(length_prob_raw = length_count/sum(length_count))

ewfg_all = ewfg_all %>% 
  left_join(tmp_lengths) %>% 
  mutate(frequency_prob = frequency/sum(frequency)) %>% 
  mutate(length_prob = length_prob_raw/length_count)


num_samples = 500
df = tibble(iter = seq(num_samples),
            frequency_sample_types = rep(NA, num_samples),
            length_sample_types = rep(NA, num_samples))

n = 65000
for (i in seq(nrow(df))){
  
  frequency_sample = ewfg_all %>% 
    slice_sample(n = n, weight_by = frequency_prob, replace = T)
  
  length_sample = ewfg_all %>% 
    slice_sample(n = n, weight_by = length_prob, replace = T)
  
  df$frequency_sample_types[i] = length(unique(frequency_sample$word))
  df$length_sample_types[i] = length(unique(length_sample$word))
  
}


df %>% 
  pivot_longer(cols = -iter, names_to = "sampling_method", values_to = "num_types") %>% 
  ggplot(aes(sampling_method, num_types, fill = sampling_method)) +
  geom_bar(stat = "summary") +
  geom_point(position = position_jitter(width = .02), size = .001) +
  labs(x = "Sampling method",
       y = "Number unique words") +
  theme_minimal() +
  coord_cartesian(ylim = c(4500, 15000))
  
    