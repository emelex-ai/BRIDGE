# 2025-07-10

# Test cases
For each case there are two files: 
- the benchmark file
- the model used
In two cases, the benchmarks are running existing models from 
torch.nn, which I did not copy over to my local folder. 

# List of cases
- benchmark_classical_full_attention.py                  
- benchmark_classical_windowed_full_attention.py         

- fast_sliding_window_model.py
- benchmark_fast_sliding_window.py                       

- true_vectorized_sliding_window_model.py
- benchmark_true_vectorized_sliding_window.py

- true_vectorized_sliding_window_outer_loop_model.py
- benchmark_true_vectorized_sliding_window_outer_loop.py

- sdpa_full_attention_model.py
- benchmark_sdpa_full_attention.py                       

- sdpa_sliding_window_model.py
- benchmark_sdpa_sliding_window.py                       

- chunked_vectorized_sliding_window_model.py
- benchmark_chunked_vectorized_sliding_window.py         
