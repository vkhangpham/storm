python run_costorm_gemini.py \
    --retriever tavily \
    --output-dir ./results/gpt4o/tavily \
    --max_search_queries 2 \
    --max_search_queries_per_turn 1 \
    --retrieve_top_k 5 \
    --total_conv_turn 3 \
    --warmstart_max_turn_per_experts 3 \
    --warmstart_max_num_experts 2 \
    --max_num_round_table_experts 2
