"""
Co-STORM pipeline powered by GPT-4o/4o-mini and Bing search engine.
You need to set up the following environment variables to run this script:
    - OPENAI_API_KEY: OpenAI API key
    - OPENAI_API_TYPE: OpenAI API type (e.g., 'openai' or 'azure')
    - AZURE_API_BASE: Azure API base URL if using Azure API
    - AZURE_API_VERSION: Azure API version if using Azure API
    - BING_SEARCH_API_KEY: Biang search API key; BING_SEARCH_API_KEY: Bing Search API key, SERPER_API_KEY: Serper API key, BRAVE_API_KEY: Brave API key, or TAVILY_API_KEY: Tavily API key

Output will be structured as below
args.output_dir/
    log.json           # Log of information-seeking conversation
    report.txt         # Final article generated
"""

import os
import json
import time
import logging
from argparse import ArgumentParser
from knowledge_storm.collaborative_storm.engine import CollaborativeStormLMConfigs, RunnerArgument, CoStormRunner
from knowledge_storm.collaborative_storm.modules.callback import LocalConsolePrintCallBackHandler
from knowledge_storm.lm import OpenAIModel, AzureOpenAIModel, GoogleModel
from knowledge_storm.logging_wrapper import LoggingWrapper
from knowledge_storm.rm import YouRM, BingSearch, BraveRM, SerperRM, DuckDuckGoSearchRM, TavilySearchRM, SearXNG
from knowledge_storm.utils import load_api_key


def main(args, topic, subject):
    # load_api_key(toml_file_path='secrets.toml')
    lm_config: CollaborativeStormLMConfigs = CollaborativeStormLMConfigs()
    
    openai_kwargs = {
        "api_key": os.getenv("OPENAI_API_KEY"),
        "api_provider": "openai",
        "temperature": 1.0,
        "top_p": 0.9,
        "api_base": None,
    } if os.getenv('OPENAI_API_TYPE') == 'openai' else {
        "api_key": os.getenv("AZURE_API_KEY"),
        "temperature": 1.0,
        "top_p": 0.9,
        "api_base": os.getenv("AZURE_API_BASE"),
        "api_version": os.getenv("AZURE_API_VERSION"),
    }
    
    GPTModelClass = OpenAIModel if os.getenv('OPENAI_API_TYPE') == 'openai' else AzureOpenAIModel
    gpt_4o_mini_model_name = 'gpt-4o-mini'
    gpt_4o_model_name = 'gpt-4o'
    if os.getenv('OPENAI_API_TYPE') == 'azure':
        openai_kwargs['api_base'] = os.getenv('AZURE_API_BASE')
        openai_kwargs['api_version'] = os.getenv('AZURE_API_VERSION')

    GeminiModelClass = GoogleModel
    gemini_model_name = 'gemini-1.5-flash'
    gemini_kwargs = {}

    # question_answering_lm = GeminiModelClass(model=gemini_model_name, max_tokens=1000, **gemini_kwargs)
    # discourse_manage_lm = GeminiModelClass(model=gemini_model_name, max_tokens=500, **gemini_kwargs)
    # utterance_polishing_lm = GeminiModelClass(model=gemini_model_name, max_tokens=2000, **gemini_kwargs)
    # warmstart_outline_gen_lm = GPTModelClass(model=gpt_4o_mini_model_name, max_tokens=500, **openai_kwargs)
    # warmstart_outline_gen_lm = GeminiModelClass(model=gemini_model_name, max_tokens=500, **gemini_kwargs)
    # question_asking_lm = GeminiModelClass(model=gemini_model_name, max_tokens=300, **gemini_kwargs)
    # knowledge_base_lm = GPTModelClass(model=gpt_4o_mini_model_name, max_tokens=1000, **openai_kwargs)
    # knowledge_base_lm = GeminiModelClass(model=gemini_model_name, max_tokens=1000, **gemini_kwargs)
    question_answering_lm = GPTModelClass(model=gpt_4o_mini_model_name, max_tokens=1000, **openai_kwargs)
    discourse_manage_lm = GPTModelClass(model=gpt_4o_mini_model_name, max_tokens=500, **openai_kwargs)
    utterance_polishing_lm = GPTModelClass(model=gpt_4o_mini_model_name, max_tokens=2000, **openai_kwargs)
    warmstart_outline_gen_lm = GPTModelClass(model=gpt_4o_model_name, max_tokens=500, **openai_kwargs)
    question_asking_lm = GPTModelClass(model=gpt_4o_mini_model_name, max_tokens=300, **openai_kwargs)
    knowledge_base_lm = GPTModelClass(model=gpt_4o_model_name, max_tokens=1000, **openai_kwargs)
    lm_config.set_question_answering_lm(question_answering_lm)
    lm_config.set_discourse_manage_lm(discourse_manage_lm)
    lm_config.set_utterance_polishing_lm(utterance_polishing_lm)
    lm_config.set_warmstart_outline_gen_lm(warmstart_outline_gen_lm)
    lm_config.set_question_asking_lm(question_asking_lm)
    lm_config.set_knowledge_base_lm(knowledge_base_lm)

    runner_argument = RunnerArgument(
        topic=topic,
        retrieve_top_k=args.retrieve_top_k,
        max_search_queries=args.max_search_queries,
        total_conv_turn=args.total_conv_turn,
        max_search_thread=args.max_search_thread,
        max_search_queries_per_turn=args.max_search_queries_per_turn,
        warmstart_max_num_experts=args.warmstart_max_num_experts,
        warmstart_max_turn_per_experts=args.warmstart_max_turn_per_experts,
        warmstart_max_thread=args.warmstart_max_thread,
        max_thread_num=args.max_thread_num,
        max_num_round_table_experts=args.max_num_round_table_experts,
        moderator_override_N_consecutive_answering_turn=args.moderator_override_N_consecutive_answering_turn,
        node_expansion_trigger_count=args.node_expansion_trigger_count)
    logging_wrapper = LoggingWrapper(lm_config)
    callback_handler = LocalConsolePrintCallBackHandler() if args.enable_log_print else None

    match args.retriever:
        case 'bing':
            rm = BingSearch(bing_search_api=os.getenv('BING_SEARCH_API_KEY'), k=runner_argument.retrieve_top_k)
        case 'you':
             rm = YouRM(ydc_api_key=os.getenv('YDC_API_KEY'), k=runner_argument.retrieve_top_k)
        case 'brave':
            rm = BraveRM(brave_search_api_key=os.getenv('BRAVE_API_KEY'), k=runner_argument.retrieve_top_k)
        case 'duckduckgo':
            rm = DuckDuckGoSearchRM(k=runner_argument.retrieve_top_k, safe_search='On', region='us-en')
        case 'serper':
            rm = SerperRM(serper_search_api_key=os.getenv('SERPER_API_KEY'), query_params={'autocorrect': True, 'num': 10, 'page': 1})
        case 'tavily':
            rm = TavilySearchRM(tavily_search_api_key=os.getenv('TAVILY_API_KEY'), k=runner_argument.retrieve_top_k, include_raw_content=True)
        case 'searxng':
            rm = SearXNG(searxng_api_key=os.getenv('SEARXNG_API_KEY'), k=runner_argument.retrieve_top_k)
        case _:
             raise ValueError(f'Invalid retriever: {args.retriever}. Choose either "bing", "you", "brave", "duckduckgo", "serper", "tavily", or "searxng"')

    costorm_runner = CoStormRunner(lm_config=lm_config,
                                   runner_argument=runner_argument,
                                   logging_wrapper=logging_wrapper,
                                   rm=rm,
                                   callback_handler=callback_handler)

    costorm_runner.warm_start()
    # print("Knowledge base after warm start:")
    # costorm_runner.print_knowledge_base()
    # print("="*50)

    user_utterance = f"I want to know about the timeline and milestones of research work in {subject}"
    costorm_runner.step(user_utterance=user_utterance)
    
    user_utterance = f"What's about the cases where research in {subject} is applied in the real world? Find some concrete examples and prioritize the most famous ones."
    costorm_runner.step(user_utterance=user_utterance)
    
    for _ in range(runner_argument.total_conv_turn):
        costorm_runner.step()
        # print("Knowledge base after this turn:")
        # costorm_runner.print_knowledge_base()
        # print("="*50)    

    costorm_runner.knowledge_base.reogranize()
    # print("Final knowledge base:")
    # costorm_runner.print_knowledge_base()
    # print("="*50)

    try:
        article = costorm_runner.generate_report()
        if article is None:
            raise ValueError("generate_report() returned None")
    except Exception as e:
        print(f"Error generating report: {str(e)}")
        article = "Error: Unable to generate report."

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, subject.lower().replace(' ', '_')), exist_ok=True)

    with open(os.path.join(args.output_dir, subject.lower().replace(' ', '_'), "report.md"), "w") as f:
        f.write(article)
        
    costorm_runner.save_mind_map(os.path.join(args.output_dir, subject.lower().replace(' ', '_'), "mind_map.json"))

    instance_copy = costorm_runner.to_dict()
    with open(os.path.join(args.output_dir, subject.lower().replace(' ', '_'), "instance_dump.json"), "w") as f:
        json.dump(instance_copy, f, indent=2)

    log_dump = costorm_runner.dump_logging_and_reset()
    with open(os.path.join(args.output_dir, subject.lower().replace(' ', '_'), "log.json"), "w") as f:
        json.dump(log_dump, f, indent=2)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--output-dir', type=str, default='./results/co-storm',
                        help='Directory to store the outputs.')
    parser.add_argument('--retriever', type=str, choices=['bing', 'you', 'brave', 'serper', 'duckduckgo', 'tavily', 'searxng'],
                        help='The search engine API to use for retrieving information.')
    parser.add_argument(
        '--retrieve_top_k',
        type=int,
        default=5,
        help='Retrieve top k results for each query in retriever.'
    )
    parser.add_argument(
        '--max_search_queries',
        type=int,
        default=2,
        help='Maximum number of search queries to consider for each question.'
    )
    parser.add_argument(
        '--total_conv_turn',
        type=int,
        default=1,
        help='Maximum number of turns in conversation.'
    )
    parser.add_argument(
        '--max_search_thread',
        type=int,
        default=5,
        help='Maximum number of parallel threads for retriever.'
    )
    parser.add_argument(
        '--max_search_queries_per_turn',
        type=int,
        default=2,
        help='Maximum number of search queries to consider in each turn.'
    )
    parser.add_argument(
        '--warmstart_max_num_experts',
        type=int,
        default=2,
        help='Max number of experts in perspective-guided QA during warm start.'
    )
    parser.add_argument(
        '--warmstart_max_turn_per_experts',
        type=int,
        default=2,
        help='Max number of turns per perspective during warm start.'
    )
    parser.add_argument(
        '--warmstart_max_thread',
        type=int,
        default=3,
        help='Max number of threads for parallel perspective-guided QA during warm start.'
    )
    parser.add_argument(
        '--max_thread_num',
        type=int,
        default=10,
        help=("Maximum number of threads to use. "
              "Consider reducing it if you keep getting 'Exceed rate limit' errors when calling the LM API.")
    )
    parser.add_argument(
        '--max_num_round_table_experts',
        type=int,
        default=2,
        help='Max number of active experts in round table discussion.'
    )
    parser.add_argument(
        '--moderator_override_N_consecutive_answering_turn',
        type=int,
        default=3,
        help=('Number of consecutive expert answering turns before the moderator overrides the conversation.')
    )
    parser.add_argument(
        '--node_expansion_trigger_count',
        type=int,
        default=10,
        help='Trigger node expansion for nodes that contain more than N snippets.'
    )
    parser.add_argument(
        '--enable_log_print',
        action='store_true',
        help='If set, enable console log pprint.'
    )

    args = parser.parse_args()

    # export args to a file
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "params.json"), "w") as f:
        json.dump(args.__dict__, f, indent=4)

    subjects = [
        "Data Structure",
        # "Computer Science",
        # "Human-Computer Interaction",
        # "Machine Learning",
        # "Explainable AI",
        # "Natural Language Processing",
        # "Computer Vision",
        # "Generative AI",
        # "Supervised Learning",
        # "Unsupervised Learning",
        # "Reinforcement Learning",
        # "Self-Supervised Learning",
        # "Transfer Learning",
        # "Deep Learning",
        # "Query Optimization",
        # "Prompt Engineering",
        # "Retrieval Augmented Generation",
        # "Natural Language to SQL",
        # "Convolutional Neural Network",
        # "Long Short-Term Memory",
        # "Hopfield Network"
    ]

        # "social science",
        # "ecology",
        # "religious studies",
        # "political science",
        # "communication",
        # "comparative literature",
        # "neurology",
        # "family medicine",
        # "microbiology",
        # "art history",
        # "african american studies",
        # "planetary sciences",
        # "dermatology",
        # "education",
        # "developmental biology",
        # "astrophysics",
        # "forensic science",
        # "cancer research",
        # "sarcoma",
        # "functional data analysis",
        # "bacteria",
        # "longevity",
        # "bias",
        # "marketing strategy",
        # "chronic pain",
        # "child-computer interaction",
        # "environmental technology",
        # "intelligent computing",
        # "liver physiology",
        # "tourism research",
        # "screen medium",
        # "theoretical economics",
        # "hydraulic research",
        # "field archaeology",
        # "resource policy",
    # ]

    start_time = time.time()
    for subject in subjects:
        print(f"Generating content for: {subject}")
        # topic = f"What is the real-world application of {subject}? And what are the main history of {subject}?"
        # topic = f"The real-world application of {subject}"
        # topic = f"The outline of {subject}"
        topic = subject
        main(args, topic, subject)
        print(f"Finished generating content for: {subject}\n")
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
