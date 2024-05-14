import os
import sys
from server.logger.logger_config import my_logger as logger


def check_env_variables():
    # LLM_NAME: Name of the language model being used, should be 'OpenAI' or 'ZhipuAI' or 'Ollama'.
    LLM_NAME = os.getenv('LLM_NAME')
    llm_name_list = ['OpenAI', 'ZhipuAI', 'Ollama']
    if LLM_NAME not in llm_name_list:
        logger.error(f"LLM_NAME: '{LLM_NAME}' is illegal! Must be in {llm_name_list}.")
        sys.exit(-1)

    if LLM_NAME == 'OpenAI':
        # OPENAI_API_KEY: API key for accessing OpenAI's services.
        OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        if OPENAI_API_KEY == 'xxxx':
            logger.error(f"OPENAI_API_KEY: '{OPENAI_API_KEY}' is illegal!")
            sys.exit(-1)

        # GPT_MODEL_NAME: Specific GPT model being used, e.g., 'gpt-3.5-turbo' or 'gpt-4-torbo'.
        GPT_MODEL_NAME = os.getenv('GPT_MODEL_NAME')
        if GPT_MODEL_NAME not in ['gpt-3.5-turbo', 'gpt-4-turbo']:
            logger.error(f"GPT_MODEL_NAME: '{GPT_MODEL_NAME}' is illegal! Must be 'gpt-3.5-turbo' or 'gpt-4-turbo'")
            sys.exit(-1)
    elif LLM_NAME == 'ZhipuAI':
        # ZHIPUAI_API_KEY: API key for accessing ZhipuAI's services.
        ZHIPUAI_API_KEY = os.getenv('ZHIPUAI_API_KEY')
        if ZHIPUAI_API_KEY == 'xxxx':
            logger.error(f"ZHIPUAI_API_KEY: '{ZHIPUAI_API_KEY}' is illegal!")
            sys.exit(-1)

        # GLM_MODEL_NAME: Specific GLM model being used, e.g., 'glm-3-turbo' or 'glm-4'.
        GLM_MODEL_NAME = os.getenv('GLM_MODEL_NAME')
        if GLM_MODEL_NAME not in ['glm-3-turbo', 'glm-4']:
            logger.error(f"GLM_MODEL_NAME: '{GLM_MODEL_NAME}' is illegal! Must be 'glm-3-turbo' or 'glm-4'")
            sys.exit(-1)
    else:
        # OLLAMA_MODEL_NAME: Specific Ollma model being used, e.g., 'llama3', 'llama3:70b', 'phi3', 'mistral', etc.
        #OLLAMA_MODEL_NAME = os.getenv('OLLAMA_MODEL_NAME')
        OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL')
        if not OLLAMA_BASE_URL.startswith('http://') and not OLLAMA_BASE_URL.startswith('https://'):
            logger.error(f"OLLAMA_BASE_URL: '{OLLAMA_BASE_URL}' is illegal! It must start with 'http://' or 'https://'")
            sys.exit(-1)

    # MIN_RELEVANCE_SCORE: Minimum score for a document to be considered relevant, and will be used in prompt, between 0.3 and 0.7.
    MIN_RELEVANCE_SCORE = os.getenv('MIN_RELEVANCE_SCORE')
    try:
        min_relevance_score = float(MIN_RELEVANCE_SCORE)
        if min_relevance_score < 0.3 or min_relevance_score > 0.7:
            logger.error(f"MIN_RELEVANCE_SCORE: {MIN_RELEVANCE_SCORE} is illegal! It should be a float number between [0.3~0.7]")
            sys.exit(-1)
    except Exception as e:
        logger.error(f"MIN_RELEVANCE_SCORE: {MIN_RELEVANCE_SCORE} is illegal! It should be a float number between [0.3~0.7]")
        sys.exit(-1)

    # BOT_TOPIC: Main topic or domain the bot is designed to handle, like 'OpenIM' or 'LangChain'.
    BOT_TOPIC = os.getenv('BOT_TOPIC')
    if BOT_TOPIC == 'xxxx':
        logger.error(f"BOT_TOPIC: '{BOT_TOPIC}' is illegal! You must set your own bot topic, such as 'OpenIM' or 'LangChain', etc.")
        sys.exit(-1)

    # URL_PREFIX: The prefix URL for accessing media and other resources, must start with 'http://' or 'https://'.
    URL_PREFIX = os.getenv('URL_PREFIX')
    if not URL_PREFIX.startswith('http://') and not URL_PREFIX.startswith('https://'):
        logger.error(f"URL_PREFIX: '{URL_PREFIX}' is illegal! It must start with 'http://' or 'https://'")
        sys.exit(-1)

    # USE_PREPROCESS_QUERY: Flag (0 or 1) indicating whether preprocessing should be applied to queries.
    USE_PREPROCESS_QUERY = os.getenv('USE_PREPROCESS_QUERY')
    try:
        use_preprocess_query = int(USE_PREPROCESS_QUERY)
        if use_preprocess_query not in [0, 1]:
            logger.error(f"USE_PREPROCESS_QUERY: {USE_PREPROCESS_QUERY} is illegal! It should be 0 or 1!")
            sys.exit(-1)
    except Exception as e:
        logger.error(f"USE_PREPROCESS_QUERY: {USE_PREPROCESS_QUERY} is illegal! It should be 0 or 1!")
        sys.exit(-1)

    # USE_RERANKING: Flag (0 or 1) indicating whether reranking should be applied to search results.
    USE_RERANKING = os.getenv('USE_RERANKING')
    try:
        use_reranking = int(USE_RERANKING)
        if use_reranking not in [0, 1]:
            logger.error(f"USE_RERANKING: {USE_RERANKING} is illegal! It should be 0 or 1!")
            sys.exit(-1)
    except Exception as e:
        logger.error(f"USE_RERANKING: {USE_RERANKING} is illegal! It should be 0 or 1!")
        sys.exit(-1)

    # USE_DEBUG: Flag (0 or 1) indicating whether to output additional debug information, such as `search`, `reranking`, `prompt`.
    USE_DEBUG = os.getenv('USE_DEBUG')
    try:
        use_debug = int(USE_DEBUG)
        if use_debug not in [0, 1]:
            logger.error(f"USE_DEBUG: {USE_DEBUG} is illegal! It should be 0 or 1!")
            sys.exit(-1)
    except Exception as e:
        logger.error(f"USE_DEBUG: {USE_DEBUG} is illegal! It should be 0 or 1!")
        sys.exit(-1)
