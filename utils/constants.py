GPT_4o = "gpt-4o"

type_mapping = {
    "number": float,
    "string": str,
    "array": list,
    'object': object
}


TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"
INPUTS_DESC_VECTOR_STORE_PATH = "vectorstores/inputs_desc_vector_store"
OUTPUT_DESC_VECTOR_STORE_PATH = "vectorstores/outputs_desc_vector_store"
FUNC_DESC_VECTOR_STORE_PATH = "vectorstores/func_desc_vector_store"
NAME_TO_FUNCTION_JSON_PATH = "function_registry/name_to_function.json"

# prompt paths
PLANNING_AGENT_PROMPT = "agent_prompts/planning_agent.txt"
EXTRACT_RETRY_AGENT_SYSTEM_PROMPT = "agent_prompts/extract_retry_agent_system_prompt.txt"
EXTRACT_RETRY_AGENT_USER_PROMPT = "agent_prompts/extract_retry_agent_user_prompt.txt"
ANSWER_QUESTION_SYSTEM_PROMPT = "agent_prompts/answer_question_system_prompt.txt"
ANSWER_QUESTION_USER_PROMPT = "agent_prompts/answer_question_user_prompt.txt"

# mode
PLANNING = "planning"
EXECUTE = "execute"
