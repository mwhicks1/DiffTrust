import difftrust.llm.abstract as abstract
import difftrust.llm.chat as chat
import difftrust.llm.gemini as gemini
import difftrust.llm.chatgpt as chatgpt
import difftrust.llm.claude as claude
import difftrust.llm.open_router as open_router
from difftrust.llm.chat import _set_log_function as set_log_function

_llm_dict = {
    "gemini_2_5_flash": gemini.gemini_2_5_flash,
    "gemini_2_0_flash_lite": gemini.gemini_2_0_flash_lite,
    "gemini_2_0_flash": gemini.gemini_2_0_flash,
    "gemini_2_5_pro": gemini.gemini_2_5_pro,
    "gemini_1_5_pro": gemini.gemini_1_5_pro,
    "gemini_1_5_flash": gemini.gemini_1_5_flash,
    "gpt_3_5": chatgpt.gpt_3_5,
    "gpt_4": chatgpt.gpt_4,
    "gpt_o4_mini": chatgpt.gpt_o4_mini,
    "gpt_4_turbo": chatgpt.gpt_4_turbo,
    "gpt_4_preview": chatgpt.gpt_4_preview,
    "gpt_4o": chatgpt.gpt_4o,
    "gpt_o1": chatgpt.gpt_o1,
    "gpt_o3_mini": chatgpt.gpt_o3_mini,
    "claude_opus_4": claude.claude_opus_4,
    "claude_sonnet_4": claude.claude_sonnet_4,
    "claude_3_7_sonnet": claude.claude_3_7_sonnet,
    "claude_3_5_sonnet": claude.claude_3_5_sonnet,
    "claude_3_5_haiku": claude.claude_3_5_haiku,
    "claude_3_opus": claude.claude_3_opus,
    "claude_3_haiku": claude.claude_3_haiku,
    "deepseek_r1": open_router.deepseek_r1,
    "deepseek_v3_0324": open_router.deepseek_v3_0324,
    "llama_3_3_70b_instruct": open_router.llama_3_3_70b_instruct,
    "llama_4_maverick_17b": open_router.llama_4_maverick_17b,
    "devstral_small": open_router.devstral_small,
    "qwen3_14b": open_router.qwen3_14b,
    "gemma_3_1b_it": open_router.gemma_3_1b_it,
    "ministral_8b": open_router.ministral_8b,
    "llama_3_1_8b_instruct": open_router.llama_3_1_8b_instruct,
    "qwen3_8b": open_router.qwen3_8b
}

general_models = {
    "gpt_4",
    "deepseek_r1",
    "gemini_2_5_flash",
    "claude_opus_4",
    "llama_3_3_70b_instruct"
}

coding_models = {
    "gpt_o4_mini",
    "deepseek_v3_0324",
    "gemini_2_5_pro",
    "claude_3_7_sonnet",
    "llama_4_maverick_17b"
}

medium_sized_models = {
    "devstral_small",
    "qwen3_14b",
    "gemma_3_1b_it"
}

small_sized_models = {
    "ministral_8b",
    "llama_3_1_8b_instruct",
    "qwen3_8b"
}


def available():
    return list(_llm_dict.keys())


def get_llm_by_name(name: str):
    try:
        return _llm_dict[name]
    except KeyError:
        raise Exception(f"LLM {name} is not available. The available llms are {available()}")


def test():
    set_log_function(print)
    for name in small_sized_models:
        llm = _llm_dict[name]
        print(llm.name)
        chat_ = chat.Chat(llm, "You are a helpful assistant.")
        chat_.ask("Hi ! My name is George !")
        chat_.ask("Do you remember my name ?")
    set_log_function(lambda x: None)


if __name__ == "__main__":
    test()
