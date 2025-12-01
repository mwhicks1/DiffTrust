try:
    from google import genai
    from google.genai import types
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    genai = None
    types = None

from difftrust.config import config
from difftrust.llm.abstract import LLM
from difftrust.llm.chat import Chat, Msg

# Getting client
failed_msg = None
if GOOGLE_AVAILABLE:
    try:
        googleai_client = genai.Client(api_key=config()["llm"]["google_ai_api_key"])
    except Exception as exception:
        failed_msg = f"Something went wrong when charging gemini client : {exception}"
else:
    failed_msg = "Google Generative AI package not installed"


class Gemini(LLM):
    def __init__(self, model: str = "gemini-2.0-flash", temperature: float = 0.0):
        super().__init__(model, temperature)
        self.model = model

    def run(self, chat: Chat) -> Msg:
        if failed_msg is not None:
            raise Exception(failed_msg)
        contents = [
            types.Content(
                role=msg.writer,
                parts=[types.Part.from_text(text=msg.content)]
            )
            for msg in chat.chatstream if msg.writer != "system"
        ]
        system_instruction = [types.Part.from_text(text=msg.content) for msg in chat.chatstream if
                              msg.writer == "system"]

        generate_content_config = types.GenerateContentConfig(
            temperature=self.temperature,
            response_mime_type="text/plain",
            system_instruction=system_instruction
        )

        answer = googleai_client.models.generate_content(
            model=self.model,
            contents=contents,
            config=generate_content_config

        )
        return Msg("model", answer.text)


gemini_2_5_pro = Gemini(model="gemini-2.5-pro-preview-05-06", temperature=0.6)
gemini_2_5_flash = Gemini(model="gemini-2.5-flash-preview-05-20", temperature=0.6)
gemini_2_0_flash = Gemini(model="gemini-2.0-flash", temperature=0.6)
gemini_2_0_flash_lite = Gemini(model="gemini-2.0-flash-lite", temperature=0.6)
gemini_1_5_pro = Gemini(model="gemini-1.5-pro", temperature=0.6)
gemini_1_5_flash = Gemini(model="gemini-1.5-flash", temperature=0.6)
