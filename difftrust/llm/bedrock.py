import boto3
from difftrust.config import config
from difftrust.llm.abstract import LLM
from difftrust.llm.chat import Chat, Msg


class BedrockModel(LLM):
    def __init__(self, model_id: str, temperature: float = 0.6, max_tokens: int = 4096):
        super().__init__(model_id, temperature)
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.failed_msg = None
        try:
            self.client = boto3.client('bedrock-runtime', region_name=config()["llm"].get("bedrock_region", "us-east-1"))
        except Exception as exception:
            self.failed_msg = f"Failed to initialize Bedrock client: {exception}"

    def run(self, chat: Chat) -> Msg:
        if self.failed_msg is not None:
            raise Exception(self.failed_msg)
        
        messages = []
        system_prompt = None
        
        for msg in chat.chatstream:
            if msg.writer == "system":
                system_prompt = msg.content
            else:
                messages.append({"role": msg.writer, "content": [{"text": msg.content}]})
        
        request_body = {
            "messages": messages,
            "inferenceConfig": {
                "temperature": self.temperature,
                "maxTokens": self.max_tokens
            }
        }
        
        if system_prompt:
            request_body["system"] = [{"text": system_prompt}]
        
        response = self.client.converse(modelId=self.model_id, **request_body)
        content = response['output']['message']['content'][0]['text']
        return Msg("assistant", content)


# Claude models via Bedrock
claude_4_5_sonnet_bedrock = BedrockModel(model_id="us.anthropic.claude-4-5-sonnet-20250514-v1:0", temperature=0.6)
claude_3_7_sonnet_bedrock = BedrockModel(model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0", temperature=0.6)
claude_3_5_sonnet_bedrock = BedrockModel(model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0", temperature=0.6)
claude_3_5_haiku_bedrock = BedrockModel(model_id="us.anthropic.claude-3-5-haiku-20241022-v1:0", temperature=0.6)
claude_3_opus_bedrock = BedrockModel(model_id="anthropic.claude-3-opus-20240229-v1:0", temperature=0.6)

# Nova models
nova_pro_bedrock = BedrockModel(model_id="us.amazon.nova-pro-v1:0", temperature=0.6)
nova_lite_bedrock = BedrockModel(model_id="us.amazon.nova-lite-v1:0", temperature=0.6)
nova_micro_bedrock = BedrockModel(model_id="us.amazon.nova-micro-v1:0", temperature=0.6)
