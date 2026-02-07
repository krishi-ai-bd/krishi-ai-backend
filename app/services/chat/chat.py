import json
import os
from typing import Any, Dict

import openai
from dotenv import load_dotenv
from .chat_schema import chatbot_request, chatbot_response


load_dotenv()


class ChatbotAgent:
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.reasoning_model = "gpt-4o"

    def chat(self, request: chatbot_request) -> chatbot_response:
        """Generate chat response based on user message."""
        prompt = self.create_prompt(request)
        response_data = self.get_ai_response(prompt)
        
        if response_data and "response" in response_data:
            return chatbot_response(response=response_data["response"])
        return chatbot_response(response="I apologize, I'm having trouble generating a response. Please try again.")

    def create_prompt(self, request: chatbot_request) -> str:
        system_prompt = """You are Krishi AI, an expert agricultural assistant specialized in crop management, farming techniques, and sustainable agriculture practices. 

Your role is to:
- Provide accurate, practical advice on crop cultivation, pest management, soil health, and irrigation
- Recommend solutions based on local farming conditions and best practices
- Help farmers optimize crop yields and reduce losses
- Explain complex agricultural concepts in simple, actionable terms
- Consider environmental sustainability and cost-effectiveness in your recommendations

Always respond in a helpful, professional tone. If you're uncertain about specific regional practices, acknowledge it and provide general best practices. Format your response as a JSON object with a 'response' key containing your answer."""
        
        payload = {
            "message": request.message
        }
        return json.dumps({
            "system": system_prompt,
            "payload": payload
        })

    def get_ai_response(self, prompt: str) -> Dict[str, Any] | None:
        """Call OpenAI API to generate affirmations."""
        try:
            payload_data = json.loads(prompt)
            system_content = payload_data.get("system", "")
            user_content = json.dumps(payload_data.get("payload", {}), ensure_ascii=False)

            response = self.client.chat.completions.create(
                model=self.reasoning_model,
                temperature=0.9,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content},
                ],
            )
            
            content = response.choices[0].message.content
            if content:
                return json.loads(content)
        except Exception:
            return None
        return None
