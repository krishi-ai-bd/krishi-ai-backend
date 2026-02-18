import os
from pydantic import ValidationError
import json
import openai
from dotenv import load_dotenv
from .daily_suggestion_schema import daily_suggestion_request, daily_suggestion_response

load_dotenv ()

class DailySuggestion:
    def __init__(self):
        self.client=openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def get_daily_suggestion(self, input_data: str) -> daily_suggestion_response:
        prompt = self.create_prompt()
        raw_response = self.get_openai_response(prompt, input_data)
        print("RAW OPENAI RESPONSE:\n", raw_response)


        try:
            cleaned = raw_response.strip()

            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]

            parsed_json = json.loads(cleaned)
            if "response" in parsed_json:
                parsed_json = parsed_json["response"]
            return daily_suggestion_response(**parsed_json)

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON returned by OpenAI: {e}")
        except ValidationError as ve:
            raise ValueError(f"Validation failed when parsing DailySuggestionResponse: {ve}")

    
    def create_prompt(self) -> str:
        return f"""You are a helpful assistant that provides daily suggestions to farmers based on their previous suggestions.Avoid repeating advices"""
    
    def get_openai_response (self, prompt:str, data:str)->str:
        completion =self.client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role":"system", "content": prompt},{"role":"user", "content": data}],
            temperature=0.3            
        )
        return completion.choices[0].message.content
    