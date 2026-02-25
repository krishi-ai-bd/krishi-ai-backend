import os
from datetime import date
from pydantic import ValidationError
import json
import openai
from dotenv import load_dotenv
from .daily_suggestion_schema import daily_suggestion_request, daily_suggestion_response

load_dotenv()

# Bangladesh agricultural seasons
def get_bangladesh_season(today: date) -> tuple[str, str]:
    """
    Returns (season_name_bangla, farming_context) based on the current month.
    Bangladesh has 6 seasons (shodritu).
    """
    month = today.month
    seasons = {
        (12, 1):  ("শীতকাল (Winter/Shit)",     "রবি ফসল মৌসুম: গম, আলু, সরিষা, ডাল, শাকসবজি চাষের উপযুক্ত সময়।"),
        (2, 3):   ("বসন্তকাল (Spring/Basanta)", "বোরো ধানের পরিচর্যা, সবজি সংগ্রহ ও গ্রীষ্মকালীন ফসলের প্রস্তুতি।"),
        (4, 5):   ("গ্রীষ্মকাল (Summer/Grishmo)", "আউশ ধান বপন, পাট চাষ, তরমুজ ও গ্রীষ্মকালীন সবজি চাষের উপযুক্ত সময়।"),
        (6, 7):   ("বর্ষাকাল (Monsoon/Borsha)",  "আমন ধান রোপণ, বন্যা ব্যবস্থাপনা, জলাবদ্ধতা সহনশীল ফসলের পরিচর্যা।"),
        (8, 9):   ("শরৎকাল (Autumn/Shorot)",    "আমন ধানের যত্ন, শাকসবজি চাষ, রবি ফসলের জমি প্রস্তুতির সময়।"),
        (10, 11): ("হেমন্তকাল (Late Autumn/Hemanto)", "আমন ধান কাটা, রবি ফসল বপন শুরু, শীতকালীন সবজি চাষ শুরু।"),
    }
    for (m1, m2), (season, context) in seasons.items():
        if month in (m1, m2):
            return season, context
    return ("শীতকাল", "রবি ফসল মৌসুম।")  # Default fallback


class DailySuggestion:
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def daily_suggestion(self, request: daily_suggestion_request) -> daily_suggestion_response:
        today = date.today()
        season_name, season_context = get_bangladesh_season(today)
        prompt = self.create_prompt(today, season_name, season_context)
        input_data = self.prepare_input(request)
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

    def prepare_input(self, request: daily_suggestion_request) -> str:
        """Format previous suggestions as input context"""
        if not request.previous_suggestions:
            return "কোনো পূর্ববর্তী পরামর্শ নেই।"
        suggestions_text = "\n".join(
            f"- {s}" for s in request.previous_suggestions
        )
        return f"পূর্ববর্তী পরামর্শসমূহ:\n{suggestions_text}"

    def create_prompt(self, today: date, season_name: str, season_context: str) -> str:
        return f"""আপনি বাংলাদেশের একজন অভিজ্ঞ কৃষি বিশেষজ্ঞ। আপনার কাজ হলো বাংলাদেশের কৃষকদের জন্য দৈনিক কৃষি পরামর্শ প্রদান করা।

আজকের তারিখ: {today.strftime("%d %B %Y")}
বর্তমান ঋতু: {season_name}
ঋতুভিত্তিক প্রেক্ষাপট: {season_context}

নির্দেশনা:
১. সর্বদা বাংলা ভাষায় উত্তর দিন।
২. পরামর্শ বাংলাদেশের আবহাওয়া, মাটি ও ফসলের ধরন অনুযায়ী হতে হবে।
৩. বর্তমান ঋতু ও কৃষি মৌসুম বিবেচনা করে প্রাসঙ্গিক পরামর্শ দিন।
৪. পূর্বের পরামর্শগুলো পুনরাবৃত্তি করবেন না।
৫. ব্যবহারিক, সহজবোধ্য এবং কার্যকর পরামর্শ দিন।
৬. প্রতিটি পরামর্শ সংক্ষিপ্ত কিন্তু তথ্যবহুল হতে হবে।

আউটপুট ফরম্যাট (শুধুমাত্র JSON):
{{"response": "আজকের কৃষি পরামর্শ এখানে লিখুন..."}}"""

    def get_openai_response(self, prompt: str, data: str) -> str:
        completion = self.client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": data}
            ],
            temperature=0.5
        )
        return completion.choices[0].message.content
