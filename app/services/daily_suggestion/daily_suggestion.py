import os
import threading
from datetime import date, datetime, timedelta
from pathlib import Path
from pydantic import ValidationError
import json
import openai
from dotenv import load_dotenv
from app.core.config import settings
from app.utils.text_to_speech import tts_client
from .daily_suggestion_schema import daily_suggestion_request, daily_suggestion_response

load_dotenv()

AUDIO_DIR = Path(settings.AUDIO_DIR)
AUDIO_DIR.mkdir(exist_ok=True)


# ─── Bangladesh Seasonal Context ─────────────────────────────────────────────

def get_bangladesh_season(today: date) -> tuple[str, str]:
    """Returns (season_name, farming_context) for the current month."""
    month = today.month
    seasons = {
        (12, 1):  ("শীতকাল (Winter)",          "রবি ফসল মৌসুম: গম, আলু, সরিষা, ডাল, শাকসবজি চাষের উপযুক্ত সময়।"),
        (2, 3):   ("বসন্তকাল (Spring)",          "বোরো ধানের পরিচর্যা, সবজি সংগ্রহ ও গ্রীষ্মকালীন ফসলের প্রস্তুতি।"),
        (4, 5):   ("গ্রীষ্মকাল (Summer)",        "আউশ ধান বপন, পাট চাষ, তরমুজ ও গ্রীষ্মকালীন সবজি চাষের উপযুক্ত সময়।"),
        (6, 7):   ("বর্ষাকাল (Monsoon)",          "আমন ধান রোপণ, বন্যা ব্যবস্থাপনা, জলাবদ্ধতা সহনশীল ফসলের পরিচর্যা।"),
        (8, 9):   ("শরৎকাল (Autumn)",             "আমন ধানের যত্ন, শাকসবজি চাষ, রবি ফসলের জমি প্রস্তুতির সময়।"),
        (10, 11): ("হেমন্তকাল (Late Autumn)",     "আমন ধান কাটা, রবি ফসল বপন শুরু, শীতকালীন সবজি চাষ শুরু।"),
    }
    for (m1, m2), (season, context) in seasons.items():
        if month in (m1, m2):
            return season, context
    return ("শীতকাল", "রবি ফসল মৌসুম।")


# ─── Audio Cleanup ─────────────────────────────────────────────────────────

def cleanup_old_audio_files():
    """Delete WAV files older than AUDIO_RETENTION_DAYS (default 7 days)."""
    cutoff = datetime.now() - timedelta(days=settings.AUDIO_RETENTION_DAYS)
    deleted = 0
    for audio_file in AUDIO_DIR.glob("*.wav"):
        try:
            file_mtime = datetime.fromtimestamp(audio_file.stat().st_mtime)
            if file_mtime < cutoff:
                audio_file.unlink()
                deleted += 1
        except Exception as e:
            print(f"[AUDIO CLEANUP] Error deleting {audio_file.name}: {e}")
    if deleted:
        print(f"[AUDIO CLEANUP] Deleted {deleted} old audio file(s)")


# ─── DailySuggestion Service ─────────────────────────────────────────────────

class DailySuggestion:
    def __init__(self):
        self.api_keys = settings.load_api_keys("openai")
        if not self.api_keys:
            raise ValueError("[DailySuggestion] No OpenAI API keys found. Add OPENAI_API_KEY_1 etc. to .env")
        self._counter = 0
        self._lock = threading.Lock()

    def _get_next_client(self):
        """Round-robin OpenAI client selection (thread-safe)"""
        with self._lock:
            key = self.api_keys[self._counter % len(self.api_keys)]
            self._counter += 1
        return openai.OpenAI(api_key=key)

    def _get_today_audio_path(self) -> Path:
        """Returns the expected file path for today's audio."""
        today_str = date.today().strftime("%Y%m%d")
        return AUDIO_DIR / f"suggestion_{today_str}.wav"

    def _get_today_audio_url(self) -> str:
        """Builds the public URL for today's audio file."""
        filename = self._get_today_audio_path().name
        return f"{settings.BASE_URL.rstrip('/')}/audio/{filename}"

    def get_today_suggestion(self) -> daily_suggestion_response:
        """
        GET handler - returns today's audio URL only if already generated.
        Returns None if not yet generated today (frontend should try again later).
        """
        today_path = self._get_today_audio_path()
        if today_path.exists():
            return daily_suggestion_response(audio_url=self._get_today_audio_url())
        return None

    def daily_suggestion(self, request: daily_suggestion_request) -> daily_suggestion_response:
        """
        POST handler - force regenerate today's audio (manual trigger/override).
        Always regenerates even if today's file already exists.
        """
        cleanup_old_audio_files()
        return self._generate_and_save(previous_suggestions=request.previous_suggestions)

    def generate_scheduled(self):
        """
        Called by the APScheduler daily job.
        Uses the same logic as the POST endpoint.
        Skips if today's audio was already generated.
        """
        today_path = self._get_today_audio_path()
        if today_path.exists():
            print(f"[SCHEDULER] Audio for today already exists, skipping.")
            return

        print(f"[SCHEDULER] Generating daily suggestion audio...")
        cleanup_old_audio_files()
        try:
            self._generate_and_save(previous_suggestions=[])
            print(f"[SCHEDULER] Daily suggestion generated successfully.")
        except Exception as e:
            print(f"[SCHEDULER] Error generating daily suggestion: {e}")

    def _generate_and_save(self, previous_suggestions: list) -> daily_suggestion_response:
        """Core logic: generate text suggestion → convert to audio → return URL."""
        today = date.today()
        season_name, season_context = get_bangladesh_season(today)

        # 1. Generate text suggestion
        prompt = self.create_prompt(today, season_name, season_context)
        input_data = self.prepare_input(previous_suggestions)
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
            suggestion_text = parsed_json.get("response", "")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON from OpenAI: {e}")

        # 2. Convert to speech — use date-based filename (idempotent)
        filename = f"suggestion_{today.strftime('%Y%m%d')}"
        audio_path = tts_client.synthesize(text=suggestion_text, filename=filename)

        if not audio_path:
            raise RuntimeError("[DailySuggestion] TTS failed — no audio generated")

        return daily_suggestion_response(audio_url=self._get_today_audio_url())

    def prepare_input(self, previous_suggestions: list) -> str:
        if not previous_suggestions:
            return "কোনো পূর্ববর্তী পরামর্শ নেই।"
        suggestions_text = "\n".join(f"- {s}" for s in previous_suggestions)
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
        client = self._get_next_client()
        completion = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": data}
            ],
            temperature=0.5
        )
        return completion.choices[0].message.content
