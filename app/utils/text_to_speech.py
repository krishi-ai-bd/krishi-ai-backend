import os
import wave
import struct
import logging
import threading
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Audio output directory
AUDIO_DIR = Path("audio")
AUDIO_DIR.mkdir(exist_ok=True)


class GeminiTTSClient:
    """
    Text-to-Speech using Gemini 2.0 Flash.
    - Loads all GEMINI_API_KEY_N keys from environment
    - Round-robin rotation with automatic failover if a key fails
    - Saves audio as WAV files to the audio/ folder
    """

    GEMINI_TTS_MODEL = "gemini-2.0-flash-preview-tts"  # TTS-capable model
    VOICE_NAME = "Aoede"  # Supports Bangla naturally

    def __init__(self):
        import threading
        self.api_keys = self._load_api_keys()

        if not self.api_keys:
            logger.error("[TTS] No Gemini API keys found. Add GEMINI_API_KEY_1, GEMINI_API_KEY_2, etc. to .env")
        else:
            logger.info(f"[TTS] Loaded {len(self.api_keys)} Gemini API key(s)")
            logger.info(f"[TTS] Round-robin with failover enabled")

        # Thread-safe round-robin counter
        self._counter = 0
        self._lock = threading.Lock()

    def _load_api_keys(self) -> list:
        """Load all GEMINI_API_KEY_N keys from environment"""
        keys = []
        i = 1
        while True:
            key = os.getenv(f"GEMINI_API_KEY_{i}", "")
            if key and key.lower() not in ("none", ""):
                keys.append(key)
                i += 1
            else:
                break

        # Fallback to single GEMINI_API_KEY
        if not keys:
            single = os.getenv("GEMINI_API_KEY", "")
            if single:
                keys.append(single)

        return keys

    def _get_next_key(self) -> Optional[str]:
        """Get next API key in round-robin order (thread-safe)"""
        if not self.api_keys:
            return None
        with self._lock:
            key = self.api_keys[self._counter % len(self.api_keys)]
            self._counter += 1
            return key

    def _save_as_wav(self, audio_data: bytes, output_path: Path) -> None:
        """Save raw PCM audio bytes as a proper WAV file"""
        with wave.open(str(output_path), "wb") as wf:
            wf.setnchannels(1)       # Mono
            wf.setsampwidth(2)       # 16-bit
            wf.setframerate(24000)   # Gemini outputs 24kHz
            wf.writeframes(audio_data)

    def synthesize(self, text: str, filename: str) -> Optional[str]:
        """
        Convert text to speech using Gemini TTS with round-robin + failover.

        Args:
            text: Bangla (or any) text to convert to speech
            filename: Output filename (without extension), saved to audio/ folder

        Returns:
            Path to saved WAV file, or None if all keys failed
        """
        if not self.api_keys:
            logger.error("[TTS] No API keys available")
            return None

        import google.generativeai as genai
        
        output_path = AUDIO_DIR / f"{filename}.wav"
        attempted_keys = set()

        # Try each key with failover
        for attempt in range(len(self.api_keys)):
            api_key = self._get_next_key()
            key_index = (self._counter - 1) % len(self.api_keys) + 1

            if api_key in attempted_keys:
                continue
            attempted_keys.add(api_key)

            try:
                logger.info(f"[TTS] Attempting with key #{key_index}/{len(self.api_keys)}")

                client = genai.Client(api_key=api_key)

                response = client.models.generate_content(
                    model=self.GEMINI_TTS_MODEL,
                    contents=text,
                    config=genai.types.GenerateContentConfig(
                        response_modalities=["AUDIO"],
                        speech_config=genai.types.SpeechConfig(
                            voice_config=genai.types.VoiceConfig(
                                prebuilt_voice_config=genai.types.PrebuiltVoiceConfig(
                                    voice_name=self.VOICE_NAME
                                )
                            )
                        ),
                    ),
                )

                # Extract audio bytes from response
                audio_data = response.candidates[0].content.parts[0].inline_data.data

                # Save as WAV
                self._save_as_wav(audio_data, output_path)

                logger.info(f"[TTS] Audio saved to: {output_path}")
                return str(output_path)

            except Exception as e:
                logger.warning(f"[TTS] Key #{key_index} failed: {str(e)}. Trying next key...")
                continue

        logger.error("[TTS] All Gemini API keys failed for TTS")
        return None


# Global singleton instance
tts_client = GeminiTTSClient()
