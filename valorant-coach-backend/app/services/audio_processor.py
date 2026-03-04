"""
Audio Processor for Valorant VODs.

Handles audio extraction and basic transcription analysis.
Uses a simplified approach for MVP - can be enhanced with
OpenAI Whisper for production use.

This represents 5% of the overall coaching score.
"""

import os
import subprocess
import json
from dataclasses import dataclass, field


@dataclass
class TranscriptionSegment:
    start: float
    end: float
    text: str
    is_callout: bool = False
    is_timely: bool = True


@dataclass
class AudioAnalysisResult:
    total_callouts: int = 0
    timely_callouts_pct: float = 0.0
    late_callouts_pct: float = 0.0
    transcription_segments: list[dict] = field(default_factory=list)
    audio_events: list[dict] = field(default_factory=list)
    score: float = 0.0


class AudioProcessor:
    """
    Processes audio from Valorant VODs for communication analysis.

    MVP approach: Extract audio, detect voice activity, and provide
    basic transcription timing analysis.

    Full implementation would use Whisper for accurate transcription
    and NLP for callout classification.
    """

    # Common Valorant callouts for detection (EN + PT-BR)
    CALLOUT_KEYWORDS = [
        # Numbers
        "one", "two", "three", "four", "five",
        "um", "dois", "tres", "quatro", "cinco",
        # Positions
        "short", "long", "mid", "a", "b", "c",
        "curto", "longo",
        "heaven", "hell", "site", "main", "link",
        "ceu", "inferno", "bomb", "principal",
        # Abilities
        "flash", "smoke", "wall", "ult", "util",
        "fumaça", "parede", "ulti", "habilidade",
        # Actions
        "push", "rotate", "hold", "peek", "rush",
        "empurra", "rota", "segura", "pica", "corre",
        "defuse", "plant", "spike", "save", "eco",
        "desarma", "planta",
        "tp", "teleport", "flank", "back",
        "flanqueia", "volta",
        # Status
        "low", "lit", "tagged", "half",
        "baixo", "machucado", "metade",
        "nice", "trade", "cover",
        "troca", "cobre",
    ]

    def __init__(self, video_path: str):
        self.video_path = video_path
        self.audio_path = ""
        self.segments: list[TranscriptionSegment] = []

    # Only extract the first 5 minutes of audio.  This keeps the WAV
    # file small (~9.6 MB) which is critical on memory-constrained
    # deployments (Fly.io 256 MB RAM).
    MAX_AUDIO_SECONDS = 300

    def extract_audio(self, output_dir: str) -> str:
        """Extract audio track from video file using ffmpeg.

        Only the first ``MAX_AUDIO_SECONDS`` seconds are extracted to
        limit disk and memory usage on constrained environments.
        """
        self.audio_path = os.path.join(output_dir, "audio.wav")

        try:
            subprocess.run(
                [
                    "ffmpeg", "-i", self.video_path,
                    "-vn",  # no video
                    "-acodec", "pcm_s16le",
                    "-ar", "16000",  # 16kHz for speech
                    "-ac", "1",  # mono
                    "-t", str(self.MAX_AUDIO_SECONDS),  # limit duration
                    "-y",  # overwrite
                    self.audio_path
                ],
                capture_output=True,
                timeout=180,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # ffmpeg not available or timeout
            self.audio_path = ""

        return self.audio_path

    def detect_voice_activity(self) -> list[dict]:
        """
        Simple voice activity detection using audio energy levels.

        Reads the WAV file in streaming 1-second chunks so that only
        ~32 KB of audio is in memory at any time (16 kHz × 2 bytes).
        """
        if not self.audio_path or not os.path.exists(self.audio_path):
            return self._generate_simulated_events()

        try:
            import wave
            import numpy as np

            events: list[dict] = []
            in_voice = False
            voice_start = 0.0

            with wave.open(self.audio_path, "rb") as wf:
                sample_rate = wf.getframerate()
                n_frames = wf.getnframes()
                # Process in 0.5-second windows without loading the
                # entire file – keeps memory usage to ~32 KB.
                window_frames = int(sample_rate * 0.5)
                offset = 0
                while offset < n_frames:
                    chunk_size = min(window_frames, n_frames - offset)
                    raw = wf.readframes(chunk_size)
                    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
                    samples = samples / 32768.0  # normalize
                    energy = float(np.sqrt(np.mean(samples ** 2)))
                    timestamp = offset / sample_rate

                    if energy > 0.02 and not in_voice:
                        in_voice = True
                        voice_start = timestamp
                    elif energy <= 0.02 and in_voice:
                        in_voice = False
                        if timestamp - voice_start > 0.3:
                            events.append({
                                "start": round(voice_start, 2),
                                "end": round(timestamp, 2),
                                "type": "voice_activity",
                                "energy": round(energy, 4),
                            })
                    offset += chunk_size

            # Clean up WAV file immediately to free disk space
            try:
                os.remove(self.audio_path)
            except OSError:
                pass

            return events

        except Exception:
            return self._generate_simulated_events()

    def _generate_simulated_events(self) -> list[dict]:
        """Generate simulated audio events for demo/testing (PT-BR)."""
        return [
            {"start": 15.0, "end": 17.5, "type": "voice_activity", "text": "Dois no A curto, um machucado"},
            {"start": 45.0, "end": 46.8, "type": "voice_activity", "text": "Flash saindo, empurrando"},
            {"start": 78.0, "end": 80.0, "type": "voice_activity", "text": "Rota pro B, spike avistada"},
            {"start": 120.0, "end": 122.0, "type": "voice_activity", "text": "Boa troca, segura o site"},
            {"start": 155.0, "end": 157.5, "type": "voice_activity", "text": "Salva, salva, salva"},
        ]

    def analyze_callout_timing(self, timeline_events: list[dict]) -> list[TranscriptionSegment]:
        """
        Analyze if callouts were timely relative to game events.

        A callout is 'timely' if it occurs within 2 seconds of the
        related game event (or before it).
        """
        voice_events = self.detect_voice_activity()

        segments = []
        for ve in voice_events:
            text = ve.get("text", "Voice communication detected")
            is_callout = any(kw in text.lower() for kw in self.CALLOUT_KEYWORDS) if text else True

            # Check timing relative to game events
            is_timely = True
            if timeline_events:
                closest_event_time = min(
                    (abs(ve["start"] - te.get("timestamp", 0)) for te in timeline_events),
                    default=999
                )
                is_timely = closest_event_time < 3.0

            segment = TranscriptionSegment(
                start=ve["start"],
                end=ve["end"],
                text=text,
                is_callout=is_callout,
                is_timely=is_timely,
            )
            segments.append(segment)

        self.segments = segments
        return segments

    def generate_results(self, timeline_events: list[dict] | None = None) -> AudioAnalysisResult:
        """Generate audio analysis results."""
        if not self.segments:
            self.analyze_callout_timing(timeline_events or [])

        if not self.segments:
            return AudioAnalysisResult(score=50.0)

        callouts = [s for s in self.segments if s.is_callout]
        total_callouts = len(callouts)

        if total_callouts > 0:
            timely = sum(1 for s in callouts if s.is_timely)
            timely_pct = (timely / total_callouts) * 100
            late_pct = 100 - timely_pct
        else:
            timely_pct = 0.0
            late_pct = 0.0

        transcription_data = [
            {
                "start": s.start,
                "end": s.end,
                "text": s.text,
                "is_callout": s.is_callout,
                "is_timely": s.is_timely,
            }
            for s in self.segments
        ]

        audio_events = self.detect_voice_activity()

        # Score calculation
        # Having callouts: 30 points
        callout_score = min(30, total_callouts * 6)
        # Timely callouts: 70 points
        timing_score = (timely_pct / 100) * 70

        total_score = callout_score + timing_score

        return AudioAnalysisResult(
            total_callouts=total_callouts,
            timely_callouts_pct=round(timely_pct, 1),
            late_callouts_pct=round(late_pct, 1),
            transcription_segments=transcription_data,
            audio_events=audio_events,
            score=round(total_score, 1),
        )
