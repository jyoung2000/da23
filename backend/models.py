from enum import Enum
from typing import Optional, Union
from pydantic import BaseModel


class JobStatus(str, Enum):
    QUEUED = "queued"
    EXTRACTING_FRAMES = "extracting_frames"
    TRANSCRIBING = "transcribing"
    ANALYZING_SCENES = "analyzing_scenes"
    GENERATING_SUMMARY = "generating_summary"
    DETECTING_CLIPS = "detecting_clips"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


class LayoutMode(str, Enum):
    """Video layout modes for multi-speaker reframing."""
    SINGLE = "single"          # One speaker centered (current behavior)
    SPLIT = "split"            # Two speakers side-by-side (vertical split)
    TRIPLE = "triple"          # Three speakers in grid
    PICTURE_IN_PICTURE = "pip" # Main speaker large + secondary small overlay
    SCREENSHARE = "screenshare" # Screen content top, speaker bottom
    GAMEPLAY = "gameplay"      # Gameplay top 70%, speaker bottom 30%


class FrameData(BaseModel):
    timestamp: float
    path: str
    base64: Optional[str] = None
    face_data: Optional[object] = None  # FrameFaces from face_detector (set after extraction)
    face_registry: Optional[object] = None  # FaceRegistry from face_registry module

    class Config:
        arbitrary_types_allowed = True


class SceneDescription(BaseModel):
    timestamp: float
    description: str
    importance_score: int  # 1-10
    thumbnail_path: str
    subject_x: int = 50  # 0-100, horizontal subject position (0=left, 50=center, 100=right)
    active_speaker_x: Optional[int] = None  # 0-100, position of the person who is talking (if detectable)
    layout_mode: str = "single"           # Recommended layout for this timestamp
    face_count: int = 0                   # Number of faces detected at this timestamp
    face_positions: list[dict] = []       # [{slot_id, x, y, w, h, is_speaking, identity_id}]
    has_screen_content: bool = False       # Whether frame contains screen share / slides / text
    primary_object_x: Optional[int] = None # Non-face object tracking position (0-100)
    primary_object_type: Optional[str] = None  # "ball", "product", "hand", "text", etc.


class WordTimestamp(BaseModel):
    start: float
    end: float
    word: str


class TranscriptSegment(BaseModel):
    start: float
    end: float
    text: str
    speaker: str  # "Speaker 1", "Speaker 2", etc.
    words: Optional[list[WordTimestamp]] = None  # per-word timestamps from Whisper
    confidence: Optional[float] = None  # 0.0-1.0, derived from avg_logprob
    avg_logprob: Optional[float] = None  # Raw Whisper log probability
    no_speech_prob: Optional[float] = None  # Probability this is not speech


class ClipCandidate(BaseModel):
    id: int
    title: str
    start_time: float
    end_time: float
    duration: float
    viral_score: int  # 1-100
    viral_score_reasoning: str
    clip_type: str
    platform: str  # tiktok | youtube_shorts | both
    suggested_caption: str
    hook_text: str
    why_this_works: str
    clip_focus: Optional[str] = None  # The focus topic used to generate this clip, if any
    focus_relevance: Optional[int] = None  # 1-100, how relevant to the focus query
    focus_tier: Optional[str] = None  # "strong", "moderate", "weak"
    # Persisted SEO data (populated by generation endpoints)
    seo_title: Optional[str] = None
    seo_description: Optional[str] = None
    seo_tags: list[str] = []
    seo_platform_tips: Optional[str] = None
    shorts_description: Optional[str] = None
    longform_description: Optional[str] = None


class VideoSummary(BaseModel):
    overview: str
    key_topics: list[str]
    tone: str
    estimated_audience: str
    content_category: str


class JobResult(BaseModel):
    job_id: str
    filename: str
    file_path: str
    language: str = ""  # ISO 639-1 code, empty = auto-detect
    subtitle_language: str = ""  # ISO 639-1 target language for subtitles, empty = same as audio
    duration: float = 0.0
    resolution: str = ""
    fps: float = 0.0
    file_size_mb: float = 0.0
    status: JobStatus = JobStatus.QUEUED
    progress: int = 0  # 0-100
    progress_message: str = ""
    provider_used: dict = {}  # {task: model_name}
    created_at: str = ""
    updated_at: str = ""
    analysis_started_at: Optional[str] = None  # ISO timestamp when analysis began
    analysis_duration_seconds: Optional[float] = None  # total wall-clock time for analysis
    summary: Optional[VideoSummary] = None
    scenes: list[SceneDescription] = []
    transcript: list[TranscriptSegment] = []
    translated_transcript: list[TranscriptSegment] = []  # Translated subtitle segments
    clips: list[ClipCandidate] = []
    speaker_names: dict[str, str] = {}  # {"Speaker 1": "Eric", "Speaker 2": "Alice"}
    scene_cut_timestamps: list[float] = []  # Timestamps of camera cuts from frame extraction
    exported_clips: list[dict] = []
    subtitle_settings: Optional[dict] = None  # Canonical subtitle settings — server is source of truth
    error: Optional[str] = None
    estimated_cost_usd: Optional[float] = None
    face_registry_data: Optional[dict] = None   # Serialized FaceRegistry with embeddings
    layout_timeline: list[dict] = []            # [{start, end, layout_mode, face_positions}]
    default_layout_mode: str = "single"         # Overall recommended layout for the video


class ClipSEO(BaseModel):
    title: str
    description: str
    tags: list[str] = []
    platform_tips: str = ""


class SubtitleSettings(BaseModel):
    font: str = "DM Sans"
    size: Union[str, int, float] = "medium"  # "small" | "medium" | "large" | numeric px (12-72)
    font_weight: Union[str, int] = "bold"  # "normal"|"bold" or numeric 100-900
    font_color: str = "#FFFFFF"  # default subtitle text color (hex)
    position: str = "bottom"  # "top" | "center" | "bottom"
    speaker_colors: dict[str, str] = {}  # {"Speaker 1": "#00D9FF"}
    use_speaker_colors: bool = True  # True = per-speaker colors, False = uniform font_color
    background_enabled: bool = False
    background_color: str = "#000000"
    background_opacity: int = 75  # 0-100
    background_radius: int = 0  # 0-20, border radius in px for subtitle background
    outline_color: str = "#000000"  # text outline color (hex)
    outline_opacity: int = 100  # 0-100, text outline opacity
    outline_width: int = 2  # 0-10, text outline thickness in reference pixels
    show_speaker_labels: bool = False  # show "Speaker:" prefix in subtitle text
    max_width: int = 90  # 20-100, max subtitle width as % of video width
    offset_v: int = 4  # 0-100, vertical offset from edge as % of video height
    max_words: int = 0  # 0 = disabled, 1-20 = max words per subtitle event
    active_word_enabled: bool = False  # highlight the currently spoken word
    active_word_color: str = "#FFD700"  # text color of the active word (gold)
    active_word_outline_color: str = "#000000"  # outline/stroke color of the active word
    active_word_bg_color: str = "#000000"  # background color behind the active word
    active_word_bg_opacity: int = 0  # 0-100, background opacity (0 = no background)
    active_word_bg_radius: int = 4  # 0-20, border radius of the active word background box


class SegmentSettings(BaseModel):
    start: float          # Absolute start time in seconds
    end: float            # Absolute end time in seconds
    volume: float = 1.0   # 0.0 to 2.0 gain
    muted: bool = False
    subtitles_enabled: bool = True
    subject_tracking_enabled: bool = True  # Per-segment subject tracking toggle
    speed: float = 1.0    # 0.25 to 4.0 playback speed


class VideoEffects(BaseModel):
    """Video effects applied via FFmpeg eq/hue/boxblur filters to match preview."""
    brightness: float = 0.0    # -100 to 100 (maps to FFmpeg eq brightness)
    contrast: float = 0.0      # -100 to 100 (maps to FFmpeg eq contrast)
    saturation: float = 0.0    # -100 to 100 (maps to FFmpeg eq saturation)
    blur: float = 0.0          # 0 to 20 (maps to FFmpeg boxblur)
    hue_rotate: float = 0.0    # 0 to 360 degrees (maps to FFmpeg hue)
    sepia: float = 0.0         # 0 to 100 (maps to FFmpeg colorchannelmixer)
    opacity: float = 1.0       # 0 to 1 (maps to FFmpeg colorchannelmixer alpha)
    # Video transform — position/size/rotation from Properties panel
    position_x: float = 50.0   # 0-100% (50 = centered)
    position_y: float = 50.0   # 0-100% (50 = centered)
    width: float = 100.0       # 0-200% of frame (100 = original)
    height: float = 100.0      # 0-200% of frame (100 = original)
    rotation: float = 0.0      # degrees
    fade_in: float = 0.0       # seconds
    fade_out: float = 0.0      # seconds


class TextOverlay(BaseModel):
    """Text overlay for FFmpeg drawtext filter."""
    item_id: str = ""          # timeline item ID for compositing order lookup
    text: str = ""
    x: float = 50              # position % (0-100)
    y: float = 50              # position % (0-100)
    font_size: float = 48
    font_color: str = "#FFFFFF"
    font_family: str = "sans-serif"
    font_weight: float = 400
    background_color: Optional[str] = None
    outline_width: float = 0
    outline_color: str = "#000000"
    start_time: float = 0.0    # relative to clip start
    end_time: float = 0.0
    rotation: float = 0.0
    opacity: float = 1.0
    fade_in: float = 0.0       # seconds
    fade_out: float = 0.0      # seconds
    animation: str = "none"
    text_align: str = "center"
    shadow_blur: float = 0
    shadow_color: str = "rgba(0,0,0,0.5)"
    shadow_offset_x: float = 0
    shadow_offset_y: float = 0
    bg_opacity: float = 0
    bg_padding: float = 8
    bg_radius: float = 4


class ImageOverlay(BaseModel):
    """Image overlay for FFmpeg overlay filter."""
    item_id: str = ""          # timeline item ID for compositing order lookup
    src: str = ""              # path or URL to image
    x: float = 50              # position % (0-100)
    y: float = 50              # position % (0-100)
    width: float = 30          # size % (0-100)
    height: float = 30         # size % (0-100)
    start_time: float = 0.0
    end_time: float = 0.0
    opacity: float = 1.0
    fade_in: float = 0.0       # seconds
    fade_out: float = 0.0      # seconds
    rotation: float = 0.0      # degrees (-360 to 360)
    # Visual effects from multi-track editor properties panel
    brightness: float = 0.0    # -100 to 100
    contrast: float = 0.0      # -100 to 100
    saturation: float = 0.0    # -100 to 100
    blur: float = 0.0          # 0 to 20 (px)
    hue_rotate: float = 0.0    # 0 to 360 (degrees)
    sepia: float = 0.0         # 0 to 100


class ShapeOverlay(BaseModel):
    """Shape overlay rendered as a temporary PNG and composited via FFmpeg overlay."""
    item_id: str = ""          # timeline item ID for compositing order lookup
    shape_type: str = "rectangle"  # rectangle | circle | ellipse | line | arrow
    x: float = 50              # center position % (0-100)
    y: float = 50              # center position % (0-100)
    width: float = 20          # size % (0-100)
    height: float = 20         # size % (0-100)
    fill_color: str = "#FF3B30"
    stroke_color: str = "#FFFFFF"
    stroke_width: float = 2
    corner_radius: float = 0   # for rectangles
    start_time: float = 0.0
    end_time: float = 0.0
    rotation: float = 0.0
    opacity: float = 1.0
    fade_in: float = 0.0      # seconds
    fade_out: float = 0.0     # seconds


class AudioOverlay(BaseModel):
    """Additional audio item from multi-track editor (background music, SFX)."""
    src: str = ""              # path or URL to audio file
    start_time: float = 0.0   # start on the timeline (seconds)
    end_time: float = 0.0     # end on the timeline (seconds)
    volume: float = 1.0       # 0.0 to 2.0 gain
    fade_in: float = 0.0      # seconds
    fade_out: float = 0.0     # seconds


class ExportRequest(BaseModel):
    start: float
    end: float
    clip_id: int
    clip_title: Optional[str] = None  # optional title — used as export filename
    aspect_ratio: Optional[str] = None  # "16:9" | "9:16" | "1:1" | "4:5" | None=source
    subtitles_enabled: bool = False    # True if ANY subtitles needed (global or per-segment)
    global_subtitles_enabled: Optional[bool] = None  # Original global toggle (before segment overrides)
    subtitle_settings: Optional[SubtitleSettings] = None
    export_quality: str = "1080p"  # "720p" | "1080p" | "4k"
    # VideoEditor params — applied during FFmpeg export
    volume: float = 1.0              # 0.0 to 2.0 gain
    speed: float = 1.0               # 0.25 to 4.0 playback speed
    trim_start_offset: float = 0.0   # Seconds trimmed from clip start
    trim_end_offset: float = 0.0     # Seconds trimmed from clip end
    segments: list[SegmentSettings] = []  # Per-segment volume/subtitle overrides
    # Multi-track editor effects — applied via FFmpeg filters to match preview
    video_effects: Optional[VideoEffects] = None
    text_overlays: list[TextOverlay] = []
    image_overlays: list[ImageOverlay] = []
    shape_overlays: list[ShapeOverlay] = []
    audio_overlays: list[AudioOverlay] = []  # Additional audio items (music, SFX)
    overlay_compositing_order: list[dict] = []  # Global render order for cross-type compositing
    edited_subtitle_segments: Optional[list[TranscriptSegment]] = None  # User-edited subtitle timing from timeline
    layout_mode: str = "auto"  # "auto" | "single" | "split" | "triple" | "pip" | "screenshare"
    pip_position: str = "bottom_right"  # For PIP mode
    pip_size_pct: float = 25.0          # For PIP mode


class FullVideoExportRequest(BaseModel):
    aspect_ratio: Optional[str] = None  # "16:9" | "9:16" | "1:1" | "4:5" | None=source
    subtitles_enabled: bool = False    # True if ANY subtitles needed (global or per-segment)
    global_subtitles_enabled: Optional[bool] = None  # Original global toggle (before segment overrides)
    subtitle_settings: Optional[SubtitleSettings] = None
    export_quality: str = "1080p"  # "720p" | "1080p" | "4k"
    # VideoEditor params — applied during FFmpeg export
    volume: float = 1.0
    speed: float = 1.0
    trim_start_offset: float = 0.0
    trim_end_offset: float = 0.0
    segments: list[SegmentSettings] = []  # Per-segment volume/subtitle overrides
    # Multi-track editor effects — applied via FFmpeg filters to match preview
    video_effects: Optional[VideoEffects] = None
    text_overlays: list[TextOverlay] = []
    image_overlays: list[ImageOverlay] = []
    shape_overlays: list[ShapeOverlay] = []
    edited_subtitle_segments: Optional[list[TranscriptSegment]] = None  # User-edited subtitle timing from timeline


class UpdateClipTitleRequest(BaseModel):
    title: str


class UpdateClipTimesRequest(BaseModel):
    start_time: Optional[float] = None
    end_time: Optional[float] = None


class GenerateClipsRequest(BaseModel):
    min_duration: float = 15
    max_duration: float = 600
    clip_count: Optional[int] = None  # None = use settings.MAX_CLIP_CANDIDATES
    clip_focus: Optional[str] = None  # Optional focus topic (e.g. "fighting", "cooking tips")
    viral_score_min: int = 0  # Minimum viral score (0-100), clips below are filtered out
    viral_score_max: int = 100  # Maximum viral score (0-100), clips above are filtered out


class SettingsUpdate(BaseModel):
    ai_provider: Optional[str] = None
    openrouter_preset: Optional[str] = None
    openrouter_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None
    whisper_model: Optional[str] = None
    frame_sample_rate: Optional[int] = None
    max_clip_candidates: Optional[int] = None
    fallback_chain: Optional[str] = None


class ClipPreset(BaseModel):
    id: str
    name: str
    settings: dict
    created_at: str


class SavePresetRequest(BaseModel):
    name: str
    settings: dict


class RenamePresetRequest(BaseModel):
    name: str


class TranslateRequest(BaseModel):
    source_language: str = ""  # Auto-detect from job if empty
    target_language: str       # ISO 639-1 code


# ── Multi-track Timeline Models ─────────────────────────────────────────────

class TimelineItem(BaseModel):
    id: str
    track_id: str
    type: str  # 'video', 'audio', 'image', 'subtitle'
    media_ref: Optional[str] = None
    start: float
    end: float
    trim_start: float = 0
    trim_end: Optional[float] = None
    volume: float = 1.0
    speed: float = 1.0
    opacity: float = 1.0
    position: Optional[dict] = None  # {x, y} percentages
    size: Optional[dict] = None      # {w, h} percentages
    fade_in: float = 0
    fade_out: float = 0
    subtitle_text: Optional[str] = None
    subtitle_style: Optional[dict] = None


class TimelineExportRequest(BaseModel):
    tracks: list[dict] = []
    items: list[TimelineItem] = []
    duration: float = 0
    resolution: str = "1080p"  # "720p", "1080p", "4K"
    quality: int = 23          # CRF value
    format: str = "mp4"


class EditorStateRequest(BaseModel):
    tracks: list[dict] = []
    items: list[dict] = []
    mediaLibrary: list[dict] = []
    duration: float = 0
