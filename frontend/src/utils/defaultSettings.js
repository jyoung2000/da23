/**
 * Shared default clip settings — SINGLE SOURCE OF TRUTH.
 * Used by ClipSettingsPanel, Analysis, ViralClips, and anywhere
 * that needs default subtitle/clip settings.
 */
export const DEFAULT_CLIP_SETTINGS = {
  clipCount: 12,
  minDuration: 30,
  maxDuration: 300,
  aspectRatio: null,
  subtitlesEnabled: true,  // Subtitles ON by default when transcript exists
  subtitleFont: 'DM Sans',
  subtitleSize: 30,
  subtitleFontWeight: 700,
  subtitleFontColor: '#FFFFFF',
  subtitlePosition: 'bottom',
  speakerColors: {},
  subtitleBgEnabled: false,
  subtitleBgColor: '#000000',
  subtitleBgOpacity: 75,
  subtitleBgRadius: 0,
  subtitleOutlineColor: '#000000',
  subtitleOutlineOpacity: 100,
  subtitleOutlineWidth: 2,
  showSpeakerLabels: false,
  subtitleMaxWidth: 90,
  subtitleOffsetV: 4,
  subtitleMaxWords: 0,
  activeWordEnabled: false,
  activeWordColor: '#FFD700',
  activeWordOutlineColor: '#000000',
  activeWordBgColor: '#000000',
  activeWordBgOpacity: 0,
  activeWordBgRadius: 4,
  useSpeakerColors: true,
  exportQuality: '1080p',
  playbackVolume: 100,
  playbackSpeed: 1.0,
};
