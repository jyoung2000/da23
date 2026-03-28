# Claude Code Opus 4.6 Implementation Prompt: Fix Subtitle Background & Active Word Highlight Positioning

## Problem Statement

The FFmpeg-exported subtitles have two critical positioning bugs visible in the rendered video:

1. **Subtitle background box extends far past the text** — The dark rounded-rectangle background behind subtitle text is significantly wider than the actual text, with ~30-50px excess on each side.

2. **Active word highlight is offset from the actual word** — The colored box that should highlight the currently-spoken word is shifted to the LEFT of the word instead of being directly behind it. It also bleeds into adjacent words.

Both bugs only affect the **drawing-based background path** (BGDRAW/AWDRAW), which is activated when `background_radius > 0` or `active_word_bg_radius > 0` in the subtitle settings.

## Root Cause

The file `backend/services/ass_generator.py` generates ASS (Advanced SubStation Alpha) subtitle files rendered by FFmpeg's libass filter. Two separate coordinate systems are in play:

1. **Text positioning**: libass auto-positions text events using its own internal FreeType metrics (alignment=2 for bottom-center, margins, etc.)
2. **Drawing positioning**: BGDRAW/AWDRAW events use absolute `\an7\pos(x,y)` coordinates calculated from **Pillow's** `font.getlength()` measurements

The root cause: **Pillow's text width measurements consistently differ from libass's rendering width** for the same font and size. Pillow typically overestimates text width by 10-25%, causing:
- `line_left_x = (video_width - text_w) / 2` to be too far LEFT
- `draw_w = text_w + 2 * pad` to be too WIDE
- `word_left_x = line_left_x + before_w` to be offset from the actual word position

Since text is auto-positioned by libass (correct) but backgrounds are positioned by Pillow (incorrect), they use different reference frames and become misaligned.

## The Fix

### Core Approach: Force text events into the same coordinate system as drawing events

When `_bg_split` or `_aw_bg_split` is True (drawing-based backgrounds active), add `\an7\pos(x,y)` overrides to ALL text events (both `base_text_events` and `pending_word_events`) using the same Pillow-measured positions used for BGDRAW/AWDRAW. This forces libass to place text at the Pillow-predicted position, ensuring perfect **relative** alignment between text and backgrounds.

### Implementation Steps

#### Step 1: Add position override post-processing section

**Location**: In `generate_ass()` function in `backend/services/ass_generator.py`, insert AFTER the BGDRAW section (after `pending_bg_draw_events` generation) and BEFORE the "FINAL OVERLAP ELIMINATION" section.

```python
# ── Force text positions when drawing-based backgrounds are active ──
_needs_pos_override = (_bg_split or _aw_bg_split) and (_bg_font_path or _aw_font_path)
if _needs_pos_override:
    import re as _re_pos
    _pos_font_path = _bg_font_path or _aw_font_path

    def _compute_line_pos(plain_text):
        """Compute top-left position for \\an7\\pos() from plain text."""
        m = _measure_text(plain_text, _pos_font_path, size_px)
        if not m:
            return None
        text_w = m[0]
        line_h = m[1] + m[2]
        lx = round((video_width - text_w) / 2)
        if alignment == 2:
            ty = round(video_height - margin_v - line_h)
        elif alignment == 5:
            ty = round((video_height - line_h) / 2)
        elif alignment == 8:
            ty = round(margin_v)
        else:
            return None
        return (lx, ty)

    def _add_pos_override(ev_text, pos_x, pos_y):
        """Prepend \\an7\\pos() to an ASS event text string."""
        pos_tag = f"\\an7\\pos({pos_x},{pos_y})"
        if ev_text.startswith("{"):
            return "{" + pos_tag + ev_text[1:]
        return "{" + pos_tag + "}" + ev_text

    # Post-process base_text_events
    new_base = []
    for ev_start, ev_end, ev_style, ev_text in base_text_events:
        plain = _re_pos.sub(r"\{[^}]*\}", "", ev_text)
        if plain.strip():
            pos = _compute_line_pos(plain)
            if pos:
                ev_text = _add_pos_override(ev_text, pos[0], pos[1])
        new_base.append((ev_start, ev_end, ev_style, ev_text))
    base_text_events[:] = new_base

    # Post-process pending_word_events
    new_words = []
    for ev_start, ev_end, ev_style, ev_text in pending_word_events:
        plain = _re_pos.sub(r"\{[^}]*\}", "", ev_text)
        if plain.strip():
            pos = _compute_line_pos(plain)
            if pos:
                ev_text = _add_pos_override(ev_text, pos[0], pos[1])
        new_words.append((ev_start, ev_end, ev_style, ev_text))
    pending_word_events[:] = new_words
```

#### Step 2: Increase AWDRAW horizontal padding

The active word background padding is currently very tight (2px), leaving no room for per-character measurement differences. Increase it:

**Location**: Line ~640 in the `_aw_bg_split` initialization:

```python
# BEFORE:
_aw_draw_pad_h = max(2, round(2 * font_scale))
# AFTER:
_aw_draw_pad_h = max(3, round(4 * font_scale))
```

### Why This Works

1. **Text events**: With `\an7\pos(line_left_x, text_top_y)`, libass places the text's top-left corner at the Pillow-computed position. The text renders from left to right starting at `line_left_x`.

2. **BGDRAW events**: The background box starts at `line_left_x - pad`, which is exactly `pad` pixels to the left of the text start. Perfect alignment.

3. **AWDRAW events**: The active word box starts at `word_left_x - pad = line_left_x + before_w - pad`. Since text is forced to start at `line_left_x`, the word at position `before_w` aligns with the drawing.

4. **Relative vs absolute accuracy**: Even if Pillow's absolute measurements differ from libass, all positions use the same Pillow reference frame. Relative errors (per-character width differences) are much smaller than the absolute centering error (~2-5px vs ~30-50px).

### Trade-offs

- Text wrapping is disabled for drawing-background events (single-line only). This is acceptable because active-word subtitles with `max_words` limits are almost always single-line.
- The text center may shift slightly from the true video center (proportional to Pillow/libass measurement difference). This is imperceptible (<1% of video width).

### Files Modified

- `backend/services/ass_generator.py` — The only file that needs changes

### Testing

1. Export a clip with `background_radius > 0` and `active_word_bg_radius > 0`
2. Verify the subtitle background box tightly wraps the text
3. Verify the active word highlight is directly behind the currently-spoken word
4. Verify the active word highlight doesn't bleed into adjacent words
5. Test with different fonts, sizes, and video resolutions
6. Test with speaker labels enabled/disabled
7. Test with both short (2-3 words) and longer (10+ words) subtitle segments
