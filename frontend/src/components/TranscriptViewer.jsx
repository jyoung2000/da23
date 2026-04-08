import React, { useState, useMemo, useCallback, useRef, useEffect } from 'react';
import useResponsive from '../hooks/useResponsive';

const SPEAKER_COLORS_LIST = [
  'var(--accent-cyan)',
  'var(--accent-amber)',
  'var(--success)',
  '#A78BFA',
];

// Must match the palette in VideoEditor / SubtitleOverlay / ClipPreview
const DEFAULT_SPEAKER_PALETTE = [
  '#00D9FF', '#F59E0B', '#10B981', '#A78BFA', '#EF4444', '#EC4899',
  '#06B6D4', '#8B5CF6', '#F97316', '#14B8A6', '#E879F9', '#84CC16',
  '#FB7185', '#38BDF8', '#FBBF24', '#34D399', '#C084FC', '#F472B6',
  '#22D3EE', '#A3E635', '#FB923C', '#2DD4BF', '#818CF8', '#F87171',
];

function formatTime(seconds) {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

function toSRT(segments) {
  return segments
    .map((seg, i) => {
      const startH = Math.floor(seg.start / 3600);
      const startM = Math.floor((seg.start % 3600) / 60);
      const startS = Math.floor(seg.start % 60);
      const startMs = Math.floor((seg.start % 1) * 1000);
      const endH = Math.floor(seg.end / 3600);
      const endM = Math.floor((seg.end % 3600) / 60);
      const endS = Math.floor(seg.end % 60);
      const endMs = Math.floor((seg.end % 1) * 1000);
      return `${i + 1}\n${String(startH).padStart(2, '0')}:${String(startM).padStart(2, '0')}:${String(startS).padStart(2, '0')},${String(startMs).padStart(3, '0')} --> ${String(endH).padStart(2, '0')}:${String(endM).padStart(2, '0')}:${String(endS).padStart(2, '0')},${String(endMs).padStart(3, '0')}\n${seg.speaker}: ${seg.text}`;
    })
    .join('\n\n');
}

function toTXT(segments) {
  return segments.map((seg) => `[${formatTime(seg.start)}] ${seg.speaker}: ${seg.text}`).join('\n');
}

export default function TranscriptViewer({ transcript, onSeek, jobId, onSpeakerRenamed, onTranscriptUpdated, onSpeakerColorChanged, onSpeakerAdded, speakerColors, timeRange, currentTime, maxHeight }) {
  const { isMobile } = useResponsive();
  const scrollContainerRef = useRef(null);
  const activeSegRef = useRef(null);
  const [search, setSearch] = useState('');
  const [editingSpeaker, setEditingSpeaker] = useState(null);
  const [editValue, setEditValue] = useState('');
  const [renaming, setRenaming] = useState(false);
  const [editingSegIdx, setEditingSegIdx] = useState(null);
  const [editSegText, setEditSegText] = useState('');
  const [savingSegment, setSavingSegment] = useState(false);

  // Multi-select state
  const [selectedIndices, setSelectedIndices] = useState(new Set());
  const lastClickedIdx = useRef(null);
  const [bulkSaving, setBulkSaving] = useState(false);

  // Add speaker state
  const [addingSpeaker, setAddingSpeaker] = useState(false);
  const [newSpeakerName, setNewSpeakerName] = useState('');

  // Insert segment state
  const [insertAfterIdx, setInsertAfterIdx] = useState(null);
  const [insertText, setInsertText] = useState('');
  const [insertSpeaker, setInsertSpeaker] = useState('');
  const [insertStart, setInsertStart] = useState('');
  const [insertEnd, setInsertEnd] = useState('');
  const [insertSaving, setInsertSaving] = useState(false);

  // Build ordered list of unique speakers and assign colors by position
  const speakers = useMemo(() => {
    const seen = [];
    transcript.forEach((seg) => {
      if (!seen.includes(seg.speaker)) seen.push(seg.speaker);
    });
    return seen;
  }, [transcript]);

  // Map speaker name -> color: use subtitle speakerColors if provided (matches subtitles),
  // otherwise fall back to DEFAULT_SPEAKER_PALETTE, then SPEAKER_COLORS_LIST
  const speakerColor = useCallback(
    (name) => {
      if (speakerColors?.[name]) return speakerColors[name];
      const idx = speakers.indexOf(name);
      return DEFAULT_SPEAKER_PALETTE[idx % DEFAULT_SPEAKER_PALETTE.length] || SPEAKER_COLORS_LIST[idx % SPEAKER_COLORS_LIST.length] || 'var(--text-secondary)';
    },
    [speakers, speakerColors]
  );

  const timeFiltered = useMemo(() => {
    if (!timeRange) return transcript;
    return transcript.filter((seg) =>
      seg.start < timeRange.end && seg.end > timeRange.start
    );
  }, [transcript, timeRange]);

  const filtered = useMemo(() => {
    if (!search.trim()) return timeFiltered;
    const q = search.toLowerCase();
    return timeFiltered.filter((seg) => seg.text.toLowerCase().includes(q) || seg.speaker.toLowerCase().includes(q));
  }, [timeFiltered, search]);

  // Determine which segment is currently playing (by original transcript index)
  const activeOriginalIdx = useMemo(() => {
    if (currentTime == null || !filtered.length) return -1;
    for (const seg of filtered) {
      if (seg.start <= currentTime && currentTime < seg.end) {
        return transcript.indexOf(seg);
      }
    }
    return -1;
  }, [currentTime, filtered, transcript]);

  // Auto-scroll to keep the active segment centered in the transcript view.
  // Uses per-frame lerp (exponential ease-out) instead of CSS smooth scroll
  // to avoid choppiness when segments change rapidly during playback.
  const scrollAnimRef = useRef(null);
  useEffect(() => {
    if (activeOriginalIdx < 0) return;
    const el = activeSegRef.current;
    const container = scrollContainerRef.current;
    if (!el || !container) return;

    // Cancel any running animation
    if (scrollAnimRef.current) {
      cancelAnimationFrame(scrollAnimRef.current);
      scrollAnimRef.current = null;
    }

    // Target: center the active segment in the container
    const targetTop = el.offsetTop - container.clientHeight / 2 + el.offsetHeight / 2;

    const animate = () => {
      const diff = targetTop - container.scrollTop;
      // If close enough, snap and stop
      if (Math.abs(diff) < 1) {
        container.scrollTop = targetTop;
        scrollAnimRef.current = null;
        return;
      }
      // Lerp: move 15% of remaining distance each frame (~60fps)
      // This gives smooth exponential ease-out without fighting CSS transitions
      container.scrollTop += diff * 0.15;
      scrollAnimRef.current = requestAnimationFrame(animate);
    };
    scrollAnimRef.current = requestAnimationFrame(animate);

    return () => {
      if (scrollAnimRef.current) {
        cancelAnimationFrame(scrollAnimRef.current);
        scrollAnimRef.current = null;
      }
    };
  }, [activeOriginalIdx]);

  const download = (content, filename) => {
    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  };

  const copyAll = () => {
    const text = toTXT(timeFiltered);
    navigator.clipboard.writeText(text);
  };

  const startEditing = (speaker) => {
    setEditingSpeaker(speaker);
    setEditValue(speaker);
  };

  const cancelEditing = () => {
    setEditingSpeaker(null);
    setEditValue('');
  };

  const submitRename = async () => {
    const newName = editValue.trim();
    if (!newName || newName === editingSpeaker || !jobId) {
      cancelEditing();
      return;
    }
    setRenaming(true);
    try {
      const res = await fetch(`/api/jobs/${jobId}/speakers`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ speaker_names: { [editingSpeaker]: newName } }),
      });
      if (res.ok) {
        // Transfer the old speaker's subtitle color to the new name
        if (onSpeakerColorChanged && speakerColors?.[editingSpeaker]) {
          onSpeakerColorChanged(newName, speakerColors[editingSpeaker], editingSpeaker);
        }
        cancelEditing();
        if (onSpeakerRenamed) onSpeakerRenamed();
      }
    } catch (err) {
      console.error('Speaker rename failed:', err);
    } finally {
      setRenaming(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      submitRename();
    } else if (e.key === 'Escape') {
      cancelEditing();
    }
  };

  // --- Segment text editing ---
  const startSegEdit = (originalIdx, text, e) => {
    e.stopPropagation();
    setEditingSegIdx(originalIdx);
    setEditSegText(text);
  };

  const cancelSegEdit = () => {
    setEditingSegIdx(null);
    setEditSegText('');
  };

  const saveSegEdit = async () => {
    const newText = editSegText.trim();
    if (editingSegIdx === null || !jobId) { cancelSegEdit(); return; }
    if (!newText || newText === transcript[editingSegIdx]?.text) { cancelSegEdit(); return; }
    setSavingSegment(true);
    try {
      const res = await fetch(`/api/jobs/${jobId}/transcript/${editingSegIdx}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: newText }),
      });
      if (res.ok) {
        cancelSegEdit();
        if (onTranscriptUpdated) onTranscriptUpdated();
      }
    } catch (err) {
      console.error('Segment update failed:', err);
    } finally {
      setSavingSegment(false);
    }
  };

  const handleSegKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      saveSegEdit();
    } else if (e.key === 'Escape') {
      cancelSegEdit();
    }
  };

  // --- Change speaker on a single segment ---
  const handleChangeSpeaker = async (originalIdx, newSpeaker) => {
    if (!jobId || !newSpeaker) return;
    try {
      const res = await fetch(`/api/jobs/${jobId}/transcript/${originalIdx}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ speaker: newSpeaker }),
      });
      if (res.ok && onTranscriptUpdated) onTranscriptUpdated();
    } catch (err) {
      console.error('Speaker change failed:', err);
    }
  };

  // --- Multi-select helpers ---
  const toggleSelect = useCallback((originalIdx, e) => {
    setSelectedIndices((prev) => {
      const next = new Set(prev);
      if (e?.shiftKey && lastClickedIdx.current != null) {
        // Range select: select all filtered segments between last click and this one
        const filteredOriginalIndices = filtered.map((seg) => transcript.indexOf(seg));
        const a = filteredOriginalIndices.indexOf(lastClickedIdx.current);
        const b = filteredOriginalIndices.indexOf(originalIdx);
        if (a >= 0 && b >= 0) {
          const [lo, hi] = a < b ? [a, b] : [b, a];
          for (let i = lo; i <= hi; i++) {
            next.add(filteredOriginalIndices[i]);
          }
        }
      } else {
        if (next.has(originalIdx)) next.delete(originalIdx);
        else next.add(originalIdx);
      }
      lastClickedIdx.current = originalIdx;
      return next;
    });
  }, [filtered, transcript]);

  const selectAllFiltered = useCallback(() => {
    const allIndices = filtered.map((seg) => transcript.indexOf(seg));
    setSelectedIndices(new Set(allIndices));
  }, [filtered, transcript]);

  const clearSelection = useCallback(() => {
    setSelectedIndices(new Set());
    lastClickedIdx.current = null;
  }, []);

  const handleBulkSpeakerChange = async (newSpeaker) => {
    if (!jobId || !newSpeaker || selectedIndices.size === 0) return;
    setBulkSaving(true);
    try {
      const res = await fetch(`/api/jobs/${jobId}/transcript/bulk-update-speaker`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ segment_indices: [...selectedIndices], speaker: newSpeaker }),
      });
      if (res.ok) {
        clearSelection();
        if (onTranscriptUpdated) onTranscriptUpdated();
      }
    } catch (err) {
      console.error('Bulk speaker change failed:', err);
    } finally {
      setBulkSaving(false);
    }
  };

  // --- Delete segment ---
  const handleDeleteSegment = async (originalIdx) => {
    if (!jobId) return;
    try {
      const res = await fetch(`/api/jobs/${jobId}/transcript/${originalIdx}`, { method: 'DELETE' });
      if (res.ok && onTranscriptUpdated) onTranscriptUpdated();
    } catch (err) {
      console.error('Segment delete failed:', err);
    }
  };

  // --- Insert segment ---
  const openInsertForm = (afterIdx) => {
    const seg = transcript[afterIdx];
    setInsertAfterIdx(afterIdx);
    setInsertText('');
    setInsertSpeaker(seg?.speaker || speakers[0] || 'Speaker 1');
    setInsertStart(seg ? seg.end.toFixed(1) : '0');
    setInsertEnd(seg ? (seg.end + 2).toFixed(1) : '2');
  };

  const cancelInsert = () => {
    setInsertAfterIdx(null);
    setInsertText('');
  };

  const saveInsert = async () => {
    const text = insertText.trim();
    if (!text || !jobId) { cancelInsert(); return; }
    setInsertSaving(true);
    try {
      const res = await fetch(`/api/jobs/${jobId}/transcript`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          start: parseFloat(insertStart) || 0,
          end: parseFloat(insertEnd) || 0,
          text,
          speaker: insertSpeaker || 'Speaker 1',
        }),
      });
      if (res.ok) {
        cancelInsert();
        if (onTranscriptUpdated) onTranscriptUpdated();
      }
    } catch (err) {
      console.error('Insert segment failed:', err);
    } finally {
      setInsertSaving(false);
    }
  };

  const btnStyle = {
    padding: '3px 8px', fontSize: 11, border: 'none', borderRadius: 'var(--radius-sm)', cursor: 'pointer',
  };

  // --- Insert form row ---
  const renderInsertForm = (afterIdx) => {
    if (insertAfterIdx !== afterIdx) return null;
    return (
      <div
        key={`insert-${afterIdx}`}
        style={{
          padding: '8px 0', borderBottom: '1px solid var(--border)',
          background: 'var(--bg-elevated)', margin: '0 -4px', paddingLeft: 4, paddingRight: 4,
          borderRadius: 'var(--radius-sm)',
        }}
      >
        <div style={{ fontSize: 10, color: 'var(--accent-cyan)', fontWeight: 600, marginBottom: 6 }}>
          + INSERT NEW LINE
        </div>
        <div style={{ display: 'flex', gap: 6, marginBottom: 6, flexWrap: 'wrap' }}>
          <select
            value={insertSpeaker}
            onChange={(e) => setInsertSpeaker(e.target.value)}
            style={{
              padding: '3px 6px', fontSize: 11, background: 'var(--bg-base)',
              border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
              color: 'var(--text-primary)', minWidth: 100,
            }}
          >
            {speakers.map((sp) => <option key={sp} value={sp}>{sp}</option>)}
            <option value="__new__">+ New Speaker</option>
          </select>
          {insertSpeaker === '__new__' && (
            <input
              type="text"
              placeholder="Speaker name"
              onChange={(e) => setInsertSpeaker(e.target.value)}
              style={{
                padding: '3px 6px', fontSize: 11, background: 'var(--bg-base)',
                border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
                color: 'var(--text-primary)', width: 100,
              }}
            />
          )}
          <input
            type="number" step="0.1" value={insertStart}
            onChange={(e) => setInsertStart(e.target.value)}
            style={{
              padding: '3px 6px', fontSize: 11, width: 60, background: 'var(--bg-base)',
              border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
              color: 'var(--text-primary)', fontFamily: 'var(--font-mono)',
            }}
            title="Start time (seconds)"
          />
          <span style={{ fontSize: 10, color: 'var(--text-muted)', alignSelf: 'center' }}>to</span>
          <input
            type="number" step="0.1" value={insertEnd}
            onChange={(e) => setInsertEnd(e.target.value)}
            style={{
              padding: '3px 6px', fontSize: 11, width: 60, background: 'var(--bg-base)',
              border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
              color: 'var(--text-primary)', fontFamily: 'var(--font-mono)',
            }}
            title="End time (seconds)"
          />
        </div>
        <textarea
          autoFocus
          value={insertText}
          onChange={(e) => setInsertText(e.target.value)}
          placeholder="Type the transcript line..."
          rows={2}
          onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); saveInsert(); } if (e.key === 'Escape') cancelInsert(); }}
          style={{
            width: '100%', fontSize: 13, lineHeight: 1.5, color: 'var(--text-primary)',
            background: 'var(--bg-base)', border: '1px solid var(--accent-cyan)',
            borderRadius: 'var(--radius-sm)', padding: '4px 8px', resize: 'vertical',
            outline: 'none', fontFamily: 'inherit', boxSizing: 'border-box', marginBottom: 6,
          }}
        />
        <div style={{ display: 'flex', gap: 4 }}>
          <button onClick={saveInsert} disabled={insertSaving || !insertText.trim()}
            style={{ ...btnStyle, background: 'var(--accent-cyan)', color: 'var(--bg-base)', fontWeight: 600, opacity: insertSaving || !insertText.trim() ? 0.5 : 1 }}>
            {insertSaving ? 'Saving...' : 'Insert'}
          </button>
          <button onClick={cancelInsert} style={{ ...btnStyle, background: 'none', color: 'var(--text-muted)', border: '1px solid var(--border)' }}>
            Cancel
          </button>
        </div>
      </div>
    );
  };

  return (
    <div>
      {/* Speaker legend — click to rename */}
      <div
        style={{
          display: 'flex',
          gap: 12,
          marginBottom: 16,
          padding: '8px 12px',
          background: 'var(--bg-elevated)',
          border: '1px solid var(--border)',
          flexWrap: 'wrap',
          alignItems: 'center',
        }}
      >
        {speakers.map((sp) => {
          const color = speakerColor(sp);
          const isEditing = editingSpeaker === sp;
          return (
            <div key={sp} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <div
                style={{
                  width: 10,
                  height: 10,
                  borderRadius: '50%',
                  background: color,
                  flexShrink: 0,
                }}
              />
              {isEditing ? (
                <input
                  autoFocus
                  value={editValue}
                  onChange={(e) => setEditValue(e.target.value)}
                  onKeyDown={handleKeyDown}
                  onBlur={submitRename}
                  disabled={renaming}
                  style={{
                    fontSize: 12,
                    color: color,
                    background: 'var(--bg-base)',
                    border: `1px solid ${color}`,
                    borderRadius: 3,
                    padding: '2px 6px',
                    width: Math.max(80, editValue.length * 8),
                    outline: 'none',
                    fontFamily: 'inherit',
                  }}
                />
              ) : (
                <span
                  onClick={() => startEditing(sp)}
                  title="Click to rename speaker"
                  style={{
                    fontSize: 12,
                    color: color,
                    cursor: 'pointer',
                    borderBottom: '1px dashed transparent',
                    transition: 'border-color 0.15s',
                  }}
                  onMouseEnter={(e) => (e.target.style.borderBottomColor = color)}
                  onMouseLeave={(e) => (e.target.style.borderBottomColor = 'transparent')}
                >
                  {String(sp ?? '')}
                </span>
              )}
            </div>
          );
        })}
        {/* Add speaker inline */}
        {addingSpeaker ? (
          <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <input
              autoFocus
              value={newSpeakerName}
              onChange={(e) => setNewSpeakerName(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && newSpeakerName.trim()) {
                  const name = newSpeakerName.trim();
                  // Assign the next palette color
                  const nextColor = DEFAULT_SPEAKER_PALETTE[speakers.length % DEFAULT_SPEAKER_PALETTE.length];
                  if (onSpeakerAdded) onSpeakerAdded(name, nextColor);
                  setAddingSpeaker(false);
                  setNewSpeakerName('');
                } else if (e.key === 'Escape') {
                  setAddingSpeaker(false);
                  setNewSpeakerName('');
                }
              }}
              onBlur={() => { setAddingSpeaker(false); setNewSpeakerName(''); }}
              placeholder="Speaker name"
              style={{
                fontSize: 12, padding: '2px 6px', width: 120,
                background: 'var(--bg-base)', border: '1px solid var(--accent-cyan)',
                borderRadius: 3, color: 'var(--text-primary)', outline: 'none',
              }}
            />
          </div>
        ) : (
          <button
            onClick={() => setAddingSpeaker(true)}
            style={{
              display: 'flex', alignItems: 'center', gap: 3,
              padding: '2px 8px', fontSize: 11, fontWeight: 600,
              background: 'transparent', color: 'var(--text-muted)',
              border: '1px dashed var(--border)', borderRadius: 3,
              cursor: 'pointer',
            }}
            title="Add a new speaker"
          >
            <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><line x1="12" y1="5" x2="12" y2="19" /><line x1="5" y1="12" x2="19" y2="12" /></svg>
            Add Speaker
          </button>
        )}
        {speakers.length > 0 && !addingSpeaker && (
          <span style={{ fontSize: 10, color: 'var(--text-muted)', marginLeft: 4 }}>
            click name to rename
          </span>
        )}
      </div>

      {/* Search + actions */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 16, flexWrap: 'wrap' }}>
        <input
          type="text"
          placeholder="Search transcript..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          style={{
            flex: 1,
            minWidth: 200,
            padding: '8px 12px',
            borderRadius: 'var(--radius-sm)',
          }}
        />
        <button
          onClick={() => download(toTXT(timeFiltered), 'transcript.txt')}
          style={{
            padding: '8px 12px',
            background: 'var(--bg-elevated)',
            color: 'var(--text-secondary)',
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius-sm)',
            fontSize: 12,
          }}
        >
          .txt
        </button>
        <button
          onClick={() => download(toSRT(timeFiltered), 'transcript.srt')}
          style={{
            padding: '8px 12px',
            background: 'var(--bg-elevated)',
            color: 'var(--text-secondary)',
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius-sm)',
            fontSize: 12,
          }}
        >
          .srt
        </button>
        <button
          onClick={copyAll}
          style={{
            padding: '8px 12px',
            background: 'var(--bg-elevated)',
            color: 'var(--text-secondary)',
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius-sm)',
            fontSize: 12,
          }}
        >
          Copy All
        </button>
        {jobId && speakers.length > 0 && (
          <button
            onClick={selectedIndices.size > 0 && selectedIndices.size === filtered.reduce((s, seg) => { s.add(transcript.indexOf(seg)); return s; }, new Set()).size ? clearSelection : selectAllFiltered}
            style={{
              padding: '8px 12px',
              background: selectedIndices.size > 0 ? 'var(--accent-cyan-dim, rgba(0,217,255,0.08))' : 'var(--bg-elevated)',
              color: selectedIndices.size > 0 ? 'var(--accent-cyan)' : 'var(--text-secondary)',
              border: selectedIndices.size > 0 ? '1px solid var(--accent-cyan)' : '1px solid var(--border)',
              borderRadius: 'var(--radius-sm)',
              fontSize: 12,
              cursor: 'pointer',
              fontWeight: selectedIndices.size > 0 ? 600 : 400,
            }}
          >
            {selectedIndices.size > 0 ? `${selectedIndices.size} Selected` : 'Select All'}
          </button>
        )}
      </div>

      {/* Editable hint */}
      {jobId && (
        <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 8, display: 'flex', alignItems: 'center', gap: 4, flexWrap: 'wrap' }}>
          <span style={{ color: 'var(--accent-cyan)' }}>{'\u270E'}</span>
          Click text to edit. Use speaker dropdown to reassign. Changes update subtitles for new and existing clip exports.
        </div>
      )}

      {/* Multi-select bulk action bar */}
      {jobId && selectedIndices.size > 0 && (
        <div style={{
          display: 'flex', alignItems: 'center', gap: 10, padding: '8px 12px',
          marginBottom: 8, background: 'var(--accent-cyan-dim, rgba(0,217,255,0.08))',
          border: '1px solid var(--accent-cyan)', borderRadius: 'var(--radius-sm)',
          flexWrap: 'wrap',
        }}>
          <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--accent-cyan)' }}>
            {selectedIndices.size} selected
          </span>
          <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>|</span>
          <span style={{ fontSize: 11, color: 'var(--text-secondary)' }}>Set speaker:</span>
          {speakers.map((sp) => (
            <button
              key={sp}
              onClick={() => handleBulkSpeakerChange(sp)}
              disabled={bulkSaving}
              style={{
                ...btnStyle, fontSize: 11, padding: '3px 10px',
                background: 'var(--bg-elevated)', color: speakerColor(sp),
                border: `1px solid ${speakerColor(sp)}`, fontWeight: 600,
                opacity: bulkSaving ? 0.5 : 1, cursor: bulkSaving ? 'wait' : 'pointer',
              }}
            >
              {sp}
            </button>
          ))}
          <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>|</span>
          <button
            onClick={selectAllFiltered}
            style={{ ...btnStyle, fontSize: 10, background: 'var(--bg-elevated)', color: 'var(--text-secondary)', border: '1px solid var(--border)' }}
          >
            Select All
          </button>
          <button
            onClick={clearSelection}
            style={{ ...btnStyle, fontSize: 10, background: 'var(--bg-elevated)', color: 'var(--text-secondary)', border: '1px solid var(--border)' }}
          >
            Deselect
          </button>
        </div>
      )}

      {/* Segments */}
      <div ref={scrollContainerRef} style={{ maxHeight: maxHeight || 500, overflow: 'auto' }}>
        {filtered.map((seg, i) => {
          const color = speakerColor(seg.speaker);
          const originalIdx = transcript.indexOf(seg);
          const isEditingSeg = editingSegIdx === originalIdx;
          const isActiveSeg = activeOriginalIdx === originalIdx;
          const highlightedText = search.trim()
            ? seg.text.replace(
                new RegExp(`(${search.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi'),
                '<mark style="background:var(--accent-cyan-dim);color:var(--accent-cyan)">$1</mark>'
              )
            : seg.text;

          const isSelected = selectedIndices.has(originalIdx);

          return (
            <React.Fragment key={`seg-${originalIdx}`}>
              <div
                ref={isActiveSeg ? activeSegRef : undefined}
                style={{
                  display: 'flex',
                  gap: isMobile ? 8 : 12,
                  padding: '8px 4px',
                  alignItems: 'flex-start',
                  borderRadius: 'var(--radius-sm)',
                  transition: 'background 0.2s, border-color 0.2s',
                  ...(isActiveSeg && !isSelected ? {
                    background: 'var(--accent-cyan-dim, rgba(0,217,255,0.08))',
                    borderLeft: '2px solid var(--accent-cyan)',
                    paddingLeft: 6,
                  } : isSelected ? {
                    background: 'var(--accent-cyan-dim, rgba(0,217,255,0.12))',
                    borderLeft: '2px solid var(--accent-cyan)',
                    paddingLeft: 6,
                  } : {
                    borderLeft: '2px solid transparent',
                  }),
                }}
              >
                {/* Selection checkbox */}
                {jobId && (
                  <div
                    style={{ flexShrink: 0, display: 'flex', alignItems: 'center', paddingTop: 2 }}
                  >
                    <input
                      type="checkbox"
                      checked={isSelected}
                      onChange={(e) => toggleSelect(originalIdx, e.nativeEvent)}
                      onClick={(e) => e.stopPropagation()}
                      title="Select line (Shift+click to select range)"
                      style={{ cursor: 'pointer', accentColor: 'var(--accent-cyan)' }}
                    />
                  </div>
                )}

                {/* Speaker + time column */}
                <div style={{ minWidth: isMobile ? 80 : 110, maxWidth: isMobile ? 110 : 150, flexShrink: 0 }}>
                  {/* Speaker dropdown */}
                  {jobId && speakers.length > 1 ? (
                    <div style={{ position: 'relative', display: 'inline-block', maxWidth: '100%' }}>
                      <select
                        value={seg.speaker}
                        onChange={(e) => handleChangeSpeaker(originalIdx, e.target.value)}
                        title="Change speaker"
                        style={{
                          fontFamily: 'var(--font-mono)', fontSize: 11, color,
                          fontWeight: 600, background: 'transparent', border: 'none',
                          cursor: 'pointer', padding: '0 14px 0 0', appearance: 'none',
                          WebkitAppearance: 'none', MozAppearance: 'none',
                          maxWidth: '100%', outline: 'none',
                        }}
                      >
                        {speakers.map((sp) => <option key={sp} value={sp}>{sp}</option>)}
                      </select>
                      <span style={{
                        position: 'absolute', right: 0, top: '50%', transform: 'translateY(-50%)',
                        pointerEvents: 'none', fontSize: 8, color: 'var(--text-muted)', lineHeight: 1,
                      }}>&#9662;</span>
                    </div>
                  ) : (
                    <span
                      style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color, fontWeight: 600 }}
                    >
                      {String(seg.speaker ?? '')}
                    </span>
                  )}
                  <br />
                  <span
                    style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-muted)', cursor: 'pointer' }}
                    onClick={() => onSeek?.(seg.start)}
                    title="Seek to this time"
                  >
                    {formatTime(seg.start)}
                  </span>
                </div>

                {/* Text content */}
                {isEditingSeg ? (
                  <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 4 }}>
                    <textarea
                      autoFocus
                      value={editSegText}
                      onChange={(e) => setEditSegText(e.target.value)}
                      onKeyDown={handleSegKeyDown}
                      disabled={savingSegment}
                      rows={Math.max(2, Math.ceil(editSegText.length / 60))}
                      style={{
                        width: '100%', fontSize: 13, lineHeight: 1.5,
                        color: 'var(--text-primary)', background: 'var(--bg-base)',
                        border: '1px solid var(--accent-cyan)', borderRadius: 'var(--radius-sm)',
                        padding: '4px 8px', resize: 'vertical', outline: 'none',
                        fontFamily: 'inherit', boxSizing: 'border-box',
                      }}
                    />
                    <div style={{ display: 'flex', gap: 4 }}>
                      <button onClick={saveSegEdit} disabled={savingSegment}
                        style={{ ...btnStyle, background: 'var(--accent-cyan)', color: 'var(--bg-base)', fontWeight: 600 }}>
                        {savingSegment ? 'Saving...' : 'Save'}
                      </button>
                      <button onClick={cancelSegEdit}
                        style={{ ...btnStyle, background: 'none', color: 'var(--text-muted)', border: '1px solid var(--border)' }}>
                        Cancel
                      </button>
                      <span style={{ fontSize: 9, color: 'var(--text-muted)', alignSelf: 'center', marginLeft: 4 }}>
                        Enter to save, Esc to cancel
                      </span>
                    </div>
                  </div>
                ) : (
                  <div
                    onClick={(e) => startSegEdit(originalIdx, seg.text, e)}
                    title="Click to edit text"
                    className="transcript-seg-editable"
                    style={{
                      fontSize: 13, lineHeight: 1.5, color: 'var(--text-primary)',
                      cursor: 'pointer', flex: 1, display: 'flex', alignItems: 'flex-start',
                      gap: 6, padding: '2px 4px', borderRadius: 'var(--radius-sm)',
                      transition: 'background 0.15s',
                    }}
                    onMouseEnter={(e) => { e.currentTarget.style.background = 'var(--bg-elevated)'; }}
                    onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent'; }}
                  >
                    <span style={{ flex: 1 }} dangerouslySetInnerHTML={{ __html: highlightedText }} />
                    <span style={{ fontSize: 10, color: 'var(--text-muted)', opacity: 0.4, flexShrink: 0, marginTop: 2 }}>
                      {'\u270E'}
                    </span>
                  </div>
                )}

                {/* Action buttons: insert + delete */}
                {jobId && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 4, flexShrink: 0 }}>
                    <button
                      onClick={() => openInsertForm(originalIdx)}
                      title="Insert line below"
                      style={{
                        ...btnStyle, background: 'var(--bg-elevated)', color: 'var(--accent-cyan)',
                        border: '1px solid var(--border)', fontSize: 10, lineHeight: 1, padding: '3px 6px',
                        opacity: 0.7, transition: 'opacity 0.15s, background 0.15s',
                        whiteSpace: 'nowrap',
                      }}
                      onMouseEnter={(e) => { e.currentTarget.style.opacity = '1'; e.currentTarget.style.background = 'var(--accent-cyan)'; e.currentTarget.style.color = 'var(--bg-base)'; }}
                      onMouseLeave={(e) => { e.currentTarget.style.opacity = '0.7'; e.currentTarget.style.background = 'var(--bg-elevated)'; e.currentTarget.style.color = 'var(--accent-cyan)'; }}
                    >
                      + Add
                    </button>
                    <button
                      onClick={() => { if (window.confirm('Delete this transcript line?')) handleDeleteSegment(originalIdx); }}
                      title="Delete line"
                      style={{
                        ...btnStyle, background: 'transparent', color: 'var(--danger, #ff3b30)',
                        border: '1px solid var(--border)', fontSize: 12, lineHeight: 1, padding: '2px 5px',
                        opacity: 0.5, transition: 'opacity 0.15s',
                      }}
                      onMouseEnter={(e) => { e.currentTarget.style.opacity = '1'; }}
                      onMouseLeave={(e) => { e.currentTarget.style.opacity = '0.5'; }}
                    >
                      {'\u2715'}
                    </button>
                  </div>
                )}
              </div>

              {/* Hover divider to insert a line */}
              {jobId && insertAfterIdx !== originalIdx && (
                <div
                  className="transcript-insert-divider"
                  onClick={() => openInsertForm(originalIdx)}
                  style={{
                    position: 'relative',
                    height: 1,
                    background: 'var(--border)',
                    cursor: 'pointer',
                    margin: '0 -4px',
                    padding: '6px 4px',
                    backgroundClip: 'content-box',
                  }}
                >
                  <span
                    className="transcript-insert-divider-btn"
                    style={{
                      position: 'absolute', left: '50%', top: '50%',
                      transform: 'translate(-50%, -50%)',
                      fontSize: 9, fontWeight: 600, lineHeight: 1,
                      padding: '2px 10px', borderRadius: 'var(--radius-sm)',
                      background: 'var(--accent-cyan)', color: 'var(--bg-base)',
                      whiteSpace: 'nowrap', pointerEvents: 'none',
                      opacity: 0, transition: 'opacity 0.15s',
                    }}
                  >
                    + Add line
                  </span>
                </div>
              )}
              {/* Plain divider when insert form is shown for this segment */}
              {insertAfterIdx === originalIdx && (
                <div style={{ height: 1, background: 'var(--border)', margin: '0 -4px' }} />
              )}

              {/* Insert form appears after this segment */}
              {renderInsertForm(originalIdx)}
            </React.Fragment>
          );
        })}
        {filtered.length === 0 && (
          <div style={{ textAlign: 'center', padding: 24, color: 'var(--text-muted)' }}>
            No matching segments found.
          </div>
        )}

        {/* Insert at the very end */}
        {jobId && transcript.length > 0 && !timeRange && (
          <div style={{ padding: '8px 0', textAlign: 'center' }}>
            <button
              onClick={() => openInsertForm(transcript.length - 1)}
              style={{
                ...btnStyle, background: 'var(--bg-elevated)', color: 'var(--accent-cyan)',
                border: '1px solid var(--border)', fontSize: 11, padding: '4px 12px',
              }}
            >
              + Add Line at End
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
