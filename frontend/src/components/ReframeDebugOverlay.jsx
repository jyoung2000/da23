import React, { useMemo } from 'react';

/**
 * ReframeDebugOverlay — shows pacing, min_hold, confidence, and fallback
 * reasons as a timeline strip below the preview player.
 *
 * Enabled via dev flag or keyboard shortcut. Reuses the RenderPlan debug
 * data — no new API needed.
 *
 * Props:
 *   renderPlan: RenderPlan JSON from /api/jobs/{id}/render_plan?debug=1
 *   currentTime: current playback position (seconds, clip-relative)
 *   clipDuration: total clip duration in seconds
 */
export default function ReframeDebugOverlay({ renderPlan, currentTime = 0, clipDuration = 0 }) {
  const debug = renderPlan?.debug;
  const ops = renderPlan?.ops;

  if (!debug && !ops) return null;

  const duration = clipDuration || renderPlan?.total_duration_sec || 1;

  // Strategy color map
  const strategyColors = {
    crop: '#3b82f6',           // blue
    tracking_crop: '#8b5cf6',  // purple
    wide_master: '#f59e0b',    // amber
    blur_fill: '#6366f1',      // indigo
    split_screen: '#10b981',   // emerald
    stacked_gameplay: '#14b8a6', // teal
    grid_2x2: '#06b6d4',      // cyan
  };

  // Confidence color
  const confColor = (conf) => {
    if (conf >= 0.7) return '#22c55e';  // green
    if (conf >= 0.5) return '#f59e0b';  // amber
    if (conf >= 0.3) return '#ef4444';  // red
    return '#991b1b';                   // dark red
  };

  // Pacing sparkline
  const pacingSvg = useMemo(() => {
    const pacing = debug?.pacing_per_sec;
    if (!pacing || pacing.length === 0) return null;

    const w = 600;
    const h = 30;
    const points = pacing.map((v, i) => {
      const x = (i / Math.max(1, pacing.length - 1)) * w;
      const y = h - v * h;
      return `${x},${y}`;
    }).join(' ');

    return (
      <svg width="100%" height={h} viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none"
           style={{ display: 'block' }}>
        <polyline points={points} fill="none" stroke="#60a5fa" strokeWidth="1.5" />
        {/* Playhead */}
        <line x1={(currentTime / duration) * w} y1={0}
              x2={(currentTime / duration) * w} y2={h}
              stroke="white" strokeWidth="1" opacity="0.6" />
      </svg>
    );
  }, [debug?.pacing_per_sec, currentTime, duration]);

  // MinHold sparkline
  const minHoldSvg = useMemo(() => {
    const minHold = debug?.min_hold_per_sec;
    if (!minHold || minHold.length === 0) return null;

    const w = 600;
    const h = 20;
    const maxHold = 3.0;
    const points = minHold.map((v, i) => {
      const x = (i / Math.max(1, minHold.length - 1)) * w;
      const y = h - (v / maxHold) * h;
      return `${x},${y}`;
    }).join(' ');

    return (
      <svg width="100%" height={h} viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none"
           style={{ display: 'block' }}>
        <polyline points={points} fill="none" stroke="#fbbf24" strokeWidth="1.5" />
      </svg>
    );
  }, [debug?.min_hold_per_sec]);

  return (
    <div style={{
      width: '100%',
      background: 'rgba(0, 0, 0, 0.85)',
      borderTop: '1px solid rgba(255, 255, 255, 0.1)',
      padding: '6px 8px',
      fontSize: 10,
      color: '#9ca3af',
      fontFamily: 'monospace',
    }}>
      {/* Pacing score line chart */}
      {pacingSvg && (
        <div style={{ marginBottom: 4 }}>
          <span style={{ color: '#60a5fa', marginRight: 8 }}>pacing</span>
          {pacingSvg}
        </div>
      )}

      {/* MinHold line chart */}
      {minHoldSvg && (
        <div style={{ marginBottom: 4 }}>
          <span style={{ color: '#fbbf24', marginRight: 8 }}>min_hold</span>
          {minHoldSvg}
        </div>
      )}

      {/* Strategy chips */}
      {ops && ops.length > 0 && (
        <div style={{ display: 'flex', gap: 1, marginBottom: 3, height: 14 }}>
          {ops.map((op, i) => {
            const widthPct = ((op.end_sec - op.start_sec) / duration) * 100;
            return (
              <div key={i} style={{
                width: `${widthPct}%`,
                minWidth: 2,
                height: '100%',
                background: strategyColors[op.kind] || '#4b5563',
                borderRadius: 2,
                overflow: 'hidden',
                fontSize: 8,
                color: 'white',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                whiteSpace: 'nowrap',
              }}>
                {widthPct > 5 ? op.kind.replace('_', ' ') : ''}
              </div>
            );
          })}
        </div>
      )}

      {/* Confidence bars */}
      {debug?.confidence_per_segment && debug.confidence_per_segment.length > 0 && ops && (
        <div style={{ display: 'flex', gap: 1, marginBottom: 3, height: 10 }}>
          {ops.map((op, i) => {
            const widthPct = ((op.end_sec - op.start_sec) / duration) * 100;
            const conf = debug.confidence_per_segment[i] ?? 1.0;
            return (
              <div key={i} style={{
                width: `${widthPct}%`,
                minWidth: 2,
                height: '100%',
                background: confColor(conf),
                borderRadius: 1,
                opacity: 0.8,
              }}
              title={`confidence: ${conf.toFixed(2)}`}
              />
            );
          })}
        </div>
      )}

      {/* Fallback reason tags */}
      {debug?.fallback_reasons && debug.fallback_reasons.some(r => r) && (
        <div style={{ display: 'flex', gap: 2, flexWrap: 'wrap', marginTop: 2 }}>
          {debug.fallback_reasons.map((reason, i) =>
            reason ? (
              <span key={i} style={{
                padding: '1px 4px',
                borderRadius: 3,
                background: 'rgba(239, 68, 68, 0.2)',
                color: '#fca5a5',
                fontSize: 9,
              }}>
                #{i}: {reason}
              </span>
            ) : null
          )}
        </div>
      )}
    </div>
  );
}
