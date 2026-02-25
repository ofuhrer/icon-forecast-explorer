export const SERIES_TYPES = ["control", "p10", "p90"];

export function seriesHasMissingValues(seriesData, requestedTypes) {
  if (!seriesData || !seriesData.lead_hours || !seriesData.values) {
    return true;
  }
  const leads = seriesData.lead_hours || [];
  const values = seriesData.values || {};
  for (const typeId of requestedTypes) {
    const arr = values[typeId] || [];
    if (arr.length !== leads.length) {
      return true;
    }
    if (arr.some((v) => v == null || !Number.isFinite(Number(v)))) {
      return true;
    }
  }
  return false;
}

export function renderMeteogram({
  canvas,
  pointEl,
  pinnedPoint,
  seriesData,
  lead,
  errorText = "",
  statusText = "",
}) {
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, w, h);

  if (!pinnedPoint) {
    if (pointEl) {
      pointEl.textContent = "Click map to pin a point";
    }
    ctx.fillStyle = "#7a8a9b";
    ctx.font = "12px IBM Plex Sans, Segoe UI, sans-serif";
    ctx.fillText("No point selected", 10, 20);
    return;
  }

  const basePointText = `${pinnedPoint.lat.toFixed(3)}, ${pinnedPoint.lon.toFixed(3)}`;
  if (pointEl) {
    pointEl.textContent = statusText ? `${basePointText} ${statusText}` : basePointText;
  }

  if (errorText) {
    ctx.fillStyle = "#c62828";
    ctx.font = "12px IBM Plex Sans, Segoe UI, sans-serif";
    ctx.fillText(errorText, 10, 20);
    return;
  }

  if (!seriesData || !seriesData.lead_hours || seriesData.lead_hours.length === 0) {
    ctx.fillStyle = "#7a8a9b";
    ctx.font = "12px IBM Plex Sans, Segoe UI, sans-serif";
    ctx.fillText("No series data", 10, 20);
    return;
  }

  const leads = seriesData.lead_hours;
  const vals = seriesData.values || {};
  const lines = {
    control: vals.control || [],
    p10: vals.p10 || [],
    p90: vals.p90 || [],
  };

  const all = [];
  Object.values(lines).forEach((arr) => {
    arr.forEach((v) => {
      if (v != null && Number.isFinite(Number(v))) {
        all.push(Number(v));
      }
    });
  });
  if (all.length === 0) {
    ctx.fillStyle = "#7a8a9b";
    ctx.font = "12px IBM Plex Sans, Segoe UI, sans-serif";
    ctx.fillText("No valid values", 10, 20);
    return;
  }

  const padL = 36;
  const padR = 8;
  const padT = 8;
  const padB = 40;
  const x0 = padL;
  const y0 = padT;
  const x1 = w - padR;
  const y1 = h - padB;

  const rawMin = Math.min(...all);
  const rawMax = Math.max(...all);
  const scale = computeNiceScale(rawMin, rawMax, 5);
  const minV = scale.min;
  const maxV = scale.max;
  const tickStep = scale.step;

  const xAt = (forecastLead) => {
    const idx = leads.indexOf(forecastLead);
    if (idx < 0) {
      return x0;
    }
    const t = leads.length === 1 ? 0 : idx / (leads.length - 1);
    return x0 + t * (x1 - x0);
  };
  const yAt = (v) => y1 - ((v - minV) / (maxV - minV)) * (y1 - y0);

  ctx.strokeStyle = "#d7dde5";
  ctx.lineWidth = 1;
  ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);

  ctx.strokeStyle = "#c4cdd8";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(x0, y1);
  ctx.lineTo(x1, y1);
  ctx.moveTo(x0, y0);
  ctx.lineTo(x0, y1);
  ctx.stroke();

  ctx.strokeStyle = "#e4e8ee";
  ctx.fillStyle = "#7a8a9b";
  ctx.font = "10px IBM Plex Sans, Segoe UI, sans-serif";
  for (let v = minV; v <= maxV + tickStep * 0.5; v += tickStep) {
    const y = yAt(v);
    ctx.beginPath();
    ctx.moveTo(x0, y);
    ctx.lineTo(x1, y);
    ctx.stroke();
    ctx.strokeStyle = "#c4cdd8";
    ctx.beginPath();
    ctx.moveTo(x0 - 4, y);
    ctx.lineTo(x0, y);
    ctx.stroke();
    ctx.strokeStyle = "#e4e8ee";
    ctx.fillText(formatTick(v), 2, y + 3);
  }

  const p10 = lines.p10;
  const p90 = lines.p90;
  const control = lines.control;
  const controlIndices = [];
  for (let i = 0; i < leads.length; i += 1) {
    const cv = control[i];
    if (cv != null && Number.isFinite(Number(cv))) {
      controlIndices.push(i);
    }
  }
  const hasFullBandForControl =
    controlIndices.length > 0 &&
    p10.length === leads.length &&
    p90.length === leads.length &&
    controlIndices.every(
      (i) =>
        p10[i] != null &&
        p90[i] != null &&
        Number.isFinite(Number(p10[i])) &&
        Number.isFinite(Number(p90[i]))
    );
  const bandHiddenReason = hasFullBandForControl
    ? ""
    : "band hidden: missing p10/p90 for some control points";

  if (hasFullBandForControl) {
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < leads.length; i += 1) {
      const lo = p10[i];
      if (lo == null) {
        continue;
      }
      const x = xAt(leads[i]);
      const y = yAt(Number(lo));
      if (!started) {
        ctx.moveTo(x, y);
        started = true;
      } else {
        ctx.lineTo(x, y);
      }
    }
    for (let i = leads.length - 1; i >= 0; i -= 1) {
      const hi = p90[i];
      if (hi == null) {
        continue;
      }
      const x = xAt(leads[i]);
      const y = yAt(Number(hi));
      ctx.lineTo(x, y);
    }
    if (started) {
      ctx.closePath();
      ctx.fillStyle = "rgba(66, 133, 244, 0.18)";
      ctx.fill();
    }
  }

  const drawLine = (arr, color, width = 1.4, drawPoints = false) => {
    if (!arr || arr.length === 0) {
      return;
    }
    ctx.beginPath();
    let started = false;
    const points = [];
    for (let i = 0; i < leads.length; i += 1) {
      const v = arr[i];
      if (v == null) {
        continue;
      }
      const x = xAt(leads[i]);
      const y = yAt(Number(v));
      points.push({ x, y });
      if (!started) {
        ctx.moveTo(x, y);
        started = true;
      } else {
        ctx.lineTo(x, y);
      }
    }
    if (started) {
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.stroke();
      if (drawPoints) {
        ctx.fillStyle = color;
        for (const p of points) {
          ctx.beginPath();
          ctx.arc(p.x, p.y, 1.9, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }
  };

  drawLine(lines.control, "#1b263b", 1.8, true);

  if (lead != null && leads.includes(lead)) {
    const x = xAt(lead);
    ctx.beginPath();
    ctx.moveTo(x, y0);
    ctx.lineTo(x, y1);
    ctx.strokeStyle = "rgba(198, 40, 40, 0.65)";
    ctx.lineWidth = 1;
    ctx.stroke();
  }

  ctx.fillStyle = "#54667a";
  ctx.font = "11px IBM Plex Sans, Segoe UI, sans-serif";
  const xTickIndices = chooseTickIndices(leads.length, 6);
  for (const idx of xTickIndices) {
    const fcLead = leads[idx];
    const x = xAt(fcLead);
    ctx.strokeStyle = "#c4cdd8";
    ctx.beginPath();
    ctx.moveTo(x, y1);
    ctx.lineTo(x, y1 + 4);
    ctx.stroke();
    const label = `+${fcLead}h`;
    const textWidth = ctx.measureText(label).width;
    const tx = Math.max(x0, Math.min(x1 - textWidth, x - textWidth / 2));
    ctx.fillText(label, tx, h - 6);
  }

  if (pointEl) {
    const parts = [];
    if (statusText) {
      parts.push(statusText);
    }
    if (bandHiddenReason) {
      parts.push(`(${bandHiddenReason})`);
    }
    pointEl.textContent = parts.length > 0 ? `${basePointText} ${parts.join(" ")}` : basePointText;
  }
}

function formatTick(v) {
  if (Math.abs(v) >= 100 || Number.isInteger(v)) {
    return String(Math.round(v));
  }
  return v.toFixed(1);
}

function computeNiceScale(min, max, desiredTicks = 5) {
  if (!(max > min)) {
    return { min: min - 1, max: max + 1, step: 0.5 };
  }
  const range = niceNumber(max - min, false);
  const step = niceNumber(range / Math.max(2, desiredTicks - 1), true);
  const niceMin = Math.floor(min / step) * step;
  const niceMax = Math.ceil(max / step) * step;
  return { min: niceMin, max: niceMax, step };
}

function niceNumber(value, round) {
  const exponent = Math.floor(Math.log10(value));
  const fraction = value / 10 ** exponent;
  let niceFraction;
  if (round) {
    if (fraction < 1.5) {
      niceFraction = 1;
    } else if (fraction < 3) {
      niceFraction = 2;
    } else if (fraction < 7) {
      niceFraction = 5;
    } else {
      niceFraction = 10;
    }
  } else if (fraction <= 1) {
    niceFraction = 1;
  } else if (fraction <= 2) {
    niceFraction = 2;
  } else if (fraction <= 5) {
    niceFraction = 5;
  } else {
    niceFraction = 10;
  }
  return niceFraction * 10 ** exponent;
}

function chooseTickIndices(count, maxTicks) {
  if (count <= 0) {
    return [];
  }
  if (count <= maxTicks) {
    return Array.from({ length: count }, (_, i) => i);
  }
  const indices = [0];
  const step = (count - 1) / (maxTicks - 1);
  for (let i = 1; i < maxTicks - 1; i += 1) {
    indices.push(Math.round(i * step));
  }
  indices.push(count - 1);
  return Array.from(new Set(indices)).sort((a, b) => a - b);
}
