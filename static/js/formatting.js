export function formatInit(initStr) {
  const y = initStr.slice(0, 4);
  const m = initStr.slice(4, 6);
  const d = initStr.slice(6, 8);
  const h = initStr.slice(8, 10);
  return `${y}-${m}-${d} ${h}:00 UTC`;
}

export function validTimeFromInitAndLead(initStr, leadHours) {
  const y = Number(initStr.slice(0, 4));
  const m = Number(initStr.slice(4, 6)) - 1;
  const d = Number(initStr.slice(6, 8));
  const h = Number(initStr.slice(8, 10));
  const dt = new Date(Date.UTC(y, m, d, h, 0, 0));
  dt.setUTCHours(dt.getUTCHours() + leadHours);
  return dt;
}

export function formatSwissLocal(date) {
  const tz = "Europe/Zurich";
  const weekday = new Intl.DateTimeFormat("en-US", {
    timeZone: tz,
    weekday: "short",
  })
    .format(date)
    .replace(",", "");
  const datePart = new Intl.DateTimeFormat("de-CH", {
    timeZone: tz,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).format(date);
  const timePart = new Intl.DateTimeFormat("de-CH", {
    timeZone: tz,
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).format(date);
  return `${weekday} ${datePart} ${timePart}`;
}

export function formatLegendValue(value) {
  if (!Number.isFinite(value)) {
    return "-";
  }
  const abs = Math.abs(value);
  let decimals = 0;
  if (abs < 1) {
    decimals = 2;
  } else if (abs < 10) {
    decimals = 1;
  }
  return value
    .toFixed(decimals)
    .replace(/\.0+$/, "")
    .replace(/(\.\d*[1-9])0+$/, "$1");
}
