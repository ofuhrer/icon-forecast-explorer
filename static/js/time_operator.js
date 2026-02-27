export function timeOperatorStepHours(timeOperator) {
  const op = String(timeOperator || "none");
  const m = op.match(/^(avg|acc|min|max)_(\d+)h$/);
  if (!m) {
    return null;
  }
  const hours = Number(m[2]);
  return Number.isFinite(hours) && hours > 0 ? hours : null;
}

export function filterLeadsForTimeOperator(leads, timeOperator, leadForDisplay) {
  const stepHours = timeOperatorStepHours(timeOperator);
  const list = Array.isArray(leads) ? leads.map((v) => Number(v)).filter(Number.isFinite) : [];
  if (!stepHours || list.length === 0) {
    return list;
  }
  const filtered = list.filter((lead) => {
    const displayLead = Number(leadForDisplay(lead));
    return Number.isFinite(displayLead) && displayLead >= 0 && displayLead % stepHours === 0;
  });
  return filtered.length > 0 ? filtered : list;
}
