export function selectedOptionText(selectEl, fallback = "-") {
  if (!selectEl || selectEl.selectedIndex < 0) {
    return fallback;
  }
  const option = selectEl.options[selectEl.selectedIndex];
  if (!option) {
    return fallback;
  }
  return String(option.textContent || fallback).trim();
}
