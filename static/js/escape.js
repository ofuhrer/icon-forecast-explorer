/**
 * HTML escaping utility.
 *
 * Exported so it can be imported by both main.js and any future module
 * that needs to build safe HTML strings without a framework.
 */

export function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
