export async function fetchJson(url, options = {}) {
  const { timeoutMs, ...fetchOptions } = options;
  let timeoutId = null;
  let didTimeout = false;

  const shouldAttachContext = (() => {
    try {
      const u = new URL(String(url), window.location.href);
      return u.origin === window.location.origin && u.pathname.startsWith("/api/");
    } catch (_err) {
      return false;
    }
  })();
  if (shouldAttachContext && typeof window.__iconForecastRequestContext === "function") {
    const ctx = window.__iconForecastRequestContext() || {};
    const headers = new Headers(fetchOptions.headers || {});
    if (ctx.viewToken) {
      headers.set("x-view-token", String(ctx.viewToken));
    }
    if (ctx.requestId) {
      headers.set("x-request-id", String(ctx.requestId));
    }
    fetchOptions.headers = headers;
  }

  if (timeoutMs != null) {
    const timeoutController = new AbortController();
    const externalSignal = fetchOptions.signal;
    if (externalSignal) {
      externalSignal.addEventListener("abort", () => timeoutController.abort(), { once: true });
    }
    fetchOptions.signal = timeoutController.signal;
    timeoutId = setTimeout(() => {
      didTimeout = true;
      timeoutController.abort();
    }, timeoutMs);
  }

  try {
    const response = await fetch(url, fetchOptions);
    if (!response.ok) {
      const body = await response.text();
      throw new Error(`${response.status}: ${body}`);
    }
    return response.json();
  } catch (err) {
    if (err.name === "AbortError") {
      if (didTimeout) {
        const timeoutError = new Error(`Request timeout for ${url}`);
        timeoutError.name = "TimeoutError";
        throw timeoutError;
      }
      throw err;
    }
    throw err;
  } finally {
    if (timeoutId != null) {
      clearTimeout(timeoutId);
    }
  }
}
