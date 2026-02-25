export async function fetchJson(url, options = {}) {
  const { timeoutMs, ...fetchOptions } = options;
  let timeoutId = null;
  let didTimeout = false;

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
