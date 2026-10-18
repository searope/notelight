// A service worker that makes the demo cross-origin isolated on hosts that
// cannot send headers, such as GitHub Pages. Isolation is what lets
// WebAssembly use threads (SharedArrayBuffer): the embedder then runs on 4
// threads, about 10 ms per query instead of 29 ms on one.
//
// It adds two headers to this origin's responses and changes nothing else:
//   Cross-Origin-Opener-Policy: same-origin
//   Cross-Origin-Embedder-Policy: require-corp
// No caching and no rewriting of bodies, and cross-origin requests (the model
// download from huggingface.co, which is a CORS request and so allowed under
// require-corp) pass through untouched. app.js registers it and reloads once,
// because a page only becomes isolated when its own navigation response
// carries the headers.
//
// Same-origin requests are made with `cache: "no-cache"`: GitHub Pages sends
// `Cache-Control: max-age=600`, so without it a browser reuses each file for
// 10 minutes and, right after a deploy, can pair the new index.html with an
// old app.js. With it, every load revalidates by ETag: an unchanged file is a
// 304 with no body, a changed one is fetched. Requests that already ask for a
// specific cache mode (reload, no-store, ...) keep it.

self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (event) => event.waitUntil(self.clients.claim()));

self.addEventListener("fetch", (event) => {
  const request = event.request;
  if (new URL(request.url).origin !== self.location.origin) return;
  // Chrome throws for this combination; let the browser handle it.
  if (request.cache === "only-if-cached" && request.mode !== "same-origin") return;
  event.respondWith(
    fetch(revalidated(request)).then((response) => {
      if (response.status === 0) return response; // opaque: headers are not ours to set
      const headers = new Headers(response.headers);
      headers.set("Cross-Origin-Opener-Policy", "same-origin");
      headers.set("Cross-Origin-Embedder-Policy", "require-corp");
      return new Response(response.body, {
        status: response.status,
        statusText: response.statusText,
        headers,
      });
    }),
  );
});

/**
 * `request` with `cache: "no-cache"` when it uses the default cache mode.
 * A navigation request cannot be copied with new options (Chrome throws), so
 * the page itself is re-requested by URL, keeping its manual redirect mode so
 * the browser still follows GitHub Pages' `/notelight` -> `/notelight/` itself.
 */
function revalidated(request) {
  if (request.cache !== "default") return request;
  if (request.mode === "navigate") {
    return new Request(request.url, {
      cache: "no-cache",
      credentials: request.credentials,
      headers: request.headers,
      redirect: "manual",
    });
  }
  return new Request(request, { cache: "no-cache" });
}
