/* AgriDoctor AI - Service Worker (app-shell cache).
 * Caches static assets for offline UI load. NEVER caches API responses. */

const CACHE = 'agridoctor-v2';
const SHELL = [
  '/',
  '/index.html',
  '/css/styles.css',
  '/js/api.js',
  '/js/ui.js',
  '/js/app.js',
  '/manifest.webmanifest',
  '/icons/icon-192.png',
  '/icons/icon-512.png',
];

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (e) => {
  const url = new URL(e.request.url);
  // Never intercept API or non-GET requests.
  if (e.request.method !== 'GET' || url.pathname.startsWith('/api/')) return;

  // Cache-first for same-origin shell assets; network fallback.
  if (url.origin === self.location.origin) {
    e.respondWith(
      caches.match(e.request).then((hit) => hit || fetch(e.request).catch(() => caches.match('/index.html')))
    );
  }
});
