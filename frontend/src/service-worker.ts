/// <reference types="@sveltejs/kit" />
/// <reference no-default-lib="true"/>
/// <reference lib="esnext" />
/// <reference lib="webworker" />

import { build, files, version } from '$service-worker';

const sw = self as unknown as ServiceWorkerGlobalScope;

const CACHE_NAME = `evoref-cache-${version}`;

// Build output + static files
const ASSETS = [...build, ...files];

sw.addEventListener('install', (event) => {
	event.waitUntil(
		caches.open(CACHE_NAME).then((cache) => cache.addAll(ASSETS)).then(() => {
			sw.skipWaiting();
		})
	);
});

sw.addEventListener('activate', (event) => {
	event.waitUntil(
		caches.keys().then((keys) => {
			return Promise.all(
				keys.filter((key) => key !== CACHE_NAME).map((key) => caches.delete(key))
			);
		}).then(() => {
			sw.clients.claim();
		})
	);
});

sw.addEventListener('fetch', (event) => {
	const url = new URL(event.request.url);

	// Skip API requests — always go to network
	if (url.pathname.startsWith('/api')) {
		return;
	}

	// For assets, try cache first
	if (ASSETS.includes(url.pathname)) {
		event.respondWith(
			caches.match(event.request).then((cached) => {
				return cached || fetch(event.request);
			})
		);
		return;
	}

	// For navigation requests, try network first, fall back to cache
	if (event.request.mode === 'navigate') {
		event.respondWith(
			fetch(event.request).catch(() => {
				return caches.match(event.request).then((cached) => {
					return cached || caches.match('/') || new Response('Offline', { status: 503 });
				});
			})
		);
		return;
	}
});
