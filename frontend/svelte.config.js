import adapter from '@sveltejs/adapter-auto';

/** @type {import('@sveltejs/kit').Config} */
const config = {
	kit: {
		// adapter-auto only supports some environments, see https://svelte.dev/docs/kit/adapter-auto for a list.
		// If your environment is not supported, or you settled on a specific environment, switch out the adapter.
		// See https://svelte.dev/docs/kit/adapters for more information about adapters.
		adapter: adapter(),
		// 未信頼コンテンツ (LLM 出力・テーマ CSS) の外送を止める多層防御 (docs/c_11 §2.4)。
		// script-src / style-src / default-src は指定しない — インライン初期化・HMR・style 属性・
		// テーマの <style> を壊すため。API は Vite proxy 経由の相対パス (/api) なので 'self' で足りる。
		csp: {
			mode: 'auto',
			directives: {
				'img-src': ['self', 'data:', 'blob:'],
				'font-src': ['self', 'data:'],
				'connect-src': ['self'],
				'object-src': ['none'],
				'base-uri': ['self'],
				'form-action': ['self'],
				'frame-ancestors': ['none']
			}
		}
	}
};

export default config;
