import { sveltekit } from '@sveltejs/kit/vite';
import tailwindcss from '@tailwindcss/vite';
import { defineConfig } from 'vite';

const apiPort = process.env.VITE_API_PORT ?? '8000';

export default defineConfig({
	plugins: [tailwindcss(), sveltekit()],
	server: {
		proxy: {
			'/api': {
				// 127.0.0.1 固定 (Node 17+ は localhost を ::1 に解決しうる)。Host を書き換えない —
				// backend の Host 許可リストが frontend のポートを見て判定する (c_06 §1.5)。
				target: `http://127.0.0.1:${apiPort}`,
				changeOrigin: false
			}
		}
	}
});
