/** サーバー制御 API */

import { request } from './_client';

export type ServerName = 'base' | 'embed';

export interface ServerActionResponse {
	name: string;
	action: string;
	success: boolean;
	message: string;
}

/** サーバー起動 */
export async function startServer(name: ServerName, force = false): Promise<ServerActionResponse> {
	return request<ServerActionResponse>('POST', '/servers/start', { name, force });
}

/** サーバー停止 */
export async function stopServer(name: ServerName): Promise<ServerActionResponse> {
	return request<ServerActionResponse>('POST', '/servers/stop', { name });
}
