/**
 * エディション判定モジュール
 *
 * VITE_EVOREF_EDITION 環境変数でエディションを制御する。
 * - 未設定 or 'pro': Pro 扱い（Pro 機能有効、Develop 機能無効）
 * - 'free': Free 版（Pro / Develop 機能無効）
 * - 'develop': Develop 版（Pro 機能 + Develop 機能の両方が有効、社内/開発者限定）
 *
 * 階層: Develop ⊇ Pro ⊇ Free
 * isPro は Develop でも true を返す (Develop は Pro の上位互換)。
 */
export type Edition = 'free' | 'pro' | 'develop';

export const edition: Edition =
	(import.meta.env.VITE_EVOREF_EDITION as Edition) || 'pro';

export const isPro: boolean = edition === 'pro' || edition === 'develop';
export const isDevelop: boolean = edition === 'develop';
