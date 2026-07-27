/** 会話履歴の型定義 */

export interface SessionSummary {
	session_id: string;
	started_at: string;
	duration_sec: number;
	mode: string;
	turn_count: number;
	summary: string | null;
	/** 未要約セッションの見出しフォールバック（最初のユーザ発話の先頭） */
	first_user_preview: string;
	matched_preview: string | null;
}

export interface SessionDetailData {
	session_id: string;
	started_at: string;
	ended_at: string;
	duration_sec: number;
	mode: string;
	modes_used: string[];
	instance_name: string;
	base_model: string;
	turns: TurnData[];
	turn_count: number;
	context_files: string[];
	cartridge_ids: string[];
	summary: string | null;
}

export interface TurnData {
	role: string;
	content: string;
	timestamp?: string;
}

export interface DateGroup {
	label: string;
	sessions: SessionSummary[];
}
