/** 設定フィールドのエラー表示とコントロールを結ぶ属性 (FieldShell と各フィールドで共用) */

/** FieldShell のエラー行の ID */
export function fieldErrorId(fieldId: string): string {
	return `${fieldId}-error`;
}

/** エラーがあるときだけコントロールに付ける aria-invalid / aria-describedby */
export function errorAria(
	fieldId: string,
	error: string
): { 'aria-invalid'?: 'true'; 'aria-describedby'?: string } {
	return error ? { 'aria-invalid': 'true', 'aria-describedby': fieldErrorId(fieldId) } : {};
}
