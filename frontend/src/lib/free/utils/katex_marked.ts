/**
 * marked 用の LaTeX 数式拡張 (KaTeX レンダリング)。
 *
 * base モデルは数式を求められると自然に LaTeX (`$$...$$` / `$k$`) を吐くが、
 * 従来はレンダラが未対応で生の文字列として表示されていた
 * (実測 2026-07-25: RRF の解説で `$$ S = \frac{1}{k+r} $$` がそのまま出た)。
 *
 * 対応する区切り:
 *   ブロック: `$$...$$` / `\[...\]`
 *   インライン: `$...$` / `\(...\)`
 *
 * `$` は通貨表記と衝突するため、インラインは以下を満たす場合のみ数式とみなす:
 *   - 開き `$` の直後が空白・`$` でない
 *   - 閉じ `$` の直前が空白・バックスラッシュでない
 *   - 閉じ `$` の直後が数字でない (「$5 と $3」を式にしない)
 *
 * KaTeX は完全にローカルで動作し (ネットワーク不要)、`throwOnError: false` で
 * 不正な TeX でもクラッシュせずエラー表示に落とす。パースに失敗した場合は
 * 元の文字列をエスケープしてそのまま出す (従来表示へのフォールバック)。
 */
import katex from 'katex';
import { marked, type MarkedExtension, type TokenizerAndRendererExtension } from 'marked';

function escapeHtml(s: string): string {
	return s
		.replace(/&/g, '&amp;')
		.replace(/</g, '&lt;')
		.replace(/>/g, '&gt;')
		.replace(/"/g, '&quot;');
}

export function renderMath(tex: string, displayMode: boolean): string {
	try {
		return katex.renderToString(tex, {
			displayMode,
			throwOnError: false,
			// 数式の途中で改行されるより横スクロールの方が読みやすい
			strict: 'ignore'
		});
	} catch {
		const raw = displayMode ? `$$${tex}$$` : `$${tex}$`;
		return escapeHtml(raw);
	}
}

const BLOCK_MATH_RE = /^(?:\$\$([\s\S]+?)\$\$|\\\[([\s\S]+?)\\\])[ \t]*(?:\n+|$)/;
const INLINE_MATH_RE = /^(?:\$(?![\s$])((?:\\.|[^\\$\n])+?)(?<![\s\\])\$(?!\d)|\\\(([\s\S]+?)\\\))/;

const blockMath: TokenizerAndRendererExtension = {
	name: 'blockMath',
	level: 'block',
	start(src: string) {
		const m = /\$\$|\\\[/.exec(src);
		return m?.index;
	},
	tokenizer(src: string) {
		const m = BLOCK_MATH_RE.exec(src);
		if (!m) return undefined;
		return {
			type: 'blockMath',
			raw: m[0],
			text: (m[1] ?? m[2] ?? '').trim()
		};
	},
	renderer(token) {
		return `<div class="math-block">${renderMath(token.text as string, true)}</div>`;
	}
};

const inlineMath: TokenizerAndRendererExtension = {
	name: 'inlineMath',
	level: 'inline',
	start(src: string) {
		const m = /\$(?![\s$])|\\\(/.exec(src);
		return m?.index;
	},
	tokenizer(src: string) {
		const m = INLINE_MATH_RE.exec(src);
		if (!m) return undefined;
		return {
			type: 'inlineMath',
			raw: m[0],
			text: (m[1] ?? m[2] ?? '').trim()
		};
	},
	renderer(token) {
		return renderMath(token.text as string, false);
	}
};

/**
 * CJK 隣接の `**強調**` を CommonMark の flanking 規則に阻まれず描画する。
 *
 * CommonMark は `**` を「左フランキング」の時だけ開き区切りとみなす。日本語では
 * `製品型番は**「BP-770」**でした` のように `**` の直前が CJK 文字・直後が
 * `「` (Unicode 句読点) になることが多く、この条件を満たさないため強調が開かず
 * `**` が生のまま画面に出る (2026-07-25 実測: ser/estar の解説や型番の想起など
 * 構造化された日本語応答の大半で発生)。
 *
 * marked はカスタム inline 拡張を組込みトークナイザより先に評価するため、
 * `**...**` の対応が取れている範囲をここで先に強調として確定させる。ASCII の
 * 通常ケースは組込みと同じ `<strong>` になり、`2 ** 3` のような閉じの無い
 * `**` や空白直後の `**` は従来どおり素通りする。
 */
const CJK_STRONG_RE = /^\*\*(?=[^\s*])((?:(?!\n\n)[\s\S])*?[^\s*])\*\*(?!\*)/;

const cjkStrong: TokenizerAndRendererExtension = {
	name: 'cjkStrong',
	level: 'inline',
	start(src: string) {
		const i = src.indexOf('**');
		return i < 0 ? undefined : i;
	},
	tokenizer(src: string) {
		const m = CJK_STRONG_RE.exec(src);
		if (!m) return undefined;
		return {
			type: 'cjkStrong',
			raw: m[0],
			tokens: this.lexer.inlineTokens(m[1])
		};
	},
	renderer(token) {
		return `<strong>${this.parser.parseInline(token.tokens ?? [])}</strong>`;
	}
};

export const katexExtension: MarkedExtension = {
	extensions: [blockMath, inlineMath, cjkStrong]
};

/**
 * KaTeX の出力を DOMPurify が落とさないための追加許可。
 *
 * 既定設定でも `<math>` と描画用 span は残るが、`<annotation
 * encoding="application/x-tex">` (元の TeX ソース = コピー用/支援技術向け) が
 * 落ちる。これらを許可しても script / on* / javascript: は従来どおり除去される
 * (2026-07-25 に XSS ベクタ 5 種で確認済み)。
 */
export const KATEX_SANITIZE_OPTIONS: {
	ADD_TAGS: string[];
	ADD_ATTR: string[];
} = {
	ADD_TAGS: ['annotation', 'semantics'],
	ADD_ATTR: ['encoding']
};

let installed = false;

/**
 * marked シングルトンへチャット向け設定を一度だけ登録する (数式拡張 + 改行の扱い)。
 *
 * `breaks: true` は CommonMark 既定 (単一改行を無視して段落内で連結) を、チャット
 * 応答で期待される「1 改行 = 1 行」へ変える。既定のままだと LLM が改行で区切った
 * 出力が 1 行に潰れ、書式指定そのものが守られていないように見える (2026-08-12
 * ライブ監査:「必ず 3 行ちょうどで」の依頼に 3 行で応答したが、画面上は
 * 「甘くてジューシーです。 vitaminも豊富です。 健康に良い果実です。」と 1 行に
 * 連結された。会話の要約も同様に箇条書きが 1 行に潰れていた)。
 * 段落・リスト・コードブロックの解釈は変わらない。
 */
export function installKatexExtension(): void {
	if (installed) return;
	installed = true;
	marked.use(katexExtension);
	marked.use({ breaks: true });
}
