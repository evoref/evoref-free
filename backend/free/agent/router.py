"""複雑度分類ルーター: クエリを3層エージェントに振り分ける"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from backend.free.agent.context_budget import resolve_meta_cognitive_loop_budget
from backend.free.agent.meta_cognitive_text import assigns_file_content
from backend.free.agent.safety_patterns import strip_command_literals
from backend.free.core.intent_vocab import (
    CALCULATE_TERM,
    CODE_SEARCH_PATTERNS,
    EXECUTABLE_QUERY_PATTERNS_EN,
    EXECUTABLE_QUERY_PATTERNS_JA,
    continuation_request,
    has_history_recall_keyword,
    persist_request,
    runtime_info_question,
    tool_inventory_question,
    GREETING_PUNCTUATION_JA,
    QUESTION_TAIL_RE,
    REFERENTIAL_WRITE_TARGET_RE,
    URL_LITERAL_RE,
    ascii_boundary_alternation,
    exact_greeting_pattern,
    looks_like_numeric_question,
    PREMISE_CONFIRMATION_RE,
)
from backend.free.core.session_mode import is_chat_mode, is_create_mode
from backend.free.core.text_quality import count_belongs_to_another_subject
from backend.free.document_nouns import (
    DOCUMENT_NOUN_LEARNABLE_JA,
    DOCUMENT_NOUNS_NEEDS_SUFFIX,
    DOCUMENT_NOUNS_NEEDS_SUFFIX_EN,
    DOCUMENT_NOUNS_STANDALONE,
    DOCUMENT_NOUNS_STANDALONE_EN,
)
from backend.log_config import get_logger
from backend.policy_helpers import get_policy_value

if TYPE_CHECKING:
    from backend.free.agent.learned_patterns import LearnedPatternStore
    from backend.free.core.policy_interpreter import PolicyInterpreter

logger = get_logger("agent.router")

# 長文生成リクエスト検出パターン（設計書 f_03 §1.2）
# 名詞群・動詞群の語彙拡張時は f_03 §1.2 の表も同時更新すること。
# ハードコード regex に加え、`LearnedPatternStore.match(category="long_form")`
# の学習済みキーワード (FeedbackCollector で蓄積、LearningScheduler で進化)
# も判定対象 (`_detect_long_form_learned()` で OR 統合)。
LONG_FORM_PATTERNS = [
    # 文書系名詞 → 動作動詞 の組合せ。
    # 「仕様書を作成して」「ドキュメントを出力して」「計画書をまとめて」等
    # を長文生成として拾う。名詞語彙は backend/free/document_nouns.py で
    # content_detector.py (EvorefGen) の TEXT_PATTERNS と共有する。router は
    # 常にサフィックス必須で運用するため、名詞単体マッチ許容語彙
    # (DOCUMENT_NOUNS_STANDALONE) もここではサフィックス必須の安全側で使う。
    # サフィックスの ``書`` は「〜を書いて」の動詞を拾うためのもの。**書式語の
    # 中の「書」に当ててはいけない** — 「箇条書き」は長さではなく **形式** の
    # 指定で、むしろ短い一覧を求めるいちばん強いシグナルなのに、
    # ``(文書名詞).*(書)`` が「チェックリストを箇条書」に一致して長文生成へ
    # 振っていた。
    #
    # 実インシデント (2026-08-31 ライブ監査 T11#4):
    # 「装置立ち上げのチェックリストを箇条書きで7項目。」が cogwriter へ回り、
    # 127 秒かけて **箇条書きゼロの散文** を返した (backend.log に
    # ``Turn marked failed (asked for a bullet list but the answer has no
    # list items)``)。
    #
    # 「まとめ」は **既存物の要約** にも使う。「このレポートをまとめて」
    # 「README を読んで要点をまとめて」「この論文を要約してまとめてください」は
    # 生成量を減らす依頼で、ユニット分割の長文生成は categorically 不適
    # (2026-09-02 監査 A2)。要約系の語 (要約 / 要点 / 読んで / 整理) と共起する
    # 「まとめ」、および指示詞 (この / その / あの) で既存の成果物を指した
    # 文書名詞に付く「まとめ」は採らない。書 / 作成 / 生成 / 出力 は従来どおり。
    re.compile(
        rf"({'|'.join(DOCUMENT_NOUNS_NEEDS_SUFFIX + DOCUMENT_NOUNS_STANDALONE)})"
        r".*((?<!箇条)(?<!横)(?<!縦)書|作成|生成|出力)",
    ),
    re.compile(
        r"^(?!.*(?:要約|要点|読んで|整理))"
        r"(?<!この)(?<!その)(?<!あの)"
        rf"(?:{'|'.join(DOCUMENT_NOUNS_NEEDS_SUFFIX + DOCUMENT_NOUNS_STANDALONE)})"
        r".*まとめ",
    ),
    # 「長文」「長編」の言及、または「長い」+ 創作文書系名詞 (物語/小説等、
    # pattern[1] の文書名詞リストには含まれない) の言及が、生成依頼のて形
    # 動詞と共起する場合に長文生成として拾う。
    # 「一番長い川」のような一般形容詞や、「長文が重複生成されている」といった
    # (受身形の) バグ報告文自体の誤爆を防ぐため、pattern[1] と異なりて形の
    # 完全一致を要求する (「生成」の bare stem だと「生成され」にも誤って一致する)。
    re.compile(
        r"(?:長(?:文|編)|長い(?:物語|小説|エッセイ|詩|お話))"
        r".*(?:書いて|作成して|生成して|まとめて|出力して)",
    ),
    re.compile(
        r"(ファイル|モジュール|クラス|プロジェクト)"
        r".*(一式|全体|まるごと|フル).*(作成|生成|実装)",
    ),
    # 「全体」「完全」は **依頼のて形** を伴うときだけ採る。旧定義
    # ``(実装|作成).*全体`` / ``(完全|網羅的|包括的).*(実装|ガイド|解説)`` は
    # 「実装全体を見直して」「作成した関数全体を見せて」「テストは完全に
    # 実装済みです」のような点検依頼・状況報告まで長文生成へ振っていた
    # (2026-09-02 監査 A2)。
    re.compile(r"全体.*(?:実装|作成|生成)(?:して|しろ|(?:を)?お願い)"),
    re.compile(
        r"(?:完全|網羅的|包括的).*(?:実装|ガイド|解説)"
        r".*(?:して|しろ|(?:を)?お願い|書いて|作って|作成|生成|出力)",
    ),
]

# LONG_FORM_PATTERNS の英語版。GUI locale に関わらず LONG_FORM_PATTERNS と
# 常に両方評価する (2026-07-22 発見: GUI locale が既定 'ja' のまま英語で
# チャットすると、以前は locale 排他選択のため英語の文書作成依頼が一切
# 検出されなかった。詳細は _detect_long_form 参照)。
LONG_FORM_PATTERNS_EN = [
    re.compile(
        rf"\b(?:write|draft|create|compose|generate|produce|prepare|put\s+together)\b"
        rf".*\b(?:{'|'.join(DOCUMENT_NOUNS_NEEDS_SUFFIX_EN + DOCUMENT_NOUNS_STANDALONE_EN)})\b"
        r"|"
        rf"\b(?:{'|'.join(DOCUMENT_NOUNS_NEEDS_SUFFIX_EN + DOCUMENT_NOUNS_STANDALONE_EN)})\b"
        rf".*\b(?:write|draft|create|compose|generate|produce|prepare|put\s+together)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:write|draft|create|compose)\b.*\b(?:long|lengthy|in-depth|comprehensive)\b"
        r".*\b(?:article|essay|story|novel|report|poem)\b"
        r"|"
        r"\b(?:long|lengthy|in-depth|comprehensive)\b.*\b(?:article|essay|story|novel|report|poem)\b"
        r".*\b(?:write|draft|create|compose)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:files?|modules?|classes?|the\s+(?:whole|entire)\s+project)\b"
        r".*\b(?:entire|whole|full|complete)\b.*\b(?:create|generate|implement|build)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:implement|create|build)\b.*\b(?:everything|the\s+whole\s+thing|the\s+entire\s+project)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:complete|comprehensive|exhaustive|full)\b"
        r".*\b(?:implementation|guide|explanation|walkthrough|write-?up)\b",
        re.IGNORECASE,
    ),
]

# URL を含み、かつファイル書込み/出力意図のあるクエリは「取得 → 書込み」の連鎖を
# 要するため meta_cognitive 層へ振る。deliberative は 1 ターン 1 ツールで連鎖できず、
# long_form は URL を取得せず散文を生成する (内容を捏造する) ため、いずれも不適。
_URL_HINT_RE = re.compile(r"https?://", re.IGNORECASE)
# 書込み **動詞** のみ。旧定義は名詞 (file / csv / excel / word …) を裸で並べて
# いたため "in a few **words**" / "user pro**file**" / "ex**cel**lent" が
# url_write_intent になり、逆に「URL を読んで要約して」まで meta_cognitive へ
# 振られていた (2026-09-02 監査 A5)。
_FILE_WRITE_INTENT_RE = re.compile(
    r"(?:出力|保存|書き出|書き込|エクスポート|セーブ"
    r"|" + ascii_boundary_alternation("export", "save") + r")",
    re.IGNORECASE,
)

# URL を伴わずローカルパスへ書き出す意図のクエリ (例: 「C:\\...\\aa 配下に Excel で
# カレンダーを作成して出力」) も「生成 → 書込み」の連鎖を要するため meta_cognitive 層へ
# 振る。long_form (文書系名詞) にも _is_url_write_intent にも当たらないデータ成果物
# (カレンダー/一覧表等) を拾い、deliberative のディレクトリ書込み除外による拒否を防ぐ。
# CJK (ひらがな / カタカナ / 漢字) の検出。クエリ長判定の分岐に使う。
_CJK_CHAR_RE = re.compile(r"[぀-ヿ㐀-䶿一-鿿]")
# ローカルファイルパス (Windows ドライブ / Unix) の存在検出。
_LOCAL_PATH_RE = re.compile(
    # Windows ドライブパス (C:\ / C:/)。先頭境界が無いと "http://x" の
    # ``p:/`` に当たる (2026-09-02 監査 A8)。
    r"(?<![A-Za-z])[A-Za-z]:[\\/]"
    r"|(?:^|[\s　])(?:/[\w._-]+){2,}",  # Unix パス (/home/user/...)
)
# 書込み「動詞」のみ。名詞 (excel / docx 等) 単独では発火させない (「report.xlsx を
# 読んで」のような read 文脈で誤検出しないため)。
#
# ``core.intent_vocab.WRITE_VERB_RE`` とは **意図的に別物** (統合しない)。
# あちらは meta_cognitive のタスク種別判定用で語彙が広く (修正 / 変更 / 更新 /
# 実装 等)、ここでルータの振り分けに使うと純粋な読み取り・編集依頼まで
# 書込み志向のプランナへ送ってしまう。共有するのは概念であって語彙ではない。
#
# 「追記」「書き足」は 2026-08-08 に追加。``書[きい]`` は 書き/書い しか拾わず、
# 「同じファイルに 3 行追記して」が書込み意図として認識されなかった。その結果
# chat で実行できる書込みツールが無い経路に落ち、ツールを 1 つも撃たないまま
# 「追記しました」と完了を捏造した (実ファイルは無変更。ライブ監査 ターン6)。
_WRITE_VERB_RE = re.compile(
    r"(?:作成|作って|生成|出力|保存|書[きい]|書込|追記|書き足|エクスポート|export"
    r"|(?<![A-Za-z])save(?![A-Za-z])"
    r"|(?<![A-Za-z])write(?![A-Za-z])"
    r"|(?<![A-Za-z])append(?![A-Za-z])"
    r"|(?<![A-Za-z])create(?![A-Za-z])"
    r"|(?<![A-Za-z])output(?![A-Za-z]))",
    re.IGNORECASE,
)
# 既に書かれた成果物を **指し示す** 連体修飾節。「いま書いたそのファイルを読み込んで」
# の「書いた」は依頼された動作ではなく対象の説明であり、書込み意図ではない。
# これを書込み動詞として拾うと、純粋な読み取り依頼が meta_cognitive (書込み志向の
# プランナ) へ振られ、read → 「合計を計算」→「中身を表示」のような非ファイル
# サブタスクまでファイルツールで実行しようとして破綻する (実インシデント
# 2026-07-29 ライブ監査: 「いま書いたそのファイルを読み込んで、金額の合計を
# 計算してください」が 3 タスクに分解され、うち 2 つが実在しないパス
# (prices.txt / unknown) の read_file になって失敗し、質問は無回答のまま
# 「1 件のタスクを完了し、2 件が失敗しました。」だけが返った)。
#
# 「〜に書いてある / 書かれている」も同様に **状態の説明** であって依頼された
# 動作ではない。``書[きい]`` が ``書いてある`` の ``書い`` に一致するため、
# 裸のファイル名を書込み先として認めるようにすると
# 「notes.txt に書いてある内容を見せて」が書込み依頼に化ける。
_DESCRIPTIVE_WRITE_CLAUSE_RE = re.compile(
    r"(?:書[きい]た|作成した|作った|生成した|保存した|出力した|書き込んだ)"
    r"\s*(?:ばかりの?|ところの?)?"
    r"\s*(?:その|この|あの|先ほどの?|さっきの?)?"
    r"\s*(?:ファイル|もの|やつ|データ|内容|中身)"
    r"|書いてあ|書かれて|書いてる"
    r"|(?:you\s+(?:just\s+)?(?:wrote|created|saved|generated))",
    re.IGNORECASE,
)
# 裸のファイル名 (ディレクトリを伴わない ``notes.txt``)。書込み動詞と揃った
# ときだけ書込み先として認める。
#
# ``_LOCAL_PATH_RE`` はドライブ接頭辞か Unix の 2 階層以上しか認めないため、
# 直前に自分が作ったファイルを名前だけで指す **最も自然な言い方** が
# ``local_write_intent`` を外れ deliberative に落ちていた。chat には実行できる
# 書込みツールが無いので、ツールを 1 つも撃たないまま完了を捏造する
# (実インシデント 2026-08-09 ライブ監査: 「inventory_notes.txt に 1 行追記して
# ください」でツール 0 回、**フルパスを補って**「E:\tmp\inventory_notes.txt の
# 末尾に追記しました」と報告。実ファイルは無変更。フルパスで同じ依頼をすると
# 正常に書き込まれ、差はパス表記だけだった)。
#
# 実際の保存先は meta_cognitive の write-fast 経路が会話から解決する
# (解決できなければ書かずに失敗する = 捏造しない)。
# 先頭境界は **否定後読み** で書く。区切り文字を列挙する方式にすると、
# 日本語の句読点直後 (「その値です。notes.txt を書き直して」) を取りこぼす。
# 日本語ではファイル名が句点・読点の直後に来るのが普通なので、列挙漏れは
# そのまま実害になる (2026-08-09 の 2 回目のライブ監査で、`。` 直後の
# 指定がルータを外れ、ツール 0 回のまま「書き直します」と応答した)。
# パス区切り・語構成文字だけを除外すれば、``E:\tmp\a.txt`` の末尾要素を
# 裸名として二重に拾うことも防げる。
_BARE_FILENAME_TARGET_RE = re.compile(
    r"(?<![\w./\\-])"
    r"[\w-]+(?:\.[\w-]+)*\."
    r"(?:txt|md|markdown|csv|tsv|json|jsonl|ya?ml|log|ini|cfg|conf|toml"
    r"|xlsx|xls|ods|docx|doc|pptx|ppt|pdf|html?|xml"
    r"|py|js|ts|tsx|jsx|sh|ps1|bat|sql|rs|go|java|rb|c|cpp|h)"
    r"(?![A-Za-z0-9])",
    re.IGNORECASE,
)
# 保存先を直前の文脈に委ねる参照表現。「同じファイルに保存し直して」のような
# 追記・修正依頼はパスを本文に持たないため _LOCAL_PATH_RE に掛からず、
# deliberative に落ちて read_file だけが走り、書込みが一度も起きないまま
# 「保存し直した」体の回答になっていた (実測 2026-07-27)。
# 定義は core.intent_vocab が SSOT (agent.feedback の訂正判定でも同じ意味で使う)。
_REFERENTIAL_WRITE_TARGET_RE = REFERENTIAL_WRITE_TARGET_RE
# 表形式データの出力先拡張子。long_form (散文ユニット分割) では表構造を
# 生成できないため、これらへの書出し意図は long_form 判定より優先して
# local_write_intent (write-fast 経路) に振る。
_TABULAR_TARGET_RE = re.compile(
    r"\.(?:csv|tsv|xlsx|ods)(?![A-Za-z])", re.IGNORECASE,
)

# 学習済み long_form パターン単独発火の抑止床。閾値以上の一致が 1 語のみの
# 場合、その重み合計がこの値以上でなければ long_form に分類しない。
_LEARNED_LONG_FORM_MIN_WEIGHT_SUM = 0.8

# how-to (教示) マーカー。「作成する方法を教えて」のような書込み動詞を含む
# 知識質問を local_write_intent / long_form から除外する。``_is_knowledge_query``
# は「一覧を…」等のデータ語を広く拾い正当な書込みコマンドまで除外してしまう
# ため、ここでは **教示** マーカーに限定する。locale で切替えず単一の正規表現に
# JA/EN 両方の教示フレーズを併記する。英語側は "How do I create this report
# at C:\reports\" のように疑問符無しで how-to 意図を書くクエリがあるため、
# 明示的な how-to フレーズを列挙する (2026-07-22 監査で判明)。
#
# **疑問形一般 (ですか / ますか / でしょうか / ?) は含めない。** 以前は含めて
# いたため、丁寧な依頼がすべて除外されていた — 「3000字のレポートを書いて
# くれますか？」が long_form を外れて deliberative へ、"Can you write a
# 3000-word article?" が reactive へ、「C:\tmp\a.txt に保存してくれますか？」
# が local_write_intent を外れて persist_intent へ落ちた (2026-09-02 監査 A1)。
# 成果物 **について** 尋ねる問いは ``_ARTIFACT_QUESTION_RE`` (文末に錨) が
# 別に受ける。
_HOWTO_QUERY_RE = re.compile(
    r"(?:教えて|おしえて|どうやって|どうすれば|どうやったら|やり方"
    r"|方法(?:は|を|が|って|に)|とは|って何"
    r"|how\s+(?:do|can|should|would)\s+i\b|how\s+to\b"
    r"|(?:could|can|would)\s+you\s+(?:tell|show)\s+me\s+how\b"
    r"|tell\s+me\s+how\b|what'?s?\s+the\s+way\s+to\b)",
    re.IGNORECASE,
)

# 既出の成果物から一部だけを抜き出す / 並べ直す依頼。生成量を **減らす**
# 依頼なので、ユニット分割の長文生成は categorically 不適
# (実インシデント 2026-07-29 ライブ監査: 800 字の記事を書かせた直後に
# 「その記事の見出しだけを箇条書きで並べてください。」と頼んだところ、
# 見出しの一覧ではなく春の話題の散文生成が始まった)。
_EXTRACTION_REQUEST_RE = re.compile(
    r"(?:見出し|表題|タイトル|項目|要点|ポイント|キーワード|一覧|リスト)"
    r"\s*(?:だけ|のみ)"
    r"|(?:だけ|のみ)\s*を?\s*(?:並べ|挙げ|列挙|抜き出|抜粋|取り出)"
    r"|(?:headings?|titles?|key\s*points?|items?)\s+only"
    r"|(?:list|extract)\s+(?:just|only)\s+the",
    re.IGNORECASE,
)

#: 「さきほどの表で上書きして」型: 書くべき本文は会話に既にあり、生成する対象が
#: 無い。書込み経路の仕事なので長文生成に落とさない。
#:
#: 実インシデント 2026-08-10 ライブ監査: 「そのファイルには表ではなく説明文が
#: 入っています。さきほどの列定義の表で上書きしてください。」が long_form に
#: 分類され、**やるべきことを説明する約 1,000 字の散文**を出力してファイルは
#: 無変更のまま「✓ 生成 完了 (1 ユニット)」と表示された。さらにその誤ルー
#: ティングが `説明文` として long_form パターンに学習・永続化された。
#:
#: 動詞はファイルへ向かうもの (上書き/保存/書き込) に限る。「先ほどの内容を
#: もとにレポートを書いて」のような正当な長文依頼を巻き込まないため。
_PRIOR_ARTIFACT_REF_RE = re.compile(
    r"(?:さきほど|さっき|先(?:ほど|程)|上記|直前|いま|今)の?\s*[^。、]{0,10}?"
    r"(?:表|リスト|一覧|コード|スクリプト|関数|クラス|定義|案|下書き|内容|文面|本文)",
)
_WRITE_TO_FILE_VERB_RE = re.compile(r"上書き|書き込|書き直|保存")

# 複雑度を示すキーワードパターン（同義語・表記揺れ対応済み）
COMPLEX_KEYWORDS = [
    "比較", "なぜ", "どのように", "違い", "分析", "解析",
    "メリット", "デメリット", "利点", "欠点", "長所", "短所",
    "原因", "理由", "要因", "根本原因",
    "仕組み", "構造", "アーキテクチャ",
]

# COMPLEX_KEYWORDS の英語版。"how"/"cons"/"pros" 等の短い英単語は
# "however"/"prospect"/"considering" 等への substring 誤爆を防ぐため、
# 単純な文字列一致ではなく ASCII 境界ガード付き正規表現で構成する
# (META_KEYWORDS_EN_PATTERNS の慣用句を踏襲)。
COMPLEX_KEYWORDS_EN_PATTERNS = [
    re.compile(r"(?<![A-Za-z])explain(?:ed|ing|s)?(?![A-Za-z])", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z])compar(?:e[ds]?|ing|ison)(?![A-Za-z])", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z])analy[sz]e[ds]?(?![A-Za-z])", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z])analy[sz]ing(?![A-Za-z])", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z])why(?![A-Za-z])", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z])how(?![A-Za-z])", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z])difference[s]?(?![A-Za-z])", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z])pros(?![A-Za-z])", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z])cons(?![A-Za-z])", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z])trade[\s-]?offs?(?![A-Za-z])", re.IGNORECASE),
]

# 履歴検索が必要なキーワード → Deliberative 層
# 「覚えて」「最初に」は当初リストに含まれておらず、20+ ターンの長い会話で
# 「この会話の最初に〜覚えてますか」型の recall 質問が reactive の既定分岐
# (step 12) まで素通りし、reactive_light 経路 (直近 REACTIVE_LIGHT_HISTORY_TURNS
# 件のみ・SemMem/STM 注入なし) に落ちて古いターンの内容を想起できない実インシデントが
# あった (2026-07-19)。deliberative に昇格させれば build_semmem_injection 経由で
# STM ノート (pin ブースト込み) が注入されるため、search_history ツール発火の有無に
# 関わらず recall 精度が回復する。
#
# 「過去の会話」「会話履歴」等の明示的な履歴検索語も対象。これらが無いと、
# ユーザーが明示的に検索を依頼しても知識質問プレフィルタで落ち、
# search_history が一度も発火しないまま「確認したが見当たらない」と
# 未確認のまま断言する (実インシデント 2026-07-27 ライブ検証:
# 「過去の会話で、登山の話題をしたことはありますか？探してください。」が
# tool_call_decision=no_tool / reason=no_match_in_any_layer で素通りし、
# 実際には 1 時間前に登山の会話があったのに「見当たりません」と回答)。
# 履歴参照キーワードは core.intent_vocab が SSOT (キーワードと「近接 / 長距離」の
# 距離分類を 1 つの表で持つ)。照合は ``has_history_recall_keyword`` 経由。

# Meta-Cognitive 層へのエスカレーションキーワード（同義語拡張済み）
# 注意: 命令形（〜して）のみ対象。「計画を教えて」等の知識質問は
# ここに含めない（RAG パイプラインで処理する）。
# 英語語彙は **ここに載せない**。素の部分一致で照合するため、裸の英単語は
# "designated" / "prefix" / "rebuild" に当たり、境界付きの
# META_KEYWORDS_EN_PATTERNS を無効化していた (2026-09-02 監査 A6)。
META_KEYWORDS = [
    "ステップ", "手順", "実装して", "作って", "作成して", "書いて",
    "リファクタ", "修正して", "デバッグ", "設計して", "組み立て",
    "構築", "ビルド", "テストして", "動かして",
]

# META_KEYWORDS の英語版。短い英単語の substring 誤爆 ("prefix"/"contest"/
# "rebuild" 等) を防ぐため、単純な文字列一致ではなく ASCII 境界ガード付き
# 正規表現で構成する (_EXECUTABLE_QUERY_PATTERNS の慣用句を踏襲)。
META_KEYWORDS_EN_PATTERNS = [
    re.compile(r"(?<![A-Za-z])steps?(?![A-Za-z])", re.IGNORECASE),
    re.compile(r"step[\s-]by[\s-]step", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z])implement(?:ed|ing|ation)?(?![A-Za-z])", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z])write(?![A-Za-z])", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z])create(?![A-Za-z])", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z])build(?![A-Za-z])", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z])refactor(?:ing)?(?![A-Za-z])", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z])fix(?:ed|ing)?(?![A-Za-z])", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z])debug(?:ging)?(?![A-Za-z])", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z])design(?:ed|ing)?(?![A-Za-z])", re.IGNORECASE),
    re.compile(r"set\s*up", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z])construct(?:ed|ing)?(?![A-Za-z])", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z])test(?:ed|ing)?(?![A-Za-z])", re.IGNORECASE),
    re.compile(r"run\s+(?:it|this|that)\b", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z])make(?![A-Za-z])", re.IGNORECASE),
]

# 知識質問パターン — ツール呼び出しではなく RAG で処理すべきクエリ
# これらにマッチするクエリはツール/Meta-Cognitive エスカレーションをスキップする
#
# 「明示的な疑問形式」(strict) と「名詞+助詞だけの緩いパターン」(loose) を
# 分けて定義し、全体はその合成として組み立てる。両者を 1 つのリストに混ぜて
# index で除外すると (旧実装は ``if i != 1``)、リストに要素を 1 つ挿入した
# 瞬間に除外対象が黙って別のパターンへすり替わる。
_KNOWLEDGE_QUERY_STRICT_PATTERNS = [
    QUESTION_TAIL_RE,
    re.compile(r"(?:について|に関して|に関する)", re.IGNORECASE),
    re.compile(r"(?:知りたい|確認したい|調べたい|わかる|分かる)", re.IGNORECASE),
    re.compile(r"(?:what is|tell me|explain|describe|how does)\b", re.IGNORECASE),
]

# 名詞 + 助詞のみで判定するため緩く、クリエイトモードの正規な生成依頼
# ("サンプルCSVデータを作成し" 等) の一部にも誤マッチする (2026-07-22 ライブ
# 検証で発覚)。strict 側 (教えて/について/知りたい 等) は明示的な疑問形式のみを
# 対象とするため誤爆が少ない。create_meta_keywords の is_knowledge 上書き判定
# では、この緩いパターンを除いた strict のみを知識質問シグナルとして扱う。
_KNOWLEDGE_QUERY_LOOSE_PATTERN = re.compile(
    r"(?:資料|情報|データ|内容|概要|詳細|特徴|一覧).*(?:は|を|が|に)", re.IGNORECASE,
)

KNOWLEDGE_QUERY_PATTERNS = [
    *_KNOWLEDGE_QUERY_STRICT_PATTERNS,
    _KNOWLEDGE_QUERY_LOOSE_PATTERN,
]

# ユーザー自身について記憶している内容の想起を求めるクエリ。
#
# reactive / reactive_light は SemMem も RAG も一切走らせないため、この種の
# クエリが short_query で reactive に落ちると、**記憶しているのに「情報が
# ありません」と答える**。知識質問を deliberative へ逃がす knowledge_query
# ルール (直上) と同じ理由だが、想起依頼は「〜ですか」で終わらない言い回しが
# 多く strict パターンに掛からない。
#
# 実インシデント (2026-08-07 ライブ監査): 同一セッションで「趣味は自転車と
# 写真です」と伝え、実際にファクトとして保持していたにもかかわらず、
# 「私の趣味をもう一度確認させてください。」(19 文字) が short_query →
# reactive_light に落ち、「あなたの趣味に関する情報は提供されていない」と
# 回答した。「私の趣味は何でしたか？」も同じ経路に落ちる (strict 側は
# ``ですか`` を持つが ``でしたか`` を持たない)。
#
# 一人称の所有格と想起動詞が近接している場合のみ対象とし、間に挟める文字数を
# 制限して「私が作ったコードを修正して」等の依頼を巻き込まない。
#
# 「私の◯◯は？」型 (想起動詞も ``ですか`` も無い最短形) も対象に含める。
# 実インシデント (2026-08-09): 「私の猫の名前は？」(9 文字) /
# 「私の好きな季節は？」が short_query → reactive に落ち、**記憶ブロックが
# 一切組み立てられないまま** 「文脈が不足しているため猫の名前を教えて
# ください」と回答した。ファクトは保持しており (類似度 0.490)、注入経路に
# 到達しさえすれば正答する。所有格 + 短い属性名 + ``は？`` で終わる形だけを
# 採るので、「私の作ったコードは動きますか」のような依頼文は掛からない。
# 属性を 2 つ以上並べる形は読点を挟む。読点を除外していたため
# 「私の好きな色と、飼っているペットの数は？」(21 文字) が personal_recall を
# 外れ、short_query → reactive_light に落ちて 6 ターン前に伝えた事実を
# 「提供されていません」と回答した (2026-08-12 ライブ監査 ターン22)。
# 句点 (文の切れ目) は従来どおり除外して依頼文を巻き込まない。
# 体言止め (「〜は。」/ 句読点なしの「〜は」) と、所有格以外の助詞 (``が`` / ``は``)
# も採る。従来は ``私の…は？`` (所有格 + 疑問符必須) だけだったため、
# 実インシデント (2026-08-29 ライブ監査 T22#3): 「音楽はジャズをよく聴きます」と
# 伝えた後、新セッションの「**私が好きな音楽のジャンルは。**」(14 文字) が
# ``の`` でも ``？`` でもないため personal_recall を外れ、
# ``short_query -> reactive -> reactive_light`` に落ちて **記憶検索を一度も
# 行わないまま「J-POPです」と捏造**した (同ターンの rag_selection 行が存在しない)。
# 前後の「私の好きな飲み物は何でしたか」(personal_recall) /
# 「好きなミュージシャンは誰でしたか」(discourse_recall) は正しく deliberative に
# 乗っており、**体言止めの短文だけが漏れていた**。
# 値の言明 (「私の猫の名前はミケです。」「私が好きな音楽はジャズです。」) は
# ``は`` の直後に文字が続くため掛からない — 想起の問いと言明は末尾で分離できる。
_PERSONAL_RECALL_RE = re.compile(
    r"(?:私|わたし|僕|ぼく|俺|おれ|自分)(?:の|が|は)[^。？?\n]{1,24}は[？?。．.]?\s*$"
    r"|(?:私|わたし|僕|ぼく|俺|おれ|自分)(?:の|は|が)[^。？?\n]{0,24}"
    r"(?:覚えて|おぼえて|記憶して|思い出|確認|言って|"
    r"何(?:です|でした|だった)|は何|でしたか|だっけ)",
)
# 双方向: 「my X — recall verb」だけでなく「what was — my X」の語順
# ("What was my cat's name?" / "What did I say my hobby was?") も採る。
# 旧定義は ``my`` が先行する形しか拾わず、想起の問いが short_query → reactive に
# 落ちて記憶を引かなかった (2026-09-02 監査 A4)。
_PERSONAL_RECALL_EN_RE = re.compile(
    r"\bmy\b[^.?!\n]{0,24}"
    r"\b(?:remember|recall|what(?:'s| is| was)|again)\b"
    r"|\bwhat(?:'s| was| is| were)\b[^.?!\n]{0,30}\bmy\b",
    re.IGNORECASE,
)

# 会話で既出のものを問い直す想起形 (談話照応)。一人称を伴わない。
#
# ``_PERSONAL_RECALL_RE`` は一人称 (``私`` / ``僕`` 等) を必須にしているため、
# **ユーザー自身ではなく会話で出てきた対象**を問い直すクエリが漏れる。
# 日本語の過去形疑問「〜でしたか」「〜だっけ」は *その場で初めて聞く事柄* には
# 使わない — 「HTTP とは何ですか」とは言うが「何でしたか」とは言わない。
# 既出であることを前提にした形なので、記憶を引くべきクエリの強いシグナルになる。
#
# 実インシデント (2026-08-19 ライブ監査): 前のセッションで「あさひプロジェクトの
# 締切は10月15日」と伝えた後、新しいセッションで
# 「あさひプロジェクトの締切はいつでしたか？」(22 文字) が一人称を含まないため
# personal_recall を外れ、short_query → reactive_light に落ちて SemMem を一度も
# 引かないまま「確認できていません。」と回答した。同一セッション内では直近窓に
# 残っていたため正答しており、**窓から出た瞬間だけ失敗する**構造だった。
#
# 誤爆しても代償は deliberative の実行コストだけ。逆に取りこぼすと「記憶して
# いるのに情報が無いと答える」ため、非対称を踏まえて広めに採る。ただし
# ``ましたか`` (「わかりましたか？」等の一般的な確認) は含めない。
_DISCOURSE_RECALL_RE = re.compile(
    r"(?:でしたか|だったか|でしたっけ|だったっけ|だっけ)"
    r"[。．.、,！!？?\s\"'」』）)]*\s*$",
)
# 英語版。"What was the deadline again" / "What did you say my name was?" の
# ように、会話で既出の値を問い直す形 (2026-09-02 監査 A4)。
_DISCOURSE_RECALL_EN_RE = re.compile(
    r"\bwhat\s+(?:was|were)\b.*\bagain\b"
    r"|\bwhat\s+did\s+(?:i|you)\s+(?:say|tell|mention)\b",
    re.IGNORECASE,
)

# 体言止めの問い (「〜は？」/「〜は。」/ 句読点なしの「〜は」)。
#
# ``_PERSONAL_RECALL_RE`` は同じ末尾形を持つが **一人称を必須** にしている。
# 日本語は主語を落とすのが常態で、話題が前のターンで確立していれば一人称は
# まず現れない。よって一人称を条件にした時点でこの形は必ず漏れる。
#
# 実インシデント (2026-09-04 ライブ監査): 直前のターンで
# 「Rust 7 割 / C++ 3 割」を伝え、SemMem にも保持していたにもかかわらず、
# 「**業務での言語比率は？**」(11 文字) が一人称を含まないため
# personal_recall を外れ、``short_query -> reactive`` に落ちて記憶を一度も
# 引かないまま「C++ が約 60%、Rust が約 40% です」と **比率を逆に捏造**した。
# 同じ会話の「業務で使う言語の比率はどうなっていましたか？」は
# discourse_recall に乗って正答している。
#
# 値の言明は ``は`` の後ろに必ず語が続く (「住まいは横浜です」) ので、
# **末尾の ``は``** だけで問いと言明を分離できる (``_PERSONAL_RECALL_RE`` /
# ``text_quality._QUESTION_ENDING_RE`` と同じ判別)。日本語の平叙文は
# ``は`` で終わらない。
#
# 例外は挨拶だけ (「こんにちは」「こんばんは」) — 日本語で ``は`` で終わる
# 定型句はこの閉じた集合しかない。
_TOPIC_STOP_QUESTION_RE = re.compile(
    r"(?:^|[。！!？?\n])\s*[^。！!？?\n]{2,40}は[？?。．.]?\s*$",
)
_GREETING_TOPIC_RE = re.compile(
    r"(?:^|[。！!？?\n])\s*(?:こんにち|こんばん|今日|今晩|おはよう?ござい)"
    r"は[？?。．.]?\s*$",
)

# 前提の同意を求める確認形。「〜ですよね？」「〜で合っていますか」など。
#
# knowledge_query の strict パターンは ``ですか|でしょうか|とは|教えて`` を持つが、
# 同意要求形はそのどれにも当たらない。短ければ short_query で reactive に落ち、
# RAG もカートリッジも引かないまま事前知識だけで答えることになる。**この形は
# 誤りに同意するのが既定の失敗**なので、他の短文より取りこぼしの代償が大きい。
#
# 実インシデント (2026-08-08 ライブ監査): 「evoref は RAG に LangChain を使って
# いますよね？」(29 文字) が short_query → reactive に落ち、検索を一切せずに
# 「はい、evoref は RAG に LangChain を使用しています」と回答した。リポジトリの
# 不変則 (LangChain / LlamaIndex / ChromaDB / FAISS を使わない) と真逆で、同じ
# セッションの「パリはイタリアの首都ですよね？」は事前知識だけで正しく訂正できて
# いたため、**知識の有無ではなく検索の有無**が分けていた。
# ``よね`` は述語の断定形に付くものだけを採る。裸の ``よね$`` にすると
# 「修正してよね」のような依頼 (て形 + よね) まで確認形として拾ってしまう。
# 定義は core.intent_vocab が SSOT (deliberative も同じ判定を使うため)。
_PREMISE_CONFIRMATION_RE = PREMISE_CONFIRMATION_RE

# KNOWLEDGE_QUERY_PATTERNS の英語版。「about」は汎用前置詞のため直訳せず、
# 限定的な表現で構成する (誤爆抑制)。
KNOWLEDGE_QUERY_PATTERNS_EN = [
    re.compile(
        r"\b(?:what\s+is|what's|what\s+are|tell\s+me\s+(?:about|more)|explain"
        r"|describe|how\s+does|how\s+do|could\s+you\s+explain)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:tell\s+me\s+about|what\s+about|information\s+(?:about|on)"
        r"|details?\s+(?:about|on)|overview\s+of|regarding)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:want\s+to\s+know|would\s+like\s+to\s+know|i'?d\s+like\s+to\s+know"
        r"|curious\s+about|want\s+to\s+understand|want\s+to\s+check)\b",
        re.IGNORECASE,
    ),
]

# ツール呼び出しを示すパターン（Meta-Cognitive 層へのエスカレーション対象）
# 注意: 単純なファイル読み書きは Deliberative 層の ToolCallJudge が処理する。
# ここには複数ツールの連携が必要なパターンのみ含める。
# 「検索」等の汎用語は知識質問にもマッチするため、コード/ファイル文脈を要求する。
# 日英共通エントリ (語彙は core.intent_vocab が SSOT)。以前は TOOL_PATTERNS と
# TOOL_PATTERNS_EN に byte 一致の 4 エントリが複製され、コード検索の ASCII 語に
# 境界が無く ("crossencoder" の 'code')、計算は ``(?:計算|calculate)\s`` の
# 末尾 ``\s`` のせいで日本語 (「計算して」) に一度も一致しなかった
# (2026-09-02 監査 A7 / B)。
_TOOL_PATTERNS_SHARED = [
    # コード/ファイル検索: 汎用「検索」ではなく、コード・ファイル文脈を要求
    *CODE_SEARCH_PATTERNS,
    # URL リテラルのみ。tool_judge_signals の web 参照語彙 (サイト / ページ /
    # 記事 …) まで広げると「この記事について教えて」のような知識質問が
    # meta_cognitive へ振られるため、層の振り分けでは明示 URL に限定する
    # (意図的分岐、f_03 §12.1)。
    URL_LITERAL_RE,
    re.compile(CALCULATE_TERM, re.IGNORECASE),
]

TOOL_PATTERNS = [
    # ファイル操作: 読み取り+変更など複合操作のみ（単一操作は Deliberative で処理）
    re.compile(r"(?:ファイル|file).*(?:読|開).*(?:書|修正|変更|削除|追加)", re.IGNORECASE),
    re.compile(r"(?:コマンド|command).*(?:実行|run)", re.IGNORECASE),
    *_TOOL_PATTERNS_SHARED,
]

# TOOL_PATTERNS の英語版。1つ目 (ファイル読み書き複合操作) は日本語版が
# 活用語尾ゲート限定のため英語文で発火しないので、英語動詞で作り直す。
# 2つ目 (コマンド実行) も日本語活用語尾 (して/する) を要求せずに発火するよう
# 緩和する。残りは ``_TOOL_PATTERNS_SHARED`` を共有する。
TOOL_PATTERNS_EN = [
    re.compile(r"\bfile\b.*\b(?:read|open)\b.*\b(?:write|modify|change|update|delete|remove|edit|append)\b", re.IGNORECASE),
    re.compile(r"\b(?:run|execute|exec)\b.*\bcommand\b|\bcommand\b.*\b(?:run|execute|exec)\b", re.IGNORECASE),
    *_TOOL_PATTERNS_SHARED,
]

# Python 実行で正確に答えられるクエリのパターン
# 知識質問パターンにマッチしてもこれらを含む場合は deliberative に昇格して
# ToolCallJudge によるツール実行を誘導する。
# 語彙は core.intent_vocab が SSOT (``EXECUTABLE_QUERY_PATTERNS_JA`` / ``_EN``)。
# tool_call_judge 側の ``_TOOL_PATTERNS`` / ``_infer_tool`` のゲート / コマンド
# 合成のルール表と同じ部品から組む — 以前は 4 箇所に書き写され、「変換」の
# 除外・「日付型」のガード・``CPU バウンド`` の除外が片方にしか入っていなかった
# (ASCII 境界も router 側だけ無く "program" の 'ram' に部分マッチしていた)。
_EXECUTABLE_QUERY_PATTERNS = EXECUTABLE_QUERY_PATTERNS_JA

# _EXECUTABLE_QUERY_PATTERNS の英語版。
_EXECUTABLE_QUERY_PATTERNS_EN = EXECUTABLE_QUERY_PATTERNS_EN

# 挨拶・簡単な定型パターン → Reactive 層で即応答
# 体裁 (クエリ全体が挨拶だけであることの要求) は core.intent_vocab から派生させる。
# 語彙は reactive.GREETING_RESPONSES とは **意図的に別** (あちらは応答文と 1 対 1 で
# 対応させるため分割が細かく、「おはようございます」等も個別に拾う)。
GREETING_PATTERNS = [
    re.compile(
        exact_greeting_pattern(
            r"こんにち[はわ]|おはよう|こんばんは|やあ|ども|hello|hi|hey",
            punctuation=GREETING_PUNCTUATION_JA,
        ),
        re.IGNORECASE,
    ),
    re.compile(
        exact_greeting_pattern(
            r"ありがと[うございます]*|thanks|thank you",
            punctuation=GREETING_PUNCTUATION_JA,
        ),
        re.IGNORECASE,
    ),
    re.compile(
        exact_greeting_pattern(
            r"おやすみ|さようなら|bye|goodbye",
            punctuation=GREETING_PUNCTUATION_JA,
        ),
        re.IGNORECASE,
    ),
]


#: 「N 文字 / N 行」型の分量指定。**桁数ではなく実際の大きさ** で判定する。
#: 旧実装は ``\d{3,}`` (3 桁以上) を「大きい」の代用にしており、300 文字程度の
#: 普通のチャット回答までユニット分割パイプラインへ流していた (実インシデント
#: 2026-08-01 ライブ監査: 「300 文字程度で説明して」が 4 ユニット 530 字になり、
#: 品質ゲートが「目標の 1.77 倍」と警告した)。桁数は大きさの代用でしかなく、
#: 100〜999 の帯で必ず外れる。捕捉した数値をそのまま閾値と比べる。
_LENGTH_REQUEST_RE = re.compile(r"(\d{2,})\s*(文字|字|行|文)")
_LENGTH_REQUEST_RE_EN = re.compile(
    r"\b(\d{2,})[\s-]?(words?|lines?|characters?|chars?)\b", re.IGNORECASE,
)
#: 単位ごとの長文しきい値。1 単位あたりの分量が違うので単位別に持つ。
#: 字/文字 1000 は「1 パス生成で収まる上限」の目安。これ未満でユニット分割
#: すると、構造化の利得より各ユニットの尺の膨らみ (実測 1.77 倍) が上回る。
_LONG_FORM_MIN_BY_UNIT: dict[str, int] = {
    "文字": 1000, "字": 1000, "文": 50, "行": 100,
    "word": 300, "words": 300, "character": 1000, "characters": 1000,
    "char": 1000, "chars": 1000, "line": 100, "lines": 100,
}
#: 分量指定 (1 桁も拾う)。``_LENGTH_REQUEST_RE`` は長文側の判定専用で
#: ``\d{2,}`` に限っているため、「3文で」のような **短さの明示** が
#: そもそも分量指定として認識されなかった (``requests_short_output`` 参照)。
_ANY_LENGTH_REQUEST_RE = re.compile(r"(\d+)\s*(文字|字|行|文|段落)")
_ANY_LENGTH_REQUEST_RE_EN = re.compile(
    r"\b(\d+)[\s-]?(sentences?|paragraphs?|words?|lines?|characters?|chars?)\b",
    re.IGNORECASE,
)
#: この値**以下**なら「短い」と確定し long_form へ振らない。長文しきい値との
#: 間 (例: 11〜49 文) はどちらでもないので、他の long_form シグナルに委ねる。
_SHORT_FORM_MAX_BY_UNIT: dict[str, int] = {
    "文": 10, "行": 20, "段落": 3, "文字": 400, "字": 400,
    "sentence": 10, "sentences": 10, "paragraph": 3, "paragraphs": 3,
    "line": 20, "lines": 20, "word": 150, "words": 150,
    "character": 400, "characters": 400, "char": 400, "chars": 400,
}


def is_environment_fact_query(query: str) -> bool:
    """クエリがこの実行環境の事実 (時刻 / OS / スペック等) を尋ねているか。

    router の executable_query 分類と、``ToolCallJudge.measurement_blocked`` の
    公開可否判定が **同じ語彙を共有** するための公開口。別々に持つと、片方だけ
    更新されて「測定要求なのに注記が付かない / 測定要求でないのに付く」の
    どちらかがずれる (純粋関数)。
    """
    return _matches_any(_EXECUTABLE_QUERY_PATTERNS, query) or _matches_any(
        _EXECUTABLE_QUERY_PATTERNS_EN, query,
    )


#: ディレクトリを指す名詞。
_DIRECTORY_NOUN_RE = re.compile(
    r"(?:ディレクトリ|フォルダ"
    r"|(?<![A-Za-z])director(?:y|ies)(?![A-Za-z])"
    r"|(?<![A-Za-z])folders?(?![A-Za-z]))",
    re.IGNORECASE,
)

#: 「そこに何があるか」を問う列挙表現。
_LISTING_QUESTION_RE = re.compile(
    r"(?:直下|配下|の中に|の下に|一覧|リスト|中身|入って(?:いま|ま)す|何があり)"
    r"|(?<![A-Za-z])list(?![A-Za-z])"
    r"|what(?:'s| is| are)\s+in",
    re.IGNORECASE,
)


def asks_directory_listing(query: str) -> bool:
    """ディレクトリの中身の列挙を求めているか (純粋関数)。

    ディレクトリ名詞と列挙表現の **連言** で判定する。片方だけでは必ず誤爆する —
    「ディレクトリとは何ですか」は名詞のみ、「利点を一覧で教えて」は列挙のみで、
    どちらも実行要求ではない。

    実インシデント (2026-08-03 ライブ監査): 「backend ディレクトリの直下には何が
    ありますか」がツールシグナル無しと判定されて knowledge query に落ち、
    ``list_directory`` が一度も走らないまま Node.js の構成
    (``config/ db/ controllers/ routes/ app.js``) を丸ごと捏造した。続く
    「そのうち Python ファイルはいくつありますか」も捏造の上に「5 個」と答えた。
    """
    return bool(
        _DIRECTORY_NOUN_RE.search(query) and _LISTING_QUESTION_RE.search(query),
    )


def requests_long_output(query: str) -> bool:
    """クエリが「長文と呼べる分量」を明示的に指定しているか (純粋関数)。

    単位ごとの閾値以上を 1 つでも指定していれば True。分量指定が無い、
    または閾値未満なら False (他の long_form シグナルの判定へ委ねる)。

    **応答以外の対象に付いた数量は分量指定ではない**
    (:func:`count_belongs_to_another_subject`)。実インシデント
    (2026-08-31 ライブ監査 T05#2): 「2つ目は、1ファイルの行数を500行以内に
    することです。」というコーディング規約の申告が ``500行`` で long_form に
    振られ、**「どのような内容・主題の文書をご希望ですか？」** と返した。
    """
    for regex in (_LENGTH_REQUEST_RE, _LENGTH_REQUEST_RE_EN):
        for m in regex.finditer(query):
            amount, unit = m.group(1), m.group(2)
            threshold = _LONG_FORM_MIN_BY_UNIT.get(unit.lower())
            if threshold is None or int(amount) < threshold:
                continue
            if count_belongs_to_another_subject(query[: m.start()]):
                continue
            return True
    return False


def requests_short_output(query: str) -> bool:
    """クエリが「短い分量」を明示的に指定しているか (純粋関数)。

    ``requests_long_output`` の対称形。明示的に小さい分量を指定した依頼は、
    文書種別名詞 (案内文 / 手順書 …) を含んでいても長文生成ではない。

    ``_LENGTH_REQUEST_RE`` は ``\\d{2,}`` (2 桁以上) しか拾わないため、
    **1 桁の指定がそもそも分量指定として認識されていなかった**。
    実インシデント (2026-08-09 2 回目のライブ監査): 「会員向けの案内文を英語で
    **3文** 書いてください」が ``案内文`` で long_form に分類され、
    ユニット分割パイプラインで 543 トークン / 49 秒を生成。3 文どころか
    数段落になり、会話で確定済みの大会日 (2026-11-07) を無視して
    「October 25th」を捏造した。
    """
    for regex in (_ANY_LENGTH_REQUEST_RE, _ANY_LENGTH_REQUEST_RE_EN):
        for m in regex.finditer(query):
            amount, unit = m.group(1), m.group(2)
            cap = _SHORT_FORM_MAX_BY_UNIT.get(unit.lower())
            if cap is None or int(amount) > cap:
                continue
            # 長文側と同じ帰属判定を通す (片方だけ直すと非対称になる)。
            if count_belongs_to_another_subject(query[: m.start()]):
                continue
            return True
    return False


def _matches_any(patterns: list[re.Pattern], query: str) -> bool:
    """パターンリストのいずれかが query にマッチするか判定する。

    JA/EN 両方のパターンリストを GUI locale に関わらず両方評価する用途で使う
    (2026-07-22 発見: 以前は locale で片方の言語のみ評価しており、GUI locale
    と実際の入力言語が食い違うと該当言語側の判定が一切効かなかった)。
    """
    return any(p.search(query) for p in patterns)


#: ルール表で meta_cognitive を表すセンチネル。実際の層はマッチ時に
#: ``_guard_meta_cognitive()`` で解決する (コンテキスト予算が足りなければ
#: deliberative へ降格するため、表に静的な層名は書けない)。
_META = "__meta_cognitive__"


class _ClassifyContext:
    """1 回の ``classify()`` 呼出で共有する導出値。

    ``is_knowledge`` / ``is_short`` 等は複数のルールが参照するため、都度
    計算せずここでキャッシュする。すべて純粋関数の結果なので、評価タイミングが
    変わっても判定は変わらない (早期マッチ時に評価されないよう遅延させている)。
    """

    __slots__ = ("_c", "_cache", "_context", "mode", "query")

    def __init__(
        self,
        classifier: "ComplexityClassifier",
        query: str,
        mode: str,
        context: str | Callable[[], str] = "",
    ) -> None:
        self._c = classifier
        self.query = query
        self.mode = mode
        #: 直近会話の本文 (または遅延評価の callable)。被演算子の片方が前ターン
        #: にしか無い計算クエリを reactive に落とさないために使う
        #: (``looks_like_numeric_question``)。消費するルールは numeric_question
        #: だけなので、そこまで到達しなかった呼出では組み立て自体を省く。
        self._context = context
        self._cache: dict[str, object] = {}

    @property
    def context(self) -> str:
        if callable(self._context):
            self._context = self._context()
        return self._context

    def _memo(self, key: str, compute):
        if key not in self._cache:
            self._cache[key] = compute()
        return self._cache[key]

    @property
    def is_knowledge(self) -> bool:
        return self._memo("k", lambda: self._c._is_knowledge_query(self.query))

    @property
    def is_short(self) -> bool:
        return self._memo(
            "s", lambda: self._c._is_short_query(self.query, self.mode),
        )

    @property
    def is_long_form_candidate(self) -> bool:
        return self._memo("lf", lambda: self._c._detect_long_form(self.query))


@dataclass(frozen=True)
class _ClassifyRule:
    """``classify()`` の 1 判定。表中の並び順がそのまま優先度。"""

    #: decision.jsonl / policy_adjuster へ渡る matched-rule 識別子。
    reason: str
    #: 遷移先。``_META`` はマッチ時にガード経由で解決する。
    layer: str
    predicate: Callable[["ComplexityClassifier", _ClassifyContext], bool]
    #: マッチ時に ``is_long_form`` へ設定する値。
    long_form: bool = False


#: 判定順序の SSOT。**並び順が優先度**であり、番号は振らない
#: (かつて 1.4 / 2.5 / 8.5 のような小数が後付け層の痕跡として残り、
#: 欠番 3 / 4 を docstring が参照する事故も起きていた)。
_CLASSIFY_RULES: tuple[_ClassifyRule, ...] = (
    _ClassifyRule(
        "greeting", "reactive",
        lambda c, x: c._is_greeting(x.query),
    ),
    # URL 取得 → ファイル書込みの連鎖は meta_cognitive でしか実行できない。
    # 学習語による long_form 誤振り分けより先に評価する。
    _ClassifyRule(
        "url_write_intent", _META,
        lambda c, x: c._is_url_write_intent(x.query, x.mode),
    ),
    # 表形式データ出力先 (.csv/.tsv/.xlsx/.ods) はユニット分割の散文生成が
    # 構造を壊すため long_form を抑止して write-fast 経路へ落とす
    # (2026-07-15: annual_events.csv が 26 unit の散文 CSV 化)。
    # long_form 候補の部分集合なので long_form ルールより前に置く。
    _ClassifyRule(
        "local_write_intent", _META,
        lambda c, x: (
            x.is_long_form_candidate and c._is_tabular_write_intent(x.query, x.mode)
        ),
    ),
    _ClassifyRule(
        "long_form", _META,
        lambda c, x: x.is_long_form_candidate,
        long_form=True,
    ),
    # long_form (文書系名詞) にも url_write_intent にも当たらないデータ成果物
    # (Excel カレンダー/一覧表等) のローカル出力を拾う。long_form 判定の後に
    # 置くことで「仕様書/ドキュメント + パス」は従来どおり long_form を維持する。
    _ClassifyRule(
        "local_write_intent", _META,
        lambda c, x: c._is_local_write_intent(x.query, x.mode),
    ),
    # create の正規な生成依頼が executable_query へ短絡するのを防ぐため、
    # executable_query より先に評価する (2026-07-22: 「CSV を集計する
    # プログラムを作成して」が staged 生成へ一切到達しなかった)。
    _ClassifyRule(
        "create_tools", _META,
        lambda c, x: (
            is_create_mode(x.mode) and not x.is_knowledge and c._needs_tools(x.query)
        ),
    ),
    # 緩い名詞+助詞パターンは正規の生成依頼にも誤マッチするため、ここでは
    # strict 側 (明示的な疑問形式) だけを知識質問として扱う。
    _ClassifyRule(
        "create_meta_keywords", _META,
        lambda c, x: (
            is_create_mode(x.mode)
            and c._has_meta_keywords(x.query)
            and not c._is_strict_knowledge_query(x.query)
        ),
    ),
    # 「明日は何月何日?」のような ? 終わりの疑問文は知識質問パターンが要求する
    # 末尾形式を満たさないため、is_knowledge を見ずに常時評価する。
    _ClassifyRule(
        "executable_query", "deliberative",
        lambda c, x: c._contains_executable_query_keywords(x.query),
    ),
    _ClassifyRule(
        "history_ref", "deliberative",
        lambda c, x: c._has_history_keywords(x.query),
    ),
    _ClassifyRule(
        "complex_keywords", "deliberative",
        lambda c, x: c._has_complex_keywords(x.query),
    ),
    # reactive / reactive_light は検索パイプラインを一切走らせないため、短い
    # 知識質問を short_query で reactive に落とすとカートリッジを参照せず
    # 事前知識だけで答えてしまう。short_query の手前に置く。
    _ClassifyRule(
        "knowledge_query", "deliberative",
        lambda c, x: x.is_knowledge,
    ),
    # 記憶の想起依頼も reactive に落とすと SemMem を引かずに「情報がありません」
    # と答える (_PERSONAL_RECALL_RE 参照)。knowledge_query と同じく short_query
    # の手前に置く。
    _ClassifyRule(
        "personal_recall", "deliberative",
        lambda c, x: c._is_personal_recall_query(x.query),
    ),
    # 一人称を伴わない想起形 (「〜でしたか」) も同じ経路へ逃がす
    # (_DISCOURSE_RECALL_RE 参照)。
    _ClassifyRule(
        "discourse_recall", "deliberative",
        lambda c, x: c._is_discourse_recall_query(x.query),
    ),
    # 計算を求めるクエリを reactive に落とすとツール判定に一度も到達せず、
    # base の暗算に倒れる (2026-08-08 ライブ監査:「時速240kmで2時間30分走ると
    # 何km進みますか。」(26 文字) が short_query → reactive で 540km と誤答。
    # 正解 600km)。knowledge_query / personal_recall と同じく short_query の
    # 手前に置く。判定は core.intent_vocab が SSOT。
    _ClassifyRule(
        "numeric_question", "deliberative",
        lambda c, x: looks_like_numeric_question(x.query, x.context),
    ),
    # 前提の同意を求める確認形も同様 (_PREMISE_CONFIRMATION_RE 参照)。
    _ClassifyRule(
        "premise_confirmation", "deliberative",
        lambda c, x: c._is_premise_confirmation_query(x.query),
    ),
    # 自分自身の構成 / ツール目録を尋ねるクエリも short_query の手前に置く。
    # 根拠はすべて決定論で取れる (ToolsRegistry / config / llama-server の
    # /props) のに、その注入はどれも deliberative 側にしか無い
    # (``deliberative._append_tool_inventory_fact`` / ツール判定層 0.6b)。
    # reactive_light へ落ちるとツール判定ごと外れ、base が事前知識だけで
    # 答えて **存在しないツール名を並べる**。実インシデント
    # (2026-08-22 ライブ監査 2 回目 ターン8): 「あなたが使えるツールを箇条書きで
    # 列挙してください。」(30 文字) が short_query → reactive_light に落ち、
    # 「web_search / fetch_url / execute_code / get_weather / get_stock_price /
    # get_sports_results」という **1 つも登録されていない** 一覧を回答した。
    # knowledge_query / personal_recall / numeric_question を short_query の
    # 手前に置いたのと同じ理由。
    # 「ファイルに保存して」型の永続化依頼は、保存先が書かれていなくても
    # reactive へ落とさない。``local_write_intent`` はパスを必須にするため、
    # パスの無い依頼はツール判定に一度も到達しない。実インシデント
    # (2026-08-22 ライブ監査 2 回目 ターン 252): 「ファイルに保存しておいて。」
    # (12 文字) が short_query → reactive に落ち、「ファイル保存機能は利用
    # できないため、保存できません。」と回答した。**同じ会話のターン 122 で
    # write_file が成功している** ので能力が無いという説明自体が誤りだった。
    # 保存先が無いなら聞き返すのが正しい応答で、それは deliberative でしか出せない。
    _ClassifyRule(
        "persist_intent", "deliberative",
        lambda c, x: persist_request(x.query),
    ),
    _ClassifyRule(
        "self_config_query", "deliberative",
        lambda c, x: tool_inventory_question(x.query) or runtime_info_question(x.query),
    ),
    # 「続けて」型の継続要求も short_query の手前に置く。切断が観測されていれば
    # 分類器の手前で ``_dispatch_continuation`` が奪うが、**切れずに完結した
    # 直後の「続けて」** はここへ降りてくる。3 文字なので必ず short_query →
    # reactive_light に落ち、直近数件の履歴しか見ないモデルが直前の user 発話を
    # 逐語で復唱する。実インシデント (2026-08-25 ライブ監査 T6-5):
    # 長文生成 (cogwriter, 972 tokens) が正常終了した次ターンの「続けて」に対し
    # 「Pythonのデコレータについて2000文字程度で詳しく解説してください。」という
    # **前ターンの依頼文そのもの** を返した。続きを書くには何をどこまで書いたかを
    # 見る必要があり、それは deliberative でしか出せない。
    _ClassifyRule(
        "continuation_request", "deliberative",
        lambda c, x: continuation_request(x.query),
    ),
    # 体言止めの問い (「〜は？」) も short_query の手前に置く
    # (_TOPIC_STOP_QUESTION_RE 参照)。上の個別ルール (numeric_question /
    # knowledge_query / personal_recall …) を先に通してから受けるので、
    # それらの matched-rule 識別子は変わらない。
    _ClassifyRule(
        "topic_stop_question", "deliberative",
        lambda c, x: c._is_topic_stop_question(x.query),
    ),
    _ClassifyRule(
        "short_query", "reactive",
        lambda c, x: x.is_short,
    ),
    _ClassifyRule(
        "tool_patterns", _META,
        lambda c, x: c._needs_tools(x.query),
    ),
    # 表の終端。ここまで落ちてきたクエリは「短くもなく、知識質問でも履歴参照でも
    # ツール要求でもない」もので、reactive の視界 (直近 6 メッセージ / 記憶なし)
    # では足りない。
    #
    # かつて終端は reactive で、その手前に ``not x.rag_results`` を条件とする
    # ``no_rag_results`` ルールが居た。しかし ``classify()`` の本番呼出は
    # ``rag_results`` を渡さない (検索は投機並列化され、分類の時点では結果が
    # 無い) ため、この条件は **恒真** で、終端の 2 ルール
    # (``tool_patterns`` / ``default``) は到達不能だった。実測でも
    # ``no_rag_results`` は 31 回発火する一方 ``tool_patterns`` は 0 回で、
    # meta_cognitive へは ``local_write_intent`` からしか入っていない。
    # 恒真ルールを消して終端を deliberative にすることで、実質の振り分けを
    # 変えずに (実測 233 ターンの再生で差分 0) ``tool_patterns`` の経路を戻す。
    #
    # 注: ``policy_interpreter`` の router ドメイン ``rag_score_threshold`` は
    # これで **消費者ゼロ** になった (唯一の読み手だった ``short_high_rag`` の
    # 閾値)。ただし探索の次元を食う問題は既に解消している — router ドメイン
    # 自体が ``policy_evolver.EVOLVABLE_DOMAINS`` から外れ、摂動の対象では
    # なくなったため (層の振り分け閾値を、モデル自身の出力由来の turn_outcome で
    # 動かすのは閉ループになる)。残っているのは永続化された進化ステートとの
    # スキーマ互換のためのキーだけで、実行時には誰も読まない。
    # 不変則は ``test_router_domain_stays_frozen`` が固定する。
    _ClassifyRule("default", "deliberative", lambda c, x: True),
)


#: 成果物 **について尋ねる問い**。生成依頼ではないので長文生成から外す。
#:
#: 生成依頼と問いを分けるのは文末の形。「〜を書いてください」は依頼、
#: 「〜はありましたか」「〜は何章ですか」は問い。語彙 (計画書 / レポート …)
#: では分けられない — 同じ語が両方に出る。
#: 文末記号。日本語の問いは疑問符を伴わず句点で閉じるのが普通。文末記号を
#: 許さないと「…ありましたか。」が 1 件も当たらない。
_SENTENCE_CLOSE = r"[。．.、,！!\s\"'」』）)]*$"
#: 過去形の問い (「〜ましたか」「〜でしたか」)。**分量指定より先に** 評価する —
#: 「3000字のレポートは出力されましたか」は分量を含んでいても依頼ではない。
#: 旧定義の ``ありました?か`` は ``た`` を任意にした誤記で「ありましか」に当たり、
#: 「切れましたか」「終わりましたか」「出力されましたか」を取りこぼしていた
#: (2026-09-02 監査 A3)。
_ARTIFACT_PAST_QUESTION_RE = re.compile(
    r"(?:ありましたか|ありませんでしたか|(?:まし|でし)たか|ましたでしょうか)"
    + _SENTENCE_CLOSE,
)
#: 現在形の問い / 疑問符閉じ。語彙は ``QUESTION_TAIL_RE`` (知識質問の strict
#: 判定と同じ) + ``ますか`` + 疑問符。丁寧な依頼 (「〜書いてくれますか？」) も
#: ここに当たるため、明示的な分量指定 (``requests_long_output``) の **後** に
#: 評価する。
_ARTIFACT_QUESTION_RE = re.compile(
    rf"(?:{QUESTION_TAIL_RE.pattern}|ますか|[?？])" + _SENTENCE_CLOSE,
)


#: ユーザー **自身の** 予定・約束の申告。生成依頼ではないので長文生成から外す。
#:
#: 実インシデント (2026-08-30 ライブ監査 T11#4): 「来週までにアーキテクチャ
#: 設計書を書くと約束しました。」が meta_cognitive (long_form) へ振られ、
#: **693 秒 (11.5 分)** かけて 577 トークンの一般論エッセイを生成した。同じ
#: テーマの他の申告 (「締切は9月30日です」「チームは4人です」) はすべて 1 行の
#: 復唱で返っている。``設計書`` + ``書`` が ``LONG_FORM_PATTERNS[0]`` に当たる。
#:
#: 依頼と申告を分けるのは ``_ARTIFACT_QUESTION_RE`` と同じく **文末の形**。
#: 語彙 (設計書 / レポート …) では分けられない。引用節 (「〜と約束しました」)
#: や予定の言い切りが文の終わりに来ていれば、それは報告であって依頼ではない。
#: 文末に錨を打つので「レポートを書くと言ったよね、書いてくれる？」のように
#: 後ろに本物の依頼が続くクエリは除外されない。
_SELF_COMMITMENT_REPORT_RE = re.compile(
    r"(?:"
    r"と(?:約束|宣言|明言|報告|連絡)(?:を)?し(?:まし)?た"
    r"|と(?:言い|伝え|話し|決め)(?:まし)?た"
    r"|ことに(?:なり|し)(?:まし)?た"
    r"|(?:予定|つもり)(?:です|だ|でした)"
    r")"
    r"[。．.、,！!\s\"'」』）)]*$",
)


class ComplexityClassifier:
    """クエリの複雑度を分類し、適切なエージェント層を決定する

    分類はすべてルールベース。LLM 呼び出しゼロ。
    ランタイムガードにより、Meta-Cognitive 層が利用できない場合は
    Deliberative 層にフォールバックする。
    """

    def __init__(
        self,
        config: dict | None = None,
        learned_patterns: LearnedPatternStore | None = None,
        policy: PolicyInterpreter | None = None,
    ):
        self._config = config or {}
        self._policy = policy
        self.is_long_form: bool = False
        self._learned_patterns = learned_patterns
        learning_cfg = self._config.get("learning", {})
        # 長文生成パターン学習の閾値 (ハードコード regex に加えて
        # `category="long_form"` の学習済みパターンも OR 判定する)
        self._long_form_threshold: float = learning_cfg.get(
            "long_form_pattern_match_threshold", 0.4,
        )

    def classify(
        self,
        query: str,
        mode: str = "chat",
        context: str | Callable[[], str] = "",
    ) -> str:
        """クエリの複雑度を分類する

        判定は ``_CLASSIFY_RULES`` の **並び順がそのまま優先度** で、最初に
        マッチしたルールの層を返す。順序と条件はすべて表側にあるので、本体は
        表を上から評価するだけ。ルールを足すときは表の適切な位置へ 1 エントリ
        加える (以前は 18 個の if が本体に並び、後付け層が 1.4 / 2.5 / 8.5 の
        ような小数コメントで表現されていた)。

        **判定は query / mode / 直近会話だけを見る**。検索結果は渡さない —
        検索は分類の後に投機起動されるため、分類の時点では存在しない。かつて
        ``rag_results`` 引数があったが本番呼出は一度も渡しておらず、それを条件
        にする 2 ルールが「永久に false」と「恒真」になっていた (表末尾の
        ``default`` のコメント参照)。

        ``context`` は文字列か、それを返す callable (遅延評価)。消費するのは
        numeric_question ルールだけなので、そこへ到達しない呼出では組み立てない。

        Returns:
            "reactive" | "deliberative" | "meta_cognitive"
        """
        self.is_long_form = False
        self._classify_mode = mode
        self._last_classify_reason = "default"

        ctx = _ClassifyContext(self, query, mode, context)
        for rule in _CLASSIFY_RULES:
            if not rule.predicate(self, ctx):
                continue
            reason = rule.reason
            layer = rule.layer
            if layer == _META:
                layer = self._guard_meta_cognitive()
                if layer != "meta_cognitive":
                    # 予算不足で deliberative へ降格したターンは long_form 経路
                    # (ユニット分割生成) に乗らない。旧実装は is_long_form=True と
                    # reason="long_form" を残したまま deliberative を返しており、
                    # decision.jsonl では降格が見えなかった (2026-09-02 監査 D1)。
                    reason = f"{reason}:budget_downgrade"
                    self.is_long_form = False
                    return self._record_classification(layer, reason, query)
            self.is_long_form = rule.long_form
            return self._record_classification(layer, reason, query)

        raise AssertionError("_CLASSIFY_RULES must end with an unconditional rule")

    def _record_classification(self, layer: str, reason: str, query: str) -> str:
        """分類結果をログし matched-rule 識別子を保持して layer を返す。

        ``self._last_classify_reason`` に matched-rule 識別子を残す。chat 側が
        primary routing を decision.jsonl に記録する際の ``reason`` に使う
        (context={"mode": ...} と併せて policy_adjuster の mode 別学習へ供給)。
        """
        self._last_classify_reason = reason
        logger.debug("Classified as %s (%s): %s", layer, reason, query[:50])
        return layer

    def _guard_meta_cognitive(self) -> str:
        """Meta-Cognitive 層のランタイムガード

        設定で無効化されている場合や、コンテキスト予算が不足する場合は
        Deliberative 層にフォールバックする。
        """
        agent_cfg = self._config.get("agent", {})

        # 明示的無効化チェック
        if not agent_cfg.get("meta_cognitive_enabled", True):
            logger.info("Meta-Cognitive disabled by config, falling back to deliberative")
            return "deliberative"

        # コンテキスト予算チェック
        mode = getattr(self, "_classify_mode", "chat")
        if not _can_use_meta_cognitive(self._config, self._policy, mode):
            logger.warning(
                "Meta-Cognitive budget insufficient, falling back to deliberative"
            )
            return "deliberative"

        return "meta_cognitive"

    def _is_short_query(self, query: str, mode: str = "chat") -> bool:
        """クエリが短い（単純）かどうかを判定

        日本語テキストはスペースで分割されないため、文字数で判定する。
        分岐は **表記体系** で決める。空白トークン数で分岐すると、日本語文に
        ASCII 断片 (パス / コマンド / 識別子) が混ざっただけで英語扱いになり、
        長い日本語クエリが「短文」と誤判定される (実インシデント 2026-07-29
        ライブ監査: 55 文字の「コマンド `dir E:\\tmp\\x` を実行して、返ってきた
        出力をそのまま報告してください。」が空白 4 トークンなので short_query
        と判定され、検索もツール判定も走らない reactive に落ちて、ベースモデルが
        実行していないコマンドの出力を捏造した)。
        """
        min_tokens = self._get_policy("router", "short_query_min_tokens", mode, 3)
        max_tokens = self._get_policy("router", "short_query_max_tokens", mode, 10)
        max_chars = self._get_policy("router", "short_query_max_chars", mode, 20)

        # 英語テキスト: 単語数で判定 (CJK を含まない場合のみ)
        if not _CJK_CHAR_RE.search(query):
            tokens = len(query.split())
            if tokens >= min_tokens:
                return tokens < max_tokens
        # 日本語テキスト: 文字数で判定
        return len(query.strip()) < max_chars

    def _is_greeting(self, query: str) -> bool:
        """挨拶パターンを検出"""
        return any(p.match(query.strip()) for p in GREETING_PATTERNS)

    def _has_complex_keywords(self, query: str) -> bool:
        """複雑度を示すキーワードを検出"""
        q_lower = query.lower()
        if any(kw in q_lower for kw in COMPLEX_KEYWORDS):
            return True
        return _matches_any(COMPLEX_KEYWORDS_EN_PATTERNS, query)

    def _has_history_keywords(self, query: str) -> bool:
        """履歴参照キーワードを検出 (locale を問わず JA / EN 両方を見る)。

        実体は ``core.intent_vocab.has_history_recall_keyword``
        (tool_call_judge の強制発火判定と同じ照合)。
        """
        return has_history_recall_keyword(query)

    def _has_meta_keywords(self, query: str) -> bool:
        """Meta-Cognitive 層へのエスカレーションキーワードを検出"""
        q_lower = query.lower()
        if any(kw in q_lower for kw in META_KEYWORDS):
            return True
        return _matches_any(META_KEYWORDS_EN_PATTERNS, query)

    def _is_url_write_intent(self, query: str, mode: str = "chat") -> bool:
        """URL からデータを取得してファイルに書き出す意図を検出する。

        chat モードのみ対象 (create モードは step 2.5 の ``_needs_tools`` が同等の
        meta_cognitive 振り分けを行う)。URL とファイル書込み/出力意図の双方が
        揃った場合のみ True を返す。
        """
        if not is_chat_mode(mode):
            return False
        if not _URL_HINT_RE.search(query):
            return False
        return bool(_FILE_WRITE_INTENT_RE.search(query))

    def _is_local_write_intent(self, query: str, mode: str = "chat") -> bool:
        """ローカルパス + ファイル書込み意図 (URL 無し) を検出する。

        chat モードのみ対象 (create モードは step 2.5 / 2.6 の ``_needs_tools`` /
        ``_has_meta_keywords`` が同等の振り分けを行う)。URL を含む場合は
        ``_is_url_write_intent`` / fetch 経路に委ね、知識質問 (「作成方法を
        教えて」等の how-to) は除外する。ローカルパスと書込み動詞の双方が
        揃った場合のみ True を返す。
        """
        if not is_chat_mode(mode):
            return False
        if _URL_HINT_RE.search(query):
            return False
        if _HOWTO_QUERY_RE.search(query):
            return False
        # コマンドリテラル (`dir E:\tmp\x`) 内のパスは実行対象の引数であって
        # 書込み先ではない。書込み動詞・パスの双方をこれを除いた本文で判定する。
        probe = strip_command_literals(query)
        # 「いま書いたそのファイル」のような既出成果物への言及は、依頼された
        # 動作ではなく対象の説明。書込み動詞判定から除外する。
        probe = _DESCRIPTIVE_WRITE_CLAUSE_RE.sub(" ", probe)
        # 書込み動詞の列挙に加えて **本文代入** の構文も受ける。「内容を『X』に
        # してください」は書込み依頼だが動詞を 1 つも含まないため、動詞だけの
        # 判定では読取へ落ちる (assigns_file_content の docstring 参照)。
        if not _WRITE_VERB_RE.search(probe) and not assigns_file_content(probe):
            return False
        if _LOCAL_PATH_RE.search(probe):
            return True
        # ディレクトリを伴わない裸のファイル名 (「notes.txt に追記して」)。
        # 直前に作ったファイルを名前だけで指す言い方で、書込み先としては
        # 参照依頼と同じ扱い (保存先は write-fast 経路が会話から解決する)。
        if _BARE_FILENAME_TARGET_RE.search(probe):
            return True
        # パスは直前ターンにしか無い参照依頼 (「同じファイルに保存し直して」)。
        # 保存先は write-fast 経路が会話から解決する。
        return bool(_REFERENTIAL_WRITE_TARGET_RE.search(probe))

    def _is_tabular_write_intent(self, query: str, mode: str = "chat") -> bool:
        """表形式データ (.csv/.tsv/.xlsx/.ods) のローカル書出し意図を検出する。

        long_form (散文ユニット分割) では表構造の成果物を生成できないため、
        long_form 判定に勝つ優先ゲートとして使う。ローカル書込み意図と
        表形式拡張子の双方が揃った場合のみ True。
        """
        if not self._is_local_write_intent(query, mode):
            return False
        return bool(_TABULAR_TARGET_RE.search(query))

    def _detect_long_form(self, query: str) -> bool:
        """長文生成リクエストかどうかを判定

        ハードコード regex (LONG_FORM_PATTERNS) と
        ``LearnedPatternStore`` の ``category="long_form"`` を OR 判定する。
        後者は FeedbackCollector / LearningScheduler により自己進化する。
        """
        # URL を含むクエリは取得 (fetch) を要するためツール経路で扱う。散文生成
        # (long_form) は URL 内容を取得できず捏造するため、学習語が一致しても除外する。
        if _URL_HINT_RE.search(query):
            return False
        # 成果物 **について尋ねる過去形の問い** は生成依頼ではない。
        #
        # 実インシデント (2026-08-27 ライブ監査 T10-7):
        # 「計画書の出力が途中で切れた部分はありましたか。」に対し
        # **「どのような内容・主題の文書をご希望ですか？」** と返した。
        # 「計画書」「出力」という語で長文生成候補になり、meta_cognitive が
        # 新規の文書生成依頼として受けてしまう。文末が問いなら生成しない。
        if _ARTIFACT_PAST_QUESTION_RE.search(query):
            return False
        # ユーザー自身の予定・約束の申告も依頼ではない
        # (``_SELF_COMMITMENT_REPORT_RE`` のコメント参照)。
        if _SELF_COMMITMENT_REPORT_RE.search(query):
            return False
        # 「さきほどの表で上書きして」型も同じ。書くべき本文は会話にあり、
        # 生成する対象が無い (_PRIOR_ARTIFACT_REF_RE のコメント参照)。
        if (
            _PRIOR_ARTIFACT_REF_RE.search(query)
            and _WRITE_TO_FILE_VERB_RE.search(query)
        ):
            return False
        # 「見出しだけ並べて」型は既出成果物の抽出であって生成ではない。
        if _EXTRACTION_REQUEST_RE.search(query):
            return False
        # 明示的な分量指定は、丁寧な依頼形 (「〜くれますか？」) や how-to 除外より
        # 先に見る。「3000字のレポートを書いてくれますか？」は疑問符で閉じて
        # いても長文生成の依頼 (2026-09-02 監査 A1)。
        if requests_long_output(query):
            return True
        # how-to は「作成方法の説明」を求めているのであり、実際の生成を
        # 依頼しているわけではない。_is_local_write_intent と同じ _HOWTO_QUERY_RE
        # で除外しないと、「手順書を作成する方法を教えて」/ "How do I create a
        # status report at <path>" のような howto 質問が実際に長文ドキュメントを
        # 生成・書込みしてしまう (2026-07-22 ライブ検証で判明。JA/EN 両方で
        # 再現し、PR#298-300 の対象範囲より前から存在した既存バグ)。
        if _HOWTO_QUERY_RE.search(query):
            return False
        # 成果物について尋ねる現在形の問い / 疑問符閉じも生成依頼ではない。
        if _ARTIFACT_QUESTION_RE.search(query):
            return False
        # 明示的な短さ指定は文書種別名詞より優先する。長文側を先に見ているので、
        # 「2000字で」のような明示的な長文指定があればそちらが勝つ。
        if requests_short_output(query):
            return False
        if _matches_any(LONG_FORM_PATTERNS, query) or _matches_any(
            LONG_FORM_PATTERNS_EN, query,
        ):
            return True
        return self._contains_learned_long_form_patterns(query)

    def _contains_learned_long_form_patterns(self, query: str) -> bool:
        """学習済み long_form パターンにマッチするか判定

        単一キーワード hit での発火は一般語の自己強化ループを招く
        (2026-07-15: 「文書」「明日」等の学習語 1 hit で朝礼メモや CSV 依頼が
        ユニット分割パイプラインへ流出)。閾値以上の一致が 2 語以上、または
        重み合計が ``_LEARNED_LONG_FORM_MIN_WEIGHT_SUM`` 以上の場合のみ発火する。

        さらに 2026-07-25 から、学習語だけで発火させるには文書種別名詞
        (``DOCUMENT_NOUN_LEARNABLE_JA``) を 1 語以上含むことを必須にした。
        学習側の許容リスト (``is_long_form_learnable``) と二重防御の関係で、
        ストアに既存の汚染語が残っていても誤発火しない
        (実測: 「ありがとう。技術的な議論の部分はとても有益だった」が
        技術的 0.65 + 技術 0.5 の 2 語成立で 6,436 字の長文生成に化けた)。
        """
        if self._learned_patterns is None:
            return False
        matches = self._learned_patterns.match(query, category="long_form")
        if not matches:
            return False
        if not any(noun in query for noun in DOCUMENT_NOUN_LEARNABLE_JA):
            return False
        eligible = [w for _, w in matches if w >= self._long_form_threshold]
        if len(eligible) >= 2:
            return True
        return sum(eligible) >= _LEARNED_LONG_FORM_MIN_WEIGHT_SUM

    def _is_knowledge_query(self, query: str) -> bool:
        """知識質問パターンを検出（RAG で処理すべきクエリ）"""
        return _matches_any(KNOWLEDGE_QUERY_PATTERNS, query) or _matches_any(
            KNOWLEDGE_QUERY_PATTERNS_EN, query,
        )

    def _is_personal_recall_query(self, query: str) -> bool:
        """ユーザー自身について記憶している内容の想起依頼かを判定する。"""
        return bool(
            _PERSONAL_RECALL_RE.search(query)
            or _PERSONAL_RECALL_EN_RE.search(query),
        )

    def _is_discourse_recall_query(self, query: str) -> bool:
        """会話で既出の事柄を問い直す想起形かを判定する (_DISCOURSE_RECALL_RE 参照)。"""
        stripped = query.strip()
        return bool(
            _DISCOURSE_RECALL_RE.search(stripped)
            or _DISCOURSE_RECALL_EN_RE.search(stripped),
        )

    def _is_topic_stop_question(self, query: str) -> bool:
        """体言止めの問いかを判定する (_TOPIC_STOP_QUESTION_RE 参照)。"""
        stripped = query.strip()
        return bool(
            _TOPIC_STOP_QUESTION_RE.search(stripped)
            and not _GREETING_TOPIC_RE.search(stripped),
        )

    def _is_premise_confirmation_query(self, query: str) -> bool:
        """前提の同意を求める確認形かを判定する (_PREMISE_CONFIRMATION_RE 参照)。"""
        return bool(_PREMISE_CONFIRMATION_RE.search(query.strip()))

    def _is_strict_knowledge_query(self, query: str) -> bool:
        """緩い名詞+助詞パターン (_KNOWLEDGE_QUERY_LOOSE_PATTERN) を除いた、
        明示的な疑問形式のみによる知識質問判定。create_meta_keywords の
        is_knowledge 上書き判定用 (詳細は _KNOWLEDGE_QUERY_STRICT_PATTERNS 参照)。
        """
        return _matches_any(_KNOWLEDGE_QUERY_STRICT_PATTERNS, query) or _matches_any(
            KNOWLEDGE_QUERY_PATTERNS_EN, query,
        )

    def _contains_executable_query_keywords(self, query: str) -> bool:
        """Python 実行で正確に答えられるクエリのキーワードを検出"""
        return is_environment_fact_query(query)

    def _needs_tools(self, query: str) -> bool:
        """ツール呼び出しが必要なパターンを検出"""
        return _matches_any(TOOL_PATTERNS, query) or _matches_any(
            TOOL_PATTERNS_EN, query,
        )

    def _get_policy(
        self, domain: str, key: str, mode: str, default: int | float,
    ) -> int | float:
        """PolicyInterpreter からパラメータを取得（フォールバック付き）"""
        return get_policy_value(self._policy, domain, key, default, mode=mode)


def _can_use_meta_cognitive(
    config: dict,
    policy: PolicyInterpreter | None = None,
    mode: str = "chat",
) -> bool:
    """コンテキスト予算が Meta-Cognitive ループに十分か判定"""
    loop_budget = resolve_meta_cognitive_loop_budget(config, mode)

    cfg_default = config.get("agent", {}).get("meta_cognitive_min_budget", 512)
    min_budget = get_policy_value(
        policy, "agent", "meta_cognitive_min_budget", cfg_default, mode=mode,
    )

    return loop_budget >= min_budget
