"""ツール判定確定後のガード列 (純粋関数 + 宣言的パイプライン)

``ToolCallJudge.judge()`` の全 exit が通る唯一の後処理 funnel の実体。各ガードは
``(ToolJudgement, GuardContext) -> ToolJudgement`` の純粋関数で、対象ツール名が
一致しなければ何もしない。適用順と適用条件は :data:`GUARD_PIPELINE` に **データと
して** 宣言してあり、ガードを足すときの編集箇所は「関数 1 つ + 表の 1 行」に閉じる。

以前は同じガード列が判定層ごとに手で書き写されており、層によって適用される部分
集合が食い違っていた。funnel へ集約したあとも、ガードは判定クラスのメソッドとして
``self`` に閉じたまま、``_finalize`` 側が引数の違いを lambda で吸収していたため、
「ガードを足す」たびに呼出側の wiring も書く必要があった。ここではシグネチャを
1 本に揃え、その wiring を無くしている。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import PurePosixPath

from backend.free.agent.tool_judge_args import (
    _coerce_positive_int,
    _extract_head_line_count,
    _normalize_path_text,
)
from backend.free.agent.tool_judge_dialogue import _dialogue_text
from backend.free.agent.tool_judge_grounding import _ungrounded_numbers
from backend.free.agent.tool_judge_history import _only_proximal_recall_keywords
from backend.free.agent.tool_judge_signals import (
    _IMMEDIATE_CHILDREN_RE,
    _READ_PATH_TOOLS,
    _RECURSIVE_LISTING_RE,
)
from backend.free.agent.tool_judge_types import ToolJudgement
from backend.free.agent.tools_registry import ToolDefinition, ToolsRegistry
from backend.log_config import get_logger

logger = get_logger("agent.tool_call_judge")


#: 実行可能コマンドを載せるツール名 (mode により片方のみ利用可能)。
_COMMAND_TOOL_NAMES = frozenset({"run_command", "run_command_readonly"})

#: 「実行するとファイル/環境の状態が変わる」ツール名。mode 制約で撃てなかった
#: 場合、``action_blocked`` を立てて完了報告の捏造を禁じる必要がある
#: (``run_command`` は読取専用の兄弟へ載せ替わる経路があるため
#: ``_COMMAND_TOOL_NAMES`` 側で measurement_blocked として扱う)。
_STATE_CHANGING_TOOL_NAMES = frozenset({"write_file", "apply_patch", "delete_file"})


#: 処理対象の本文そのものを引数に取るツール。判定プロンプトの会話は切り詰めて
#: あるため、aux の転記をそのまま使うと断片だけが処理される。
_TEXT_OPERAND_TOOLS = frozenset({"summarize", "translate"})


#: 同じ能力を持ち権限だけが違うツールの対応表 (優先順)。aux が mode 外の
#: 兄弟名を返したとき、撃てる方へ載せ替えて判定の意図を保つ。緩い側から厳しい
#: 側への一方向にだけ張る (逆向きに張ると chat が特権ツールへ昇格してしまう)。
_MODE_CAPABILITY_SIBLINGS: dict[str, tuple[str, ...]] = {
    "run_command": ("run_command_readonly",),
}


@dataclass
class GuardContext:
    """ガード列が参照する判定文脈と、列を通す間に立つ「実行できなかった」印。

    ``measurement_blocked`` / ``action_blocked`` はガードの **出力**。判定を
    no_tool へ落とした理由が「そもそも不要」ではなく「撃てなかった」ことを
    呼出側 (``deliberative``) へ伝え、測っていない値の断定や、実行していない
    操作の完了報告を禁じる注記に使う。

    ``tools_registry`` は参照するガード (可用性チェック / hidden 抑止) だけが
    要求する。単体のガードを 1 つだけ掛ける経路では ``None`` のままでよい。
    """

    tools_registry: ToolsRegistry | None = None
    mode: str = ""
    query: str = ""
    conversation: list[dict] | None = None
    #: aux (層4) 専用ガードも適用するか (``_finalize`` の同名引数)。
    aux_guards: bool = False
    #: モデルへ提示したツール一覧に hidden ツールを含めたか。
    hidden_tools_offered: bool = False
    measurement_blocked: bool = False
    action_blocked: bool = False


Guard = Callable[[ToolJudgement, "GuardContext"], ToolJudgement]


def _suppress_proximal_recall_cross_session(
    result: ToolJudgement, ctx: GuardContext,
) -> ToolJudgement:
    """近接リコール語だけを根拠に現在セッションを除外した検索を撃たせない。

    「さっき」「先ほど」等が指しているのは**進行中の会話**であり、それを
    ``exclude_session_id`` で除外した検索先に答えは無い。結果は 2 通りしか
    なく、どちらも有害:

    - 0 件 → 10 秒前後を捨てたうえ「該当なし」が根拠として base に渡り、
      進行中の会話の内容まで否定させる (2026-07-28 ライブ検証)
    - 別セッションがヒット → その内容が「さっきの話」として提示される
      (2026-08-05 ライブ監査: 「さっき E:\\tmp\\...txt に書き込んで
      もらったはず」に対し、**別セッション**の
      ``E:\\tmp\\監査メモ.txt に「検証コードはアオサギ42」`` を今回の
      会話で依頼されたファイル操作として列挙した)

    このガードは元々 ``judge()`` の層5.5 (history_keyword_forced_fallback)
    にインラインで書かれており、**aux 経路 (層4) には掛かっていなかった**。
    上記 2026-08-05 の実インシデントはまさに ``source=aux`` の判定で
    起きている。同責務の抑止は層ごとに書き写さず ``_finalize`` の funnel
    へ集約する (funnel の docstring 参照)。

    ``query`` 未指定 (既定 "") の呼出では ``_only_proximal_recall_keywords``
    が False を返すため安全に no-op。
    """
    if result.tool_name != "search_history" or not result.tool_needed:
        return result
    if not (result.tool_args or {}).get("exclude_session_id"):
        return result
    if not _only_proximal_recall_keywords(ctx.query):
        return result
    logger.debug(
        "Suppressing search_history: proximal recall word refers to the "
        "ongoing session, which is excluded from the search: %s",
        ctx.query[:50],
    )
    return ToolJudgement(tool_needed=False, source=result.source)

def _validate_tool_availability(
    result: ToolJudgement, ctx: GuardContext,
) -> ToolJudgement:
    """tool_name が実在し、かつ現在の mode で利用可能かを最終チェックする。

    ``_judge_with_rules`` / ``_judge_with_learned_patterns`` 等の各判定層は
    ``tools_registry.has()`` (存在チェックのみ、mode 非考慮) で判定するため、
    chat モードで ``modes=["create"]`` のツール (例: run_command) が
    ``tool_needed=True`` のまま返り得る。実行自体は
    ``deliberative._execute_tool`` のモードゲートで阻止されるが、判定結果が
    True のまま turn_outcome=failed → 次ターンの訂正誤検出 → 無関係語の
    correction 誤学習、というカスケードを引き起こす (2026-07-18 の会話
    ログで実際に発生・確認済み)。

    また ``_json_to_judgement`` は ``tool == "no_tool"`` の完全一致でのみ
    no-tool と判定するため、tool_judgment 応答が max_tokens 到達で
    ``{"tool": "no_`` のように途中切断され ``json_repair`` が
    ``tool_name="no_"`` へ復元した場合、存在しないツール名のまま
    ``tool_needed=True`` で返ってしまう (2026-07-18 実インシデントで確認)。
    レジストリに存在しないツール名も判定確定の最終防衛としてここで
    no_tool に倒す。
    """
    if not result.tool_needed or not result.tool_name:
        return result
    tool_def = ctx.tools_registry.get(result.tool_name)
    if tool_def is None:
        logger.info(
            "Tool %s not found in registry (truncated JSON or "
            "hallucinated name?); downgrading to no_tool before "
            "returning judgement",
            result.tool_name,
        )
        if result.tool_name in _STATE_CHANGING_TOOL_NAMES:
            # 未登録のまま _STATE_CHANGING_TOOL_NAMES に載っている名前
            # (delete_file / apply_patch) は、この分岐が mode ゲートより
            # 先に立つため下の action_blocked に到達しない。状態を変える
            # 意図が選ばれた事実は同じなので、ここでも記録する
            # (2026-08-12 ライブ監査で削除完了の捏造を確認)。
            ctx.action_blocked = True
        return ToolJudgement(tool_needed=False, source=result.source)
    if ctx.mode not in tool_def.modes:
        remapped = _remap_to_mode_sibling(result, ctx, tool_def)
        if remapped is not None:
            return remapped
        logger.info(
            "Tool %s not available in mode=%s (allowed: %s); "
            "downgrading to no_tool before returning judgement",
            result.tool_name, ctx.mode, tool_def.modes,
        )
        if result.tool_name in _COMMAND_TOOL_NAMES:
            # 実測しようとして mode 制約で撃てなかったケース。
            # 「ツール不要」と区別して記録する (measurement_blocked 参照)。
            ctx.measurement_blocked = True
        elif result.tool_name in _STATE_CHANGING_TOOL_NAMES:
            # 状態を変える操作を選んだのに mode 制約で撃てなかった。
            # 黙って no_tool に落とすと base が完了報告を捏造する
            # (chat の ``write_file`` は create 限定で、書込みは
            # meta_cognitive 経路が担う)。実インシデント 2026-08-09
            # ライブ監査: 裸のファイル名を指した追記依頼がツール 0 回のまま
            # 「E:\tmp\inventory_notes.txt の末尾に追記しました」と
            # **フルパスまで補って** 報告され、実ファイルは無変更だった。
            ctx.action_blocked = True
        return ToolJudgement(tool_needed=False, source=result.source)
    return result

def _scope_list_directory_depth(
    result: ToolJudgement, ctx: GuardContext,
) -> ToolJudgement:
    """「直下だけ」の一覧依頼で ``list_directory`` を 1 階層に絞る。

    既定の 3 階層ツリーを返すと、受け取ったモデルがインデントを読み違えて
    入れ子の項目を直下の項目として並べる (実インシデント 2026-08-01 再検証:
    「直下にあるファイルとフォルダを一覧して」に対し backend/ の下の
    develop/ api/ tests/ を直下として列挙した)。深さは依頼文から決まる
    決定論的な値なので、モデルの転記に委ねず code 側で確定させる。
    """
    if result.tool_name != "list_directory" or not ctx.query:
        return result
    if not _IMMEDIATE_CHILDREN_RE.search(ctx.query):
        return result
    if _RECURSIVE_LISTING_RE.search(ctx.query):
        return result
    args = dict(result.tool_args or {})
    if _coerce_positive_int(args.get("max_depth")) == 1:
        return result
    args["max_depth"] = 1
    logger.info(
        "list_directory scoped to immediate children for query: %s",
        ctx.query[:60],
    )
    return ToolJudgement(
        tool_needed=True,
        tool_name=result.tool_name,
        tool_args=args,
        source=result.source,
    )

def _scope_read_file_line_range(
    result: ToolJudgement, ctx: GuardContext,
) -> ToolJudgement:
    """「先頭 N 行だけ」の読取依頼で ``read_file`` を範囲指定に絞る。

    ``list_directory`` の深さ絞り (:meth:`_scope_list_directory_depth`) と
    同じく、依頼文から決まる**引数の確定**であって抑止ではない。したがって
    層を問わず適用してよい (ツール名が違えば no-op)。

    層ごとに書き写していたため実際に食い違っていた: ``_infer_tool`` と
    ``_referential_read_judgement`` は ``_extract_head_line_count`` を見るが、
    **文法制約ツール分類 (層5.9) は見ない**。分類器は ``file_path`` しか
    埋めないため、そこで確定した瞬間に範囲指定が消える。

    実インシデント (2026-08-16 再測定ターン 14):「そのファイルの先頭2行だけ
    見せてください。」→ ``Tool classifier selected: read_file({'file_path': ...})``
    で全文 3,405 文字が返り、2 行の依頼に対して全文が渡っていた。同じ意味の
    「全文は長すぎます。そのファイルの先頭5行だけをそのまま見せてください。」は
    決定論層が拾って ``showing lines 1-5`` になる。**言い回しで挙動が割れていた**。

    既に範囲が入っている判定 (決定論層が確定させた場合) は触らない。
    """
    if result.tool_name != "read_file" or not ctx.query:
        return result
    args = dict(result.tool_args or {})
    if args.get("start_line") is not None or args.get("end_line") is not None:
        return result
    head = _extract_head_line_count(ctx.query)
    if head is None:
        return result
    args["start_line"] = 1
    args["end_line"] = head
    logger.info(
        "read_file scoped to head %d lines for query: %s", head, ctx.query[:60],
    )
    return ToolJudgement(
        tool_needed=True,
        tool_name=result.tool_name,
        tool_args=args,
        source=result.source,
    )

def _restore_truncated_text_operand(
    result: ToolJudgement, ctx: GuardContext,
) -> ToolJudgement:
    """text 引数へ転記された「切り詰め済み会話」の断片を全文へ復元する。

    判定用プロンプトは会話を 1 メッセージ ``_JUDGE_CONTEXT_CHARS`` 文字で
    切って aux に見せる。summarize / translate のように**処理対象の本文
    そのものを引数に取る**ツールでは、aux はその切り詰められた断片しか
    転記できず、後段は断片だけを処理してしまう (実インシデント 2026-08-01
    ライブ監査: 4 ユニット 530 文字の四季の文章を「1 行に要約して」と頼んだ
    ところ、先頭 100 文字 = 春の節だけが要約された)。

    引数が会話中メッセージの真の接頭辞になっている場合、それは切り詰めの
    産物であって「その部分だけを対象にする」という意図ではない。元メッセージ
    全文へ差し替える。引数が全文と一致していれば何もしない。

    文字数上限を引き上げる対処では、上限を超える長さで同じ欠落が再発する。
    転記させず code 側で解決するのが構造的な解。
    """
    if not result.tool_needed or result.tool_name not in _TEXT_OPERAND_TOOLS:
        return result
    excerpt = (result.tool_args or {}).get("text")
    if not isinstance(excerpt, str) or not excerpt.strip():
        return result
    excerpt = excerpt.strip()
    for msg in reversed(ctx.conversation or []):
        content = msg.get("content")
        if not isinstance(content, str):
            continue
        full = content.strip()
        if len(full) > len(excerpt) and full.startswith(excerpt):
            logger.info(
                "Restored truncated text operand for %s: %d -> %d chars",
                result.tool_name, len(excerpt), len(full),
            )
            args = dict(result.tool_args or {})
            args["text"] = full
            return ToolJudgement(
                tool_needed=True,
                tool_name=result.tool_name,
                tool_args=args,
                source=result.source,
            )
    return result

def _remap_to_mode_sibling(
    result: ToolJudgement, ctx: GuardContext, tool_def: ToolDefinition,
) -> ToolJudgement | None:
    """mode 外のツール名を、同じ能力を持つ mode 内の兄弟ツールへ載せ替える。

    aux へ渡すカタログは mode で絞ってあるが、モデルは学習事前分布から
    カタログ外の兄弟名を返すことがある (実インシデント 2026-08-01 ライブ監査:
    chat モードで ``run_command`` を返し、ガードが no_tool へ落として
    「ツールを実行できなかった」という曖昧な断りだけが残った。実際には
    ``run_command_readonly`` が同じ ``command`` 引数で撃てた)。

    判定そのものは正しいのに名前だけが外れている場合に意図を捨てないための
    載せ替え。引数スキーマが被覆関係にある (兄弟の必須引数をすべて満たせる)
    ときだけ行い、満たせなければ従来どおり no_tool へ落とす。権限は兄弟側の
    ツールが自前で検証するため、ここで緩むことはない。

    Returns:
        載せ替えた判定。該当が無ければ None (純粋な判定、副作用なし)。
    """
    for sibling in _MODE_CAPABILITY_SIBLINGS.get(result.tool_name, ()):
        if not ctx.tools_registry.is_available(sibling, ctx.mode):
            continue
        supplied = set(result.tool_args or {})
        if not ctx.tools_registry.required_params(sibling) <= supplied:
            continue
        logger.info(
            "Tool %s not available in mode=%s (allowed: %s); remapping to "
            "same-capability sibling %s",
            result.tool_name, ctx.mode, tool_def.modes, sibling,
        )
        return ToolJudgement(
            tool_needed=True,
            tool_name=sibling,
            tool_args=dict(result.tool_args or {}),
            source=result.source,
        )
    return None

def _suppress_unfetchable_fetch_url(
    result: ToolJudgement, ctx: GuardContext,
) -> ToolJudgement:
    """url 引数を補完できない ``fetch_url`` を ``tool_needed=False`` に格下げ.

    URL リコール (``_maybe_recall_url`` / ``_judge_with_url_recall``) でも
    補完できず、LLM 判定でも url が validate されなかったケースを
    ``deliberative._execute_tool`` の手前で抑制する。これにより:

    - "fetch_url() requires args but none provided, skipping" の警告と
      UI 上の空ステップ表示を防ぐ
    - LLM が事前知識のみで応答する経路にクリーンに落ちる

    ``fetch_url`` 以外のツール / url が補完済みの fetch_url はそのまま返す。
    """
    if not result.tool_needed or result.tool_name != "fetch_url":
        return result
    args = result.tool_args or {}
    if args.get("url"):
        return result
    logger.info(
        "Suppressing fetch_url with no URL argument (no recall hit, "
        "no in-query URL); LLM will respond from prior knowledge",
    )
    return ToolJudgement(
        tool_needed=False,
        tool_name="",
        tool_args={},
        source=result.source,
    )

def _suppress_commandless_run_command(
    result: ToolJudgement, ctx: GuardContext,
) -> ToolJudgement:
    """command 引数を補完できない ``run_command`` を ``tool_needed=False`` に格下げ.

    ``_suppress_unfetchable_fetch_url`` と対称。rule / learned 層は字句
    マッチだけで ``run_command`` を選ぶことがあり (learned 層は
    ``_infer_tool`` が推定できない場合の run_command フォールバックを持つ)、
    ルール層のコマンド解決でも command が
    埋まらなかった場合、実行段階で "requires args but none provided" と
    空振りするだけの判定が残る。実インシデント (2026-07-20 ライブ検証):
    学習済み tool_routing パターン「説明」(w=0.630) が知識質問
    「〜を説明して」にマッチし、create モードで引数なし run_command が
    返り得た (chat モードは run_command の mode ゲートで偶然無害化されて
    いた)。ここで no_tool に倒し、クエリを通常の LLM 応答パスに落とす。
    """
    if not result.tool_needed or result.tool_name not in (
        "run_command", "run_command_readonly",
    ):
        return result
    if (result.tool_args or {}).get("command"):
        return result
    logger.info(
        "Suppressing %s with no command argument (no rule "
        "inference, no aux synthesis); downgrading to no_tool",
        result.tool_name,
    )
    return ToolJudgement(
        tool_needed=False,
        tool_name="",
        tool_args={},
        source=result.source,
    )

def _suppress_hidden_tool_from_aux(
    result: ToolJudgement, ctx: GuardContext,
) -> ToolJudgement:
    """提示していない hidden ツール名が返された場合 no_tool に格下げ.

    hidden ツール (run_command_readonly 等) はプロンプトのツール一覧
    (``get_descriptions_text``) に出ないため、それを見て判定したモデルが
    その名前を返すのは定義上 hallucination。``_validate_tool_availability``
    は登録済み + mode 適合なら通してしまう (chat で modes=["chat"] の
    hidden ツールは素通り) ため、free-form 判定層の防衛としてここで弾く。
    judge のコード側注入経路 (early-return / recall / fallback /
    _infer_tool) は本メソッドを通らないため影響しない。

    **hidden ツールを提示した経路には掛けない** (``_finalize`` の
    ``hidden_tools_offered``)。文法制約ツール分類は enum に hidden も
    載せるので、そこで選ばれた名前は hallucination ではない。
    """
    if not result.tool_needed or not result.tool_name:
        return result
    tool_def = ctx.tools_registry.get(result.tool_name)
    if tool_def is None or not tool_def.hidden:
        return result
    logger.warning(
        "Suppressing hidden tool %s: it was not in the tool list shown to "
        "the model (source=%s); downgrading to no_tool",
        result.tool_name, result.source,
    )
    return ToolJudgement(tool_needed=False, source=result.source)

def _suppress_expressionless_calculate(
    result: ToolJudgement, ctx: GuardContext,
) -> ToolJudgement:
    """expression 引数を補完できない ``calculate`` を ``tool_needed=False`` に格下げ.

    ``_suppress_commandless_run_command`` と対称。rule / learned 層の
    ``_infer_tool`` は「計算」の字句マッチだけで calculate を選ぶが式抽出
    ロジックを持たず常に空 args を返し、aux 層も free-form args のため
    ``{"tool": "calculate", "args": {}}`` があり得る。実行段の必須引数
    チェックで "requires args but none provided" と空振りするだけの判定が
    残る。実インシデント (2026-07-21 ライブ検証 ターン35):
    「フィボナッチ数列の10番目を計算して」が rule 層で ``calculate, {}``
    になり WARN + UI 空ステップ (running フレームのみで完了フレーム無し)。
    ここで no_tool に倒し、クエリを通常の LLM 応答パスに落とす
    (格下げ後の応答は LLM 知識で正解した実測あり)。
    """
    if not result.tool_needed or result.tool_name != "calculate":
        return result
    if (result.tool_args or {}).get("expression"):
        return result
    logger.info(
        "Suppressing calculate with no expression argument; "
        "downgrading to no_tool",
    )
    return ToolJudgement(
        tool_needed=False,
        tool_name="",
        tool_args={},
        source=result.source,
    )

def _suppress_ungrounded_calculate(
    result: ToolJudgement, ctx: GuardContext,
) -> ToolJudgement:
    """クエリにも会話にも無い数値を含む ``calculate`` を no_tool へ格下げ.

    層5.2 (``_judge_with_calculate_fallback``) は合成式へ
    ``_synthesized_expression_grounded`` を掛けて捏造数値を弾くが、層4
    (aux ``tool_judgment``) には同じ検証が無く、素通りしていた。ツールの
    戻り値は「確かめた事実」として base に最優先で渡されるため、捏造された
    式の結果は **正しく計算された嘘** になり、素の暗算より有害になる
    (実インシデント 2026-07-29 ライブ監査: 「本当にそれで合っていますか？
    計算を見直してください。」に対し aux が ``57.8 - 4 * 1.5`` を合成。
    ``1.5`` はクエリにも会話にも無く、結果 51.8 が「正解」として提示された。
    正しくは ``57.8 - 4 * 3.4`` = 44.2)。

    格下げ後は後続層 (5.2) がグラウンディング検証付きで式を組み直す機会を
    得るため、計算そのものを諦めることにはならない。

    ホワイトリストには層5.2 の直近 4 ターンではなく **判定に渡された会話
    全体** を使う。層5.2 は式を *合成* するので狭い窓で捏造の余地を絞るのが
    正しいが、こちらは既に選ばれた式を *拒否* するだけなので、窓を狭めると
    「その距離を時速12キロで…」のような照応で数ターン前の数値を参照した
    正当な式まで巻き込む。捏造の信号 (会話のどこにも無い数値) は広い窓でも
    変わらず立つ。
    """
    if not result.tool_needed or result.tool_name != "calculate":
        return result
    expression = str((result.tool_args or {}).get("expression") or "")
    if not expression:
        return result
    context = _dialogue_text(ctx.conversation)
    unexplained = _ungrounded_numbers(expression, ctx.query, context)
    if not unexplained:
        return result
    # 格下げが正当なのは、式を組み直す層が実際に走れるときだけ。その層
    # (旧 5.2 の式合成) は撤去済みなので、格下げの落ち先は base の暗算に
    # なる。実測では暗算のほうが誤答しやすく、しかも式が見えないぶん
    # 誤りに気づけない (2026-08-08 以降のライブ監査で 4 回連続。いずれも
    # 「式は正しかったのに 1 つの数値が書かれていない」ケースで、
    # 42.195*5.5 / 3*52 / 8*5*(18-9)*2 などが棄却された)。式は残し、
    # 説明できない数値を回答側で開示させる。
    logger.info(
        "Keeping calculate with unexplained numbers %s in %r; "
        "the answer must disclose them",
        list(unexplained), expression[:80],
    )
    return replace(result, unexplained_numbers=unexplained)

def _suppress_ungrounded_read_path(
    result: ToolJudgement, ctx: GuardContext,
) -> ToolJudgement:
    """クエリにも会話にも現れないパスの読み取りツールを no_tool へ格下げ.

    層4 (aux ``tool_judgment``) は「ファイルの金額を合計して」のような
    パスを含まないタスクに対しても read 系ツールを選び、``file_path`` を
    でっち上げることがある。捏造パスの読み取りは構造的に必ず失敗するか、
    最悪の場合まったく別のファイルを読むため、実行する価値が無い
    (実インシデント 2026-07-29 ライブ監査: 「合計を計算して / 中身も
    見せて」という 2 サブタスクが ``read_file(prices.txt)`` と
    ``read_file(unknown)`` になり、直前ステップで内容は取れていたのに
    「1 件のタスクを完了し、2 件が失敗しました。」だけが返った)。

    格下げすると後続層およびツールループが、パスを持たないタスクとして
    扱い直す機会を得る。書込み系は対象外 — 出力先の既定解決 (
    ``_resolve_write_path`` 等) が別途あり、パス未指定の生成依頼を
    巻き込むため。
    """
    if not result.tool_needed or result.tool_name not in _READ_PATH_TOOLS:
        return result
    args = result.tool_args or {}
    path = str(
        args.get("file_path") or args.get("path") or args.get("directory") or "",
    )
    if not path:
        return result
    haystack = _normalize_path_text(
        f"{ctx.query}\n{_dialogue_text(ctx.conversation)}",
    )
    if _normalize_path_text(path) in haystack:
        return result
    # ドライブ / ディレクトリ表記の揺れを許すためベース名でも照合する。
    basename = _normalize_path_text(PurePosixPath(path.replace("\\", "/")).name)
    if basename and basename in haystack:
        return result
    logger.info(
        "Suppressing ungrounded %s path %r (absent from query and dialogue); "
        "downgrading to no_tool", result.tool_name, path[:120],
    )
    return ToolJudgement(
        tool_needed=False,
        tool_name="",
        tool_args={},
        source=result.source,
    )


def _always(ctx: GuardContext) -> bool:
    """層を問わず適用する (対象ツール名が違えば no-op なので安全)。"""
    return True


def _aux_only(ctx: GuardContext) -> bool:
    """aux (層4) の free-form args 向けガード。

    コード側がツール名や引数を注入する経路に掛けると正当な判定を潰すため、
    aux 経路でのみ有効化する。
    """
    return ctx.aux_guards


def _aux_without_hidden_offer(ctx: GuardContext) -> bool:
    """hidden ツールを **提示していない** aux 経路のみ。

    「プロンプトの一覧に出ない名前が返るのは hallucination」という前提は、
    提示していないときにしか成り立たない。文法制約ツール分類は hidden も enum に
    載せるため、そこで選ばれた名前を潰すと chat で唯一の実行系ツールが恒久的に
    到達不能になる (実インシデント 2026-08-08 ライブ監査 ターン13)。
    """
    return ctx.aux_guards and not ctx.hidden_tools_offered


@dataclass(frozen=True)
class GuardSpec:
    """ガード 1 件の宣言。``applies`` が False の文脈では列から外れる。"""

    name: str
    fn: Guard
    applies: Callable[[GuardContext], bool] = _always


#: 適用順。上から順に掛け、no_tool へ降格した時点で打ち切る。
#:
#: 可用性チェックは **常に最後**。mode 制約で撃てなかった場合に
#: ``measurement_blocked`` を立てるため、引数欠落で先に降格したケースと
#: 区別できる位置に置く必要がある。
GUARD_PIPELINE: tuple[GuardSpec, ...] = (
    GuardSpec("unfetchable_fetch_url", _suppress_unfetchable_fetch_url),
    GuardSpec("commandless_run_command", _suppress_commandless_run_command),
    GuardSpec("expressionless_calculate", _suppress_expressionless_calculate),
    # 深さ絞り / 行範囲絞りは抑止ではなく、依頼文から決まる引数の確定なので層を
    # 問わず安全 (対象ツール名が違えば no-op)。aux 限定にしていたため rule 層の
    # list_directory が素通りし、同じ依頼でも層が変わると再発した (実インシデント
    # 2026-08-04: source=aux では効き、source=rule では既定 3 階層のまま
    # 5,523 字が切り詰められた)。
    GuardSpec("immediate_children_depth", _scope_list_directory_depth),
    GuardSpec("read_file_line_range", _scope_read_file_line_range),
    GuardSpec(
        "proximal_recall_excluded_session", _suppress_proximal_recall_cross_session,
    ),
    GuardSpec("truncated_text_operand", _restore_truncated_text_operand, _aux_only),
    GuardSpec("ungrounded_calculate", _suppress_ungrounded_calculate, _aux_only),
    GuardSpec("ungrounded_read_path", _suppress_ungrounded_read_path, _aux_only),
    GuardSpec(
        "hidden_tool_from_aux",
        _suppress_hidden_tool_from_aux,
        _aux_without_hidden_offer,
    ),
    GuardSpec("tool_availability", _validate_tool_availability),
)


def apply_guards(result: ToolJudgement, ctx: GuardContext) -> ToolJudgement:
    """:data:`GUARD_PIPELINE` を順に掛け、確定した判定を返す。

    no_tool へ降格した時点で打ち切る (以降のガードは no_tool を素通しするだけ)。
    どのガードが降格させたかは切り分けに効くのでログへ残す。
    """
    for spec in GUARD_PIPELINE:
        if not spec.applies(ctx):
            continue
        was_needed = result.tool_needed
        result = spec.fn(result, ctx)
        if was_needed and not result.tool_needed:
            logger.debug(
                "Judge guard %s downgraded the judgement to no_tool", spec.name,
            )
            break
    return result
