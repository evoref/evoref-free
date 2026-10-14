"""CLI⇔API 対応の分類表 (docs/f_06 §2.9.1 / §2.9.4)

全 ``/api`` ルートは次のどれか 1 つに分類される。どれにも載らないルートが増えると
``tests/test_api_parity.py`` が落ちる。

- CLI のサブコマンドが使う: 登録表 (``registry``) の ``CommandSpec.api``
- CLI が API を介さず同等の機能を直接実装している: ``CommandSpec.direct_for``
- REPL (対話チャットのスラッシュコマンド) が使う: :data:`REPL_APIS`
- GUI 専用 (CLI を作らない): :data:`GUI_ONLY_APIS`
- CLI が未対応の既知の穴: :data:`KNOWN_CLI_GAPS`

キーの書式は ``"/api/x"`` (全メソッド) か ``"GET /api/x"`` (そのメソッドだけ)。パスは
ルートの表記のまま (``{param}`` を含む) の接頭辞で、ルートのパスが等しいか
``キー + "/"`` で始まるとき一致する。1 本のルートに複数のキーが一致したら、最も具体的な
キー (パスが長い、同じ長さならメソッド付き) の分類が勝つ — 例: ``template`` の
``DELETE /api/cartridges`` は REPL の ``/api/cartridges`` より具体的なので template に入る。
"""

from __future__ import annotations

#: エディションごとの API の接頭辞。ここに無いものは Free。Pro / Develop の行は
#: そのパッケージが同梱されているときだけ実在を検査する。
API_EDITION_PREFIXES: dict[str, tuple[str, ...]] = {
    "pro": ("/api/pro",),
    "develop": ("/api/develop",),
}

#: REPL のスラッシュコマンドが使う API → コマンド名
REPL_APIS: dict[str, str] = {
    "/api/memory": "/memory・/pin・/unpin・/pinned",
    "GET /api/rag/stats": "/rag",
    "GET /api/learning/status": "/learn",
    "POST /api/learning/trigger": "/learn",
    "POST /api/learning/feedback": "/feedback (REPL のみ。非対話のサブコマンドは作らない、f_06 §2.9.6)",
    "/api/history": "/history",
    # DELETE /api/cartridges/{id} は REPL の /cartridge uninstall も使うが、より具体的な
    # 登録表の template の "DELETE /api/cartridges" (template remove) が分類を持つ
    "/api/cartridges": "/cartridge",
    # Pro が差し込む REPL のサブアクション (/cartridge create、backend/pro/cli/cartridge_create.py)。
    # 分類表は API → コマンド名のデータで、Free のコードからの呼び出しではない
    "POST /api/pro/cartridges/create": "/cartridge create",
    "GET /api/pro/cartridges/{cartridge_id}/download": "/cartridge create (作成した ZIP の取得)",
    "GET /api/themes": "/theme",
    "POST /api/themes/activate": "/theme",
    "GET /api/commands/last-output": "/page",
}

#: GUI 専用の API → CLI を作らない理由
GUI_ONLY_APIS: dict[str, str] = {
    "/api/pro/widgets": "UI 部品用の proxy (Pro)",
    "GET /api/sessions": "GUI のタブ管理の一覧。CLI は自分のセッションの register / delete だけを使う",
    "POST /api/cartridges/install/stream": "SSE の進捗 UI。CLI は非ストリームの install (REPL /cartridge) で代わりにする",
    "POST /api/cartridges/install/cancel": "SSE の進捗 UI (install/stream) の取り消し",
    "POST /api/pro/cartridges/create/stream": "SSE の進捗 UI。CLI は非ストリームの create (REPL /cartridge create) で代わりにする",
    "POST /api/pro/cartridges/create/cancel": "SSE の進捗 UI (create/stream) の取り消し",
    "GET /api/themes/{theme_id}/preview": "GUI のテーマの見本表示",
    "/api/export": "GUI の書き出し。REPL の /export は CLI 側で直接書き出す (export_commands)",
    "POST /api/files/extract": "GUI のファイル添付。REPL の /file は CLI 側で直接読む (file_reader)",
    "POST /api/system/auto-tune/decision": "GUI の承諾 UI。evoref tune は --yes で決める",
    "GET /api/system/vram_status": "GUI の VRAM 表示のポーリング",
    "DELETE /api/commands/last-output": "GUI の出力パネルの消去",
    "/api/servers": (
        "GUI の設定画面のサーバ制御の互換層 (応答形 ServerActionResponse を保つ)。所有者は"
        " LlamaProcessManager で、CLI は evoref servers が /api/model/process/* を使う (c_06 §2.8)"
    ),
    "GET /api/version": (
        "CLI は作らない: evoref --version はローカルの配布物の版、稼働中の backend の版は evoref status"
        " (/api/status の version / free_version / pro_version) が出す。外部の監視・スクリプト向けの読み口"
    ),
    "POST /api/develop/reset-local-data": "Develop の GUI のリセット。実体は evoref reset (API はそれをデタッチで起こすだけ)",
}

#: CLI が未対応の既知の穴 → 計画 (どの CLI で埋めるか)
KNOWN_CLI_GAPS: dict[str, str] = {
    "GET /api/model/info": "計画: evoref model status に載せる (f_06 §2.9.6)",
    "GET /api/model/quality": "計画: evoref model status に載せる (f_06 §2.9.6)",
}
