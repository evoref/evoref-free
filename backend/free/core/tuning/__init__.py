"""環境調整 (auto-tune、c_16 §7.2.3) — PC のスペックに依存する値を測る / 見積もる横断基盤。

pillar には属さない。測る持ち主は llama-server を起こす側 (起動スクリプト / ``evoref tune``) で、
backend は結果ファイル ``cache/auto_tune.json`` と確認状態を読むだけ。

- :mod:`.hardware` — ``HardwareProfile`` (RAM / GPU / コア / OS)。プローブは注入できる
- :mod:`.items` — 調整項目のレジストリ (``TuneSpec`` / ``TuneItem``)。項目は ``tuners/`` に 1 項目 1 モジュール
- :mod:`.store` — ``cache/auto_tune.json`` (形式 ``cache.auto_tune``)
- :mod:`.runner` — 項目を依存順に実行して保存する
- :mod:`.gate` — 環境移行の確認状態 (起動経路の全入口が同じ 1 実装を見る)
"""
