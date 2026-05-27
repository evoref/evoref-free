"""App factory のサブモジュール群

旧 ``backend/app_factory.py`` の責務をドメイン別に分割した置き場:

- ``_memory_init`` : EvorefMem 初期化と起動時 SemMem ブートストラップ
- ``_pillar_wirer`` : pillar (Gen / Mem / Loop / Learn) 配線と DI

公開エントリポイントは ``backend.app_factory.create_app`` のままで、
本パッケージ内のシンボルは ``app_factory.py`` の lifespan / wire_pillars
から内部的に呼び出される。
"""
