"""検証結果の 3 値 — 合格 / 不合格 / 未検査 (理由) (f_10 §12.4)。

create の検査 (import / 静的検査・契約テスト・参考テスト …) の結果を 1 つの型で持つ。
各モジュールで「合格」「実行できず」の文字列を作らない — 表示は :meth:`CheckOutcome.render`
が i18n (``create.check.*``) で組む。

**未検査は不合格ではない**: 検査できなかった (作業フォルダの外へ書こうとした・外部依存が無い・
入出力例が無い) のはコードの誤りの証拠ではないので、``tasks_failed`` にも作り直しにも数えない。
合格とも書かない。学習では「ラベル無し」として扱う (成否の教師にしない)。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class CheckKind(StrEnum):
    """検査の種類 (i18n ``create.check.label.<値>``)。"""

    #: import / 静的検査 (構文検査器)。
    STATIC = "static"
    #: 契約テスト (骨組みの入出力例)。
    CONTRACT = "contract"
    #: 参考テスト (生成したテスト)。警告だけで未完了にも作り直しにも数えない。
    ADVISORY = "advisory"
    #: 使い方を配信形で 1 回実行する (Python のパッケージ形と平置き、f_10 §11.1-3)。不合格は ``tasks_failed`` に数える。
    USAGE = "usage"


#: 経験の成否に効く検査 (docs/f_04 §2.5)。ここが未検査ならそのターンは学習で
#: ラベル無し。参考テストは警告だけなので含めない。
OUTCOME_BEARING_CHECKS: frozenset[CheckKind] = frozenset({CheckKind.STATIC, CheckKind.CONTRACT})


class CheckStatus(StrEnum):
    """検査の状態。"""

    PASSED = "passed"
    FAILED = "failed"
    UNCHECKED = "unchecked"


class UncheckedReason(StrEnum):
    """未検査の理由 (i18n ``create.check.reason.<値>``)。

    検査の種類ごとに言い方を変える理由は ``create.check.reason_by_check.<検査>.<理由>`` (構文検査器が無い
    静的検査は :attr:`MISSING_DEPENDENCY`)。
    """

    #: 評価中のコードが作業フォルダの外へ書こうとして止められた (f_10 §11.1-4)。
    SANDBOX_VIOLATION = "sandbox_violation"
    #: 外部依存 (pygame 等) が未インストール。静的検査では構文検査器 (tree-sitter-language-pack) が無い。
    MISSING_DEPENDENCY = "missing_dependency"
    #: 骨組みに契約へ組める入出力例が無い。
    NO_EXAMPLES = "no_examples"
    #: 骨組みに入出力例はあるが、形が崩れていて (式の構文エラー / 期待値がリテラルでない) 1 つも組めない。
    INVALID_EXAMPLES = "invalid_examples"
    #: 入出力例はあるが、どれもファイル (前の例・前の実行が残したデータ) に依存するので契約から外した。
    STATEFUL_EXAMPLES = "stateful_examples"
    #: 入出力例はあるが、どれも同じ呼出しで値が変わる (乱数・時刻など) ので契約から外した。
    NONDETERMINISTIC_EXAMPLES = "nondeterministic_examples"
    #: 入出力例を全件外したが、理由が混在する・結果が長すぎる・理由が読めない (どれか 1 つを名指せない)。
    UNCOMPARABLE_EXAMPLES = "uncomparable_examples"
    #: 参考テストが残らなかった (生成できない・テスト関数が無い)。
    NO_TESTS = "no_tests"
    #: 参考テストを生成したが lint が全件落とした (detail は落とした件数、f_10 §11.1-4)。
    LINT_DROPPED = "lint_dropped"
    #: 使い方が入力 (置き場所の引数・成果物に無いファイル・対話) か画面 (tkinter 等) を要求する (f_10 §11.1-3)。
    NEEDS_INPUT = "needs_input"
    #: 使い方の実行が隔離の外へ作用する恐れ (シェル・別プロセス・ソケット・ブラウザ等) があるので実行しなかった
    #: (detail はその書き方、f_10 §11.1-3)。
    SIDE_EFFECTS = "side_effects"
    #: 前の検査 (import / 静的検査) が不合格だったので、この検査は飛ばした (detail は不合格だった検査、f_10 §12.4)。
    #: こちらの検査の不合格ではないので ``tasks_failed`` には数えない。
    PREREQUISITE_FAILED = "prerequisite_failed"
    #: テストを実行できなかった (その他)。
    NOT_RUN = "not_run"


@dataclass(frozen=True)
class CheckOutcome:
    """検査 1 件の結果。

    Attributes:
        check: 検査の種類 (i18n ``create.check.label.<check>``。``static`` / ``contract`` / ``advisory``)。
        status: 合格 / 不合格 / 未検査。
        reason: 未検査の理由 (``status == UNCHECKED`` のときだけ)。
        detail: 理由の補足 (止めた書込み先・欠けた依存名)。
        count: 検査の規模 (契約テストの例の件数)。
        errors: 不合格の件数 (静的検査のエラー数・失敗したテストの数)。
        failures: 失敗したテストの行の要約 (先頭の数件)。
    """

    check: str
    status: CheckStatus
    reason: UncheckedReason | None = None
    detail: str = ""
    count: int = 0
    errors: int = 0
    failures: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def passed(cls, check: str, *, count: int = 0) -> "CheckOutcome":
        return cls(check, CheckStatus.PASSED, count=count)

    @classmethod
    def failed(
        cls, check: str, *, count: int = 0, errors: int = 0, failures: tuple[str, ...] | list[str] = (),
    ) -> "CheckOutcome":
        return cls(check, CheckStatus.FAILED, count=count, errors=errors, failures=tuple(failures))

    @classmethod
    def unchecked(
        cls, check: str, reason: UncheckedReason, *, detail: str = "", count: int = 0,
    ) -> "CheckOutcome":
        return cls(check, CheckStatus.UNCHECKED, reason=reason, detail=detail, count=count)

    @property
    def is_failure(self) -> bool:
        return self.status is CheckStatus.FAILED

    @property
    def is_unchecked(self) -> bool:
        return self.status is CheckStatus.UNCHECKED

    def status_text(self) -> str:
        """状態の文面 (「合格」「不合格 (エラー 2 件)」「未検査 (入出力例なし)」)。"""
        from backend.i18n_helper import msg

        if self.status is CheckStatus.PASSED:
            return msg("create.check.passed")
        if self.status is CheckStatus.FAILED:
            if self.check == "static" and self.errors:
                return msg("create.check.failed_errors", count=self.errors)
            return msg("create.check.failed")
        reason_value = self.reason or UncheckedReason.NOT_RUN
        # 検査の種類ごとの言い方を先に引く (静的検査の外部依存の欠け = 構文検査器が無い)
        specific = f"create.check.reason_by_check.{self.check}.{reason_value}"
        reason = msg(specific, detail=self.detail)
        if reason == specific:
            reason = msg(f"create.check.reason.{reason_value}", detail=self.detail)
        return msg("create.check.unchecked", reason=reason)

    def label(self) -> str:
        from backend.i18n_helper import msg

        if self.check == "contract" and self.count:
            return msg("create.check.label.contract_examples", count=self.count)
        return msg(f"create.check.label.{self.check}")

    def render(self) -> str:
        """verification / SPEC.md の 1 行 (「契約テスト (入出力例 4 件): 合格」)。"""
        from backend.i18n_helper import msg

        return msg("create.check.line", label=self.label(), status=self.status_text())

    def render_failures(self) -> str:
        """失敗したテストの行 (無ければ空文字列)。"""
        if not self.failures:
            return ""
        from backend.i18n_helper import msg

        return msg("create.check.failures", label=self.label(), items="; ".join(self.failures))

    def to_dict(self) -> dict:
        """結果辞書 (``notes.checks``) の 1 件。"""
        return {
            "check": self.check,
            "status": self.status.value,
            "reason": self.reason.value if self.reason else None,
            "detail": self.detail,
            "count": self.count,
            "errors": self.errors,
            "failures": list(self.failures),
        }


def unchecked_count(outcomes: list[CheckOutcome]) -> int:
    """未検査の件数 (結果辞書の別キー。0 と「動いたが効果ゼロ」を区別する)。"""
    return sum(1 for o in outcomes if o.is_unchecked)


def unchecked_labels(checks: list[dict] | None) -> list[str]:
    """結果辞書 (:meth:`CheckOutcome.to_dict` の列) から、成否に効く検査の未検査を並べる。

    ``"contract:no_examples"`` の形。経験の成否をラベル無しにする材料
    (``FeedbackCollector.record(unchecked_checks=)``、docs/f_04 §2.5)。

    契約テストの行が無く、成否に効く検査に合格でないもの (smoke の不合格で契約テストを飛ばした回) が
    あるときも契約の未検査で、``contract:not_run`` (2026-10-02 ライブ監査 #10: 静的検査が落ちた回が
    成功の教師になっていた)。成否に効く検査が全部合格で契約の行が無いのはテスト工程の無い構成
    (Free、``tests_enabled=False``) で、従来どおり成功にする。行が 1 つも無い (staged 以外の経路) は対象外。
    """
    labels: list[str] = []
    seen: set[str] = set()
    all_passed = True
    for check in checks or []:
        if not isinstance(check, dict):
            continue
        name = str(check.get("check") or "")
        seen.add(name)
        if name in OUTCOME_BEARING_CHECKS and check.get("status") != CheckStatus.PASSED.value:
            all_passed = False
        if name in OUTCOME_BEARING_CHECKS and check.get("status") == CheckStatus.UNCHECKED.value:
            labels.append(f"{name}:{check.get('reason') or UncheckedReason.NOT_RUN.value}")
    if seen and CheckKind.CONTRACT not in seen and not all_passed:
        labels.append(f"{CheckKind.CONTRACT.value}:{UncheckedReason.NOT_RUN.value}")
    return labels


__all__ = [
    "OUTCOME_BEARING_CHECKS",
    "CheckKind",
    "CheckOutcome",
    "CheckStatus",
    "UncheckedReason",
    "unchecked_count",
    "unchecked_labels",
]
