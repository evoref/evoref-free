"""永続レコードの前計算表コーデック (c_05 §0.5.2)。

dataclass ごとの表を **import 時に 1 回だけ** ``dataclasses.fields()`` と型注釈から
機械生成し、以後の encode / decode はその表をなぞるだけにする (キーの手書き列挙を
しない / ``asdict`` の深いコピーをしない)。

規約:

- **入れ子の各階層が自分の ``_extra`` を持つ**。未知キーはその階層の ``_extra`` へ
  退避し、書き出し時にその階層のトップへ原形のまま戻す。既知フィールドと同名の
  ``_extra`` は既知側が勝つ。メモリ上の ``_extra`` は空なら ``None``。
- **null 境界**: JSON の ``null`` / キー欠損は、読むときにここで型付きの既定値へ
  正規化する (``X | None`` は ``None``、それ以外はフィールドの既定値)。既定値の
  無い必須フィールドの null / 欠損は :class:`CodecError` (呼出側がそのレコードだけ
  飛ばして件数を数える)。
- 列挙 (``Literal``) は ``{v: v}`` の 1 回の lookup で intern する。未知値は落とさず
  そのまま持つ (使うかどうかは読み手の規則、c_05 §0.4.5)。
- 自由形の ``dict[str, Any]`` / ``Any`` は丸ごと往復する (opaque)。
- ``omit_defaults=True`` の階層は既定値と同じ値を書かない (入れ子の attrs /
  provenance 向け)。Evidence の中核は全部書く。読むときは存在するキーだけを
  なぞる (入れ子は大半のキーが省かれているため。50k 件の性能 lock)。
- ``intern=(...)`` で宣言した文字列フィールドは :func:`intern_str` の ``{v: v}``
  表で 1 つの実体に寄せる (低カーディナリティの文字列の常駐メモリ、G1 設計 §17.4)。
- 型付きの入れ子の階層に素の ``dict`` が入っていれば (構築後に代入した書き手)、
  書き出しはその dict をそのまま書く。
- decode は表から生成した関数 1 本で、凍結していない dataclass はスロットへ直接
  入れる (``__post_init__`` は書き手の入力を整えるためのもので、型付き済みの
  decode の値には通さない)。
- キー単位で patch する素の dict の階層 (kind 別 attrs 等) は :meth:`RecordCodec.check_mapping`
  で既知キーの型だけを検査する (未知キーはその dict のトップに残る)。
- ``transient=(...)`` で宣言したフィールドは永続化しない (メモリ上だけの派生値。読むときは
  既定値、書くときは出さない。同名のキーがディスクにあれば捨てる)。
- 入れ子のリスト / dict の要素は ``X | None`` を持てる (null はそのまま)。
- 型付きレコードのリストは :func:`decode_skipping` で要素ごとに読める (読めない要素だけを
  飛ばして数える。c_05 §0.5.2 の「そのレコードだけ飛ばす」)。
"""

from __future__ import annotations

import dataclasses
import types
import typing
from collections.abc import Callable
from dataclasses import MISSING
from typing import Any, Generic, Literal, TypeVar

T = TypeVar("T")

EXTRA_FIELD = "_extra"
_ABSENT = object()

_SCALARS: dict[type, tuple[type, ...]] = {
    str: (str,),
    bool: (bool,),
    # JSON は 1 と 1.0 を区別しない。bool は int の部分型なので弾く。
    int: (int,),
    float: (float, int),
}


class CodecError(ValueError):
    """1 レコードが読めない (必須欠損・型違い)。呼出側はそのレコードだけ飛ばす。"""


_Decode = Callable[[Any], Any]
_Encode = Callable[[Any], Any]

#: :func:`intern_str` の表の上限。高カーディナリティの値で膨らませない
#: (上限を超えた値は寄せずにそのまま持つ)。
INTERN_MAX = 1 << 16
_INTERN: dict[str, str] = {}


def intern_str(value: str) -> str:
    """低カーディナリティの文字列を 1 つの実体に寄せる (``{v: v}`` の 1 回の lookup)。"""
    found = _INTERN.get(value)
    if found is not None:
        return found
    if len(_INTERN) < INTERN_MAX:
        _INTERN[value] = value
    return value


@dataclasses.dataclass(frozen=True, slots=True)
class _Field:
    name: str
    optional: bool
    required: bool
    default: Any
    default_factory: Callable[[], Any] | None
    decode: _Decode
    encode: _Encode | None  # None = そのまま書く

    def default_value(self) -> Any:
        if self.default_factory is not None:
            return self.default_factory()
        return self.default


def _identity(value: Any) -> Any:
    return value


def _scalar(tp: type, where: str, *, intern: bool = False) -> _Decode:
    accepted = _SCALARS[tp]

    if tp is str and intern:
        table = _INTERN

        def decode_interned(value: Any) -> Any:
            if type(value) is not str:
                raise CodecError(f"{where}: expected str, got {type(value).__name__}")
            found = table.get(value)
            return found if found is not None else intern_str(value)

        return decode_interned

    if tp is str:

        def decode_str(value: Any) -> Any:
            if type(value) is not str:
                raise CodecError(f"{where}: expected str, got {type(value).__name__}")
            return value

        return decode_str

    def decode(value: Any) -> Any:
        if type(value) is bool and tp is not bool:
            raise CodecError(f"{where}: expected {tp.__name__}, got bool")
        if not isinstance(value, accepted):
            raise CodecError(f"{where}: expected {tp.__name__}, got {type(value).__name__}")
        return float(value) if tp is float else value

    return decode


def _literal(values: tuple[Any, ...], where: str) -> _Decode:
    table = {v: v for v in values}
    kinds = tuple({type(v) for v in values})

    if kinds == (str,):

        def decode_str_literal(value: Any) -> Any:
            if type(value) is not str:
                raise CodecError(f"{where}: expected one of {sorted(values)}")
            found = table.get(value)
            return value if found is None else found  # 未知値は保持する

        return decode_str_literal

    def decode(value: Any) -> Any:
        if not isinstance(value, kinds) or type(value) is bool and bool not in kinds:
            raise CodecError(f"{where}: expected one of {sorted(map(str, values))}")
        return table.get(value, value)  # 未知値は保持する (open / closed の扱いは読み手)

    return decode


def _strip_optional(tp: Any) -> tuple[Any, bool]:
    origin = typing.get_origin(tp)
    if origin in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        if len(args) != len(typing.get_args(tp)):
            if len(args) != 1:
                raise TypeError(f"unsupported union: {tp!r}")
            return args[0], True
        raise TypeError(f"unsupported union: {tp!r}")
    return tp, False


def _build(tp: Any, where: str, *, intern: bool = False) -> tuple[_Decode, _Encode | None]:
    """型注釈 1 つ分の (decode, encode) を作る。対応外の型は import 時に ``TypeError``。"""
    if intern and tp is not str:
        raise TypeError(f"{where}: only str fields can be interned")
    inner, optional = _strip_optional(tp)
    if optional:
        # 入れ子の要素の ``X | None`` (フィールド自体の Optional は RecordCodec が剥がす)
        inner_dec, inner_enc = _build(inner, where)

        def decode_optional(value: Any) -> Any:
            return None if value is None else inner_dec(value)

        encode_optional = None if inner_enc is None else (
            lambda v: None if v is None else inner_enc(v)
        )
        return decode_optional, encode_optional
    if tp is Any:
        return _identity, None
    if tp in _SCALARS:
        return _scalar(tp, where, intern=intern), None
    origin = typing.get_origin(tp)
    if origin is Literal:
        return _literal(typing.get_args(tp), where), None
    if dataclasses.is_dataclass(tp) and isinstance(tp, type):
        nested = codec_for(tp)
        nested_encode = nested.encode

        def encode_nested(value: Any) -> Any:
            # 構築後に素の dict を代入した書き手の値はそのまま書く
            return value if type(value) is dict else nested_encode(value)

        return nested.decode, encode_nested
    if origin is list:
        (item_tp,) = typing.get_args(tp) or (Any,)
        item_dec, item_enc = _build(item_tp, f"{where}[]")

        def decode_list(value: Any) -> list[Any]:
            if not isinstance(value, list):
                raise CodecError(f"{where}: expected a list, got {type(value).__name__}")
            if item_dec is _identity:
                return list(value)
            return [item_dec(v) for v in value]

        encode_list = None if item_enc is None else (lambda v: [item_enc(x) for x in v])
        return decode_list, encode_list
    if origin is dict:
        key_tp, val_tp = typing.get_args(tp) or (str, Any)
        if key_tp is not str:
            raise TypeError(f"{where}: dict keys must be str, got {key_tp!r}")
        val_dec, val_enc = _build(val_tp, f"{where}{{}}")

        def decode_dict(value: Any) -> dict[str, Any]:
            if not isinstance(value, dict):
                raise CodecError(f"{where}: expected an object, got {type(value).__name__}")
            if val_dec is _identity:
                return dict(value)
            return {k: val_dec(v) for k, v in value.items()}

        encode_dict = None if val_enc is None else (lambda v: {k: val_enc(x) for k, x in v.items()})
        return decode_dict, encode_dict
    raise TypeError(f"{where}: unsupported field type {tp!r}")


class RecordCodec(Generic[T]):
    """1 dataclass 分の表。:func:`codec_for` で作る (同じクラスには同じ表を返す)。"""

    def __init__(
        self, cls: type[T], *, omit_defaults: bool = False, intern: tuple[str, ...] = (),
        exclude: frozenset[str] = frozenset(), transient: tuple[str, ...] = (),
    ) -> None:
        """``exclude`` は表から外すフィールド (lock R8 が「そのフィールドを知らない
        1 つ前の版の読み手」を模すためだけに使う。外したフィールドは未知キーとして
        ``_extra`` を往復する)。:func:`codec_for` の表には効かない。"""
        if not (dataclasses.is_dataclass(cls) and isinstance(cls, type)):
            raise TypeError(f"{cls!r} is not a dataclass")
        self.cls = cls
        self.omit_defaults = omit_defaults
        self.intern = tuple(intern)
        self.transient = tuple(transient)
        hints = typing.get_type_hints(cls, include_extras=False)
        names = {f.name for f in dataclasses.fields(cls)}
        if EXTRA_FIELD not in names:
            raise TypeError(f"{cls.__qualname__} must declare `{EXTRA_FIELD}: dict[str, Any] | None = None`")
        unknown_intern = sorted(set(self.intern) - names)
        if unknown_intern:
            raise TypeError(f"{cls.__qualname__}: intern names unknown fields {unknown_intern}")
        unknown_exclude = sorted(set(exclude) - names)
        if unknown_exclude:
            raise TypeError(f"{cls.__qualname__}: exclude names unknown fields {unknown_exclude}")
        unknown_transient = sorted(set(self.transient) - names)
        if unknown_transient:
            raise TypeError(f"{cls.__qualname__}: transient names unknown fields {unknown_transient}")
        #: 永続化しないフィールド (読むときは既定値で埋める)。
        self._transient_fields = tuple(
            f for f in dataclasses.fields(cls) if f.name in self.transient
        )
        fields_: list[_Field] = []
        for f in dataclasses.fields(cls):
            if f.name == EXTRA_FIELD or not f.init or f.name in exclude or f.name in self.transient:
                continue
            tp, optional = _strip_optional(hints[f.name])
            decode, encode = _build(
                tp, f"{cls.__qualname__}.{f.name}", intern=f.name in self.intern,
            )
            factory = None if f.default_factory is MISSING else f.default_factory
            required = f.default is MISSING and factory is None
            fields_.append(_Field(
                name=f.name,
                optional=optional,
                required=required,
                default=None if f.default is MISSING else f.default,
                default_factory=factory,
                decode=decode,
                encode=encode,
            ))
        self.fields: tuple[_Field, ...] = tuple(fields_)
        self.known: frozenset[str] = frozenset(f.name for f in fields_)
        #: ``_extra`` に入れない名前 (既知 + transient)。
        self._not_extra: frozenset[str] = self.known | frozenset(self.transient)
        # decode の走査用に属性参照を前計算する (50k 件の性能 lock、c_05 §0.7.2)。
        # ``exact`` はそのまま受け取れる型 (素のスカラ)。一致すれば decoder を呼ばない
        # (型が違うときだけ decoder が変換するか CodecError を送出する)。
        exact: dict[str, type] = {}
        #: 文字列の ``{v: v}`` 表で読むフィールド → (表, 未知値を表へ足すか)。
        str_tables: dict[str, tuple[dict[str, str], bool]] = {}
        for f in fields_:
            tp = _strip_optional(hints[f.name])[0]
            if f.name in self.intern:
                str_tables[f.name] = (_INTERN, True)
            elif tp in _SCALARS:
                exact[f.name] = tp
            elif typing.get_origin(tp) is Literal and all(type(a) is str for a in typing.get_args(tp)):
                str_tables[f.name] = ({a: a for a in typing.get_args(tp)}, False)
        #: 名前 → (optional, decode, 表の名前, exact)。素の dict の階層の検査用。
        self._by_name: dict[str, tuple[bool, _Decode, str, type | None]] = {
            f.name: (f.optional, f.decode, f.name, exact.get(f.name)) for f in fields_
        }
        # 生成した関数でメソッドを覆う (呼出 1 段ぶんを省く。入れ子の decoder もこれを掴む)
        self.decode = self._compile_decoder(exact, str_tables)  # type: ignore[method-assign]

    def _compile_decoder(
        self, exact: dict[str, type], str_tables: dict[str, tuple[dict[str, str], bool]],
    ) -> Callable[[Any], T]:
        """decode を表から 1 本の関数に生成する (import 時に 1 回)。

        キーの手書き列挙ではなく表 (``dataclasses.fields()`` 由来) から機械生成する
        (dataclass の ``__init__`` と同じ作り方)。フィールドごとの反復と kwargs の
        dict を無くし、素のスカラの型検査と ``{v: v}`` の lookup は関数を呼ばずに
        行う (50k 件の性能 lock、c_05 §0.7.2)。型が違う値は各フィールドの decoder に
        渡し、decoder が変換するか :class:`CodecError` を送出する。
        """
        where = self.cls.__qualname__
        env: dict[str, Any] = {
            "_cls": self.cls, "_ABSENT": _ABSENT, "_CodecError": CodecError,
            "_known": self._not_extra, "_intern": intern_str,
        }
        lines = [
            "def decode(data):",
            "    if not isinstance(data, dict):",
            f"        raise _CodecError({where!r} + ': expected an object, got ' + type(data).__name__)",
            "    get = data.get",
            "    n = 0",
        ]
        args: list[str] = []
        for i, f in enumerate(self.fields):
            v = f"v{i}"
            env[f"_dec{i}"] = f.decode
            env[f"_exact{i}"] = exact.get(f.name)
            if f.default_factory is not None:
                env[f"_factory{i}"] = f.default_factory
                default = f"_factory{i}()"
            else:
                env[f"_const{i}"] = f.default
                default = f"_const{i}"
            missing = f"raise _CodecError({f'{where}: missing required key {f.name!r}'!r})"
            lines.append(f"    {v} = get({f.name!r}, _ABSENT)")
            lines.append(f"    if {v} is _ABSENT:")
            lines.append(f"        {missing}" if f.required else f"        {v} = {default}")
            lines.append("    else:")
            lines.append("        n += 1")
            lines.append(f"        if {v} is None:")
            if f.optional:
                lines.append("            pass  # 明示の null は None のまま")
            elif f.required:
                lines.append(f"            {missing}")
            else:
                lines.append(f"            {v} = {default}")
            if f.name in str_tables:
                table, grow = str_tables[f.name]
                env[f"_tbl{i}"] = table
                lines.append(f"        elif type({v}) is str:")
                lines.append(f"            x = _tbl{i}.get({v})")
                lines.append("            if x is not None:")
                lines.append(f"                {v} = x")
                if grow:
                    lines.append("            else:")
                    lines.append(f"                {v} = _intern({v})")
                lines.append("        else:")
            elif f.name in exact:
                lines.append(f"        elif type({v}) is not _exact{i}:")
            else:
                lines.append("        else:")
            lines.append(f"            {v} = _dec{i}({v})")
            args.append(f"{f.name}={v}")
        lines += [
            "    if len(data) > n:",
            "        extra = {k: x for k, x in data.items() if k not in _known} or None",
            "    else:",
            "        extra = None",
        ]
        for i, f in enumerate(self._transient_fields):
            if f.default_factory is not MISSING:
                env[f"_tfactory{i}"] = f.default_factory
                args.append(f"{f.name}=_tfactory{i}()")
            else:
                env[f"_tconst{i}"] = f.default
                args.append(f"{f.name}=_tconst{i}")
        has_non_init = any(not f.init for f in dataclasses.fields(self.cls))
        if self.cls.__dataclass_params__.frozen or has_non_init:  # type: ignore[attr-defined]
            lines.append(f"    return _cls({', '.join([*args, f'{EXTRA_FIELD}=extra'])})")
        else:
            # __init__ の引数解析を省いてスロットへ直接入れる。decode の値は型付き済み
            # なので、書き手の入力を整える __post_init__ は通さない。
            env["_new"] = object.__new__
            lines.append("    o = _new(_cls)")
            lines += [f"    o.{a.replace('=', ' = ', 1)}" for a in args]
            lines += [f"    o.{EXTRA_FIELD} = extra", "    return o"]
        namespace: dict[str, Any] = {}
        exec("\n".join(lines), env, namespace)  # noqa: S102 — 自前の表から生成したコードだけ
        return namespace["decode"]

    def decode(self, data: Any) -> T:
        """JSON オブジェクト → インスタンス。読めなければ :class:`CodecError`。

        実体は :meth:`_compile_decoder` が作ってインスタンス属性で覆う関数。
        """
        raise NotImplementedError  # pragma: no cover — __init__ で覆われる

    def check_mapping(self, data: dict[str, Any]) -> dict[str, Any]:
        """素の dict のまま持つ階層を表で検査する (値の型と intern。違反は :class:`CodecError`)。

        キーの集合は変えない — 既知キーの値を decode したもの (null はそのまま) と
        未知キーの原形を持つ新しい dict を返す。キーの実体は 1 つに寄せる (既知は表の
        名前、未知は :func:`intern_str`。読み込みごとに作られたキーの文字列を残さない)。
        """
        by_name = self._by_name
        out: dict[str, Any] = {}
        for key, value in data.items():
            entry = by_name.get(key)
            if entry is None:
                out[intern_str(key)] = value
            elif value is None:
                out[entry[2]] = value
            elif type(value) is entry[3]:
                out[entry[2]] = value
            else:
                out[entry[2]] = entry[1](value)
        return out

    def encode(self, obj: T) -> dict[str, Any]:
        """インスタンス → JSON オブジェクト (浅い dict)。``_extra`` はトップへ戻す。"""
        out: dict[str, Any] = {}
        omit = self.omit_defaults
        for f in self.fields:
            value = getattr(obj, f.name)
            if omit and not f.required and value == f.default_value():
                continue
            out[f.name] = value if f.encode is None or value is None else f.encode(value)
        extra = getattr(obj, EXTRA_FIELD)
        if extra:
            for k, v in extra.items():
                if k not in self._not_extra:
                    out[k] = v
        return out

    def decode_many(self, items: Any) -> tuple[list[T], int]:
        """JSON 配列の各要素を読む。読めない要素は飛ばし、``(読めた値, 飛ばした数)`` を返す。

        配列でなければ :class:`CodecError`。
        """
        if not isinstance(items, list):
            raise CodecError(f"{self.cls.__qualname__}: expected a list, got {type(items).__name__}")
        out: list[T] = []
        skipped = 0
        decode = self.decode
        for item in items:
            try:
                out.append(decode(item))
            except CodecError:
                skipped += 1
        return out, skipped


_CODECS: dict[type, RecordCodec[Any]] = {}
_OPTIONS: dict[type, tuple[bool, tuple[str, ...], tuple[str, ...]]] = {}


def persisted(
    *, omit_defaults: bool = False, intern: tuple[str, ...] = (), transient: tuple[str, ...] = (),
) -> Callable[[type[T]], type[T]]:
    """dataclass を永続レコードとして宣言し、表を import 時に作るデコレータ。

    入れ子の型は先に定義しておく (型注釈をこの時点で解決するため)。``intern`` は
    :func:`intern_str` で寄せる文字列フィールドの名前、``transient`` は永続化しない
    フィールドの名前。
    """

    def wrap(cls: type[T]) -> type[T]:
        _OPTIONS[cls] = (omit_defaults, tuple(intern), tuple(transient))
        _CODECS.pop(cls, None)
        codec_for(cls)
        return cls

    return wrap


def codec_for(cls: type[T]) -> RecordCodec[T]:
    """``cls`` の表 (初回だけ作る)。"""
    codec = _CODECS.get(cls)
    if codec is None:
        omit_defaults, intern, transient = _OPTIONS.get(cls, (False, (), ()))
        codec = RecordCodec(cls, omit_defaults=omit_defaults, intern=intern, transient=transient)
        _CODECS[cls] = codec
    return codec


def decode_skipping(cls: type[T], data: Any, *, each: tuple[str, ...]) -> tuple[T, int]:
    """``cls`` で読む。``each`` の名前のフィールド (型付きレコードのリスト) は要素ごとに読み、
    読めない要素だけを飛ばす。``(値, 飛ばした要素の数)`` を返す。

    ``each`` のフィールドが配列でない / それ以外が読めないときは :class:`CodecError`。
    """
    codec = codec_for(cls)
    if not isinstance(data, dict):
        raise CodecError(f"{cls.__qualname__}: expected an object, got {type(data).__name__}")
    head = dict(data)
    raw_lists = {name: head.pop(name, None) for name in each}
    obj = codec.decode(head)
    hints = typing.get_type_hints(cls)
    skipped = 0
    for name, raw in raw_lists.items():
        if raw is None:
            continue  # 欠損 / null は既定値 (decode が入れた)
        (item_cls,) = typing.get_args(_strip_optional(hints[name])[0])
        items, bad = codec_for(item_cls).decode_many(raw)
        setattr(obj, name, items)
        skipped += bad
    return obj, skipped


__all__ = [
    "EXTRA_FIELD",
    "INTERN_MAX",
    "CodecError",
    "RecordCodec",
    "codec_for",
    "decode_skipping",
    "intern_str",
    "persisted",
]
