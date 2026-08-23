"""A small, safe DSL for business rules / constraints.

Syntax (no eval, hand-written tokenizer + recursive-descent parser):

  rule      := if_expr
  if_expr   := "if" or_expr "then" or_expr ["else" or_expr] | or_expr
  or_expr   := and_expr ("or" and_expr)*
  and_expr  := not_expr ("and" not_expr)*
  not_expr  := "not" not_expr | comparison
  comparison:= add ((">"|">="|"<"|"<="|"=="|"!=") add)?
  add       := mul (("+"|"-") mul)*
  mul       := unary (("*"|"/"|"%") unary)*
  unary     := "-" unary | postfix
  postfix   := primary ("." name)*
  primary   := number | string | bool | null | ident | func_call | "(" or_expr ")"
  ident     := bare_ident | quoted_ident
  bare_ident:= [A-Za-z_][A-Za-z0-9_]*
  quoted_id := "`" (any char | "``") + "`"

Identifiers: ``col`` (current row) or ``alias.col`` (parent table via FK alias,
e.g. ``user.region``). Functions: abs, min, max, len, lower, upper, coalesce,
days_between, now.

A column whose name is not a bare identifier -- ``ZIP code``,
``Timely response?``, ``Sub-product`` -- is written in backticks:
``` `ZIP code` == '02139' ```. A literal backtick inside one is doubled, as in
SQL. Use :func:`quote_identifier` to produce the right form for a name rather
than deciding by hand.

Backticks rather than double quotes because ``"..."`` already means a string
literal here; redefining it would silently change what every existing spec
means.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

# Word-like tokens are matched by ONE alternative -- the identifier rule --
# and classified afterwards against the keyword/literal tables below. This is
# the standard maximal-munch-plus-keyword-table lexer design, and it is the
# design because the alternative does not work: listing `true|false|null` as
# their own regex alternatives ahead of the identifier rule makes the engine
# match a *prefix* of a longer word. `nullable == true` tokenized as
# NULL + IDENT("able"), so a rule about a column named `nullable` -- or
# `trueup`, `falsework`, `nulls_allowed` -- silently parsed as something else
# entirely. Alternation order cannot fix it; only matching the whole word
# first and then asking what it is can.
#
# QIDENT sits ahead of IDENT, which is safe for exactly the reason the literal
# alternatives were not: a backtick cannot begin a bare identifier, so the two
# alternatives cannot match overlapping prefixes of the same input. The
# keyword/literal classification below still applies to whole words only, and
# never to a quoted name -- `true` in backticks is a column called "true".
_TOKEN_RE = re.compile(
    r"""
    (?P<WS>\s+)
  | (?P<NUMBER>\d+\.\d+|\d+)
  | (?P<STRING>'[^']*'|"[^"]*")
  | (?P<QIDENT>`(?:[^`]|``)*`)
  | (?P<IDENT>[A-Za-z_][A-Za-z0-9_]*)
  | (?P<OP><=|>=|==|!=|<|>|\+|-|\*|/|%|\(|\)|\.|,)
    """,
    re.VERBOSE,
)

KEYWORDS = {"if", "then", "else", "and", "or", "not"}
BOOL_LITERALS = {"true", "false"}
NULL_LITERALS = {"null"}

_BARE_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_RESERVED = KEYWORDS | BOOL_LITERALS | NULL_LITERALS


def quote_identifier(name: str) -> str:
    """Render ``name`` as an identifier this DSL will read back as ``name``.

    Plain identifiers are returned unchanged, so generated rule text stays
    readable and existing specs keep the shape they have. Anything else --
    spaces, punctuation, a leading digit, the empty string, or a word the
    tokenizer would classify as a keyword or literal -- comes back in
    backticks with any embedded backtick doubled.

    Anything that builds rule text from a column name should go through this
    rather than deciding for itself; the profiler does.
    """
    if _BARE_IDENT_RE.match(name or "") and name not in _RESERVED:
        return name
    return "`" + (name or "").replace("`", "``") + "`"


@dataclass
class Token:
    kind: str
    value: str


def tokenize(s: str) -> List[Token]:
    tokens: List[Token] = []
    pos = 0
    while pos < len(s):
        m = _TOKEN_RE.match(s, pos)
        if not m:
            raise ValueError(f"Unexpected character at {pos}: {s[pos]!r}")
        pos = m.end()
        kind = m.lastgroup
        text = m.group()
        if kind == "WS":
            continue
        if kind == "OP":
            tokens.append(Token("OP", text))
        elif kind == "NUMBER":
            tokens.append(Token("NUMBER", text))
        elif kind == "STRING":
            tokens.append(Token("STRING", text[1:-1]))
        elif kind == "QIDENT":
            # A quoted name is a column name and nothing else: it is never
            # reclassified as a keyword or a literal, and never a function.
            tokens.append(Token("QIDENT", text[1:-1].replace("``", "`")))
        elif kind == "IDENT":
            # whole word matched; now decide what it is
            if text in KEYWORDS:
                tokens.append(Token("KW", text))
            elif text in BOOL_LITERALS:
                tokens.append(Token("BOOL", text))
            elif text in NULL_LITERALS:
                tokens.append(Token("NULL", text))
            else:
                tokens.append(Token("IDENT", text))
        else:
            raise ValueError(f"Bad token: {text}")
    return tokens


# ---------------------------------------------------------------------------
# AST
# ---------------------------------------------------------------------------

@dataclass
class Node:
    pass


@dataclass
class Literal(Node):
    value: Any


@dataclass
class Ident(Node):
    name: str
    # True when the name was written in backticks. Kept because a quoted name
    # is opaque: ```a.b``` is one column called "a.b", not column "b" of
    # parent alias "a", and :func:`_lookup` must not split it on the dot.
    quoted: bool = False


@dataclass
class Attr(Node):
    base: Node
    attr: str


@dataclass
class Call(Node):
    name: str
    args: List[Node] = field(default_factory=list)


@dataclass
class Unary(Node):
    op: str
    operand: Node


@dataclass
class BinOp(Node):
    op: str
    left: Node
    right: Node


@dataclass
class Compare(Node):
    op: str
    left: Node
    right: Node


@dataclass
class IfExpr(Node):
    cond: Node
    then: Node
    els: Optional[Node] = None


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class Parser:
    def __init__(self, tokens: List[Token]):
        self.toks = tokens
        self.i = 0

    def peek(self) -> Optional[Token]:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def next(self) -> Token:
        t = self.toks[self.i]
        self.i += 1
        return t

    def expect(self, kind: str, value: Optional[str] = None) -> Token:
        t = self.peek()
        if t is None or t.kind != kind or (value is not None and t.value != value):
            raise ValueError(f"Expected {kind} {value!r} but got {t}")
        return self.next()

    def at_keyword(self, word: str) -> bool:
        """Whether the next token is the keyword ``word``.

        The kind matters as much as the text. Before quoted identifiers
        existed the tokenizer guaranteed that a token spelled ``if`` was the
        keyword, so testing ``value`` alone was safe; now ```if``` is a
        perfectly good column name that arrives as QIDENT with the same text.
        Testing text alone would parse ```if` == 1`` as a malformed
        if-expression.
        """
        t = self.peek()
        return t is not None and t.kind == "KW" and t.value == word

    def parse(self) -> Node:
        node = self.parse_if()
        if self.i != len(self.toks):
            raise ValueError(f"Trailing tokens at {self.i}")
        return node

    def parse_if(self) -> Node:
        if self.at_keyword("if"):
            self.next()
            cond = self.parse_or()
            self.expect("KW", "then")
            then = self.parse_or()
            els = None
            if self.at_keyword("else"):
                self.next()
                els = self.parse_or()
            return IfExpr(cond, then, els)
        return self.parse_or()

    def parse_or(self) -> Node:
        left = self.parse_and()
        while self.at_keyword("or"):
            self.next()
            right = self.parse_and()
            left = BinOp("or", left, right)
        return left

    def parse_and(self) -> Node:
        left = self.parse_not()
        while self.at_keyword("and"):
            self.next()
            right = self.parse_not()
            left = BinOp("and", left, right)
        return left

    def parse_not(self) -> Node:
        if self.at_keyword("not"):
            self.next()
            return Unary("not", self.parse_not())
        return self.parse_comparison()

    def parse_comparison(self) -> Node:
        left = self.parse_add()
        t = self.peek()
        if t and t.kind == "OP" and t.value in ("<", "<=", ">", ">=", "==", "!="):
            self.next()
            right = self.parse_add()
            return Compare(t.value, left, right)
        return left

    def parse_add(self) -> Node:
        left = self.parse_mul()
        while self.peek() and self.peek().kind == "OP" and self.peek().value in ("+", "-"):  # type: ignore[union-attr]
            op = self.next().value
            right = self.parse_mul()
            left = BinOp(op, left, right)
        return left

    def parse_mul(self) -> Node:
        left = self.parse_unary()
        while self.peek() and self.peek().kind == "OP" and self.peek().value in ("*", "/", "%"):  # type: ignore[union-attr]
            op = self.next().value
            right = self.parse_unary()
            left = BinOp(op, left, right)
        return left

    def parse_unary(self) -> Node:
        t = self.peek()
        if t and t.kind == "OP" and t.value in ("-", "+"):
            op = self.next().value
            if op == "-":
                return Unary("-", self.parse_unary())
            return self.parse_unary()
        return self.parse_postfix()

    def parse_postfix(self) -> Node:
        node = self.parse_primary()
        while self.peek() and self.peek().kind == "OP" and self.peek().value == ".":  # type: ignore[union-attr]
            self.next()
            t = self.peek()
            if t is None or t.kind not in ("IDENT", "QIDENT"):
                raise ValueError(f"Expected an identifier after '.' but got {t}")
            node = Attr(node, self.next().value)
        return node

    def parse_primary(self) -> Node:
        t = self.peek()
        if t is None:
            raise ValueError("Unexpected end of rule")
        if t.kind == "NUMBER":
            self.next()
            val = float(t.value) if "." in t.value else int(t.value)
            return Literal(val)
        if t.kind == "STRING":
            self.next()
            return Literal(t.value)
        if t.kind == "BOOL":
            self.next()
            return Literal(t.value == "true")
        if t.kind == "NULL":
            self.next()
            return Literal(None)
        if t.kind == "OP" and t.value == "(":
            self.next()
            node = self.parse_or()
            self.expect("OP", ")")
            return node
        if t.kind == "QIDENT":
            # Deliberately never a function call: quoting says "this is the
            # name of a column", so ```abs`(1)`` is a syntax error rather
            # than a call to abs().
            self.next()
            return Ident(t.value, quoted=True)
        if t.kind == "IDENT":
            self.next()
            if self.peek() and self.peek().kind == "OP" and self.peek().value == "(":
                self.next()
                args: List[Node] = []
                if not (self.peek() and self.peek().kind == "OP" and self.peek().value == ")"):
                    args.append(self.parse_or())
                    while self.peek() and self.peek().kind == "OP" and self.peek().value == ",":
                        self.next()
                        args.append(self.parse_or())
                self.expect("OP", ")")
                return Call(t.value, args)
            return Ident(t.value)
        raise ValueError(f"Unexpected token: {t.value}")


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

_FUNCS = {
    "abs": abs,
    "min": min,
    "max": max,
    "len": len,
    "lower": lambda x: str(x).lower(),
    "upper": lambda x: str(x).upper(),
    "coalesce": lambda *a: next((v for v in a if v is not None), None),
    "days_between": lambda a, b: _to_dt(a).toordinal() - _to_dt(b).toordinal(),
    "now": lambda: datetime.now(),
}


def _to_dt(v: Any) -> Any:
    if isinstance(v, datetime):
        return v
    if hasattr(v, "toordinal"):
        return v
    if isinstance(v, str):
        return datetime.fromisoformat(v)
    return v


def _truthy(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, (int, float)):
        return v != 0
    if isinstance(v, (str, list, dict, tuple)):
        return len(v) > 0
    return bool(v)


def _lookup(name: str, ctx: Dict[str, Any], parents: Dict[str, Dict[str, Any]],
            quoted: bool = False) -> Any:
    if "." in name and not quoted:
        head, _, rest = name.partition(".")
        if head in parents:
            obj = parents[head]
            for part in rest.split("."):
                if obj is None:
                    return None
                obj = obj.get(part)
            return obj
    if name in ctx:
        return ctx[name]
    if name in parents:
        return parents[name]
    raise ValueError(f"Unknown identifier: {name}")


def _eval_node(node: Node, ctx: Dict[str, Any], parents: Dict[str, Dict[str, Any]]) -> Any:
    if isinstance(node, Literal):
        return node.value
    if isinstance(node, Ident):
        return _lookup(node.name, ctx, parents, node.quoted)
    if isinstance(node, Attr):
        base = _eval_node(node.base, ctx, parents)
        if base is None:
            return None
        return base.get(node.attr)
    if isinstance(node, Call):
        fn = _FUNCS.get(node.name)
        if fn is None:
            raise ValueError(f"Unknown function: {node.name}")
        args = [_eval_node(a, ctx, parents) for a in node.args]
        return fn(*args)
    if isinstance(node, Unary):
        v = _eval_node(node.operand, ctx, parents)
        if node.op == "-":
            return -v
        if node.op == "not":
            return not _truthy(v)
    if isinstance(node, BinOp):
        l = _eval_node(node.left, ctx, parents)
        if node.op == "and":
            return _truthy(l) and _truthy(_eval_node(node.right, ctx, parents))
        if node.op == "or":
            return _truthy(l) or _truthy(_eval_node(node.right, ctx, parents))
        r = _eval_node(node.right, ctx, parents)
        if node.op == "+":
            return l + r
        if node.op == "-":
            return l - r
        if node.op == "*":
            return l * r
        if node.op == "/":
            return l / r
        if node.op == "%":
            return l % r
    if isinstance(node, Compare):
        l = _eval_node(node.left, ctx, parents)
        r = _eval_node(node.right, ctx, parents)
        return _apply_compare(node.op, l, r)
    if isinstance(node, IfExpr):
        if _truthy(_eval_node(node.cond, ctx, parents)):
            return _eval_node(node.then, ctx, parents)
        if node.els is not None:
            return _eval_node(node.els, ctx, parents)
        return None
    raise ValueError(f"Cannot evaluate node: {node}")


def _apply_compare(op: str, l: Any, r: Any) -> bool:
    if op == "==":
        return l == r
    if op == "!=":
        return l != r
    if l is None or r is None:
        return False
    if op == ">":
        return l > r
    if op == ">=":
        return l >= r
    if op == "<":
        return l < r
    if op == "<=":
        return l <= r
    raise ValueError(f"Unknown compare op: {op}")


class Rule:
    """A compiled business rule."""

    def __init__(self, expr: str):
        self.expr = expr
        self.ast = Parser(tokenize(expr)).parse()

    def check(self, ctx: Dict[str, Any], parents: Dict[str, Dict[str, Any]]) -> bool:
        return _truthy(_eval_node(self.ast, ctx, parents))

    def parent_aliases(self) -> List[str]:
        aliases: set[str] = set()

        def walk(n: Node):
            if isinstance(n, Attr) and isinstance(n.base, Ident):
                aliases.add(n.base.name)
            for child in vars(n).values():
                if isinstance(child, Node):
                    walk(child)
                elif isinstance(child, list):
                    for c in child:
                        if isinstance(c, Node):
                            walk(c)

        walk(self.ast)
        return list(aliases)

    def column_refs(self) -> List[str]:
        """Return current-row column identifiers referenced by this rule.

        The base identifier in ``parent.col`` is deliberately excluded because
        it is a relationship alias, not a column in the current row.
        """
        refs: set[str] = set()

        def walk(n: Node, attr_base: bool = False):
            if isinstance(n, Ident):
                if not attr_base:
                    refs.add(n.name)
                return
            if isinstance(n, Attr):
                walk(n.base, attr_base=True)
                return
            for child in vars(n).values():
                if isinstance(child, Node):
                    walk(child)
                elif isinstance(child, list):
                    for item in child:
                        if isinstance(item, Node):
                            walk(item)

        walk(self.ast)
        return sorted(refs)


def compile_rules(exprs: List[str]) -> List[Rule]:
    return [Rule(e) for e in exprs]
