#!/usr/bin/env python3

from __future__ import annotations
import sys
from dataclasses import dataclass, field
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Any


# -----------------------------
# Grammar representation
# -----------------------------

@dataclass(frozen=True)
class Rule:
    lhs: str
    rhs: Tuple[str, ...]
    prob: float


class Grammar:
    def __init__(self) -> None:
        self.rules_by_lhs: Dict[str, List[Rule]] = defaultdict(list)
        self.nonterminals: set[str] = set()

    @staticmethod
    def from_file(path: str) -> "Grammar":
        grammar = Grammar()

        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 4 or "->" not in parts:
                    raise ValueError(f"Invalid grammar line {line_no}: {line}")

                try:
                    prob = float(parts[0])
                except ValueError as e:
                    raise ValueError(f"Invalid probability on line {line_no}: {line}") from e

                lhs = parts[1]
                if parts[2] != "->":
                    raise ValueError(f"Expected '->' on line {line_no}: {line}")

                rhs = tuple(parts[3:])
                if not rhs:
                    raise ValueError(f"Epsilon rules are not allowed on line {line_no}: {line}")

                rule = Rule(lhs, rhs, prob)
                grammar.rules_by_lhs[lhs].append(rule)
                grammar.nonterminals.add(lhs)

        return grammar

    def is_nonterminal(self, symbol: str) -> bool:
        return symbol in self.nonterminals

    def start_symbol(self) -> str:
        if "S" in self.rules_by_lhs:
            return "S"
        return next(iter(self.rules_by_lhs))


# -----------------------------
# Packed derivations
# -----------------------------

@dataclass(frozen=True)
class Derivation:
    parts: Tuple[Any, ...]
    score: float


@dataclass
class PackedState:
    rule: Rule
    dot: int
    start: int
    end: int
    derivations: List[Derivation] = field(default_factory=list)
    _seen_derivations: set = field(default_factory=set)

    def key(self) -> Tuple[str, Tuple[str, ...], int, int, int]:
        return (self.rule.lhs, self.rule.rhs, self.dot, self.start, self.end)

    def is_complete(self) -> bool:
        return self.dot == len(self.rule.rhs)

    def next_symbol(self) -> Optional[str]:
        if self.is_complete():
            return None
        return self.rule.rhs[self.dot]

    def add_derivation(self, derivation: Derivation) -> bool:
        signature = (derivation.parts, round(derivation.score, 15))
        if signature in self._seen_derivations:
            return False
        self._seen_derivations.add(signature)
        self.derivations.append(derivation)
        return True


# -----------------------------
# Earley parser
# -----------------------------

class EarleyParser:
    def __init__(self, grammar: Grammar) -> None:
        self.grammar = grammar
        self.aug_start = "ROOT"

    def parse(
        self, words: List[str]
    ) -> Tuple[List[Dict[Tuple, PackedState]], List[Tuple[PackedState, int, float]]]:
        n = len(words)
        chart: List[Dict[Tuple, PackedState]] = [dict() for _ in range(n + 1)]
        agendas: List[deque[PackedState]] = [deque() for _ in range(n + 1)]

        start_rule = Rule(self.aug_start, (self.grammar.start_symbol(),), 1.0)
        start_state = self._get_or_create_state(chart[0], start_rule, 0, 0, 0)
        if start_state.add_derivation(Derivation(parts=(), score=1.0)):
            agendas[0].append(start_state)

        for j in range(n + 1):
            while agendas[j]:
                state = agendas[j].popleft()

                if state.is_complete():
                    self._complete(state, chart, agendas)
                else:
                    sym = state.next_symbol()
                    if sym is None:
                        continue
                    if self.grammar.is_nonterminal(sym):
                        self._predict(sym, j, chart[j], agendas[j])
                    else:
                        self._scan(state, words, chart, agendas)

        final_parses: List[Tuple[PackedState, int, float]] = []
        for st in chart[n].values():
            if (
                st.rule.lhs == self.aug_start
                and st.is_complete()
                and st.start == 0
                and st.end == n
            ):
                for idx, derivation in enumerate(st.derivations):
                    final_parses.append((st, idx, derivation.score))

        final_parses.sort(key=lambda x: x[2], reverse=True)
        return chart, final_parses

    def _get_or_create_state(
        self,
        column: Dict[Tuple, PackedState],
        rule: Rule,
        dot: int,
        start: int,
        end: int,
    ) -> PackedState:
        key = (rule.lhs, rule.rhs, dot, start, end)
        if key not in column:
            column[key] = PackedState(rule=rule, dot=dot, start=start, end=end)
        return column[key]

    def _add_derivation_to_state(
        self,
        state: PackedState,
        derivation: Derivation,
        agenda: deque[PackedState],
    ) -> None:
        if state.add_derivation(derivation):
            agenda.append(state)

    def _predict(
        self,
        nonterminal: str,
        j: int,
        column: Dict[Tuple, PackedState],
        agenda: deque[PackedState],
    ) -> None:
        for rule in self.grammar.rules_by_lhs.get(nonterminal, []):
            predicted = self._get_or_create_state(column, rule, 0, j, j)
            self._add_derivation_to_state(
                predicted,
                Derivation(parts=(), score=rule.prob),
                agenda,
            )

    def _scan(
        self,
        state: PackedState,
        words: List[str],
        chart: List[Dict[Tuple, PackedState]],
        agendas: List[deque[PackedState]],
    ) -> None:
        if state.end >= len(words):
            return

        expected = state.next_symbol()
        word = words[state.end]
        if expected != word:
            return

        advanced = self._get_or_create_state(
            chart[state.end + 1],
            state.rule,
            state.dot + 1,
            state.start,
            state.end + 1,
        )

        for derivation in state.derivations:
            new_derivation = Derivation(
                parts=derivation.parts + (word,),
                score=derivation.score,
            )
            self._add_derivation_to_state(
                advanced,
                new_derivation,
                agendas[state.end + 1],
            )

    def _complete(
        self,
        completed: PackedState,
        chart: List[Dict[Tuple, PackedState]],
        agendas: List[deque[PackedState]],
    ) -> None:
        origin_col = completed.start
        current_col = completed.end
        completed_lhs = completed.rule.lhs

        for waiting in list(chart[origin_col].values()):
            if waiting.next_symbol() != completed_lhs:
                continue

            advanced = self._get_or_create_state(
                chart[current_col],
                waiting.rule,
                waiting.dot + 1,
                waiting.start,
                current_col,
            )

            completed_key = completed.key()

            for waiting_der in waiting.derivations:
                for child_idx, child_der in enumerate(completed.derivations):
                    child_ref = ("STATE", completed_key, child_idx)
                    new_derivation = Derivation(
                        parts=waiting_der.parts + (child_ref,),
                        score=waiting_der.score * child_der.score,
                    )
                    self._add_derivation_to_state(
                        advanced,
                        new_derivation,
                        agendas[current_col],
                    )


# -----------------------------
# Tree reconstruction
# -----------------------------

def build_tree_with_spans(
    chart: List[Dict[Tuple, PackedState]],
    state: PackedState,
    derivation_index: int,
) -> Any:
    derivation = state.derivations[derivation_index]
    children = []

    for part in derivation.parts:
        if isinstance(part, tuple) and len(part) == 3 and part[0] == "STATE":
            _, child_key, child_der_index = part
            child_col = child_key[4]
            child_state = chart[child_col][child_key]
            children.append(build_tree_with_spans(chart, child_state, child_der_index))
        else:
            children.append(part)

    return {
        "label": state.rule.lhs,
        "start": state.start,
        "end": state.end,
        "children": children,
    }


def strip_spans(node: Any) -> Any:
    if isinstance(node, str):
        return node
    return {
        "label": node["label"],
        "children": [strip_spans(child) for child in node["children"]],
    }


# -----------------------------
# Tree formatting
# -----------------------------

def format_tree_no_spans(node: Any) -> str:
    if isinstance(node, str):
        return node

    label = node["label"]
    children = node["children"]

    if not children:
        return f"({label})"

    child_text = " ".join(format_tree_no_spans(child) for child in children)
    return f"({label} {child_text})"


def format_tree_with_spans(node: Any) -> str:
    if isinstance(node, str):
        return node

    label = node["label"]
    start = node["start"]
    end = node["end"]
    children = node["children"]

    if not children:
        return f"({label} [{start},{end}])"

    child_text = " ".join(format_tree_with_spans(child) for child in children)
    return f"({label} [{start},{end}] {child_text})"


# -----------------------------
# Sentence parsing
# -----------------------------

def parse_sentences(grammar_path: str, sentence_path: str) -> None:
    grammar = Grammar.from_file(grammar_path)
    parser = EarleyParser(grammar)

    with open(sentence_path, "r", encoding="utf-8") as f:
        for line in f:
            sent = line.strip()
            if not sent:
                continue

            words = sent.split()
            chart, final_parses = parser.parse(words)

            if not final_parses:
                print("NO PARSE")
                print("NO PARSE")
                continue

            best_state, best_idx, _ = final_parses[0]
            tree_with_spans = build_tree_with_spans(chart, best_state, best_idx)
            tree_no_spans = strip_spans(tree_with_spans)

            print(format_tree_no_spans(tree_no_spans))
            print(format_tree_with_spans(tree_with_spans))


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python parse.py foo.gr foo.sen", file=sys.stderr)
        sys.exit(1)

    grammar_path = sys.argv[1]
    sentence_path = sys.argv[2]
    parse_sentences(grammar_path, sentence_path)


if __name__ == "__main__":
    main()