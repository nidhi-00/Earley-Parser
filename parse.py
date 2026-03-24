#!/usr/bin/env python3

from __future__ import annotations
import sys
from dataclasses import dataclass, field
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Iterable, Any


# -----------------------------
# Grammar representation
# -----------------------------

@dataclass(frozen=True)
class Rule:
    lhs: str
    rhs: Tuple[str, ...]
    prob: float

    def __str__(self) -> str:
        rhs_str = " ".join(self.rhs)
        return f"{self.prob} {self.lhs} -> {rhs_str}"


class Grammar:
    def __init__(self) -> None:
        self.rules_by_lhs: Dict[str, List[Rule]] = defaultdict(list)
        self.nonterminals: set[str] = set()

    @staticmethod
    def from_file(path: str) -> "Grammar":
        grammar = Grammar()
        raw_rules: List[Tuple[float, str, Tuple[str, ...]]] = []

        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                # Expected rigid format:
                #   0.4 NP -> N
                #   1.0 S -> NP VP
                parts = line.split()
                if len(parts) < 4 or "->" not in parts:
                    raise ValueError(
                        f"Invalid grammar line {line_no}: {line}"
                    )

                try:
                    prob = float(parts[0])
                except ValueError as e:
                    raise ValueError(
                        f"Invalid probability on line {line_no}: {line}"
                    ) from e

                lhs = parts[1]
                if parts[2] != "->":
                    raise ValueError(
                        f"Expected '->' on line {line_no}: {line}"
                    )

                rhs = tuple(parts[3:])
                if len(rhs) == 0:
                    raise ValueError(
                        f"Epsilon rules are not allowed on line {line_no}: {line}"
                    )

                raw_rules.append((prob, lhs, rhs))
                grammar.nonterminals.add(lhs)

        for prob, lhs, rhs in raw_rules:
            rule = Rule(lhs, rhs, prob)
            grammar.rules_by_lhs[lhs].append(rule)

        return grammar

    def is_nonterminal(self, symbol: str) -> bool:
        return symbol in self.nonterminals

    def start_symbol(self) -> str:
        if "S" in self.rules_by_lhs:
            return "S"
        # Fallback: first defined LHS
        return next(iter(self.rules_by_lhs))


# -----------------------------
# Earley state
# -----------------------------

@dataclass
class State:
    rule: Rule
    dot: int
    start: int
    end: int
    score: float
    backpointers: Tuple[Any, ...] = field(default_factory=tuple)

    def key(self) -> Tuple[str, Tuple[str, ...], int, int, int]:
        return (self.rule.lhs, self.rule.rhs, self.dot, self.start, self.end)

    def is_complete(self) -> bool:
        return self.dot == len(self.rule.rhs)

    def next_symbol(self) -> Optional[str]:
        if self.is_complete():
            return None
        return self.rule.rhs[self.dot]

    def advance_with_terminal(self, word: str) -> "State":
        return State(
            rule=self.rule,
            dot=self.dot + 1,
            start=self.start,
            end=self.end + 1,
            score=self.score,
            backpointers=self.backpointers + (word,),
        )

    def advance_with_completed_child(self, child: "State") -> "State":
        return State(
            rule=self.rule,
            dot=self.dot + 1,
            start=self.start,
            end=child.end,
            score=self.score * child.score,
            backpointers=self.backpointers + (child,),
        )

    def pretty(self) -> str:
        rhs = list(self.rule.rhs)
        rhs.insert(self.dot, "·")
        rhs_str = " ".join(rhs)
        return (
            f"[{self.rule.lhs} -> {rhs_str}, {self.start}, {self.end}, "
            f"score={self.score:.10g}]"
        )


# -----------------------------
# Earley parser with Viterbi updates
# -----------------------------

class EarleyParser:
    def __init__(self, grammar: Grammar) -> None:
        self.grammar = grammar
        self.aug_start = "γ"

    def parse(self, words: List[str]) -> Tuple[List[Dict[Tuple, State]], List[State]]:
        n = len(words)
        chart: List[Dict[Tuple, State]] = [dict() for _ in range(n + 1)]
        agendas: List[deque[State]] = [deque() for _ in range(n + 1)]

        start_rule = Rule(self.aug_start, (self.grammar.start_symbol(),), 1.0)
        start_state = State(
            rule=start_rule,
            dot=0,
            start=0,
            end=0,
            score=1.0,
            backpointers=(),
        )
        self._add_to_chart(start_state, chart[0], agendas[0])

        for j in range(n + 1):
            while agendas[j]:
                state = agendas[j].popleft()

                # Skip stale state if a better version replaced it
                current = chart[j].get(state.key())
                if current is not state:
                    continue

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

        final_states: List[State] = []
        for st in chart[n].values():
            if (
                st.rule.lhs == self.aug_start
                and st.is_complete()
                and st.start == 0
                and st.end == n
            ):
                final_states.append(st)

        final_states.sort(key=lambda s: s.score, reverse=True)
        return chart, final_states

    def _add_to_chart(
        self,
        new_state: State,
        column: Dict[Tuple, State],
        agenda: deque[State],
    ) -> None:
        """
        Viterbi collision rule:
        If exact same state exists, keep the one with larger score.
        """
        k = new_state.key()
        old_state = column.get(k)

        if old_state is None or new_state.score > old_state.score:
            column[k] = new_state
            agenda.append(new_state)

    def _predict(
        self,
        nonterminal: str,
        j: int,
        column: Dict[Tuple, State],
        agenda: deque[State],
    ) -> None:
        for rule in self.grammar.rules_by_lhs.get(nonterminal, []):
            predicted = State(
                rule=rule,
                dot=0,
                start=j,
                end=j,
                score=rule.prob,
                backpointers=(),
            )
            self._add_to_chart(predicted, column, agenda)

    def _scan(
        self,
        state: State,
        words: List[str],
        chart: List[Dict[Tuple, State]],
        agendas: List[deque[State]],
    ) -> None:
        if state.end >= len(words):
            return

        expected = state.next_symbol()
        if expected == words[state.end]:
            advanced = state.advance_with_terminal(words[state.end])
            self._add_to_chart(advanced, chart[state.end + 1], agendas[state.end + 1])

    def _complete(
        self,
        completed: State,
        chart: List[Dict[Tuple, State]],
        agendas: List[deque[State]],
    ) -> None:
        origin_col = completed.start
        current_col = completed.end
        completed_lhs = completed.rule.lhs

        for waiting in list(chart[origin_col].values()):
            sym = waiting.next_symbol()
            if sym == completed_lhs:
                advanced = waiting.advance_with_completed_child(completed)
                self._add_to_chart(advanced, chart[current_col], agendas[current_col])


# -----------------------------
# Tree reconstruction
# -----------------------------

def build_tree(state: State) -> Any:
    """
    Convert a completed state into a nested tuple tree:
    ('S', subtree1, subtree2, ...)
    Terminals remain as strings.
    """
    if not state.is_complete():
        raise ValueError("Can only build a tree from a completed state.")

    children = []
    for bp in state.backpointers:
        if isinstance(bp, State):
            children.append(build_tree(bp))
        else:
            children.append(bp)

    return tuple([state.rule.lhs] + children)


def tree_to_string(tree: Any, indent: int = 0) -> str:
    space = "  " * indent
    if not isinstance(tree, tuple):
        return f"{space}{tree}"

    label = tree[0]
    if len(tree) == 1:
        return f"{space}({label})"

    lines = [f"{space}({label}"]
    for child in tree[1:]:
        lines.append(tree_to_string(child, indent + 1))
    lines[-1] += ")"
    return "\n".join(lines)


# -----------------------------
# Utility printing
# -----------------------------

def print_chart(chart: List[Dict[Tuple, State]]) -> None:
    for idx, column in enumerate(chart):
        print(f"Chart[{idx}]")
        states = list(column.values())
        states.sort(key=lambda s: (s.start, s.end, s.rule.lhs, s.rule.rhs, s.dot))
        for st in states:
            print(st.pretty())
        print()


def parse_sentences(grammar_path: str, sentence_path: str) -> None:
    grammar = Grammar.from_file(grammar_path)
    parser = EarleyParser(grammar)

    with open(sentence_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            sent = line.strip()
            if not sent:
                continue

            words = sent.split()
            chart, final_states = parser.parse(words)

            print("=" * 80)
            print(f"Sentence {line_no}: {sent}")
            print("=" * 80)
            print_chart(chart)

            if not final_states:
                print("No parse found.\n")
                continue

            best = final_states[0]
            best_tree = build_tree(best)

            print("Best parse probability:", f"{best.score:.10g}")
            print("Best parse tree:")
            print(tree_to_string(best_tree))
            print()

            if len(final_states) > 1:
                print("Other complete parses:")
                for rank, st in enumerate(final_states[1:], start=2):
                    print(f"Parse {rank} probability: {st.score:.10g}")
                    print(tree_to_string(build_tree(st)))
                    print()


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: ./parse.py foo.gr foo.sen", file=sys.stderr)
        sys.exit(1)

    grammar_path = sys.argv[1]
    sentence_path = sys.argv[2]
    parse_sentences(grammar_path, sentence_path)


if __name__ == "__main__":
    main()