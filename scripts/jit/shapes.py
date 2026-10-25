#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.13"
# ///
"""Time bytecode shapes on the evm2 interpreter and JIT.

Each suite pair runs variants of one shape inside the same counted loop, keeps
the result live by returning an accumulator, and reports the time per loop
iteration on both engines, the gas, and whether the JIT emitted the same
assembly. Arbitrary runtime bytecode (`.hex`, `.bin`, or `.evm` assembly) can
be timed the same way with `--code`. With `--bundle`, it replays deploy-and-call
bundles through `evm2 bundle` and reports per-case totals for each variant.

Examples:
    ./scripts/jit/shapes.py --suite m1,m3
    ./scripts/jit/shapes.py --list
    ./scripts/jit/shapes.py --code a.hex b.hex --calldata 0xa9059cbb...
    ./scripts/jit/shapes.py --bundle base-bundle.json literal-bundle.json
"""

import argparse
import difflib
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field

from utils import cargo_build, parse_duration, perf_counts, repo_root, total_iters

GAS_BUDGET = 14_000_000
MIX = "DUP2 XOR PUSH1 7 ADD"


@dataclass
class Variant:
    name: str
    rep: str
    setup: str = ""
    tail: str = ""


@dataclass
class Pair:
    name: str
    variants: list[Variant]
    reps: int = 8
    spec: str = "osaka"
    note: str = ""


def loop_asm(v: Variant, reps: int, iters: int) -> str:
    """Wraps `reps` copies of a shape in a counted loop.

    The body sees `[.., cp, i, acc]` with `acc` on top, where `cp` is calldata
    word 0 (zero, but unknown to the JIT) and `i` is the loop counter.
    """
    body = "\n".join(
        v.rep.replace("{k}", str(k))
        .replace("{mk}", str(32 * k + 0x80))
        .replace("{ck}", rand_word(k))
        for k in range(reps)
    )
    tail = (
        "\n".join(v.tail.replace("{k}", str(k)) for k in range(reps)) if "{k}" in v.tail else v.tail
    )
    return f"""{v.setup}
PUSH0 CALLDATALOAD
PUSH0
PUSH {iters}
loop: JUMPDEST
SWAP1
{body}
SWAP1
PUSH1 1 SWAP1 SUB
DUP1 PUSH2 %loop JUMPI
POP
PUSH0 MSTORE PUSH1 32 PUSH0 RETURN
{tail}
"""


def rand_word(k: int) -> str:
    """Returns a fixed odd 256-bit constant for repetition `k`."""
    return "0x" + hashlib.sha256(f"shapes{k}".encode()).hexdigest()[:-1] + "1"


def ff(n: int) -> str:
    return "ff" * n


ADDR = "d8da6bf26964af9d7eed9e03e53415d37aa96045"
FILL18 = "CALLVALUE " * 18
COND = {
    "pred": "DUP3 ISZERO",
    "alt": "DUP2 PUSH1 1 AND",
    # Top bit of i^2 * C_k: equidistributed, and unlike a linear sequence not learnable by branch
    # predictors; each repetition uses its own constant.
    "rand": "DUP2 DUP1 MUL PUSH32 {ck} MUL PUSH1 255 SHR",
}
BRANCHLESS = "PUSH1 7 DUP4 SUB MUL PUSH1 7 ADD ADD DUP2 XOR"
BRANCHY = (
    "PUSH2 %t{k} JUMPI PUSH1 7 PUSH2 %e{k} JUMP t{k}: JUMPDEST DUP2 e{k}: JUMPDEST ADD DUP2 XOR"
)
FUNC = "f: JUMPDEST SWAP1 DUP3 XOR PUSH1 7 ADD SWAP1 JUMP"
CALL = "PUSH2 %r{k} PUSH2 %f JUMP r{k}: JUMPDEST"
# A jump into a table of two 5-byte entries, indexed by calldata word 0 after a bounds check.
BOUNDED_JUMP = (
    "PUSH0 CALLDATALOAD PUSH1 2 DUP2 LT PUSH2 %ok JUMPI PUSH0 DUP1 REVERT "
    "ok: JUMPDEST PUSH1 5 MUL PUSH2 %tbl ADD JUMP "
    "tbl: JUMPDEST PUSH2 %s JUMP JUMPDEST PUSH2 %s JUMP s: JUMPDEST"
)


def const_pair(name: str, a: str, b: str, note: str = "") -> Pair:
    return Pair(
        name,
        [Variant("full", f"{a} ADD DUP2 XOR"), Variant("compact", f"{b} ADD DUP2 XOR")],
        note=note,
    )


def suites() -> dict[str, list[Pair]]:
    m1 = [
        const_pair("m1.ones", f"PUSH32 0x{ff(32)}", "PUSH0 NOT"),
        const_pair("m1.hi_mask", f"PUSH32 0x{ff(16)}{'00' * 16}", "PUSH0 NOT PUSH1 0x80 SHL"),
        const_pair("m1.addr_shl", f"PUSH32 0x{ADDR}{'00' * 12}", f"PUSH20 0x{ADDR} PUSH1 0x60 SHL"),
        const_pair("m1.not_one", f"PUSH32 0x{ff(31)}fe", "PUSH1 1 NOT"),
        Pair(
            "m1.push_width",
            [
                Variant("push1", "PUSH1 0x2a ADD DUP2 XOR"),
                Variant("push32", "PUSH32 0x2a ADD DUP2 XOR"),
            ],
        ),
        Pair(
            "m1.push0",
            [Variant("push0", f"PUSH0 ADD {MIX}"), Variant("push1_0", f"PUSH1 0 ADD {MIX}")],
        ),
    ]
    m2 = [
        Pair("m2.jumpdest", [Variant("base", MIX), Variant("jumpdest", f"JUMPDEST {MIX}")]),
        Pair("m2.dup1_pop", [Variant("base", MIX), Variant("dup1_pop", f"DUP1 POP {MIX}")]),
        Pair(
            "m2.swap_depth",
            [Variant("base", MIX, FILL18)]
            + [
                Variant(n, f"{s} {s} {MIX}", FILL18)
                for n, s in [
                    ("swap1x2", "SWAP1"),
                    ("swap16x2", "SWAP16"),
                    ("swapn17x2", "SWAPN 17"),
                    ("exchange_1_2x2", "EXCHANGE 1 2"),
                    ("exchange_1_16x2", "EXCHANGE 1 16"),
                ]
            ],
            spec="amsterdam",
        ),
        Pair(
            "m2.dup_depth",
            [Variant("base", MIX, FILL18)]
            + [
                Variant(n, f"{s} POP {MIX}", FILL18)
                for n, s in [("dup1", "DUP1"), ("dup16", "DUP16"), ("dupn17", "DUPN 17")]
            ],
            spec="amsterdam",
        ),
    ]
    m3 = [
        Pair(
            "m3.block_split",
            [
                Variant("straight", MIX),
                Variant("jump_split", f"{MIX} PUSH2 %l{{k}} JUMP l{{k}}: JUMPDEST"),
                Variant("jumpdest_split", f"{MIX} l{{k}}: JUMPDEST"),
            ],
        )
    ]
    m4 = []
    for cond, c in COND.items():
        m4.append(
            Pair(
                f"m4.select_{cond}",
                [
                    Variant("branchless", f"{c} {BRANCHLESS}"),
                    Variant("branchless_norm", f"{c} ISZERO ISZERO {BRANCHLESS}"),
                    Variant("branchy", f"{c} {BRANCHY}"),
                ],
            )
        )
    table_setup = " ".join(f"PUSH2 %l{k} PUSH2 {32 * k + 0x80} MSTORE" for k in range(8))
    m5 = [
        Pair(
            "m5.jump_kind",
            [
                Variant("static", f"DUP3 POP PUSH2 %l{{k}} JUMP l{{k}}: JUMPDEST {MIX}"),
                Variant("computed_add", f"DUP3 PUSH2 %l{{k}} ADD JUMP l{{k}}: JUMPDEST {MIX}"),
                Variant(
                    "mload_table", f"PUSH2 {{mk}} MLOAD JUMP l{{k}}: JUMPDEST {MIX}", table_setup
                ),
                Variant(
                    "codecopy_table",
                    f"PUSH1 3 PUSH2 %d{{k}} PUSH1 29 CODECOPY PUSH0 MLOAD PUSH2 0xffff AND JUMP l{{k}}: JUMPDEST {MIX}",
                    tail="d{k}: PUSH2 %l{k}",
                ),
            ],
        ),
        Pair(
            "m5.contagion_calls",
            [
                Variant("no_dyn", CALL, tail=FUNC),
                Variant(
                    "one_dyn_outside_loop",
                    CALL,
                    "PUSH0 CALLDATALOAD PUSH2 %s ADD JUMP s: JUMPDEST",
                    FUNC,
                ),
                Variant("one_bounded_dyn_outside_loop", CALL, BOUNDED_JUMP, FUNC),
            ],
            reps=3,
            note="3 call sites of f; variant 2 adds one unresolvable jump executed once before the loop, "
            "variant 3 one into a jump table behind a bounds check",
        ),
        Pair(
            "m5.contagion_consts",
            [
                Variant("no_dyn", "PUSH0 NOT PUSH1 0x80 SHL ADD DUP2 XOR"),
                Variant(
                    "one_dyn_outside_loop",
                    "PUSH0 NOT PUSH1 0x80 SHL ADD DUP2 XOR",
                    "PUSH0 CALLDATALOAD PUSH2 %s ADD JUMP s: JUMPDEST",
                ),
            ],
        ),
    ]
    m6 = [
        Pair(
            f"m6.call_{n}site",
            [Variant("inlined", MIX), Variant("called", CALL, tail=FUNC)],
            reps=n,
        )
        for n in (1, 3, 8)
    ]
    spill_setup = "PUSH0 PUSH1 0x80 MSTORE PUSH0 CALLDATALOAD " + "CALLVALUE " * 8
    m7 = [
        Pair(
            "m7.spill",
            [
                Variant("stack_depth12", f"DUP12 ADD {MIX} DUP1 SWAP12 POP", spill_setup),
                Variant(
                    "memory_0x80", f"PUSH1 0x80 MLOAD ADD {MIX} DUP1 PUSH1 0x80 MSTORE", spill_setup
                ),
            ],
        )
    ]
    m8 = [
        Pair(
            "m8.mul_vs_shl",
            [Variant("mul16", f"PUSH1 0x10 MUL {MIX}"), Variant("shl4", f"PUSH1 4 SHL {MIX}")],
        ),
        Pair(
            "m8.div_vs_shr",
            [
                Variant("div16", f"PUSH1 0x10 SWAP1 DIV {MIX}"),
                Variant("shr4", f"PUSH1 4 SHR {MIX}"),
            ],
        ),
        Pair(
            "m8.exp_vs_shl_const",
            [
                Variant("exp_2_4", f"PUSH1 4 PUSH1 2 EXP MUL {MIX}"),
                Variant("shl4", f"PUSH1 4 SHL {MIX}"),
            ],
        ),
        Pair(
            "m8.exp_vs_shl_dyn",
            [
                Variant("exp_2_k", f"DUP2 PUSH1 7 AND PUSH1 2 EXP MUL {MIX}"),
                Variant("shl_k", f"DUP2 PUSH1 7 AND SHL {MIX}"),
            ],
        ),
        Pair(
            "m8.iszero2_jumpi",
            [
                Variant("plain", f"{COND['rand']} {BRANCHY}"),
                Variant("iszero_iszero", f"{COND['rand']} ISZERO ISZERO {BRANCHY}"),
            ],
        ),
    ]
    return {"m1": m1, "m2": m2, "m3": m3, "m4": m4, "m5": m5, "m6": m6, "m7": m7, "m8": m8}


@dataclass
class RunResult:
    gas: int = 0
    output: str = ""
    tx_secs: dict = field(default_factory=dict)


def evm2_cmd(args, path: str, n: int, spec: str, calldata: str | None) -> list[str]:
    cmd = [args.bin, "run", path, str(n), "--spec-id", spec]
    if calldata:
        cmd += ["--calldata", calldata]
    if args.core is not None:
        cmd = ["taskset", "-c", str(args.core)] + cmd
    return cmd


def run_evm2(
    args, path: str, n: int, spec: str, calldata: str | None, dump: str | None = None
) -> RunResult:
    """Runs `evm2 run`, checking that the interpreter and the JIT agree, and parses its output."""
    cmd = evm2_cmd(args, path, n, spec, calldata) + (["-o", dump] if dump else [])
    p = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if p.returncode != 0:
        sys.exit(f"{' '.join(cmd)} failed:\n{p.stderr[-3000:]}")
    r = RunResult()
    for line in p.stderr.splitlines():
        if m := re.search(r": gas_used=(\d+) output=(\S*)", line):
            r.gas, r.output = int(m[1]), m[2]
        elif m := re.search(r"/(interpreter|jit):(.*)", line):
            r.tx_secs[m[1]] = parse_duration(m[2])
    return r


def run_perf(
    args, path: str, n: int, spec: str, calldata: str | None, engine: str
) -> tuple[int, int]:
    """Returns user-mode (cycles, instructions) of one `evm2 run` process on one engine."""
    flag = "--interpret" if engine == "interpreter" else "--jit-only"
    cmd = evm2_cmd(args, path, n, spec, calldata) + [flag]
    ins, cyc = perf_counts(cmd[0], cmd[1:])
    return cyc, ins


def asm_body(path: str, stem: str) -> list[str]:
    """Returns the instructions and labels of a JIT `opt.s` dump, without directives or comments."""
    with open(path) as f:
        lines = [line.split("#")[0].strip() for line in f]
    return [s.replace(stem, "X") for s in lines if s and not s.startswith(".")]


def asm_cmp(a: list[str], b: list[str]) -> str:
    """Compares two assembly listings, ignoring immediates such as gas costs."""
    if a == b:
        return "identical"
    changed = (
        sum(1 for line in difflib.unified_diff(a, b, n=0, lineterm="") if line[:1] in "+-") - 2
    )
    imm = re.compile(r"-?\b(0x[0-9a-f]+|\d+)\b")
    if [imm.sub("N", line) for line in a] == [imm.sub("N", line) for line in b]:
        return f"imm only ({changed // 2} lines)"
    return f"differs ({changed} lines)"


def asm_insts(lines: list[str]) -> int:
    return sum(1 for s in lines if not s.endswith(":"))


def measure(args, paths: list[str], spec: str, calldata: str | None, tx_secs: dict) -> dict:
    """Measures user-mode cycles and instructions per transaction on both engines.

    Each engine runs in its own process at two transaction counts; the per-transaction cost is
    the slope between them, which cancels startup and JIT compilation. Cycles take the minimum
    over `--runs` interleaved processes; user-mode counters skip time spent descheduled.
    """
    n1 = 10
    plan = {}
    for p in paths:
        for engine in ("interpreter", "jit"):
            extra = int(min(max(args.target_secs / tx_secs[p][engine], 10), 20_000))
            plan[p, engine] = (n1, n1 + extra)
    samples = {key: {n: [] for n in ns} for key, ns in plan.items()}
    for _ in range(args.runs):
        for key, ns in plan.items():
            for n in ns:
                samples[key][n].append(run_perf(args, key[0], n, spec, calldata, key[1]))
    res = {}
    for (p, engine), (a, b) in plan.items():
        dtx = total_iters(b) - total_iters(a)
        cyc = (
            min(c for c, _ in samples[p, engine][b]) - min(c for c, _ in samples[p, engine][a])
        ) / dtx
        ins = (
            min(i for _, i in samples[p, engine][b]) - min(i for _, i in samples[p, engine][a])
        ) / dtx
        res[p, engine] = (cyc, ins)
    return res


def run_pair(args, pair: Pair, out: list[str]):
    d = os.path.join(args.out, pair.name)
    os.makedirs(d, exist_ok=True)

    def write(v: Variant, iters: int) -> str:
        path = os.path.join(d, f"{v.name}.evm")
        with open(path, "w") as f:
            f.write(loop_asm(v, pair.reps, iters))
        return path

    probe_iters = 1000
    gas_per_iter = 0
    for v in pair.variants:
        r = run_evm2(args, write(v, probe_iters), 2, pair.spec, None)
        gas_per_iter = max(gas_per_iter, r.gas / probe_iters)
    iters = int(min(args.iters, GAS_BUDGET / gas_per_iter))

    rows, paths, tx_secs = [], [], {}
    ref_asm = None
    for v in pair.variants:
        path = write(v, iters)
        paths.append(path)
        dump = os.path.join(d, "dump")
        r = run_evm2(args, path, 3, pair.spec, None, dump)
        tx_secs[path] = r.tx_secs
        size = os.path.getsize(os.path.join(dump, v.name, "bytecode.bin"))
        asm = asm_body(os.path.join(dump, v.name, "opt.s"), v.name)
        if ref_asm is None:
            ref_asm = asm
        rows.append((v.name, size, r.gas / iters, asm_insts(asm), asm_cmp(ref_asm, asm), r.output))
    m = measure(args, paths, pair.spec, None, tx_secs)

    start = len(out)
    out.append(f"\n### {pair.name} ({pair.spec}, {pair.reps} reps/iter, {iters} iters/tx)\n")
    if pair.note:
        out.append(f"{pair.note}\n")
    out.append(
        "| variant | bytes | gas/iter | interp cyc/iter | interp ins/iter | JIT cyc/iter | JIT ins/iter "
        "| Δinterp cyc/rep | ΔJIT cyc/rep | JIT asm insts | JIT asm vs first |"
    )
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    per = lambda p, e: (m[p, e][0] / iters, m[p, e][1] / iters)
    bi, bj = per(paths[0], "interpreter")[0], per(paths[0], "jit")[0]
    for (name, size, gas, insts, same, _), p in zip(rows, paths):
        (ic, ii), (jc, ji) = per(p, "interpreter"), per(p, "jit")
        out.append(
            f"| {name} | {size} | {gas:.0f} | {ic:.1f} | {ii:.1f} | {jc:.2f} | {ji:.1f} "
            f"| {(ic - bi) / pair.reps:+.2f} | {(jc - bj) / pair.reps:+.3f} | {insts} | {same} |"
        )
    outputs = {row[5] for row in rows}
    if len(outputs) != 1:
        out.append(f"\nVariants return different values: {', '.join(sorted(outputs))}")
    print("\n".join(out[start:]), flush=True)


def run_code(args, out: list[str]):
    rows, tx_secs = [], {}
    for path in args.code:
        stem = os.path.splitext(os.path.basename(path))[0]
        dump = os.path.join(args.out, "code-dump")
        r = run_evm2(args, path, 3, args.spec, args.calldata, dump)
        tx_secs[path] = r.tx_secs
        size = os.path.getsize(os.path.join(dump, stem, "bytecode.bin"))
        asm = asm_body(os.path.join(dump, stem, "opt.s"), stem)
        rows.append((path, size, r.gas, asm_insts(asm), r.output))
    m = measure(args, args.code, args.spec, args.calldata, tx_secs)
    out.append(f"\n### --code ({args.spec}, calldata {args.calldata or 'none'})\n")
    out.append(
        "| code | bytes | gas | interp cyc/tx | interp ins/tx | JIT cyc/tx | JIT ins/tx | JIT asm insts | output |"
    )
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for path, size, gas, insts, output in rows:
        (ic, ii), (jc, ji) = m[path, "interpreter"], m[path, "jit"]
        short = output if len(output) <= 20 else output[:18] + "…"
        out.append(
            f"| {path} | {size} | {gas} | {ic:.0f} | {ii:.0f} | {jc:.0f} | {ji:.0f} | {insts} | {short} |"
        )
    print("\n".join(out), flush=True)


def run_bundles(args, out: list[str]):
    """Runs `evm2 bundle` for every bundle `--rounds` times and reports per-case minimum costs."""
    variants = []
    best = {}
    for r in range(args.rounds):
        for path in args.bundle:
            jsonl = os.path.join(args.out, f"bundle-{r}-{os.path.basename(path)}l")
            cmd = [args.bin, "bundle", path, "--batches", str(args.batches), "--out", jsonl]
            if args.core is not None:
                cmd = ["taskset", "-c", str(args.core)] + cmd
            p = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if p.returncode != 0:
                sys.exit(f"{' '.join(cmd)} failed:\n{p.stderr[-3000:]}")
            if r == 0:
                print(
                    "\n".join(line for line in p.stderr.splitlines() if "!= anvil" in line),
                    flush=True,
                )
            with open(jsonl) as f:
                calls = [json.loads(line) for line in f]
            for c in calls:
                if c["variant"] not in variants:
                    variants.append(c["variant"])
                key = (c["variant"], c["test_id"], c["label"])
                costs = [
                    c[k]
                    for k in (
                        "interp_cycles",
                        "interp_instructions",
                        "jit_cycles",
                        "jit_instructions",
                    )
                ]
                best[key] = [min(a, b) for a, b in zip(best.get(key, costs), costs)]

    cases = list(dict.fromkeys(t for _, t, _ in best))
    sums = {}
    for (v, t, _), costs in best.items():
        acc = sums.setdefault((v, t), [0.0] * 4)
        for i, c in enumerate(costs):
            acc[i] += c
    base = variants[0]
    out.append(f"\n### bundles: kcycles per case (min of {args.rounds} rounds), Δ vs {base}\n")
    out.append("| case | " + " | ".join(f"{v} interp | {v} JIT" for v in variants) + " |")
    out.append("|---|" + "---:|---:|" * len(variants))
    totals = {v: [0.0] * 4 for v in variants}
    # Only cases that every variant has, so that the totals compare the same calls.
    for t in (t for t in cases if all((v, t) in sums for v in variants)):
        cells = []
        for v in variants:
            s = sums[v, t]
            for i in range(4):
                totals[v][i] += s[i]
            b = sums[base, t]
            if v == base:
                cells += [f"{s[0] / 1e3:.1f}", f"{s[2] / 1e3:.1f}"]
            else:
                cells += [f"{(s[0] / b[0] - 1) * 100:+.1f}%", f"{(s[2] / b[2] - 1) * 100:+.1f}%"]
        out.append(f"| {t} | " + " | ".join(cells) + " |")
    out.append(
        "\n| variant | interp Mcyc | interp Mins | JIT Mcyc | JIT Mins | Δinterp cyc | ΔJIT cyc |"
    )
    out.append("|---|---:|---:|---:|---:|---:|---:|")
    for v in variants:
        s, b = totals[v], totals[base]
        out.append(
            f"| {v} | {s[0] / 1e6:.3f} | {s[1] / 1e6:.3f} | {s[2] / 1e6:.3f} | {s[3] / 1e6:.3f} "
            f"| {(s[0] / b[0] - 1) * 100:+.2f}% | {(s[2] / b[2] - 1) * 100:+.2f}% |"
        )
    print("\n".join(out), flush=True)


def main():
    root = repo_root()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--suite", default="all", help="comma-separated suites or pair names (default: all)"
    )
    ap.add_argument("--list", action="store_true", help="list suite pairs and exit")
    ap.add_argument("--code", nargs="+", help="time these bytecode files instead of the suites")
    ap.add_argument("--calldata", help="calldata hex for --code")
    ap.add_argument("--bundle", nargs="+", help="time these deploy-and-call bundle JSON files")
    ap.add_argument(
        "--rounds", type=int, default=5, help="rounds over all bundles (minimum per call)"
    )
    ap.add_argument(
        "--batches", type=int, default=7, help="timed batches per call in a bundle round"
    )
    ap.add_argument("--spec", default="osaka", help="spec for --code")
    ap.add_argument(
        "--iters", type=int, default=100_000, help="max loop iterations per transaction"
    )
    ap.add_argument("--runs", type=int, default=5, help="processes per measurement (minimum)")
    ap.add_argument("--target-secs", type=float, default=0.2, help="extra execution time per slope")
    ap.add_argument("--core", type=int, default=None, help="pin to this CPU core")
    ap.add_argument("--out", default=os.path.join(root, "target", "shapes"))
    ap.add_argument("--bin", default=None, help="evm2 binary (default: build it)")
    ap.add_argument("--markdown", help="also write the tables to this file")
    args = ap.parse_args()

    all_suites = suites()
    if args.list:
        for s, pairs in all_suites.items():
            for p in pairs:
                print(f"{s}: {p.name}: {', '.join(v.name for v in p.variants)}")
        return

    if args.bin is None:
        args.bin = cargo_build(root)
    os.makedirs(args.out, exist_ok=True)

    out: list[str] = []
    if args.bundle:
        run_bundles(args, out)
    elif args.code:
        run_code(args, out)
    else:
        wanted = args.suite.split(",")
        for s, pairs in all_suites.items():
            for p in pairs:
                if "all" in wanted or s in wanted or p.name in wanted:
                    run_pair(args, p, out)
    if args.markdown:
        with open(args.markdown, "w") as f:
            f.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
