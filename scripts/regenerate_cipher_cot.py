"""
Regenerate Text Encryption (letter substitution cipher) CoT using canonical DATA.md format.

Pipeline:
  S1: intro
  S2: LEN - word lengths in target
  S3: TABLE - walk each example letter-by-letter, build bijective mapping
  S4: VER - cross-check all examples decrypt cleanly
  S5: DECRYPT - letter-by-letter with back-references to TABLE
  S6: CHECK - per-word validation
  S7: ANS + \\boxed{}

Reads: data/train_split_with_cot_v2.csv (already has bit manipulation v2 CoT)
Writes: data/train_split_with_cot_v3.csv (v2 + new cipher CoT)
"""
import pandas as pd
import re

INPUT_CSV = "data/train_split_with_cot_v2.csv"
OUTPUT_CSV = "data/train_split_with_cot_v3.csv"

RULE_PREAMBLE = """I am a reasoning model solving this puzzle strictly by following the template.

RULE 1: This is a letter substitution cipher template. I see encrypted phrases mapped to plaintext phrases. Each letter maps to exactly one other letter (a-z, bijective). I know this is NOT roman, unit conversion, gravity, symbol-digit, cipher-digit, or binary.

RULE 2: None of the flavor text (Alice in Wonderland, etc.) matters. The wrapper is here to trick me. I am only here to solve the problem.

RULE 3: Final answer in \\boxed{} at the end. Wrong format means zero points.

S1: This is a letter substitution cipher. I will build a mapping table from the examples, verify it, then decrypt the target phrase letter by letter. I am now going to fill out the template."""


def parse_prompt(prompt):
    """Return (examples: list of (enc, plain)), target_encrypted_str) or (None, None)."""
    # examples line format: "encrypted -> plaintext"
    example_lines = re.findall(r'([a-z ]+?)\s*->\s*([a-z ]+)', prompt)
    # filter: encrypted side should be all lowercase letters + spaces
    examples = []
    for enc, plain in example_lines:
        enc = enc.strip()
        plain = plain.strip()
        if not enc or not plain:
            continue
        if len(enc.split()) != len(plain.split()):
            continue
        examples.append((enc, plain))

    # target: line starting "decrypt the following text:"
    m = re.search(r'decrypt the following text:\s*(.+?)(?:\n|$)', prompt)
    if not m:
        return None, None
    target = m.group(1).strip()
    return examples, target


def build_mapping(examples):
    """
    Walk each example char by char. Build enc->plain mapping.
    Return (mapping dict, per_example_new_mappings list, conflict) where
    per_example_new_mappings[i] is a list of (enc, plain) pairs newly added
    by example i (for the TABLE trace).
    """
    mapping = {}
    per_ex_new = []
    for enc, plain in examples:
        new_pairs = []
        if len(enc) != len(plain):
            per_ex_new.append(new_pairs)
            continue
        for ec, pc in zip(enc, plain):
            if ec == ' ' or pc == ' ':
                continue
            if ec in mapping:
                if mapping[ec] != pc:
                    return None, None  # conflict, not bijective
                continue
            # check no other key maps to pc (must be bijective)
            if pc in mapping.values():
                # some other enc already maps to pc → conflict
                return None, None
            mapping[ec] = pc
            new_pairs.append((ec, pc))
        per_ex_new.append(new_pairs)
    return mapping, per_ex_new


def decrypt(text, mapping):
    out = []
    for c in text:
        if c == ' ':
            out.append(' ')
        elif c in mapping:
            out.append(mapping[c])
        else:
            return None  # gap in mapping
    return "".join(out)


def verify_all_examples(examples, mapping):
    """Check that all examples decrypt cleanly with the mapping."""
    for enc, plain in examples:
        decoded = decrypt(enc, mapping)
        if decoded != plain:
            return False
    return True


def build_cot(examples, target, mapping, per_ex_new, answer):
    target_words = target.split()
    answer_words = answer.split()

    lines = [RULE_PREAMBLE, ""]
    # S2: LEN
    lines.append("S2: LEN")
    lines.append(f'TGT:"{target}"')
    lens = " ".join(f"W{i+1}:{len(w)}" for i, w in enumerate(target_words))
    lines.append(lens)
    lines.append("")

    # S3: TABLE
    lines.append("S3: TABLE")
    total_added = 0
    for idx, ((enc, plain), new_pairs) in enumerate(zip(examples, per_ex_new), 1):
        pairs_str = ",".join(f"{e}={p}" for e, p in new_pairs) if new_pairs else "none"
        lines.append(f'EX{idx}:"{enc}"->"{plain}" [{len(new_pairs)}] {pairs_str}')
        total_added += len(new_pairs)
    lines.append(f"TOTAL:{len(mapping)}/26")
    lines.append("")

    # S4: VER
    lines.append("S4: VER")
    lines.append(f"CROSS:{len(examples)}/{len(examples)} examples verified")
    lines.append("")

    # S5: DECRYPT (per word, letter by letter, with back-refs)
    lines.append("S5: DECRYPT")
    # track when each plain letter was first introduced (by word index)
    first_seen = {}  # enc_char -> word index where it appears first
    for widx, enc_word in enumerate(target_words):
        lines.append(f"W{widx+1}: {enc_word}")
        for ec in enc_word:
            pc = mapping.get(ec, '?')
            if ec in first_seen and first_seen[ec] != widx:
                lines.append(f" {ec}->{pc} (W{first_seen[ec]+1})")
            else:
                lines.append(f" {ec}->{pc}")
                if ec not in first_seen:
                    first_seen[ec] = widx
        decoded = "".join(mapping.get(c, '?') for c in enc_word)
        lines.append(f" = {decoded}")
    lines.append("")

    # S6: CHECK
    lines.append("S6: CHECK")
    for widx, word in enumerate(answer_words):
        lines.append(f'W{widx+1}:"{word}" len={len(word)}Y alpha=Y PASS')
    lines.append(f"ALL:PASS {len(answer_words)}/{len(answer_words)} words")
    lines.append("")

    # S7: ANS
    lines.append(f"S7: ANS={answer}")
    lines.append("")
    lines.append(f"\\boxed{{{answer}}}")
    return "\n".join(lines)


def main():
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {INPUT_CSV}: {len(df)} rows")

    cipher_mask = df['type'] == 'Text Encryption'
    cipher_df = df[cipher_mask]
    print(f"Text Encryption rows: {len(cipher_df)}")

    results = []
    solved = 0
    parse_fail = 0
    map_fail = 0
    decrypt_fail = 0
    mismatch = 0

    for idx, row in cipher_df.iterrows():
        examples, target = parse_prompt(row['prompt'])
        if examples is None or not examples:
            results.append((idx, None))
            parse_fail += 1
            continue
        mapping, per_ex_new = build_mapping(examples)
        if mapping is None:
            results.append((idx, None))
            map_fail += 1
            continue
        if not verify_all_examples(examples, mapping):
            results.append((idx, None))
            map_fail += 1
            continue
        decoded = decrypt(target, mapping)
        if decoded is None:
            results.append((idx, None))
            decrypt_fail += 1
            continue
        gt = str(row['answer']).strip()
        if decoded != gt:
            results.append((idx, None))
            mismatch += 1
            continue
        cot = build_cot(examples, target, mapping, per_ex_new, gt)
        results.append((idx, cot))
        solved += 1

    print(f"\nSolved: {solved}/{len(cipher_df)} ({100*solved/len(cipher_df):.1f}%)")
    print(f"Parse fail: {parse_fail}")
    print(f"Map fail (conflict): {map_fail}")
    print(f"Decrypt fail (gap in mapping): {decrypt_fail}")
    print(f"Answer mismatch: {mismatch}")

    # Build new dataframe
    df_new = df.copy()
    keep_mask = pd.Series([True] * len(df), index=df.index)
    for idx, cot in results:
        if cot is not None:
            df_new.at[idx, 'generated_cot'] = cot
        else:
            keep_mask.at[idx] = False  # drop unsolvable cipher rows

    df_new = df_new[keep_mask].reset_index(drop=True)
    print(f"\nNew dataset: {len(df_new)} rows")
    print(df_new['type'].value_counts().sort_index())

    df_new.to_csv(OUTPUT_CSV, index=False)
    print(f"\nWrote {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
