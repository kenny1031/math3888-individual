
"""GO Biological Process labels for the HLJ1 network."""

from collections import defaultdict
from pathlib import Path
import re

import pandas as pd


ORF_RE = re.compile(
    r"Y[A-P][LR]\d{3}[CW](?:-[A-G])?",
    re.IGNORECASE,
)


def load_ontology(path, include_part_of=False):
    """Read GO terms and parent relationships from OBO."""

    terms = {}
    aliases = {}
    current = None

    def save_term():
        if not current or current.get("obsolete"):
            return

        if "id" not in current:
            return

        term_id = current["id"]
        terms[term_id] = current

        for alt in current["alt_ids"]:
            aliases[alt] = term_id

    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()

            if line == "[Term]":
                save_term()
                current = {
                    "parents": set(),
                    "alt_ids": [],
                    "obsolete": False,
                }
                continue

            if line.startswith("["):
                save_term()
                current = None
                continue

            if current is None:
                continue

            if line.startswith("id: "):
                current["id"] = line[4:]

            elif line.startswith("namespace: "):
                current["namespace"] = line[11:]

            elif line.startswith("is_a: "):
                current["parents"].add(line.split()[1])

            elif line.startswith("relationship: part_of "):
                if include_part_of:
                    current["parents"].add(line.split()[2])

            elif line.startswith("alt_id: "):
                current["alt_ids"].append(line[8:])

            elif line == "is_obsolete: true":
                current["obsolete"] = True

    save_term()
    return terms, aliases


def selected_go_terms(terms, aliases, roots):
    """Find selected GO roots and all their descendants."""

    roots = {aliases.get(r, r) for r in roots}

    missing = roots - set(terms)

    if missing:
        raise ValueError(f"GO roots absent from ontology: {missing}")

    matched = set()

    for term_id, term in terms.items():
        if term.get("namespace") != "biological_process":
            continue

        stack = [term_id]
        visited = set()

        while stack:
            node = stack.pop()
            node = aliases.get(node, node)

            if node in visited:
                continue

            visited.add(node)

            if node in roots:
                matched.add(term_id)
                break

            parent_term = terms.get(node)

            if parent_term is not None:
                stack.extend(parent_term["parents"])

    return matched


def build_preferred_name_map(nodes):
    """Map unambiguous preferred gene names to ORFs."""

    mapping = defaultdict(set)

    for row in nodes.itertuples():
        if pd.isna(row.orf):
            continue

        name = str(row.preferred_name).upper().strip()
        orf = str(row.orf).upper().strip()

        if name:
            mapping[name].add(orf)

    return {
        name: next(iter(orfs))
        for name, orfs in mapping.items()
        if len(orfs) == 1
    }


def build_go_labels(
    gaf_path,
    obo_path,
    nodes,
    roots,
    include_part_of=False,
    exclude_evidence=None,
):
    """Return protein-level binary GO labels."""

    terms, aliases = load_ontology(
        obo_path,
        include_part_of=include_part_of,
    )

    target_terms = selected_go_terms(
        terms, aliases, roots
    )

    preferred_map = build_preferred_name_map(nodes)
    exclude_evidence = set(exclude_evidence or [])

    # Read GAF records, retaining all annotation fields
    # needed for protein ID mapping and filtering.
    records = []

    with open(gaf_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("!"):
                continue

            fields = line.rstrip("\n").split("\t")

            if len(fields) < 11:
                continue

            records.append({
                "object_id": fields[1],
                "symbol": fields[2],
                "qualifier": fields[3],
                "go_id": fields[4],
                "evidence": fields[6],
                "aspect": fields[8],
                "synonyms": fields[10],
            })

    # Resolve GAF object IDs to ORFs using gene symbols
    # and synonyms. Require a unique ORF assignment.
    object_to_orfs = defaultdict(set)

    for row in records:
        tokens = [
            row["symbol"],
            *row["synonyms"].split("|"),
        ]

        candidates = set()

        for token in tokens:
            token = token.upper().strip()

            if ORF_RE.fullmatch(token):
                candidates.add(token)

        if not candidates:
            name = row["symbol"].upper().strip()

            if name in preferred_map:
                candidates.add(preferred_map[name])

        object_to_orfs[row["object_id"]].update(candidates)

    positive_orfs = set()
    matched_annotations = 0

    for row in records:
        if row["aspect"] != "P":
            continue

        if "NOT" in row["qualifier"].upper().split("|"):
            continue

        if row["evidence"] in exclude_evidence:
            continue

        term = aliases.get(row["go_id"], row["go_id"])

        if term not in target_terms:
            continue

        orfs = object_to_orfs[row["object_id"]]

        if len(orfs) != 1:
            continue

        positive_orfs.update(orfs)
        matched_annotations += 1

    labels = nodes[
        [
            "node_index",
            "string_id",
            "orf",
            "preferred_name",
        ]
    ].copy()

    labels["go_positive"] = (
        labels["orf"].isin(positive_orfs)
    ).astype(int)

    print(f"    OBO terms: {len(terms):,}")
    print(f"    Selected GO terms: {len(target_terms):,}")
    print(f"    GAF records: {len(records):,}")
    print(f"    Matched positive annotations: {matched_annotations:,}")
    print(f"    GO-positive proteins in component: {labels['go_positive'].sum():,}")

    return labels
