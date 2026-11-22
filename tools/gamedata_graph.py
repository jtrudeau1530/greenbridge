"""Build a content-set and reference graph for the ``common.rpf`` XML catalog.

``gamedata_xml.py`` catalogs the XML documents recovered from an RPF9 carve and counts reference
tokens, but it drops the target strings. This tool keeps them long enough to build and measure a
directed graph:

    python -I gamedata_graph.py <carve_dir> [--json FILE]
    python -I gamedata_graph.py <file.xml> [<file.xml> ...] [--json FILE]

A carve directory is read through ``gamedata_xml.load_documents``, so multi-stream documents are
rebuilt the same way as the index and catalog. For each parsed document the graph has:

* a document node;
* a content-set node for every ``<id>`` (a top-level id or an id inside ``<contentSets>``);
* a target node for every ``<filename>``, ``parentPath`` and ``parentFile`` value and every
  ``<loadPath>`` value in ``<includedChangesets>``.

Edges are ``defines`` (document to its sets), ``includes`` (document to an ``<includedSets>`` id),
``contains_file`` (set or document to a filename target), ``loads_changeset`` (document to a
``<loadPath>`` target), and ``parent_path`` / ``parent_file`` for the two attributes. An include is
joined to a document only when its text equals a set id exactly. Missing ids become dangling
content-set nodes. They are never guessed from a similar spelling.

The four documented prefix rules are measured: ``$(common)``, ``$(root)``, ``$(target)``, and
``platform:/`` split into ``levels`` and everything else. ``common:/`` occurs in the corpus but is
not one of those rules, so it is reported as an unrecognized scheme and is not normalized to
``$(common)``. ``parentPath`` values are XPath-like locators, not archive paths.

The output is aggregate and schema-only: counts, root names, category names and reference prefixes.
Target strings, ids, offsets, source names and element values are never written. This keeps the JSON reviewable without copying game data, and the tests are
synthetic. The encrypted RPF9 table of contents is still the reason an archive path cannot be
mapped to a carved document; those targets remain target nodes rather than invented document
edges. This is a mount/reference graph, not a full runtime schema: the catalog carries names and
categories, not field types.
"""
import argparse
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict, deque

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import gamedata_xml as gx  # noqa: E402

MANIFEST_ROOTS = ("CDataFileMgr__ContentChangeSetDef", "CDataFileMgr__ContentDef")
MANIFEST_ROOT_SET = frozenset(MANIFEST_ROOTS)
KNOWN_MACROS = frozenset(("common", "target", "root"))
PARENT_ATTRIBUTES = ("parentPath", "parentFile")

MACRO_REF = re.compile(r"^\$\(([A-Za-z_][\w.-]*)\)(.*)$", re.S)
PLATFORM_REF = re.compile(r"^platform:/([^/]*)(.*)$", re.I | re.S)
SCHEME_REF = re.compile(r"^([A-Za-z][\w.-]*):/(.*)$", re.S)
RELATIVE = "relative/locator"


def classify_target(text: str) -> tuple:
    """Return (namespace, remainder, resolution reason) for one reference string."""
    value = text.strip()
    match = MACRO_REF.match(value)
    if match:
        name = match.group(1)
        namespace = "$(" + name + ")"
        remainder = match.group(2).lstrip("/")
        if name.lower() in KNOWN_MACROS:
            return namespace, remainder, "document_unmapped"
        return namespace, remainder, "unknown_macro"
    match = PLATFORM_REF.match(value)
    if match:
        first = (match.group(1) or "").lower()
        namespace = "platform:/levels" if first == "levels" else "platform:/other"
        return namespace, (match.group(1) or "") + match.group(2), "document_unmapped"
    match = SCHEME_REF.match(value)
    if match:
        return match.group(1).lower() + ":/", match.group(2), "unrecognized_prefix"
    return RELATIVE, value, "relative_locator"


def _text(element) -> str:
    return (element.text or "").strip() if element is not None else ""


def _direct_id(element) -> str:
    for child in element:
        if child.tag == "id":
            value = _text(child)
            if value:
                return value
    return ""


def _add_filename_refs(report: dict, scope, owner: str, seen: set) -> None:
    if scope is None:
        return
    for filename in scope.iter("filename"):
        if filename in seen:
            continue
        seen.add(filename)
        value = _text(filename)
        if value:
            report["refs"].append({"kind": "filename", "target": value, "owner": owner})


def scan_document(doc: dict) -> dict:
    """Parse one payload and keep only the graph-relevant names and reference strings."""
    report = {
        "root": "",
        "parsed": False,
        "complete": bool(doc.get("complete", False)),
        "sets": [],
        "refs": [],
        "includes": [],
    }
    try:
        root = ET.fromstring(doc["data"])
    except (ET.ParseError, ValueError):
        # A truncated tail can still carry a closing-root hint from the index. It is a named but
        # unparsed document, not a rootless fragment; only an empty hint is rootless.
        report["root"] = doc.get("root_hint") or ""
        return report
    report["parsed"] = True
    report["root"] = root.tag
    report["complete"] = bool(doc.get("complete", True))

    own = _direct_id(root)
    if own:
        report["sets"].append(own)

    content_sets = root.find("contentSets")
    if content_sets is not None:
        for item in content_sets:
            set_id = _direct_id(item)
            if set_id:
                report["sets"].append(set_id)

    seen = set()
    report["sets"] = [set_id for set_id in report["sets"]
                      if not (set_id in seen or seen.add(set_id))]

    filenames = set()
    _add_filename_refs(report, root.find("fileList"), own, filenames)

    if content_sets is not None:
        for item in content_sets:
            set_id = _direct_id(item)
            if set_id:
                _add_filename_refs(report, item.find("fileList"), set_id, filenames)

    _add_filename_refs(report, root, "", filenames)

    changesets = root.find("includedChangesets")
    if changesets is not None:
        for load_path in changesets.iter("loadPath"):
            value = _text(load_path)
            if value:
                report["refs"].append({"kind": "loadPath", "target": value, "owner": ""})

    included = root.find("includedSets")
    if included is not None:
        for item in included:
            value = "".join(item.itertext()).strip()
            if value:
                report["includes"].append(value)

    for element in root.iter():
        for attribute in PARENT_ATTRIBUTES:
            value = (element.attrib.get(attribute) or "").strip()
            if value:
                report["refs"].append({"kind": attribute, "target": value, "owner": ""})
    return report


def _strongly_connected(nodes: list, adjacency: dict) -> list:
    """Tarjan's algorithm with an explicit stack for deep include chains."""
    index = {}
    low = {}
    stack = []
    on_stack = set()
    components = []
    pending = []

    def enter(node):
        index[node] = low[node] = len(index)
        stack.append(node)
        on_stack.add(node)
        pending.append((node, iter(adjacency.get(node, ()))))

    for root in nodes:
        if root in index:
            continue
        enter(root)
        while pending:
            node, neighbours = pending[-1]
            neighbour = next(neighbours, None)
            if neighbour is not None:
                if neighbour not in index:
                    enter(neighbour)
                elif neighbour in on_stack:
                    low[node] = min(low[node], index[neighbour])
                continue

            pending.pop()
            if low[node] == index[node]:
                component = []
                while True:
                    member = stack.pop()
                    on_stack.remove(member)
                    component.append(member)
                    if member == node:
                        break
                components.append(component)
            if pending:
                parent = pending[-1][0]
                low[parent] = min(low[parent], low[node])
    return components


def build_graph(documents: list) -> dict:
    """Build nodes, edges, unresolved buckets and aggregates from scanned documents."""
    reports = [scan_document(doc) for doc in documents]
    set_docs = defaultdict(list)
    all_set_ids = set()
    for index, report in enumerate(reports):
        for set_id in report["sets"]:
            set_docs[set_id].append(index)
            all_set_ids.add(set_id)
        for set_id in report["includes"]:
            all_set_ids.add(set_id)

    target_ids = {}
    for report in reports:
        for ref in report["refs"]:
            namespace, remainder, _ = classify_target(ref["target"])
            key = (namespace, remainder)
            if key not in target_ids:
                target_ids[key] = len(target_ids)

    edges = Counter()
    references = Counter()
    prefixes = Counter()
    resolution = Counter()
    unresolved_sites = Counter()
    unresolved_namespaces = defaultdict(Counter)
    unique_resolution = defaultdict(set)
    target_edges = []
    include_edges = []
    dangling_sets = set()

    for index, report in enumerate(reports):
        for set_id in report["sets"]:
            edges["defines"] += 1
        for set_id in report["includes"]:
            references["includedSets"] += 1
            edges["includes"] += 1
            if set_id in set_docs:
                resolution["document"] += 1
                for target_doc in set_docs[set_id]:
                    include_edges.append((index, target_doc))
            else:
                resolution["include_id_missing"] += 1
                unresolved_sites["include_id_missing"] += 1
                dangling_sets.add(set_id)

        for ref in report["refs"]:
            kind = ref["kind"]
            namespace, remainder, reason = classify_target(ref["target"])
            references[kind] += 1
            prefixes[namespace] += 1
            resolution[reason] += 1
            unique_resolution[reason].add((namespace, remainder))
            if reason == "document_unmapped":
                unresolved_sites["document_unmapped"] += 1
                unresolved_namespaces["document_unmapped"][namespace] += 1
            elif reason == "unrecognized_prefix":
                unresolved_sites["unrecognized_prefix"] += 1
                unresolved_namespaces["unrecognized_prefix"][namespace] += 1
            elif reason == "unknown_macro":
                unresolved_sites["unknown_macro"] += 1
                unresolved_namespaces["unknown_macro"][namespace] += 1
            elif reason == "relative_locator":
                unresolved_sites["relative_locator"] += 1
            if kind == "filename":
                edges["contains_file"] += 1
                edge_kind = "contains_file"
            elif kind == "loadPath":
                edges["loads_changeset"] += 1
                edge_kind = "loads_changeset"
            elif kind == "parentPath":
                edges["parent_path"] += 1
                edge_kind = "parent_path"
            else:
                edges["parent_file"] += 1
                edge_kind = "parent_file"
            target_edges.append((index, ref["owner"], edge_kind, (namespace, remainder)))

    include_edges = sorted(set(include_edges))
    adjacency = defaultdict(set)
    for source, target in include_edges:
        adjacency[source].add(target)
    in_degree = Counter()
    for _, target in include_edges:
        in_degree[target] += 1

    manifest_documents = [index for index, report in enumerate(reports)
                          if report["parsed"] and report["root"] in MANIFEST_ROOT_SET]
    roots = [index for index in manifest_documents if in_degree[index] == 0]
    root_fallback = False
    if not roots and manifest_documents:
        roots = list(manifest_documents)
        root_fallback = True

    depth = {index: 0 for index in roots}
    queue = deque(roots)
    reachable_docs = set()
    while queue:
        source = queue.popleft()
        if source in reachable_docs:
            continue
        reachable_docs.add(source)
        for target in adjacency.get(source, ()):
            if target not in depth:
                depth[target] = depth[source] + 1
                queue.append(target)

    components = _strongly_connected(list(range(len(reports))), adjacency)
    cycle_components = [component for component in components
                        if len(component) > 1 or component[0] in adjacency.get(component[0], ())]
    cycle_nodes = sum(len(component) for component in cycle_components)

    reachable_sets = set()
    reachable_targets = set()
    reachable_target_sites = 0
    for index in reachable_docs:
        report = reports[index]
        reachable_sets.update(report["sets"])
        reachable_sets.update(report["includes"])
    for index, owner, _, key in target_edges:
        if index in reachable_docs and (not owner or owner in reachable_sets):
            reachable_target_sites += 1
            reachable_targets.add(key)

    category_stats = defaultdict(Counter)
    root_stats = defaultdict(Counter)
    for report in reports:
        root = report["root"]
        category = gx.category_for(root) if root else gx.UNCATALOGUED
        for stats in (category_stats[category], root_stats[root]):
            stats["documents"] += 1
            stats["parsed"] += int(report["parsed"])
            stats["complete"] += int(report["complete"])
            stats["content_sets"] += len(report["sets"])
            stats["include_references"] += len(report["includes"])
            stats["target_references"] += len(report["refs"])
            stats["references"] += len(report["refs"]) + len(report["includes"])
        if root in MANIFEST_ROOT_SET:
            category_stats[category]["manifest_documents"] += 1
            root_stats[root]["manifest_documents"] += 1

    # Root names are stable schema names; category is the only derived label.
    for root, stats in root_stats.items():
        stats["category"] = gx.category_for(root) if root else gx.UNCATALOGUED

    unique_by_namespace = Counter()
    for namespace, _ in target_ids:
        unique_by_namespace[namespace] += 1

    summary = {
        "mode": "",
        "payloads": len(reports),
        "documents": sum(1 for report in reports if report["complete"]),
        "parsed": sum(1 for report in reports if report["parsed"]),
        "unparsed": sum(1 for report in reports if not report["parsed"]),
        "root_count": len([root for root in root_stats if root]),
        "rootless": sum(1 for report in reports if not report["root"]),
        "manifest_documents": len(manifest_documents),
        "manifest_roots": len(MANIFEST_ROOTS),
        "content_sets": len(all_set_ids),
        "defined_content_sets": len(set_docs),
        "dangling_content_sets": len(dangling_sets),
        "duplicate_set_ids": sum(1 for indexes in set_docs.values() if len(indexes) > 1),
        "reference_sites": sum(references.values()),
        "target_sites": sum(references[kind] for kind in
                            ("filename", "loadPath", "parentPath", "parentFile")),
        "include_sites": references["includedSets"],
        "references": dict(sorted(references.items())),
        "prefixes": dict(sorted(prefixes.items())),
        "resolution": dict(sorted(resolution.items())),
        "unique": {
            "targets": len(target_ids),
            "targets_by_namespace": dict(sorted(unique_by_namespace.items())),
            "document_unmapped": len(unique_resolution["document_unmapped"]),
            "unknown_macro": len(unique_resolution["unknown_macro"]),
            "unrecognized_prefix": len(unique_resolution["unrecognized_prefix"]),
            "relative_locator": len(unique_resolution["relative_locator"]),
            "content_set_ids": len(all_set_ids),
            "include_ids": len({set_id for report in reports for set_id in report["includes"]}),
        },
        "nodes": {
            "documents": len(reports),
            "content_sets": len(all_set_ids),
            "targets": len(target_ids),
            "total": len(reports) + len(all_set_ids) + len(target_ids),
        },
        "edges": {
            "total": sum(edges.values()),
            "defines": edges["defines"],
            "includes": edges["includes"],
            "contains_file": edges["contains_file"],
            "loads_changeset": edges["loads_changeset"],
            "parent_path": edges["parent_path"],
            "parent_file": edges["parent_file"],
            "document_edges": edges["defines"] + edges["includes"],
            "target_edges": (edges["contains_file"] + edges["loads_changeset"] +
                             edges["parent_path"] + edges["parent_file"]),
        },
        "content_graph": {
            "include_edges": len(include_edges),
            "documents_with_incoming_includes": len(in_degree),
            "documents_with_outgoing_includes": sum(1 for report in reports
                                                    if report["includes"]),
            "root_documents": len(roots),
            "root_fallback": root_fallback,
            "reachable_documents": len(reachable_docs),
            "max_depth": max(depth.values(), default=0),
            "cycles": len(cycle_components),
            "cycle_nodes": cycle_nodes,
        },
        "reachable": {
            "documents": len(reachable_docs),
            "content_sets": len(reachable_sets),
            "targets": len(reachable_targets),
            "target_sites": reachable_target_sites,
        },
        "categories": dict(sorted(category_stats.items(), key=lambda kv: (-kv[1]["documents"],
                                                                          kv[0]))),
        "roots": dict(sorted(root_stats.items(), key=lambda kv: (-kv[1]["documents"], kv[0]))),
    }
    # Count roots per category from the root table, not from documents. A root belongs to exactly
    # one category, so this also checks that the category totals line up with the 118 named roots.
    for category, stats in summary["categories"].items():
        stats["roots"] = sum(1 for root in summary["roots"]
                             if (gx.category_for(root) if root else gx.UNCATALOGUED) == category)
        stats.setdefault("manifest_documents", 0)
    for stats in summary["roots"].values():
        stats.setdefault("manifest_documents", 0)
        stats.setdefault("category", gx.UNCATALOGUED)

    return {
        "summary": summary,
        "unresolved": {
            "document_unmapped": dict(sorted(unresolved_namespaces["document_unmapped"].items())),
            "unknown_macro": dict(sorted(unresolved_namespaces["unknown_macro"].items())),
            "unrecognized_prefix": dict(sorted(unresolved_namespaces["unrecognized_prefix"].items())),
            "relative_locator": unresolved_sites["relative_locator"],
            "include_id_missing": unresolved_sites["include_id_missing"],
            "total_sites": sum(unresolved_sites.values()),
        },
    }


def print_summary(graph: dict) -> None:
    summary = graph["summary"]
    print(f'{summary["payloads"]} payloads ({summary["documents"]} complete, '
          f'{summary["parsed"]} parsed), {summary["manifest_documents"]} manifest documents')
    print(f'{summary["content_sets"]} content-set ids '
          f'({summary["defined_content_sets"]} defined, '
          f'{summary["dangling_content_sets"]} dangling, '
          f'{summary["duplicate_set_ids"]} duplicate)')
    print(f'{summary["reference_sites"]:,} reference sites: '
          f'{summary["references"].get("filename", 0):,} filename, '
          f'{summary["references"].get("loadPath", 0)} loadPath, '
          f'{summary["references"].get("parentPath", 0)} parentPath, '
          f'{summary["references"].get("parentFile", 0)} parentFile, '
          f'{summary["include_sites"]} includedSets')
    prefixes = ", ".join(f"{name} {count:,}" for name, count in summary["prefixes"].items())
    print(f'prefixes: {prefixes}')
    print(f'resolution: {summary["resolution"]}')
    print(f'graph: {summary["nodes"]["total"]:,} nodes, {summary["edges"]["total"]:,} edges '
          f'({summary["edges"]["defines"]} defines, {summary["edges"]["includes"]} includes, '
          f'{summary["edges"]["contains_file"]:,} file, '
          f'{summary["edges"]["loads_changeset"]} changeset, '
          f'{summary["edges"]["parent_path"]} parentPath, '
          f'{summary["edges"]["parent_file"]} parentFile)')
    content = summary["content_graph"]
    print(f'content graph: {content["root_documents"]} roots, '
          f'{content["reachable_documents"]} reachable documents, '
          f'{summary["reachable"]["content_sets"]} sets, depth {content["max_depth"]}, '
          f'{content["cycles"]} cycles')
    print(f'reachable: {summary["reachable"]["targets"]:,} unique targets '
          f'({summary["reachable"]["target_sites"]:,} sites)')
    unresolved = graph["unresolved"]
    print(f'unresolved: {unresolved["total_sites"]:,} sites '
          f'({len(unresolved["document_unmapped"])} archive namespaces, '
          f'{sum(unresolved["unrecognized_prefix"].values())} unknown scheme, '
          f'{unresolved["relative_locator"]} relative, '
          f'{unresolved["include_id_missing"]} missing include id)')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("targets", nargs="+", help="carve directory or XML file(s)")
    parser.add_argument("--json", help="write the aggregate graph summary to this JSON")
    args = parser.parse_args()

    try:
        documents = list(gx.load_documents(args.targets))
    except FileNotFoundError as exc:
        print(f"not found: {exc}", file=sys.stderr)
        return 2
    graph = build_graph(documents)
    graph["summary"]["mode"] = gx.document_mode(args.targets)
    print_summary(graph)
    if args.json:
        try:
            with open(args.json, "w", encoding="utf-8") as f:
                json.dump(graph, f, indent=1)
        except OSError as exc:
            print(f"cannot write {args.json}: {exc.strerror or exc}", file=sys.stderr)
            return 2
        print("wrote", args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
