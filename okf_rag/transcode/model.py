from dataclasses import dataclass


@dataclass
class ConceptDoc:
    path: str                 # bundle-relative POSIX path, e.g. "tables/orders.md"
    okf_type: str             # OKF required field `type`
    title: str
    description: str | None
    resource: str | None
    tags: list[str]
    timestamp: str | None     # ISO 8601
    status: str               # stable | draft | deprecated
    x_source: dict            # verbatim source frontmatter (spec §4.1)
    body: str                 # markdown body; links already bundle-absolute


@dataclass
class Edge:
    src: str                  # bundle-relative path of source doc
    dst: str                  # bundle-relative path of target doc
    kind: str                 # child | curated | related | prose | sibling
    weight: float
