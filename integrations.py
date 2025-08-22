"""
integrations.py
===============

Illustrative integrations with external services.  These functions are not
used directly in the POC but demonstrate how the modular code could be
extended to work with AWS S3 or NVIDIA cuGraph.
"""

from __future__ import annotations

from typing import Dict

from .knowledge_graph import KnowledgeGraph


def upload_to_s3(file_path: str, bucket_name: str, object_name: str = None) -> None:
    """Uploads a file to an S3 bucket using boto3.  Requires AWS credentials."""
    import boto3  # type: ignore
    s3 = boto3.client("s3")
    if object_name is None:
        import os
        object_name = os.path.basename(file_path)
    s3.upload_file(file_path, bucket_name, object_name)


def run_pagerank_with_cugraph(kg: KnowledgeGraph) -> Dict[str, float]:
    """Runs PageRank using cuGraph on a GPU.  Requires RAPIDS."""
    try:
        import cudf  # type: ignore
        import cugraph  # type: ignore
    except ImportError as e:
        raise ImportError("cuDF and cuGraph are required for GPU PageRank.") from e
    src = []
    dst = []
    weights = []
    for u, neighbors in kg.adj.items():
        for v, w in neighbors:
            src.append(u)
            dst.append(v)
            weights.append(w)
    gdf = cudf.DataFrame({"src": src, "dst": dst, "weight": weights})
    G = cugraph.Graph()
    G.from_cudf_edgelist(gdf, source="src", destination="dst", edge_attr="weight")
    pr_df = cugraph.pagerank(G, alpha=0.85)
    return {row["vertex"]: float(row["pagerank"]) for row in pr_df.to_pandas().to_dict("records")}