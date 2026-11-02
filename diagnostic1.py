from pathlib import Path
import pandas as pd

tracks = pd.read_csv("data/processed/louvain_tracks.csv")
labels = pd.read_csv("data/processed/go_labels.csv")

seeds = {"DER1", "HRD3", "UBC7", "CUE1", "HRD1"}

for path in sorted(Path("outputs").glob("gnn_rankings_k*_seed3888.csv")):
    df = (
        pd.read_csv(path)
        .merge(tracks[["node_index", "track"]], on="node_index")
        .merge(labels[["node_index", "go_positive"]], on="node_index")
    )

    boundary = df[df["track"] == "Boundary"]
    top_before = boundary.nlargest(20, "gnn_cosine")
    top_after = boundary[
        ~boundary["preferred_name"].isin(seeds)
    ].nlargest(20, "gnn_cosine")

    print(
        path.stem,
        "before:", int(top_before["go_positive"].sum()),
        "after:", int(top_after["go_positive"].sum()),
        "seeds in top20:",
        top_before.loc[
            top_before["preferred_name"].isin(seeds),
            "preferred_name"
        ].tolist()
    )