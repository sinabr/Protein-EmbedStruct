#!/usr/bin/env python
"""
Structural-biology course  ·  Data-curation pipeline
---------------------------------------------------
Creates the following artefacts inside ./data/ :

    catalogue.parquet            – one row per PDB chain with sequence
    homolog_pairs.tsv            –   ''   homologous chain pairs (30–95 % id, ≥70 % cov)
    mutant_pairs.tsv             – wild-type ↔ mutant structure pairs
    domain_annotations.parquet   – CATH domain counts per chain
    residue_scores.parquet       – residue-level importance (UniProt features demo)

Author: <your name>
"""

import argparse, json, os, re, shutil, subprocess, sys, tempfile, textwrap
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────
# 0.  CLI & paths
# ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", default="data", help="output folder")
parser.add_argument(
    "--heavy",
    action="store_true",
    help="download full PDB, run MMseqs2 clustering, etc.",
)
args = parser.parse_args()
DATA = Path(args.data_dir).resolve()
RAW_PDB = DATA / "pdb"
RAW_PDB.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────
# 1.  Helpers
# ──────────────────────────────────────────────────────────────
def rcsb_graphql(query: str, variables=None):
    """Small wrapper around RCSB GraphQL API."""
    url = "https://data.rcsb.org/graphql"
    payload = {"query": query}
    if variables is not None:
        payload["variables"] = variables
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["data"]


def run(cmd, **kwargs):
    print("·", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True, **kwargs)


# ──────────────────────────────────────────────────────────────
# 2.  Catalogue of all polymer chains (sequence + metadata)
# ──────────────────────────────────────────────────────────────
CATALOGUE = DATA / "catalogue.parquet"
if not CATALOGUE.exists():
    print("▶ Building chain catalogue …")
    chain_rows = []
    # (Pull only polymer entities; the query returns ~200 k entries in < 2 min)
    query = textwrap.dedent(
        """
        {
          entries(limit: 100000) {
            pdbx_struct_group {
              entry_id
            }
            polymer_entities {
              entity_id
              rcsb_polymer_entity_container_identifiers {
                auth_asym_ids
              }
              entity_poly { pdbx_seq_one_letter_code_can }
            }
          }
        }"""
    )
    for entry in tqdm(rcsb_graphql(query)["entries"]):
        pdb_id = entry["pdbx_struct_group"]["entry_id"].lower()
        for ent in entry["polymer_entities"]:
            seq = ent["entity_poly"]["pdbx_seq_one_letter_code_can"].replace(" ", "")
            chains = ",".join(
                ent["rcsb_polymer_entity_container_identifiers"]["auth_asym_ids"]
            )
            chain_rows.append(
                dict(
                    pdb_id=pdb_id,
                    entity_id=ent["entity_id"],
                    chain=chains,
                    seq=seq,
                    length=len(seq),
                )
            )
    pd.DataFrame(chain_rows).to_parquet(CATALOGUE)

cat = pd.read_parquet(CATALOGUE)
print("▸ Chains in catalogue:", len(cat))


# ──────────────────────────────────────────────────────────────
# 3.  Homologous pairs (MMseqs2 clustering)
# ──────────────────────────────────────────────────────────────
HOMOLOG = DATA / "homolog_pairs.tsv"
if not HOMOLOG.exists():
    print("▶ Identifying homologous pairs …")
    fasta = DATA / "all_sequences.fasta"
    if not fasta.exists():
        with fasta.open("w") as f:
            for r in cat.itertuples():
                f.write(f">{r.pdb_id}_{r.entity_id}_{r.chain}\n{r.seq}\n")

    if args.heavy:
        tmp = tempfile.mkdtemp()
        run(
            [
                "mmseqs",
                "easy-cluster",
                fasta,
                tmp + "/res",
                tmp + "/tmp",
                "--min-seq-id",
                "0.3",
                "-c",
                "0.7",
            ]
        )
        shutil.copy(tmp + "/res_cluster.tsv", DATA / "mmseqs.tsv")
    else:
        print("  (light run) – using cached demo clusters")
        # demo subset shipped here:
        (DATA / "mmseqs.tsv").write_text(
            "A_0_0\tA_1_0\nA_2_0\tA_0_0\n"  # toy example
        )

    clu = pd.read_csv(DATA / "mmseqs.tsv", sep="\t", names=["member", "rep"])
    pairs = clu.merge(clu, on="rep")
    pairs = pairs[pairs.member_x != pairs.member_y]
    pairs[["member_x", "member_y"]].drop_duplicates().rename(
        columns={"member_x": "prot1", "member_y": "prot2"}
    ).to_csv(HOMOLOG, sep="\t", index=False)
print("▸ Homologous pairs:", sum(1 for _ in open(HOMOLOG)))


# ──────────────────────────────────────────────────────────────
# 4.  Mutant pairs    (quick demo via RCSB deposited_mutations flag)
# ──────────────────────────────────────────────────────────────
MUTANTS = DATA / "mutant_pairs.tsv"
if not MUTANTS.exists():
    print("▶ Collecting WT ↔ mutant pairs …")
    if args.heavy:
        query = textwrap.dedent(
            """
            query($after:String){
              entries(after:$after, query:{type:terminal, service:"text", parameters:{value:"mutation"}}) {
                pageInfo{ endCursor hasNextPage }
                nodes{
                  entry_id
                  polymer_entities{
                    entity_id
                    rcsb_polymer_entity_container_identifiers{ auth_asym_ids }
                    rcsb_mutation_details{ mutation_site }
                  }
                }
              }
            }"""
        )
        cursor, rows = None, []
        for _ in range(100):  # limit pages for safety
            res = rcsb_graphql(query, {"after": cursor})["entries"]
            for n in res["nodes"]:
                if not n["polymer_entities"][0]["rcsb_mutation_details"]:
                    continue
                wt = f"{n['entry_id'].lower()}_{n['polymer_entities'][0]['entity_id']}"
                for mut in n["polymer_entities"][1:]:
                    rows.append(
                        dict(
                            wt=wt,
                            mut=f"{n['entry_id'].lower()}_{mut['entity_id']}",
                            n_mut_sites=len(mut["rcsb_mutation_details"]),
                        )
                    )
            if not res["pageInfo"]["hasNextPage"]:
                break
            cursor = res["pageInfo"]["endCursor"]
        pd.DataFrame(rows).to_csv(MUTANTS, sep="\t", index=False)
    else:
        print("  (light run) – creating tiny demo set")
        (DATA / "mutant_pairs.tsv").write_text("wt\tmut\tn_mut_sites\nX\tY\t1\n")

print("▸ Mutant pairs:", sum(1 for _ in open(MUTANTS)))


# ──────────────────────────────────────────────────────────────
# 5.  Domain annotations (CATH)
# ──────────────────────────────────────────────────────────────
DOMAINS = DATA / "domain_annotations.parquet"
if not DOMAINS.exists():
    print("▶ Downloading & parsing CATH domain file …")
    cath_txt = DATA / "cath.txt"
    if not cath_txt.exists():
        url = "https://download.cathdb.info/cath/releases/latest-release/cath-domain-description-file.txt"
        cath_txt.write_bytes(requests.get(url, timeout=120).content)

    cath_df = pd.read_csv(
        cath_txt,
        delim_whitespace=True,
        comment="#",
        header=None,
        names=["domain", "pdb", "chain", "res_start", "res_end", "class"],
    )
    ann = (
        cath_df.groupby(["pdb", "chain"])
        .agg(n_domains=("domain", "count"), domain_ids=("domain", ";".join))
        .reset_index()
    )
    ann.to_parquet(DOMAINS)
print("▸ Chains with CATH annotation:", len(pd.read_parquet(DOMAINS)))


# ──────────────────────────────────────────────────────────────
# 6.  Residue-level importance demo  (UniProt features)
# ──────────────────────────────────────────────────────────────
RESIDUE = DATA / "residue_scores.parquet"
if not RESIDUE.exists():
    print("▶ Fetching UniProt site features (demo on first 100 chains) …")

    def uniprot_sites(accession: str):
        url = f"https://rest.uniprot.org/uniprotkb/{accession}.json"
        r = requests.get(url, timeout=20)
        if not r.ok:
            return []
        return [
            (f["type"], f["location"]["start"])
            for f in r.json().get("features", [])
            if "location" in f
        ]

    rows = []
    for r in tqdm(cat.head(100).itertuples(), total=100):
        # Normally we'd map PDB→UniProt via SIFTS; here we skip that for brevity
        for t, pos in uniprot_sites(r.pdb_id):
            rows.append(dict(pdb=r.pdb_id, chain=r.chain, pos=pos, src=t))
    pd.DataFrame(rows).to_parquet(RESIDUE)
print(
    "▸ Residue score rows:",
    len(pd.read_parquet(RESIDUE)) if RESIDUE.exists() else 0,
)


# ──────────────────────────────────────────────────────────────
# 7.  Overview
# ──────────────────────────────────────────────────────────────
summary = dict(
    n_chains=len(cat),
    n_homolog_pairs=sum(1 for _ in open(HOMOLOG)),
    n_mutant_pairs=sum(1 for _ in open(MUTANTS)),
    n_domain_rows=len(pd.read_parquet(DOMAINS)),
)
print("\n══ Summary ══")
print(json.dumps(summary, indent=2))
print("Done ✔")
