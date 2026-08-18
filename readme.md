# Size-Bounded Community Search over HINs

This repository provides the open-source implementation, datasets, and experimental resources for our research on **size-bounded community search over heterogeneous information networks (HINs)**.

The repository accompanies the paper:

> **Efficient Size Constraint Community Search over Heterogeneous Information Networks**  
> Accepted at **IEEE ICDE 2026**

The project studies how to identify cohesive communities containing a query vertex while satisfying a user-specified size constraint. It provides exact and heuristic algorithms designed for large heterogeneous information networks and supports reproducible evaluation on multiple real-world datasets.

## Repository Contents

The repository includes:

- Python implementations of the proposed community search algorithms;
- scripts for generating experimental queries;
- datasets used in the experiments;
- scripts for reproducing experimental runs; and
- the accompanying technical report.

## Environment

The code was developed and tested under the following environment:

- **Hardware:** Intel(R) Xeon(R) W-2133 CPU @ 3.60 GHz, 32 GB RAM
- **Operating System:** Windows 10 Enterprise (22H2)
- **Python:** 3.8

## Getting Started

### 1. Install Dependencies

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

### 2. Prepare the Datasets

Four datasets used in our experiments are provided with this repository. They are derived from the public HIN datasets available from:

https://github.com/librahu/HIN-Datasets-for-Recommendation-and-Network-Embedding

The **Freebase** dataset is too large to be hosted directly in this repository and can be downloaded from:

https://freebase-easy.cs.uni-freiburg.de/dump/freebase-easy-14-04-14.zip

For custom datasets, please follow the same input format as the provided datasets.

### 3. Generate Queries

Run:

```bash
python generateQueries.py
```

The dataset can be changed by modifying the `dataname` variable in `generateQueries.py`.

### 4. Run Community Search

Use the following command format:

```bash
python <algorithm>.py <dataset> <meta-path> <query-node> <size>
```

For example:

```bash
python enumerateVertex.py Amazon item,view,item i164 21
```

This command searches the `Amazon` HIN for a size-bounded community containing query node `i164`, under the meta-path `item,view,item`, with a size bound of `21`.

To execute all predefined experimental queries on Windows, run:

```powershell
.\run.ps1
```

## Research Use

This repository is intended to support reproducible research on topics including:

- community search;
- heterogeneous information networks;
- graph mining;
- dense subgraph discovery; and
- size-constrained graph analysis.

Researchers are welcome to use the implementation as a baseline, reproduce the reported experiments, or extend the algorithms to related community search problems.

## Citation

If you use this repository in your research, please cite:

```bibtex
@inproceedings{zhang2026efficient,
  title={Efficient Size Constraint Community Search over Heterogeneous Information Networks},
  author={Xinjian Zhang, Chengfei Liu, Lu Chen, Rui Zhou and Bo Ning},
  booktitle={Proceedings of the IEEE International Conference on Data Engineering (ICDE)},
  year={2026}
}
```

The complete BibTeX entry will be updated with the final publication metadata.

## License

An open-source license will be provided for this repository.
