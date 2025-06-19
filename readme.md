# Size-Bounded Community Search over HINs

This repository contains the Python implementation and datasets for the following paper:

> **Efficient Size-Bounded Community Search Algorithms over Heterogeneous Information Networks**

---

## Environment

The code was developed and tested under the following setup:

- **Hardware:** Intel(R) Xeon(R) W-2133 CPU @ 3.60GHz, 32GB RAM  
- **Operating System:** Windows 10 Enterprise (22H2)  
- **Python Version:** Python 3.8

---

## How to Run the Code

### 1. Install Dependencies

Use the following command to install the required packages:

```
pip install -r requirements.txt
```

### 2. Prepare Datasets

Four datasets are included under the `datasets/` directory and they are downloaded from [this repository](https://github.com/librahu/HIN-Datasets-for-Recommendation-and-Network-Embedding). The **Freebase** dataset is too large to be hosted here. You can download it from [freebase-easy-14-04-14.zip](https://freebase-easy.cs.uni-freiburg.de/dump/freebase-easy-14-04-14.zip)

If you want to use a custom dataset, please ensure it follows the same format as the provided ones in the `datasets/` folder.

### 3. Generate the queries

Run `generateQueries.py` to generate queries. You can modify the `dataname` variable in the script to generate queries for other datasets.


### 4. Run the Code

Use the following command structure to execute the program:

```
python code.py <dataset> <meta-path> <query-node> <size>
```

**Example:**

```
python enumerateVertex.py Amazon item,view,item i164 21
```

This command searches for a size-bounded community starting from node `i164` in the `Amazon` dataset using the meta-path `item,view,item`, with a size bound of `21`.

To run all queries, simply execute the `run.ps1` script.