# PULP — Phytochemical Uncertainty Lookup Portal: 
A generalized hypothesis generating tool for phytochemicals.

*A reproducible computational framework for phytochemical bioactivity prediction and similarity-supported target inference.*


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Status: Research Pipeline](https://img.shields.io/badge/Status-Research%20Pipeline-green.svg)]()
[![Phytochemistry](https://img.shields.io/badge/Domain-Phytochemistry-purple.svg)]()
[![Machine Learning](https://img.shields.io/badge/Field-Machine%20Learning-orange.svg)]()
[![Reproducibility](https://img.shields.io/badge/Focus-Reproducibility-blueviolet.svg)]()



**PULP (Phytochemical Uncertainty Lookup Portal)** is a computational platform for analyzing phytochemicals using molecular descriptors, three-dimensional quantitative structure–activity relationship (3D-QSAR) features, learned molecular representations, calibrated bioactivity prediction, and similarity-based target/off-target inference.

The repository organizes the computational code, data resources, model-development components, target-inference resources, and supporting artifacts required to document and reproduce the PULP workflow.

The current inference architecture uses a **40-dimensional (40D) molecular representation** constructed from three complementary feature groups:

* **10 molecular descriptors**
* **9 three-dimensional QSAR descriptors**
* **21 selected MoLFormer embedding dimensions**

These features are combined into a fixed representation used by the calibrated bioactivity model and the similarity-based target-inference workflow.

PULP is designed for computational screening, compound prioritization, and biological hypothesis generation. Its predictions are not substitutes for experimental validation.

## Key Features

* **Single-molecule analysis:** Submit a chemical structure as a SMILES string.
* **Bulk processing:** Process multiple molecules from a text file.
* **Integrated 40D representation:** Combine molecular, 3D-QSAR, and learned molecular features.
* **Three-dimensional processing:** Generate and optimize molecular conformers for 3D feature calculation.
* **Calibrated bioactivity prediction:** Obtain active and inactive probability estimates.
* **Uncertainty-related outputs:** Examine probability-derived certainty, entropy, confidence level, and probability margin where provided by the deployed model.
* **Similarity-based target inference:** Retrieve the nearest reference molecules using cosine similarity.
* **Off-target hypothesis generation:** Identify additional possible target associations supported by reference evidence.
* **Structured outputs:** Export molecular analysis and prediction results in JSON format through the service workflow.
* **Reproducibility-oriented organization:** Preserve code, data, model artifacts, feature definitions, and supporting resources in a version-controlled repository.

## Scientific Workflow

```text
                 INPUT
          Single SMILES / Bulk TXT
                     |
                     v
           Molecular Structure
             Validation
                     |
                     v
          Molecular Feature Generation
                     |
       +-------------+-------------+
       |             |             |
       v             v             v
  10 Molecular    9 3D-QSAR    MoLFormer
  Descriptors     Features     Embedding
                                  |
                            21 Selected
                              Features
       |             |             |
       +-------------+-------------+
                     |
                     v
            Fixed 40D Representation
                     |
          +----------+-----------+
          |                      |
          v                      v
   Calibrated Bioactivity   Standardized 40D
       Prediction           Similarity Search
          |                      |
          v                      v
   Active / Inactive       Top 20 Nearest
   Probabilities           Reference Neighbors
   Uncertainty Measures           |
          |                       v
          |                 Target Evidence
          |                       |
          +-----------+-----------+
                      |
                      v
             Structured Results
               and JSON Output
```

### 1. Molecular input and validation

PULP accepts molecular structures encoded as SMILES. Input structures are validated and converted into molecular objects for subsequent descriptor calculation and inference.

For bulk processing, the service reads a text file containing one molecule per line. An optional tab-separated compound name can be supplied alongside each SMILES.

### 2. Construction of the 40D representation

The deployed representation consists of:

| Feature group               | Dimensions | Purpose                                                       |
| --------------------------- | ---------: | ------------------------------------------------------------- |
| Molecular descriptors       |         10 | Describe molecular and physicochemical properties             |
| 3D-QSAR descriptors         |          9 | Capture information derived from optimized molecular geometry |
| Selected MoLFormer features |         21 | Encode learned molecular information                          |
| **Total**                   |     **40** | **Unified molecular representation**                          |

The learned representation is derived from **`ibm-research/MoLFormer-XL-both-10pct`**. The underlying embedding is 768-dimensional, from which the 21 dimensions defined by the feature-selection workflow are retained.

The selected features and their order are part of the model specification. The deployed workflow uses the established feature definitions and training-derived preprocessing parameters rather than fitting a new representation to each query molecule.

### 3. Three-dimensional conformer processing

PULP generates molecular conformers and performs energy optimization before calculating the 3D-QSAR features.

The documented service workflow uses RDKit conformer generation with ETKDGv3 and MMFF94s optimization, with UFF as a fallback where required. The optimized conformer selected by the implementation is used for downstream 3D feature calculation.

The precise conformer settings and descriptor definitions should be taken from the version-controlled implementation and associated feature manifest.

### 4. Calibrated bioactivity prediction

The calibrated binary classifier operates on the fixed 40D representation and produces probability estimates for the active and inactive classes.

Depending on the output fields produced by the service, the result may include:

* Active probability (`p_active`)
* Inactive probability (`p_inactive`)
* Certainty
* Entropy uncertainty
* Confidence level
* Probability margin

These quantities describe the model's output and its probability distribution. They do not establish experimental activity or guarantee that a prediction will generalize to every chemical domain.

The final calibrated model and its associated feature specification should be treated as a matched set. Do not substitute a model artifact without verifying feature compatibility and the corresponding preprocessing requirements.

### 5. Similarity-based target and off-target inference

PULP uses the standardized 40D representation to compare an input molecule with a reference collection. Cosine similarity is used to retrieve the top 20 nearest reference molecules in the deployed inference workflow.

Target evidence associated with the retrieved neighbors is used to generate candidate target and possible off-target hypotheses.

These results depend on reference coverage, the available target annotations, and the similarity relationship between the query and reference molecules. They are **similarity-supported hypotheses, not experimentally confirmed molecular interactions**.

## Repository Structure

The repository currently exposes the following top-level organization:

```text
PULP/
├── Bioactivity/
│   └── Bioactivity-related resources and model-development components
│
├── Code_Base/
│   └── Computational implementation and workflow scripts
│
├── Data/
│   └── Data resources used in model development and inference
│
├── Target_model/
│   └── Target-inference resources and associated model artifacts
│
├── LICENSE
└── README.md
```

The descriptions above summarize the intended role of each directory. For exact file-level provenance, refer to the committed contents, script documentation, manifests, and data documentation within each directory.

### `Bioactivity/`

Contains bioactivity-related resources associated with the development and evaluation of the bioactivity prediction component.

Relevant documentation should identify the training data, activity-label definitions, feature representation, model-training procedure, calibration procedure, evaluation strategy, and final model artifacts.

The production inference model should remain distinguishable from intermediate experiments, comparison models, and development outputs.

### `Code_Base/`

Contains the computational code used to implement and reproduce the analytical workflow.

The codebase should document the relationship between the feature-generation scripts, model-development scripts, reference-data preparation, target-inference implementation, and any service or command-line entry points.

Scripts should be executed in their intended order, with their required inputs and outputs documented explicitly. Do not assume that every script is an independent entry point.

### `Data/`

Contains data resources used by the computational pipeline.

For reproducibility, each dataset should have a documented source, version or retrieval date where available, description, relevant identifier fields, preprocessing history, and any applicable license or access restrictions.

Generated intermediate files should be distinguished from original source data and final reference resources.

### `Target_model/`

Contains target-inference resources and associated model or reference artifacts.

The target-inference implementation uses a standardized 40D reference representation and annotated target evidence. Documentation should distinguish the molecular similarity index from the biological target annotations used to interpret retrieved neighbors.

Reference files, annotation tables, identifiers, feature order, and preprocessing statistics must remain compatible with the corresponding inference scripts.

### `descriptor_summary.txt`

A supporting text file describing molecular descriptor information. Consult its contents and the corresponding feature manifest to determine the exact definitions, names, units, and order of the descriptors used in a particular model version.

### `test_20_molecular_descriptors.tsv`

A tab-separated test or reference file associated with molecular descriptor processing. Its precise role, column definitions, and relationship to training or validation data should be documented alongside the file.

**Reproducibility note:** The directory names above reflect the repository's visible top-level structure. Before publishing a release, verify this tree against the actual commit and document the exact paths of the production model, reference matrix, preprocessing artifacts, and runnable entry points.

## Software Requirements

The current workflow depends on a Python environment with compatible scientific-computing, cheminformatics, machine-learning, and transformer-model packages.

Core dependencies used by the described pipeline include:

* Python
* RDKit
* NumPy
* pandas
* scikit-learn
* joblib
* PyTorch
* Transformers

Additional packages may be required by individual scripts.

The exact versions should be recorded in a versioned environment specification, such as `environment.yml`, a pinned `requirements.txt`, or an equivalent dependency lock file. A reproducible release should include the tested Python version, package versions, model revision, and any system-level requirements.

The list above is a summary of the pipeline's dependencies, not a substitute for a verified environment file.

## Installation and Environment Setup

Clone the repository:

```bash
git clone https://github.com/AnanyaSourav/PULP.git
cd PULP
```

Create an isolated environment using the environment specification provided with the release. If no environment specification is available, create a compatible Python environment and install the dependencies documented by the individual scripts.

For example, using Conda when an `environment.yml` is available:

```bash
conda env create -f environment.yml
conda activate pulp
```

If the repository does not yet contain `environment.yml`, do not run this command until that file has been added. Use the actual dependency file or documented installation procedure associated with the selected commit.

### Pretrained MoLFormer model

The learned molecular representation uses:

`ibm-research/MoLFormer-XL-both-10pct`

The model and tokenizer may be downloaded through the Hugging Face Transformers ecosystem. Initial setup requires access to the model repository and sufficient disk space for its weights and cache.

For repeatable results, record the exact model revision used in the analysis rather than relying indefinitely on a moving repository revision. The selected model implementation may require execution of repository-provided custom model code; inspect and trust the relevant code before enabling it.

The model cache directory must be writable by the account executing the pipeline. Web-server deployments may run under a different operating-system account than interactive shell sessions.

## Running the Pipeline

The exact commands depend on the entry points and input artifacts included in the selected repository commit. Review the scripts in `Code_Base/` and the documentation within `Bioactivity/` and `Target_model/` before executing a workflow.

A reproducible run should follow these stages:

1. Verify that the required input data and model artifacts are available.
2. Install the documented software dependencies.
3. Generate or validate the molecular feature representation.
4. Load the matching calibrated bioactivity model and preprocessing artifacts.
5. Load the compatible standardized reference representation and target annotations.
6. Run inference on the query molecules.
7. Preserve the generated results, logs, configuration, and exact repository commit identifier.

### Single-molecule input

The deployed service interface supports a single SMILES string and an optional compound name.

Example input:

```text
SMILES: CCO
Name: Ethanol
```

### Bulk input

A plain-text file can contain one SMILES per line:

```text
CCO
CC(=O)Oc1ccccc1C(=O)O
C1=CC=C(C=C1)O
```

The service also supports a tab-separated name-and-SMILES format:

```text
Ethanol<TAB>CCO
Aspirin<TAB>CC(=O)Oc1ccccc1C(=O)O
```

Blank lines and lines beginning with `#` are ignored by the described service parser.

Use the actual service entry point and command-line options present in the selected commit. A deployment copy of the web service should not be assumed to exist in the repository root unless it has been committed there.

## Output and Interpretation

The service produces structured JSON output for the submitted molecules. Depending on the result and the service version, the output can include:

* Input and canonical SMILES
* Compound name and molecular formula
* Molecular descriptor values
* 3D-QSAR descriptor values
* Conformer-generation and optimization information
* Selected MoLFormer features
* Final 40D feature representation
* Active and inactive probabilities
* Probability-derived uncertainty information
* Nearest reference molecules and cosine similarity scores
* Probable target candidates
* Possible off-target candidates
* Supporting reference evidence
* Per-molecule status and error information

For bulk runs, the output also records a summary of input, successful, and failed molecule counts.

### Interpretation guidelines

**Bioactivity probability:** Indicates the model's estimated probability for a class under its learned decision function. It is not an experimental measurement.

**Certainty and entropy:** Describe aspects of the predicted probability distribution. They should be interpreted according to the formulas implemented in the corresponding code.

**Similarity score:** Quantifies similarity in the defined 40D representation. It is not a binding affinity or direct measure of biological activity.

**Probable target:** A candidate supported by the available similarity-based target evidence.

**Possible off-target:** An additional candidate for investigation, not a confirmed adverse interaction.

A missing target association does not establish that a compound cannot interact with that target. Likewise, a positive computational prediction does not prove that the interaction occurs experimentally.

## Reproducibility and Version Control

To reproduce a result as closely as possible, record the following information for every analysis:

1. Git commit hash or release tag.
2. Input file and a checksum of its contents.
3. Python and dependency versions.
4. MoLFormer model identifier and exact revision.
5. Molecular descriptor definitions and feature order.
6. 3D conformer-generation and energy-optimization settings.
7. Calibrated bioactivity model artifact and its checksum.
8. Training-derived preprocessing artifacts and their checksums.
9. Reference matrix and target-annotation versions.
10. Inference parameters, including the number of nearest neighbors.
11. Output JSON and execution logs.

The bioactivity model, feature order, scaler or preprocessing statistics, and reference representation should be treated as interdependent artifacts. Mixing artifacts from different model-development runs can invalidate predictions even when the code executes successfully.

### Data and artifact integrity

Where feasible, provide checksums for large or critical files:

```bash
sha256sum path/to/artifact
```

Store the resulting hashes in a manifest associated with the release.

If a required file is too large for normal Git storage, document its source, version, checksum, and retrieval procedure. Git Large File Storage or a persistent archival repository may be appropriate, depending on the file size and distribution permissions.

Never commit credentials, API tokens, private keys, personal data, or local server configuration containing secrets.

## Deployment

PULP can be exposed through a web interface that accepts single or bulk SMILES input and presents the resulting predictions.

A production deployment may separate:

* The web interface and user-input validation
* The Python inference service
* The trained bioactivity model
* The MoLFormer model and tokenizer cache
* The reference representation and target annotations
* Generated result files and logs

The web interface should invoke the computational pipeline rather than independently reimplementing the scientific calculations.

Deployment-specific paths, web-server users, writable cache directories, upload limits, and generated-output permissions must be configured for the target system. Local filesystem paths should not be treated as portable repository paths.

The public repository should contain the code and documentation necessary to understand and reproduce the workflow without exposing private server settings or credentials.

## Limitations

PULP is a computational screening and hypothesis-generation platform. Its results are subject to several limitations:

* Performance depends on the chemical and biological coverage of the training and reference data.
* Molecular representation and descriptor calculations may not capture every determinant of biological activity.
* The selected 40D feature space represents a defined modeling choice rather than a complete description of molecular behavior.
* Similarity-based inference may produce unreliable hypotheses for compounds that are poorly represented in the reference collection.
* Target annotations can be incomplete, inconsistent, or dependent on the source database.
* Prediction uncertainty does not account automatically for every source of experimental, biological, or dataset uncertainty.
* Computational target and off-target hypotheses require independent validation.

PULP should be used to prioritize compounds and formulate testable hypotheses, not as a replacement for experimental assays or expert scientific assessment.

## Contributing and Reuse

Contributions that improve documentation, reproducibility, testing, data provenance, and scientific transparency are welcome.

When modifying the workflow:

* Preserve the correspondence between the feature manifest and trained model.
* Document changes to descriptor definitions or preprocessing.
* Record new dependencies and tested versions.
* Avoid silently replacing reference data or model artifacts.
* Include validation results for changes to the inference pipeline.
* Clearly distinguish experimental results from computational predictions.

Please consult the repository's [MIT License](LICENSE) and the licenses and terms of use associated with external datasets and pretrained models before redistribution or commercial reuse.

## Citation

If you use PULP in a publication, thesis, or research report, cite the repository and the associated scientific work when a formal publication becomes available.

A provisional software citation is:

> Sourav, A. PULP: Phytochemical Uncertainty Lookup Portal. GitHub repository. https://github.com/AnanyaSourav/PULP

For reproducible research, also cite the specific release or commit used, the original sources of the datasets, and the pretrained MoLFormer model.

## Contact

For questions about the implementation or scientific workflow, please open an issue in the repository:

**Repository:** https://github.com/AnanyaSourav/PULP

**Web Page:** https://db.nipgr.ac.in/PULP/home.php
---

**Scientific statement:** PULP generates computational bioactivity predictions and similarity-supported target/off-target hypotheses. These results should not be interpreted as experimentally confirmed biological interactions.
