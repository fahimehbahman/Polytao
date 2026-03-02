# PolyTAO – Property-Conditioned Polymer Generation

This repository contains the implementation of teacher–student transformer models for property-conditioned polymer generation, including evaluation and validation workflows.

📁 Repository Structure
poly_fold_rx/
│
├── checkpoints/         # All pretrained teacher and student models are stored in.Evaluation scripts and validation   notebooks load models directly from this directory
│       ├── polytao_student_distilled    # Trained student model checkpoints.
│       └── teacher                      # Trained teacher model checkpoints.
├── data/
│    ├── polymers_with_properties_normalized_train.csv  # train dataSet with 80% data of main dataset.
│    ├── polymers_with_properties_normalized_test.csv   # test dataSet with 20% data of main dataset.
│    └── polymers_with_properties_normalized_Total.csv  # main dataSet with 99k data.This file contains normalized        molecular property vectors and SMILES representations.
│
├── src/                             # Core implementation
│   ├── run_evaluation.py
│   ├── run_evaluation_error.py
│   └── ...
│
├── validation/ 
│     └──  validate_teacher_student.ipynb      # Final validation notebooks 
│
├── results/                    # Evaluation Result , histogram , valid generated molecules based on model 
│     ├── eval_student_20
│     ├── eval_teacher
│     └── ...                 
│
├── requirements.txt
└── README.md

# Environment Setup (Conda – Recommended)

1️⃣ Create Environment
conda create -n polytao python=3.10 -y
conda activate polytao

2️⃣ Install RDKit (via conda-forge)
conda install -c conda-forge rdkit --solver=libmamba -y

3️⃣ Install PyTorch (CPU version)
pip install torch --index-url https://download.pytorch.org/whl/cpu

4️⃣ Install Project Dependencies (please check disk drive has enough space!)
pip install -r poly_fold_rx\requirements.txt

5️⃣ Register Jupyter Kernel (Optional but Recommended)
python -m ipykernel install --user --name polytao --display-name "Python (polytao)"

▶️ Running the Validation Notebook ( open validation\validate_teacher_student.ipynb )

1️⃣Activate environment:

run this command in terminal : conda activate polytao

2️⃣Select kernel:

Python (polytao)

3️⃣Run cells sequentially.

⚙️ Dependencies

Main dependencies:

Python 3.10

PyTorch

PyTorch Lightning

Transformers

RDKit

NumPy

Pandas

Scikit-learn

Matplotlib

tqdm

See requirements.txt for full list.
