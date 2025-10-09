## README

This folder contains the code and the bond formation mechanism prediction results for 132,300 combinations involved in the article "Unveiling Mechanistic Patterns of Radical Bond Formation in Copper-Catalyzed Radical Transformation Reactions through Predictive Linear Free Energy Relationship Modelling". The folder contains the following files:

## Dependences

The code in this folder requires the following Python libraries:

- python = 3.11.5
- numpy == 1.26.4
- pandas == 2.2.3
- pandas == 2.2.2
- scikit-learn == 1.7.2
- umap-learn == 0.5.9.post2

## Usage

1. Constructing and Validating LFERs:

- Constructing LFER: "Solve_Equation_Groups.py" is the Python code for solving systems of equations using the least squares method, which is used to fit the reactivity parameters for each reaction component from the free energy barriers obtainable in the initial set (for details, see the article).
- kFold Cross Validation: "LFER_KFold_IP.ipynb", "LFER_KFold_IP.ipynb", and "LFER_KFold_IP.ipynb" are the Python notebook files for conduct kFold-cv on LFER.
- Input file examples: "dG_RE_num.xlsx",  "dG_RS_num.xlsx", "dG_IP_num.xlsx".

Before using this script, you need to provide a formatted data file (in xlsx format) of free energy barriers within the initial set: the first column should contain the name or serial number of the ligand, the second column should contain the name or serial number of the radical, the third column should contain the name or serial number of the nucleophile, and the fourth column should contain the values of the free energy barrier corresponding to a specific bond formation mechanism for that ligand-radical-nucleophile combination. Note that a single list should only contain free energy data for one type of bond formation mechanism; and if the free energy barrier for a certain combination is not available (for example, because its transition state is hard to be located), the value for that entry should be NaN (not "-" or "None", etc.).

2. Multiple Linear Regression:

- Seach for the best model: "MLR_on_R_IP_search.ipynb" is the Python notebook for searching and estimating MLR models, used to establish a fit between reactivity parameters and physical organic parameters. The performance of each MLR model is estimated under loo-cv.
- KFold-cv and loo-cv: "MLR_on_R_IP_kfold_vs_loo.ipynb" is the Python notebook for comparing kFold-cv and loo-cv.
- Linear transformation from LFER parameter N_RE to N_RS: "Linear_Transformation.ipynb" is the Python notebook for generating N_RS derived from N_RE.
- Input file examples: "R_IP_vs_Parameters.xlsx", "LFER_Parameters_N.xlsx".

Before using this script, you need to provide a formatted data file (in xlsx format) that includes reactivity parameters and other physical organic parameters: the first column should contain the reactivity parameter for a specific reaction component under a certain mechanism (for example, the reactivity parameter R_RE for radicals under reductive elimination), and from the second column onwards should contain the values of the physical organic parameters corresponding to that reaction component. Note that the name or serial number of the reaction component does not need to be provided; it is only necessary to align the reactivity parameter and physical organic parameters of the same reaction component in the same row; all data must not be empty.

3. UMAP dimensionality reduction of the chemical space:

- Dimensionality reduction: "UMAP_by_Finger_Prints.ipynb" and "UMAP_by_Properties.ipynb" are the Python notebook files for UMAP dimensionality reduction of the chemical space by molecular finger prints and physical organic properties, respectively.
- Input file examples: for "UMAP_by_Finger_Prints.ipynb", use "other_smis.xlsx", "space_smis.xlsx", "test_smis.xlsx", and "training_smis.xlsx"; for "UMAP_by_Properties.ipynb", use "Properties_L.xlsx", "Properties_N.xlsx", "Properties_R.xlsx", and "Merged_Coms_Valid.xlsx"

4. Data source:

- 132,300 L-R-N combinations: The mechanism selectivity prediction result for the 132,300 ligand-radical-nucleophile combinations is shown in the file "Distinguished_Mechanisms_Sorted_SMILES-version.xlsx", where all chemicals are named in SMILES. Corresponding to Fig. 8 in the main text.
- Calculation Results under Different DFT Functionals: All DFT calculation results under different functionals are displayed in the file "Calculation_Results_under_Different_DFT_Functionals.xlsx". Corresponding to Supplementary Table 8 in the Supplementary Information.
- Solvent Effect: DFT calculation results with different solvents are displayed in the file "Solvent_Effect.xlsx". Corresponding to Supplementary Table 9 in the Supplementary Information.
- Optimal Reference for LFERs: The source data of evaluating the optimal reference for LFERs is displayed in the files "Evaluating_the_Optimal_Reference_RE", "Evaluating_the_Optimal_Reference_RS", and "Evaluating_the_Optimal_Reference_IP". Corresponding to Supplementary Table 17 in the Supplementary Information.
