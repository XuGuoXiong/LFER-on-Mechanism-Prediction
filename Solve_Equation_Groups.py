import numpy as np
import pandas as pd
from scipy.optimize import leastsq

df = pd.read_excel("your_data.xlsx", engine="openpyxl") # Your data file name

# Retrieve the labels of the independent variables
ligand_labels = df["ligand"].dropna().unique()
radical_labels = df["radical"].dropna().unique()
nucleophile_labels = df["nucleophile"].dropna().unique()

# Create the coefficient matrix and the right-hand side vector
coefficient_matrix = []
G = []

# Populate the coefficient matrix and the right-hand side vector
for _, row in df.iterrows():
    if pd.notna(row["dG"]):
        i = np.where(ligand_labels == row["ligand"])[0][0]
        j = np.where(radical_labels == row["radical"])[0][0]
        k = np.where(nucleophile_labels == row["nucleophile"])[0][0]
        coefficients = [0] * (
            len(ligand_labels) + len(radical_labels) + len(nucleophile_labels)
        )
        coefficients[i] = 1
        coefficients[len(ligand_labels) + j] = 1
        coefficients[len(ligand_labels) + len(radical_labels) + k] = 1
        coefficient_matrix.append(coefficients)
        G.append(row["dG"])

coefficient_matrix = np.array(coefficient_matrix)
G = np.array(G)


# Define the residual function
def residuals(params, coefficient_matrix, G):
    return coefficient_matrix.dot(params) - G


# Initial guess
initial_guess = np.zeros(
    len(ligand_labels) + len(radical_labels) + len(nucleophile_labels)
)

# Solve using the leastsq function
solution, ier = leastsq(residuals, initial_guess, args=(coefficient_matrix, G))

# Check if the solution was successful
if ier in [1, 2, 3, 4]:
    results = pd.DataFrame(
        {
            "Ligand": [ligand_labels[i] for i in range(len(ligand_labels))]
            + [radical_labels[i] for i in range(len(radical_labels))]
            + [nucleophile_labels[i] for i in range(len(nucleophile_labels))],
            "Value": np.concatenate(
                (
                    solution[: len(ligand_labels)],
                    solution[
                        len(ligand_labels) : len(ligand_labels) + len(radical_labels)
                    ],
                    solution[-len(nucleophile_labels) :],
                )
            ),
        }
    )

    results.to_excel("solution_output.xlsx", index=False)
else:
    print("The least squares method did not converge to a solution.")
