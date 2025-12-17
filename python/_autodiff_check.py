# Template script to check the correctness of the autodiff implementation
# against hand-computed gradients and Hessians.

import torch

mu_ = 10
lambda_ = 20
alpha_ = 1 + (mu_/lambda_) - (mu_/4) * lambda_

def f(F):
    """
    Computes the value of the function:
    f(F) = (mu / 2) * (I_C - 3) + (lambda / 2) * (J - alpha)^2 - (mu / 2) * ln(I_C + 1)

    Parameters:
    - F (torch.Tensor): A 3x3 matrix.

    Returns:
    - torch.Tensor: The computed value of the function, scalar
    """
    # Ensure F is a torch tensor
    if not isinstance(F, torch.Tensor):
        raise ValueError("F must be a torch.Tensor")

    # Compute necessary terms
    J = torch.det(F)
    FtF = F.T @ F
    I_C = torch.trace(FtF)

    # Compute the function value
    term1 = (mu_ / 2) * (I_C - 3)
    term2 = (lambda_ / 2) * (J - alpha_)**2
    term3 = -(mu_ / 2) * torch.log(I_C + 1)

    return term1 + term2 + term3


def gradient_hand(F):
    """
    Computes the hand-computed gradient of the function.

    Gradient formula:
    grad_f(F) = mu * F + lambda * (J - alpha) * J * F^(-T) - (mu * F) / (I_C + 1)

    Parameters:
    - F (torch.Tensor): A 3x3 matrix.

    Returns:
    - torch.Tensor: The hand-computed gradient, 3x3 matrix.
    """
    # Compute necessary terms
    J = torch.det(F)
    FtF = F.T @ F
    I_C = torch.trace(FtF)

    # Compute individual gradient terms
    term1 = mu_ * F
    term2 = lambda_ * (J - alpha_) * J * torch.inverse(F).T
    term3 = -(mu_ * F) / (I_C + 1)

    # Combine terms
    return term1 + term2 + term3


def compute_M(F):
    """
    Computes the matrix M where:
    M_ij = -(F^(-T) E_j^T F^(-T))_i
    with j interpreted based on the flattened index of F.

    Parameters:
    - F (torch.Tensor): A 3x3 matrix.

    Returns:
    - torch.Tensor: A 9x9 matrix M.
    """
    # Compute the inverse transpose of F
    FinvT = torch.inverse(F).T

    # Initialize M as a 9x9 matrix
    M = torch.zeros(9, 9)
    
    # Loop over j (column index of M)
    for j in range(9):
        # Convert j to a (row, col) index in the 3x3 matrix
        row_j = j // 3
        col_j = j % 3

        # Define E_j: a matrix with a 1 at (row_j, col_j) and 0 elsewhere
        E_j = torch.zeros((3, 3))
        E_j[row_j, col_j] = 1.0

        # Compute the matrix product F^(-T) E_j^T F^(-T)
        term = FinvT @ E_j.T @ FinvT

        # Fill in the components of M corresponding to column j
        for i in range(9):
            # Convert i to (row, col) index
            row_i = i // 3
            col_i = i % 3

            # Assign the corresponding entry to M_ij
            M[i, j] = -term[row_i, col_i]

    return M
    

def hessian_hand(F):
    """
    Computes the hand-computed hessian of the function.

    Hessian formula:
    Hessian_f(F) = mu * I + 
                   (2J - alpha) * lambda * J * (F^(-T) outerproduct F^(-T)) + 
                   lambda * (J - alpha) * J * M + 
                   ((-mu I / (I_C + 1)) + (2 mu F outerproduct F) / (I_C + 1)^2))
                   
    Parameters:
    - F (torch.Tensor): A 3x3 matrix.

    Returns:
    - torch.Tensor: The hand-computed Hessian, 9 x 9 matrix.
    """
    
    # Compute necessary terms
    J = torch.det(F)                   # Determinant of F
    FtF = F.T @ F                      # F^T F
    I_C = torch.trace(FtF)             # Trace of F^T F

    # Identity matrix in 3D (9x9)
    I = torch.eye(9)

    # Term 1: mu * I (9x9)
    term1 = mu_ * I

    # Term 2: (2J - alpha) * lambda * J * (F^(-T) ⊗ F^(-T))
    FinvT = torch.inverse(F).T
    FinvT_outer = torch.outer(FinvT.flatten(), FinvT.flatten())
    term2 = (2 * J - alpha_) * lambda_ * J * FinvT_outer

    # Term 3: lambda * (J - alpha) * J * M
    M = compute_M(F)  # Define this function based on your instructions
    term3 = lambda_ * (J - alpha_) * J * M

    # Term 4: ((-mu * I) / (I_C + 1)) + (2 * mu * (F ⊗ F)) / (I_C + 1)^2
    F_outer = torch.outer(F.flatten(), F.flatten())
    term4 = ((-mu_ * I) / (I_C + 1)) + ((2 * mu_ * F_outer) / ((I_C + 1)**2))

    # Final Hessian
    hessian = term1 + term2 + term3 + term4
    return hessian


# Function to flatten the 4D autodiff Hessian into a 9x9 matrix
def flatten_autodiff_hessian(autodiff_hessian):
    """
    Flattens a 4D Hessian tensor of shape (3, 3, 3, 3) into a 9x9 matrix.
    Each 3x3 slice is flattened row-wise and inserted as a row in the matrix.

    Parameters:
    - autodiff_hessian (torch.Tensor): A 4D tensor of shape (3, 3, 3, 3).

    Returns:
    - torch.Tensor: A 9x9 flattened Hessian matrix.
    """
    # Reshape each 3x3 slice into a row of the 9x9 matrix
    rows = []
    for i in range(3):
        for j in range(3):
            # Flatten the (i, j) slice row-wise and append to the rows list
            rows.append(autodiff_hessian[i, j].reshape(-1))
    
    # Stack all rows to form the 9x9 matrix
    return torch.stack(rows)


# Generate a random invertible 3x3 matrix
def random_invertible_matrix():
    while True:
        matrix = torch.rand(3, 3)
        if torch.det(matrix) != 0:  # Ensure the matrix is invertible
            return matrix


# Generate a sample of 10 random invertible matrices
sample_matrices = [random_invertible_matrix() for _ in range(2)]

"""
# Compare gradients for each matrix
for i, matrix in enumerate(sample_matrices):
    matrix.requires_grad_(True)
    
    # Compute gradients
    hand_grad = gradient_hand(matrix)
    autodiff_grad = torch.autograd.functional.jacobian(f, matrix, create_graph=True)
    
    # Calculate error between gradients
    error = torch.norm(autodiff_grad - hand_grad)
    
    # Print the results
    print(f"Sample {i + 1}:")
    print("Matrix:\n", matrix)
    print("Hand-computed gradient:\n", hand_grad)
    print("Autodiff gradient:\n", autodiff_grad)
    print("Error between gradients:", error.item())
    print("-" * 50)
"""
    
# Compare Hessians for each matrix
for i, matrix in enumerate(sample_matrices):
    matrix.requires_grad_(True)
    
    # Compute gradients
    hand_hessian = hessian_hand(matrix)
    autodiff_hessian = torch.autograd.functional.jacobian(gradient_hand, matrix, create_graph=True)

    reshaped_autodiff_hessian = flatten_autodiff_hessian(autodiff_hessian)
  
    # Calculate error between gradients
    error = torch.abs(reshaped_autodiff_hessian - hand_hessian).max()
    
    # Print the results
    print(f"Sample {i + 1}:")
    print("Matrix:\n", matrix)
    print("Hand-computed Hessian:\n", hand_hessian)
    print("Autodiff Hessian:\n", reshaped_autodiff_hessian)
    print("Error between gradients:", error.item())
    print("-" * 50)