import numpy as np
import time
import matplotlib.pyplot as plt

def sparsify_matrix(M, l_val, seed=None):
    """
    Element-wise sparsification of matrix M.
    
    Parameters:
    -----------
    M : ndarray - input matrix
    l_val : float - sparsity parameter (larger = denser)
    seed : int - random seed (optional)
    
    Returns:
    --------
    M_sparse : ndarray - sparsified matrix
    p_ij : ndarray - probabilities used
    nnz : int - number of non-zero elements
    """
    if seed is not None:
        np.random.seed(seed)
    
    m, n = M.shape
    
    # Compute Frobenius norm squared
    frob_norm_sq = np.sum(M**2)
    
    # Compute probabilities for each element
    p_ij = np.minimum(1.0, (l_val * M**2) / frob_norm_sq)
    
    # Create sparsified matrix
    M_sparse = np.zeros_like(M)
    
    # Generate random mask
    mask = np.random.random((m, n)) < p_ij
    
    # Keep and scale elements
    # Avoid division by zero
    p_safe = np.where(p_ij > 0, p_ij, 1.0)
    M_sparse[mask] = M[mask] / p_safe[mask]
    
    # Count non-zeros
    nnz = np.sum(mask)
    
    return M_sparse, p_ij, nnz


def matmul_alg_2(A, B, l, l_prime, seed=None):
    """
    Algorithm 2: Element-wise sparsification for matrix multiplication.
    
    Parameters:
    -----------
    A : ndarray (m x n)
    B : ndarray (n x p)
    l : float - sparsity parameter for A
    l_prime : float - sparsity parameter for B
    seed : int - random seed (optional)
    
    Returns:
    --------
    M : ndarray (m x p) - approximation of AB
    sparsity_A : float - fraction of non-zeros in A
    sparsity_B : float - fraction of non-zeros in B
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Sparsify A and B
    A_sparse, p_A, nnz_A = sparsify_matrix(A, l, seed)
    B_sparse, p_B, nnz_B = sparsify_matrix(B, l_prime, seed+1 if seed else None)
    
    # Compute sparsity (fraction of non-zeros)
    sparsity_A = nnz_A / A.size
    sparsity_B = nnz_B / B.size
    
    # Multiply sparsified matrices
    M = A_sparse @ B_sparse
    
    return M, sparsity_A, sparsity_B


def compare_algorithms(A, B, c_values, l_values, n_trials=5, seed=42):
    """
    Compare Algorithm 1 (column/row sampling) and Algorithm 2 (element-wise).
    """
    np.random.seed(seed)
    
    # Compute exact product
    print("Computing exact product...")
    C_exact = A @ B
    exact_norm = np.linalg.norm(C_exact, 'fro')
    
    # Precompute probabilities for Algorithm 1
    norms_A = np.linalg.norm(A, axis=0)
    norms_B = np.linalg.norm(B, axis=1)
    p_probs = norms_A * norms_B
    p_probs = p_probs / np.sum(p_probs)
    
    results = {
        'alg1_c': [], 'alg1_error': [], 'alg1_time': [],
        'alg2_l': [], 'alg2_error': [], 'alg2_time': [],
        'alg2_sparsity_A': [], 'alg2_sparsity_B': []
    }
    
    print("\n" + "="*80)
    print("ALGORITHM 1: Column/Row Sampling")
    print("="*80)
    print(f"{'c':>8} | {'Rel. Error':>12} | {'Time (s)':>10}")
    print("-"*80)
    
    for c in c_values:
        errors = []
        times = []
        
        for trial in range(n_trials):
            t0 = time.time()
            
            # Algorithm 1
            m, n, p = A.shape[0], A.shape[1], B.shape[1]
            M = np.zeros((m, p))
            sampled_indices = np.random.choice(n, size=c, p=p_probs)
            
            for k in sampled_indices:
                outer_prod = A[:, k:k+1] @ B[k:k+1, :]
                M += outer_prod / (c * p_probs[k])
            
            elapsed = time.time() - t0
            rel_error = np.linalg.norm(C_exact - M, 'fro') / exact_norm
            errors.append(rel_error)
            times.append(elapsed)
        
        mean_error = np.mean(errors)
        mean_time = np.mean(times)
        
        results['alg1_c'].append(c)
        results['alg1_error'].append(mean_error)
        results['alg1_time'].append(mean_time)
        
        print(f"{c:8d} | {mean_error:12.4e} | {mean_time:10.4f}")
    
    print("\n" + "="*80)
    print("ALGORITHM 2: Element-wise Sparsification")
    print("="*80)
    print(f"{'l':>8} | {'Rel. Error':>12} | {'Time (s)':>10} | {'Sparsity A':>12} | {'Sparsity B':>12}")
    print("-"*80)
    
    for l in l_values:
        errors = []
        times = []
        sparsity_A_list = []
        sparsity_B_list = []
        
        for trial in range(n_trials):
            t0 = time.time()
            
            # Algorithm 2
            M, sp_A, sp_B = matmul_alg_2(A, B, l, l, seed+trial)
            
            elapsed = time.time() - t0
            rel_error = np.linalg.norm(C_exact - M, 'fro') / exact_norm
            errors.append(rel_error)
            times.append(elapsed)
            sparsity_A_list.append(sp_A)
            sparsity_B_list.append(sp_B)
        
        mean_error = np.mean(errors)
        mean_time = np.mean(times)
        mean_sp_A = np.mean(sparsity_A_list)
        mean_sp_B = np.mean(sparsity_B_list)
        
        results['alg2_l'].append(l)
        results['alg2_error'].append(mean_error)
        results['alg2_time'].append(mean_time)
        results['alg2_sparsity_A'].append(mean_sp_A)
        results['alg2_sparsity_B'].append(mean_sp_B)
        
        print(f"{l:8.1f} | {mean_error:12.4e} | {mean_time:10.4f} | {mean_sp_A:12.4f} | {mean_sp_B:12.4f}")
    
    return results, C_exact


def plot_comparison(results, C_exact, save_path=None):
    """Plot comparison of Algorithm 1 and Algorithm 2."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Error vs parameter (c or l)
    ax1 = axes[0, 0]
    ax1.semilogx(results['alg1_c'], results['alg1_error'], 'bo-', 
                 label='Alg 1 (Column/Row)', markersize=8, linewidth=2)
    ax1.semilogx(results['alg2_l'], results['alg2_error'], 'rs-', 
                 label='Alg 2 (Element-wise)', markersize=8, linewidth=2)
    ax1.set_xlabel('Parameter (c for Alg1, l for Alg2)', fontsize=12)
    ax1.set_ylabel('Relative Error', fontsize=12)
    ax1.set_title('Accuracy Comparison', fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Plot 2: Time vs parameter
    ax2 = axes[0, 1]
    ax2.semilogx(results['alg1_c'], results['alg1_time'], 'bo-', 
                 label='Alg 1 (Column/Row)', markersize=8, linewidth=2)
    ax2.semilogx(results['alg2_l'], results['alg2_time'], 'rs-', 
                 label='Alg 2 (Element-wise)', markersize=8, linewidth=2)
    ax2.set_xlabel('Parameter (c for Alg1, l for Alg2)', fontsize=12)
    ax2.set_ylabel('Execution Time (s)', fontsize=12)
    ax2.set_title('Speed Comparison', fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    # Plot 3: Error vs Time (trade-off)
    ax3 = axes[1, 0]
    ax3.loglog(results['alg1_time'], results['alg1_error'], 'bo-', 
               label='Alg 1', markersize=8, linewidth=2)
    ax3.loglog(results['alg2_time'], results['alg2_error'], 'rs-', 
               label='Alg 2', markersize=8, linewidth=2)
    ax3.set_xlabel('Execution Time (s)', fontsize=12)
    ax3.set_ylabel('Relative Error', fontsize=12)
    ax3.set_title('Accuracy-Time Trade-off', fontsize=13)
    ax3.grid(True, which='both', alpha=0.3)
    ax3.legend(fontsize=10)
    
    # Plot 4: Sparsity vs l (for Algorithm 2)
    ax4 = axes[1, 1]
    ax4.semilogx(results['alg2_l'], results['alg2_sparsity_A'], 'g^-', 
                 label='Matrix A', markersize=8, linewidth=2)
    ax4.semilogx(results['alg2_l'], results['alg2_sparsity_B'], 'mv-', 
                 label='Matrix B', markersize=8, linewidth=2)
    ax4.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Dense (100%)')
    ax4.set_xlabel('Sparsity Parameter l', fontsize=12)
    ax4.set_ylabel('Fraction of Non-Zeros', fontsize=12)
    ax4.set_title('Algorithm 2: Sparsity Control', fontsize=13)
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=10)
    
    plt.suptitle('Algorithm 1 vs Algorithm 2 Comparison', fontsize=15, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    
    plt.show()


def main():
    """Main experiment."""
    print("="*80)
    print("COMPARING ALGORITHM 1 vs ALGORITHM 2")
    print("="*80)
    
    # Configuration
    np.random.seed(42)
    m, n, p = 500, 500, 500  # Smaller for faster comparison
    
    print(f"\nMatrix dimensions: A({m}×{n}), B({n}×{p})\n")
    
    # Generate matrices
    A = np.random.randn(m, n).astype(np.float64)
    B = np.random.randn(n, p).astype(np.float64)
    
    # Parameters to test
    c_values = [50, 100, 200, 300, 400, 500]
    l_values = [10, 25, 50, 75, 100, 150, 200]
    
    print("Running experiments...")
    results, C_exact = compare_algorithms(A, B, c_values, l_values, n_trials=3)
    
    # Plot comparison
    print("\nGenerating plots...")
    plot_comparison(results, C_exact, save_path='alg1_vs_alg2_comparison.png')
    
    print("\n" + "="*80)
    print("Experiment complete!")
    print("="*80)


if __name__ == "__main__":
    main()