import numpy as np

# ============================================================
# HGP utilities (binary 0/1) + non-binary exponent solver + LaTeX exporter
# Unified random seed for full reproducibility across runs.
# ============================================================

GLOBAL_SEED = 424242
np.random.seed(GLOBAL_SEED)

from itertools import product  # (kept for future use / parity w/ original code)


# -----------------------------
# Linear-algebra helpers (GF(2))
# -----------------------------
def nullspace_mod2(H: np.ndarray) -> np.ndarray:
    """
    Compute a basis of the nullspace over GF(2):
        { x in {0,1}^n : H x^T = 0 (mod 2) }.

    Returns
    -------
    G : (k, n) int ndarray in {0,1}
        Each row is a basis vector of the nullspace (free-variable parametrization).
    """
    m, n = H.shape
    H = H.copy() % 2
    A = np.concatenate((H, np.eye(m, dtype=int)), axis=1)
    pivots = []
    row = 0
    for col in range(n):
        pivot = np.argmax(A[row:, col]) + row
        if A[pivot, col] == 0:
            continue
        A[[row, pivot]] = A[[pivot, row]]
        for i in range(m):
            if i != row and A[i, col] == 1:
                A[i, :] ^= A[row, :]
        pivots.append(col)
        row += 1
        if row == m:
            break
    free_cols = [c for c in range(n) if c not in pivots]
    G = np.zeros((len(free_cols), n), dtype=int)
    for idx, free in enumerate(free_cols):
        G[idx, free] = 1
        for r, c in enumerate(pivots):
            if A[r, free] == 1:
                G[idx, c] = 1
    return G % 2  # :contentReference[oaicite:1]{index=1}


def rowspace_mod2(H: np.ndarray) -> np.ndarray:
    """
    Compute a basis of the row space over GF(2) using simple row elimination.

    Returns
    -------
    R : (r, n) int ndarray in {0,1}
        Independent row vectors spanning the row space.
    """
    H = H.copy() % 2
    R = []
    for row in H:
        # Reduce against already collected rows
        for r in R:
            if (row[r != 0] == r[r != 0]).all():
                row ^= r
        if np.any(row):
            R.append(row)
    return np.array(R, dtype=int)  # :contentReference[oaicite:2]{index=2}


# ------------------------------------------------------------
# CSS coset representatives: CX/CZ^⊥ and CZ/CX^⊥ (binary case)
# ------------------------------------------------------------
def enumerate_coset_basis(HX: np.ndarray, HZ: np.ndarray):
    """
    Enumerate coset basis representatives of CX/CZ^⊥ and CZ/CX^⊥.

    Notes
    -----
    - CX = Null(HX), CZ = Null(HZ) (both over GF(2)).
    - CZ^⊥ = RowSpace(HZ), CX^⊥ = RowSpace(HX).
    """
    CX = nullspace_mod2(HX)
    CZ = nullspace_mod2(HZ)
    CZp = rowspace_mod2(HZ)
    CXp = rowspace_mod2(HX)

    def span_contains(B, v):
        if B.size == 0:
            return False
        M = np.vstack([B, v]) % 2
        return np.linalg.matrix_rank(B % 2) == np.linalg.matrix_rank(M % 2)

    reps_X = [x for x in CX if not span_contains(CZp, x)]
    reps_Z = [z for z in CZ if not span_contains(CXp, z)]

    print("\n=== C_X / C_Z^⊥ ===")
    for v in reps_X:
        print(v)
    print("\n=== C_Z / C_X^⊥ ===")
    for v in reps_Z:
        print(v)
    return np.array(reps_X, dtype=int), np.array(reps_Z, dtype=int)  # :contentReference[oaicite:3]{index=3}


def compute_generator_matrix(H: np.ndarray) -> np.ndarray:
    """
    Compute a generator matrix G (over GF(2)) such that H G^T = 0 (mod 2).

    Returns
    -------
    G : (k, n) int ndarray in {0,1}
        Each row is a generator; k is the nullity of H over GF(2).
    """
    H = H.copy().astype(int) % 2
    m, n = H.shape
    A = np.concatenate((H, np.eye(m, dtype=int)), axis=1) % 2

    pivots = []
    row = 0
    for col in range(n):
        pivot = np.argmax(A[row:, col]) + row
        if A[pivot, col] == 0:
            continue
        A[[row, pivot]] = A[[pivot, row]]
        for i in range(m):
            if i != row and A[i, col] == 1:
                A[i, :] ^= A[row, :]
        pivots.append(col)
        row += 1
        if row == m:
            break

    free_cols = [c for c in range(n) if c not in pivots]
    k = len(free_cols)
    if k == 0:
        return np.zeros((0, n), dtype=int)

    G = np.zeros((k, n), dtype=int)
    for idx, free in enumerate(free_cols):
        G[idx, free] = 1
        for r, c in enumerate(pivots):
            if A[r, free] == 1:
                G[idx, c] = 1
    return G % 2  # :contentReference[oaicite:4]{index=4}


# -------------------------------------------------------------
# 4-term congruence constraints for non-binary exponent assignment
# -------------------------------------------------------------
def enumerate_cd_equations(
    binary_HX: np.ndarray,
    binary_HZ: np.ndarray,
    q_minus_1: int,
    *,
    to_latex: bool = False
) -> list[str]:
    """
    Enumerate 4-term equations of the form
        c(i,j) - c(i,j') + d(i',j) - d(i',j') ≡ 0 (mod q-1)
    for pairs of rows (i from HX, i' from HZ) that share exactly two columns.

    Returns
    -------
    eqs : list[str]
        Human-readable (or LaTeX) strings describing the constraints.
    """
    eqs = []
    m_x, n = binary_HX.shape
    m_z, _ = binary_HZ.shape

    print("\n=== Enumeration of 4-term equations in c,d ===")
    for i in range(m_x):
        hx_cols = np.where(binary_HX[i] == 1)[0]
        for ip in range(m_z):
            hz_cols = np.where(binary_HZ[ip] == 1)[0]
            common = np.intersect1d(hx_cols, hz_cols)
            if len(common) == 2:
                j, jp = int(common[0]), int(common[1])
                if not to_latex:
                    s = (
                        f"c({i},{j}) - c({i},{jp}) + d({ip},{j}) - d({ip},{jp}) "
                        f"≡ 0  (mod {q_minus_1})"
                    )
                else:
                    s = (
                        r"c(" + f"{i},{j}" + r")"
                        r" - c(" + f"{i},{jp}" + r")"
                        r" + d(" + f"{ip},{j}" + r")"
                        r" - d(" + f"{ip},{jp}" + r")"
                        r"\equiv 0~(\bmod~" + f"{q_minus_1}" + r")"
                    )
                print(s)
                eqs.append(s)
    if not eqs:
        print("No pairs with exactly two common columns found. No equations generated.")
    return eqs  # :contentReference[oaicite:5]{index=5}


# ---------------------------------------
# Lightweight real-valued RREF (for A * x)
# ---------------------------------------
def custom_rref_solver(A: np.ndarray):
    """
    Compute a reduced row echelon form (RREF) basis for the homogeneous system A x = 0.

    Notes
    -----
    - Uses float arithmetic but guards that the resulting entries are close to {0, 1, -1}.
    - Returns a list of basis vectors for the nullspace of A (over reals, then rounded).
    - Intended only as a helper to produce one valid integer solution pattern for exponents.
    """
    if A.size == 0:
        return []
    M, N = A.shape
    A_rref = np.array(A, dtype=float)
    pivot_row = 0
    pivot_cols = []

    print("\n--- RREF (real-valued) start ---")
    for j in range(N):
        if pivot_row >= M:
            break
        i = pivot_row
        while i < M and A_rref[i, j] == 0:
            i += 1
        if i < M:
            pivot_cols.append(j)
            A_rref[[i, pivot_row]] = A_rref[[pivot_row, i]]
            pv = A_rref[pivot_row, j]
            if pv != 1:
                A_rref[pivot_row, :] /= pv
            for ii in range(M):
                if ii != pivot_row:
                    A_rref[ii, :] -= A_rref[ii, j] * A_rref[pivot_row, :]
            if not np.all(np.isin(np.round(A_rref), [0, 1, -1])):
                print(f"❌ Stop: Values outside {{0,1,-1}} appeared in column {j}")
                print("Current matrix:\n", A_rref)
                return None
            pivot_row += 1
    print("✅ RREF finished (entries ~ in {0,1,-1}).")
    print("Final RREF:\n", A_rref)

    free_cols = [j for j in range(N) if j not in pivot_cols]
    basis = []
    for free in free_cols:
        v = np.zeros(N)
        v[free] = 1
        for i, pc in enumerate(pivot_cols):
            v[pc] = -A_rref[i, free]
        basis.append(np.round(v).astype(int))
    return basis  # :contentReference[oaicite:6]{index=6}


# ----------------------------------------
# Random utilities and HGP matrix builder
# ----------------------------------------
def generate_random_matrix_no_zero_cols(m: int, n: int) -> np.ndarray:
    """
    Generate an m x n random binary matrix with no all-zero columns.
    """
    if m <= 0 or n <= 0:
        raise ValueError("Matrix dimensions (m, n) must be positive.")
    A = np.random.randint(0, 2, size=(m, n), dtype=np.uint8)
    col_sums = A.sum(axis=0)
    for j in np.where(col_sums == 0)[0]:
        i = int(np.random.randint(0, m))
        A[i, j] = 1
    return A & 1  # :contentReference[oaicite:7]{index=7}


def generate_hgp_matrices(H1: np.ndarray, H2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct (HX, HZ) via the hypergraph product (binary case).

    Shapes
    ------
    H1: (m1, n1), H2: (m2, n2)
    HX: (m1*n2 + m2*n1, n1*n2 + m1*m2)
    HZ: (n1*m2 + n2*m1, n1*n2 + m1*m2)
    """
    H1 = (np.asarray(H1, dtype=np.uint8) & 1)
    H2 = (np.asarray(H2, dtype=np.uint8) & 1)
    m1, n1 = H1.shape
    m2, n2 = H2.shape
    I_n1 = np.eye(n1, dtype=np.uint8)
    I_m1 = np.eye(m1, dtype=np.uint8)
    I_n2 = np.eye(n2, dtype=np.uint8)
    I_m2 = np.eye(m2, dtype=np.uint8)
    HX = (np.hstack((np.kron(H1, I_n2), np.kron(I_m1, H2.T))) & 1).astype(np.uint8)
    HZ = (np.hstack((np.kron(I_n1, H2), np.kron(H1.T, I_m2))) & 1).astype(np.uint8)
    return HX, HZ  # :contentReference[oaicite:8]{index=8}


# -------------------------------------------------------------------
# Solve non-binary exponents (c,d) modulo q-1 from the 4-term constraints
# -------------------------------------------------------------------
def solve_exponents_by_congruence(
    binary_HX: np.ndarray,
    binary_HZ: np.ndarray,
    q_minus_1: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve for exponent tables C_gamma (for HX entries) and D_delta (for HZ entries)
    that satisfy the 4-term congruences modulo (q-1).

    Returns
    -------
    (Cg, Dd) : pair of int ndarrays shaped like HX and HZ, filled with -1 where not used.
    """
    m_x, n = binary_HX.shape
    m_z, _ = binary_HZ.shape
    hx_idx = list(zip(*np.where(binary_HX != 0)))
    hz_idx = list(zip(*np.where(binary_HZ != 0)))
    num_c = len(hx_idx)
    num_d = len(hz_idx)
    num_total = num_c + num_d

    # Variable mapping: ('c', i, j) or ('d', i', j) -> column index in A
    var_map = {('c', r, c): i for i, (r, c) in enumerate(hx_idx)}
    var_map.update({('d', r, c): i + num_c for i, (r, c) in enumerate(hz_idx)})

    # Build linear system A * z = 0  (over integers, later mod (q-1))
    rows = []
    for i in range(m_x):
        hx_cols = np.where(binary_HX[i] == 1)[0]
        for ip in range(m_z):
            hz_cols = np.where(binary_HZ[ip] == 1)[0]
            common = np.intersect1d(hx_cols, hz_cols)
            if len(common) == 2:
                j, jp = int(common[0]), int(common[1])
                row = np.zeros(num_total, dtype=int)
                row[var_map[('c', i, j)]] = 1
                row[var_map[('c', i, jp)]] = -1
                row[var_map[('d', ip, j)]] = 1
                row[var_map[('d', ip, jp)]] = -1
                rows.append(row)

    if not rows:
        print("No constraints found.")
        return np.full(binary_HX.shape, -1, dtype=int), np.full(binary_HZ.shape, -1, dtype=int)

    A = np.array(rows, dtype=int)
    basis = custom_rref_solver(A)
    if basis is None:
        print("Failed to obtain a valid basis for the constraints.")
        return np.full(binary_HX.shape, -1, dtype=int), np.full(binary_HZ.shape, -1, dtype=int)

    # Pick one random linear combination of basis vectors (mod q-1)
    if len(basis) == 0:
        sol = np.zeros(num_total, dtype=int)
    else:
        B = np.array(basis, dtype=int)
        coeff = np.random.randint(0, q_minus_1, size=B.shape[0])
        print(f"Random coefficients used for combining basis vectors: {coeff}")
        sol = (coeff @ B) % q_minus_1

    # Scatter back into Cg, Dd
    Cg = np.full(binary_HX.shape, -1, dtype=int)
    Dd = np.full(binary_HZ.shape, -1, dtype=int)
    for i, (r, c) in enumerate(hx_idx):
        Cg[r, c] = int(sol[i])
    for i, (r, c) in enumerate(hz_idx):
        Dd[r, c] = int(sol[i + num_c])
    return Cg, Dd  # :contentReference[oaicite:9]{index=9}


# -------------------------------------------------
# Sanity checks: binary commutation & NB constraints
# -------------------------------------------------
def verify_binary_commutation(HX: np.ndarray, HZ: np.ndarray) -> bool:
    """
    Check HX * HZ^T has only even overlaps (mod 2), i.e., CSS commutation holds.
    """
    overlap = HX @ HZ.T
    comm = overlap & 1
    odd = int(np.count_nonzero(comm))
    print("\n=== Binary commutation check ===")
    if odd == 0:
        print("✔ OK: Commuting (mod 2).")
        return True
    else:
        print(f"✖ FAIL: {odd} odd-overlap entries detected.")
        return False  # :contentReference[oaicite:10]{index=10}


def verify_nb_constraints(
    HX: np.ndarray, HZ: np.ndarray, Cg: np.ndarray, Dd: np.ndarray, q_minus_1: int
):
    """
    Verify the 4-term non-binary congruence constraints for all relevant pairs.

    Returns
    -------
    (checks, viol) : (int, int)
        Number of checked constraints and number of violations.
    """
    checks = 0
    viol = 0
    for i in range(HX.shape[0]):
        hx_cols = np.where(HX[i] == 1)[0]
        for ip in range(HZ.shape[0]):
            hz_cols = np.where(HZ[ip] == 1)[0]
            common = np.intersect1d(hx_cols, hz_cols)
            if len(common) == 2:
                j, jp = int(common[0]), int(common[1])
                if Cg[i, j] < 0 or Cg[i, jp] < 0 or Dd[ip, j] < 0 or Dd[ip, jp] < 0:
                    continue
                val = (Cg[i, j] - Cg[i, jp] + Dd[ip, j] - Dd[ip, jp]) % q_minus_1
                checks += 1
                if val != 0:
                    viol += 1
    print(f"\n=== Non-binary 4-term check ===\n#Checked: {checks}, #Violations: {viol}")
    return checks, viol  # :contentReference[oaicite:11]{index=11}


# ----------------------
# Minimal LaTeX exporter
# ----------------------
def _bmatrix_latex(A: np.ndarray) -> str:
    """
    Format a binary matrix as a LaTeX bmatrix (no alignment or spacing tweaks).
    """
    rows = [" ".join(str(int(x)) for x in r) for r in A]
    body = r"\\".join(rows)
    return r"\begin{bmatrix}" + body + r"\end{bmatrix}"  # :contentReference[oaicite:12]{index=12}


def export_latex_example(filename="example_output.tex", *, seed=None, q_minus_1=255):
    """
    Emit a tiny LaTeX snippet that documents one reproducible toy run.
    """
    if seed is not None:
        np.random.seed(seed)

    M1, N1 = 2, 3
    M2, N2 = 2, 3
    # H1 = generate_random_matrix_no_zero_cols(M1, N1)
    # H2 = generate_random_matrix_no_zero_cols(M2, N2)
    H1 = np.array([[1, 1, 1],
                   [0, 1, 0]], dtype=np.uint8)
    H2 = np.array([[1, 0, 0],
                   [1, 1, 1]], dtype=np.uint8)

    HX, HZ = generate_hgp_matrices(H1, H2)
    ok_bin = verify_binary_commutation(HX, HZ)
    Cg, Dd = solve_exponents_by_congruence(HX, HZ, q_minus_1)
    checks, viol = verify_nb_constraints(HX, HZ, Cg, Dd, q_minus_1)

    lines = [
        "% Auto-generated by unified-seed HGP generator",
        f"% seed={GLOBAL_SEED}",
        r"\begin{example}[Reproducible toy run]",
        "Binary inputs $H_1, H_2$:",
        "$H_1= " + _bmatrix_latex(H1) + ",\\quad H_2= " + _bmatrix_latex(H2) + "$.",
        "Binary HGP matrices:",
        "$H_X= " + _bmatrix_latex(HX) + ",\\quad H_Z= " + _bmatrix_latex(HZ) + "$.",
        "Binary commutation: " + (r"\textbf{OK}" if ok_bin else r"\textbf{FAIL}") + ".",
    ]

    eq_lines = enumerate_cd_equations(HX, HZ, q_minus_1, to_latex=True)
    if eq_lines:
        lines.append(r"\begin{align*}")
        for s in eq_lines:
            lines.append(s + r"\\")
        lines.append(r"\end{align*}")
    else:
        lines.append("No 4-term equations generated.")

    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"LaTeX example written to: {filename}")  # :contentReference[oaicite:13]{index=13}


# ============================================================
#  SMITH NORMAL FORM–BASED NON-BINARY EXPONENT SOLVER
#  (for equations A z ≡ 0 mod (q-1))
# ============================================================


# ------------------------------------------------------------
# 1.  Extended GCD and Modular Arithmetic Helpers
# ------------------------------------------------------------
def _egcd(a: int, b: int):
    """Extended Euclidean algorithm: returns (g, x, y) with ax + by = g = gcd(a,b)."""
    if b == 0:
        return (abs(a), 1 if a >= 0 else -1, 0)
    g, x1, y1 = _egcd(b, a % b)
    return (g, y1, x1 - (a // b) * y1)


def _inv_mod(a: int, mod: int) -> int | None:
    """Return the modular inverse of a modulo mod, or None if not invertible."""
    a %= mod
    g, x, _ = _egcd(a, mod)
    if g != 1:
        return None
    return x % mod


def _sign(x: int) -> int:
    """Return sign(x) as ±1 (used in Smith normalization)."""
    return -1 if x < 0 else 1



# ------------------------------------------------------------
# 2.  Smith Normal Form Computation over Integers
# ------------------------------------------------------------
def smith_normal_form(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Safe Smith Normal Form (U*A*V = S) over integers.

    Handles zero pivots gracefully (avoids divide-by-zero warnings).
    U, V are unimodular (det = ±1).
    """
    S = np.array(A, dtype=int).copy()
    m, n = S.shape
    U = np.eye(m, dtype=int)
    V = np.eye(n, dtype=int)
    i = j = 0

    while i < m and j < n:
        # --- Step 1: Find smallest nonzero element ---
        piv = None
        minval = None
        for r in range(i, m):
            for c in range(j, n):
                val = abs(S[r, c])
                if val != 0 and (minval is None or val < minval):
                    minval = val
                    piv = (r, c)
        if piv is None:
            break  # all remaining entries are zero → done

        pr, pc = piv
        if pr != i:
            S[[i, pr]] = S[[pr, i]]; U[[i, pr]] = U[[pr, i]]
        if pc != j:
            S[:, [j, pc]] = S[:, [pc, j]]; V[:, [j, pc]] = V[:, [pc, j]]

        # --- Step 2: ensure pivot positive ---
        if S[i, j] < 0:
            S[i, :] *= -1
            U[i, :] *= -1

        # If pivot is 0 (rare, but possible after swaps), skip
        if S[i, j] == 0:
            j += 1
            continue

        # --- Step 3: eliminate other rows (safe divide) ---
        for r in range(m):
            if r == i:
                continue
            while S[r, j] != 0 and S[i, j] != 0:
                q = S[r, j] // S[i, j]
                S[r, :] -= q * S[i, :]
                U[r, :] -= q * U[i, :]
                # If new remainder smaller than pivot, swap to maintain gcd shrinkage
                if abs(S[r, j]) < abs(S[i, j]) and S[r, j] != 0:
                    S[[i, r]] = S[[r, i]]
                    U[[i, r]] = U[[r, i]]

        # --- Step 4: eliminate other columns (safe divide) ---
        for c in range(n):
            if c == j:
                continue
            while S[i, c] != 0 and S[i, j] != 0:
                q = S[i, c] // S[i, j]
                S[:, c] -= q * S[:, j]
                V[:, c] -= q * V[:, j]
                if abs(S[i, c]) < abs(S[i, j]) and S[i, c] != 0:
                    S[:, [j, c]] = S[:, [c, j]]
                    V[:, [j, c]] = V[:, [c, j]]

        i += 1
        j += 1

    return U, S, V



# ------------------------------------------------------------
# 3.  Nullspace Computation over Z_mod via Smith Form
# ------------------------------------------------------------
def nullspace_mod_via_smith(A: np.ndarray, mod: int) -> np.ndarray:
    """
    Compute the right nullspace of A over Z_mod using its Smith Normal Form.

    Given U*A*V = S = diag(d1,...,dr,0,...,0), solve S w ≡ 0 mod mod, and
    then transform back z = V w (mod mod).

    Returns
    -------
    N : ndarray (k × n)
        Each row is one basis vector (mod mod) of the solution space of A z ≡ 0.
    """
    if A.size == 0:
        return np.zeros((0, 0), dtype=int)

    U, S, V = smith_normal_form(A)
    S = np.array(S, dtype=int)
    m, n = S.shape
    diag = [S[i, i] for i in range(min(m, n))]

    # Determine effective rank (number of nonzero diagonal elements)
    eff_rank = sum(1 for d in diag if d != 0)

    gens_w = []
    # --- For each diagonal constraint d_i * w_i ≡ 0 mod m ---
    for i in range(eff_rank):
        d = abs(S[i, i])
        g = np.gcd(d, mod)
        step = (mod // g) % mod
        if step % mod != 0:  # freedom when gcd(d,mod)<mod
            w = np.zeros(n, dtype=int)
            w[i] = step
            gens_w.append(w)
    # --- Remaining free variables (w_i = 1) ---
    for j in range(eff_rank, n):
        w = np.zeros(n, dtype=int)
        w[j] = 1
        gens_w.append(w)

    if not gens_w:
        return np.zeros((0, n), dtype=int)

    # --- Transform back to z = V * w (mod mod) ---
    V = np.array(V, dtype=int)
    gens_z = []
    for w in gens_w:
        z = (V @ w) % mod
        gens_z.append(z)

    return np.array(gens_z, dtype=int)



# ------------------------------------------------------------
# 4.  Solve Exponent Assignment Using Smith Normal Form
# ------------------------------------------------------------
def solve_exponents_by_smith(binary_HX: np.ndarray,
                             binary_HZ: np.ndarray,
                             q_minus_1: int,
                             *,
                             print_solution: bool = True,
                             max_print_lines: int = 50
                             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve for exponent tables (C_gamma, D_delta) in Z_{q-1}
    using the Smith-form nullspace of the constraint matrix.

    Returns
    -------
    (C_gamma, D_delta, sol) :
        - C_gamma : shaped like HX, -1 where unused
        - D_delta : shaped like HZ, -1 where unused
        - sol     : 1-D vector of length (#nonzero(HX) + #nonzero(HZ))
                    concatenated as [C-part | D-part] (values mod q-1)
    """
    m_x, n = binary_HX.shape
    m_z, _ = binary_HZ.shape
    hx_idx = list(zip(*np.where(binary_HX != 0)))
    hz_idx = list(zip(*np.where(binary_HZ != 0)))
    num_c = len(hx_idx)
    num_d = len(hz_idx)
    num_total = num_c + num_d

    var_map = {('c', r, c): i for i, (r, c) in enumerate(hx_idx)}
    var_map.update({('d', r, c): i + num_c for i, (r, c) in enumerate(hz_idx)})

    rows = []
    for i in range(m_x):
        hx_cols = np.where(binary_HX[i] == 1)[0]
        for ip in range(m_z):
            hz_cols = np.where(binary_HZ[ip] == 1)[0]
            common = np.intersect1d(hx_cols, hz_cols)
            if len(common) == 2:
                j, jp = int(common[0]), int(common[1])
                row = np.zeros(num_total, dtype=int)
                row[var_map[('c', i, j)]]  =  1
                row[var_map[('c', i, jp)]] = -1
                row[var_map[('d', ip, j)]] =  1
                row[var_map[('d', ip, jp)]] = -1
                rows.append(row)

    if not rows:
        print("No constraints found (no 2-overlap pairs).")
        Cg = np.full(binary_HX.shape, -1, dtype=int)
        Dd = np.full(binary_HZ.shape, -1, dtype=int)
        sol = np.zeros(num_total, dtype=int)
        return Cg, Dd, sol

    A = np.array(rows, dtype=int)

    # Nullspace over Z_{q-1} via Smith form
    N = nullspace_mod_via_smith(A, q_minus_1)

    # Choose one concrete solution
    if N.size == 0:
        sol = np.zeros(num_total, dtype=int)
    else:
        coeff = np.random.randint(0, q_minus_1, size=N.shape[0])
        sol = (coeff @ N) % q_minus_1

    # Scatter into C_gamma and D_delta
    Cg = np.full(binary_HX.shape, -1, dtype=int)
    Dd = np.full(binary_HZ.shape, -1, dtype=int)
    for i, (r, c) in enumerate(hx_idx):
        Cg[r, c] = int(sol[i])
    for i, (r, c) in enumerate(hz_idx):
        Dd[r, c] = int(sol[i + num_c])

    # Optional on-screen print
    if print_solution:
        print_nb_solution(sol, q_minus_1, hx_idx, hz_idx, max_lines=max_print_lines)

    return Cg, Dd, sol

# ------------------------------------------------------------
# 5.  Pretty-print helpers for the non-binary solution
# ------------------------------------------------------------
def print_nb_solution(
    sol: np.ndarray,
    q_minus_1: int,
    hx_idx: list[tuple[int, int]],
    hz_idx: list[tuple[int, int]],
    *,
    max_lines: int = 50
):
    """
    Nicely print one concrete solution vector `sol` modulo (q-1),
    with its mapping to C_gamma (HX positions) and D_delta (HZ positions).

    Parameters
    ----------
    sol : shape (num_c + num_d,)
        Linearized solution [c-variables..., d-variables...] (mod q-1).
    q_minus_1 : int
        Modulus (q-1).
    hx_idx : list[(i,j)]
        Locations of 1s in HX (row i, col j) → mapped to C_gamma[i,j].
    hz_idx : list[(i',j)]
        Locations of 1s in HZ (row i', col j) → mapped to D_delta[i',j].
    max_lines : int
        Maximum number of lines to print (to avoid overly long logs).
    """
    print("\n=== Non-binary exponent solution (one instance) ===")
    print(f"modulus = {q_minus_1}, vector length = {len(sol)}")

    lines = []
    # C_gamma part
    for k, (i, j) in enumerate(hx_idx):
        val = int(sol[k] % q_minus_1)
        lines.append(f"C_gamma[{i},{j}] = {val}")
    # D_delta part
    offset = len(hx_idx)
    for t, (ip, j) in enumerate(hz_idx):
        val = int(sol[offset + t] % q_minus_1)
        lines.append(f"D_delta[{ip},{j}] = {val}")

    # Possibly truncate for safety in very large instances
    if len(lines) > max_lines:
        head = lines[: max_lines // 2]
        tail = lines[-max_lines // 2 :]
        for s in head: print(s)
        print(f"... (omitted {len(lines) - len(head) - len(tail)} lines) ...")
        for s in tail: print(s)
    else:
        for s in lines: print(s)
# ------------------------------------------------------------
# 5.  Pretty-print helpers for the non-binary solution
#      (with explicit c(i,j), d(i',j) style)
# ------------------------------------------------------------
def print_nb_solution_cd(
    sol: np.ndarray,
    q_minus_1: int,
    hx_idx: list[tuple[int, int]],
    hz_idx: list[tuple[int, int]],
    *,
    max_lines: int = 100
):
    """
    Nicely print the solution values for all c(i,j) and d(i',j) modulo (q-1).

    Parameters
    ----------
    sol : ndarray
        Concatenated vector [c-part | d-part] (mod q-1).
    q_minus_1 : int
        Modulus of exponents (q-1).
    hx_idx : list[(i,j)]
        Positions of 1's in HX → c(i,j).
    hz_idx : list[(i',j)]
        Positions of 1's in HZ → d(i',j).
    max_lines : int
        Maximum number of printed lines (truncate if too long).
    """
    print("\n=== Non-binary solution (explicit c(i,j), d(i',j)) ===")
    print(f"(mod {q_minus_1})  total length = {len(sol)}")

    num_c = len(hx_idx)
    num_d = len(hz_idx)
    lines = []

    for k, (i, j) in enumerate(hx_idx):
        val = int(sol[k] % q_minus_1)
        lines.append(f"c({i},{j}) = {val}")
    for t, (ip, j) in enumerate(hz_idx):
        val = int(sol[num_c + t] % q_minus_1)
        lines.append(f"d({ip},{j}) = {val}")

    if len(lines) > max_lines:
        head = lines[: max_lines // 2]
        tail = lines[-max_lines // 2 :]
        for s in head:
            print(s)
        print(f"... (omitted {len(lines) - len(head) - len(tail)} lines) ...")
        for s in tail:
            print(s)
    else:
        for s in lines:
            print(s)

            
# -----------------
# Reproducible demo
# -----------------
if __name__ == "__main__":
    np.random.seed(GLOBAL_SEED)
    np.set_printoptions(linewidth=200, threshold=np.inf)

    M1, N1 = 2, 3
    M2, N2 = 2, 3
    Q_MINUS_1 = 255

    print("=== 1. Generation of binary HGP codes ===")
    # H1 = generate_random_matrix_no_zero_cols(M1, N1)
    # H2 = generate_random_matrix_no_zero_cols(M2, N2)
    H1 = np.array([[1, 1, 1],
                   [0, 1, 0]], dtype=np.uint8)
    H2 = np.array([[1, 0, 0],
                   [1, 1, 1]], dtype=np.uint8)
    HX, HZ = generate_hgp_matrices(H1, H2)
    print("H1:\n", H1)
    print("H2:\n", H2)
    print("HX:\n", HX)
    print("HZ:\n", HZ)
    verify_binary_commutation(HX, HZ)
    GX = compute_generator_matrix(HX)
    GZ = compute_generator_matrix(HZ)

    print("\n=== Generator matrices ===")
    print("G_X:\n", GX)
    print("G_Z:\n", GZ)
    print(f"dim(C_X) = {GX.shape[0]}, dim(C_Z) = {GZ.shape[0]}")

    CX_CZp, CZ_CXp = enumerate_coset_basis(HX, HZ)

    print("\n=== 3. Verification of coset bases ===")
    # Check CX/CZ⊥ representatives
    for idx, v in enumerate(CX_CZp):
        lhs_HX = (HX @ v) % 2
        lhs_GX = (GX @ v) % 2 if GX.size > 0 else np.zeros(0, dtype=int)
        print(f"\n[CX/CZ⊥] Representative #{idx}")
        print("v =", v)
        print("HX v^T =", lhs_HX)
        print("GX v^T =", lhs_GX)
        print("→ HXv = 0?", np.all(lhs_HX == 0))
        print("→ GXv ≠ 0?", not np.all(lhs_GX == 0))

    # Check CZ/CX⊥ representatives
    for idx, v in enumerate(CZ_CXp):
        lhs_HZ = (HZ @ v) % 2
        lhs_GZ = (GZ @ v) % 2 if GZ.size > 0 else np.zeros(0, dtype=int)
        print(f"\n[CZ/CX⊥] Representative #{idx}")
        print("v =", v)
        print("HZ v^T =", lhs_HZ)
        print("GZ v^T =", lhs_GZ)
        print("→ HZv = 0?", np.all(lhs_HZ == 0))
        print("→ GZv ≠ 0?", not np.all(lhs_GZ == 0))

    #print("\n=== 2. Non-binarization ===")
    #Cg, Dd = solve_exponents_by_congruence(HX, HZ, Q_MINUS_1)
    print("\n=== 2. Non-binarization (Smith-based) ===")
    Cg, Dd, sol = solve_exponents_by_smith(HX, HZ, Q_MINUS_1, print_solution=True, max_print_lines=80)
    print_nb_solution_cd(sol, Q_MINUS_1, list(zip(*np.where(HX != 0))), list(zip(*np.where(HZ != 0))))
    print("C_gamma:\n", Cg)
    print("D_delta:\n", Dd)
    verify_nb_constraints(HX, HZ, Cg, Dd, Q_MINUS_1)

    export_latex_example(filename="example_output.tex", seed=GLOBAL_SEED, q_minus_1=Q_MINUS_1)
    print("\n✓ Reproducibility test completed under unified seed.")  # :contentReference[oaicite:14]{index=14}
