# -*- coding: utf-8 -*-
import numpy as np
np.set_printoptions(precision=6,threshold=1e3)
import cvxpy as cp

def subband_assign(libopt,H_RB,h_UR,h_d,theta,p,A0,d_max,tol_sca):
    mu_initial = 1
    max_iter = 100
    mu_record =[]
    obj_record = []

    def compute_objective(A, A0, h_norm_squared, p_broadcasted, mu):
        product_1 = cp.multiply(A, h_norm_squared)
        product_2 = cp.multiply(product_1, p_broadcasted)
        x = 2 * cp.sum(product_2, axis=1) / ((libopt.sigma/1000) ** 2)
        term1 = 1 / 6 * cp.exp(-2 * x)
        term2 = 1 / 12 * cp.exp(-x)
        term3 = 1 / 4 * cp.exp(-x / 2)
        obj1 = 1 - 2 * cp.sum(term1 + term2 + term3)
        R = A - 2 * cp.multiply(A, A0) + cp.square(A0)
        obj2 = cp.sum(R)
        return obj1 - mu * obj2


    h = np.zeros((libopt.J, libopt.K, libopt.M), dtype=complex)
    for k in range(libopt.K):
        for m in range(libopt.M):
            h[:, k, m] = np.dot(H_RB[k, m, :, :], np.diag(theta) @ h_UR[m, :, k]) + h_d[m, :, k]
    h_norm_squared = np.linalg.norm(h, axis=0) ** 2

    mu = mu_initial
    mu_record.append(mu)
    obj_record.append(0.0)
    for d in range(d_max):
        print(f'This is SCA iteration:{d}')
        A = cp.Variable((libopt.K, libopt.M))
        p_broadcasted = np.broadcast_to(p[:, None], A.shape)
        objective = compute_objective(A, A0,h_norm_squared,p_broadcasted, 0)
        constraints = [
            cp.sum(A, axis=0) <= 1,
            cp.sum(A, axis=1) == 1,
            A >= 0,
            A <= 1,
        ]
        problem = cp.Problem(cp.Maximize(objective), constraints)
        problem.solve(solver=cp.SCS)
        optimal_without_penalty = problem.value

        objective = compute_objective(A, A0,h_norm_squared,p_broadcasted, mu)
        problem = cp.Problem(cp.Maximize(objective), constraints)
        problem.solve(solver=cp.SCS)
        R = A - 2 * cp.multiply(A, A0) + cp.square(A0)
        obj2 = cp.sum(R)
        magnitude_penalty = mu * obj2
        i=0
        for _ in range(max_iter):
            i=i+1
            magnitude_penalty_value = magnitude_penalty.value

            if problem.status is None or magnitude_penalty_value is None or optimal_without_penalty is None: # 628:复现时候没第三个判断条件
                print("Problem status: ", problem.status)
                print("Objective value: ", magnitude_penalty)
                break
            if not np.isclose(np.log10(abs(optimal_without_penalty)), np.log10(abs(magnitude_penalty_value)), atol=1):
                mu = mu * abs(optimal_without_penalty) / abs(magnitude_penalty_value)
                objective = compute_objective(A, A0,h_norm_squared,p_broadcasted, mu)
                problem = cp.Problem(cp.Maximize(objective), constraints)
                problem.solve(solver=cp.SCS)
                R = A - 2 * cp.multiply(A, A0) + cp.square(A0)
                obj2 = cp.sum(R)
                magnitude_penalty = mu * obj2
            else:
                break
        print(f'The max iter for mu is:{i}')
        print(f'The tuned mu is: {mu}')
        mu_record.append(mu)
        objective = compute_objective(A, A0, h_norm_squared, p_broadcasted, mu)
        problem = cp.Problem(cp.Maximize(objective), constraints)
        solver_failed = False
        try:
            problem.solve(solver=cp.SCS, max_iters=3000)
            A0 = A.value
            obj_record.append(problem.value)
        except cp.SolverError:
            print("Sub-band assignment MOSEK solver failed. Trying SCS...")
            try:
                problem.solve(solver=cp.SCS)
                A0 = A.value
                obj_record.append(problem.value)
            except cp.SolverError:
                print(f"Sub-band assignment both solvers failed in iteration {d}. Stopping the SCA iterations...")
                solver_failed = True
        if solver_failed or (abs(mu_record[d+1] - mu_record[d]) < tol_sca and abs(obj_record[d+1] - obj_record[d]) < tol_sca):
            break

    if problem.status == 'optimal' or solver_failed:
        return A0,mu_record,obj_record
    else:
        print("Problem of sub-band assignment is infeasible")
        return A0,mu_record,obj_record


def power_assign (libopt,A,theta,p_init,P0,H_RB,h_UR,h_d,l_init,u_init,epsilon,imax,jmax):
    zeta_init = np.ones(libopt.K)
    rho = 0.5
    lamb = 1e-4
    h = np.zeros((libopt.J, libopt.K, libopt.M), dtype=complex)
    for k in range(libopt.K):
        for m in range(libopt.M):
            h[:, k, m] = np.dot(H_RB[k, m, :, :], np.diag(theta) @ h_UR[m, :, k]) + h_d[m, :, k]
    h_norm_squared = np.linalg.norm(h, axis=0) ** 2

    def f1(A,h_norm_squared,p_broadcasted):
        product_1 = cp.multiply(A, h_norm_squared)
        product_2 = cp.multiply(product_1, p_broadcasted)
        x = 2 * cp.sum(product_2, axis=1) / libopt.sigma ** 2
        term1 = 1 / 6 * cp.exp(-2 * x)
        term2 = 1 / 12 * cp.exp(-x)
        term3 = 1 / 4 * cp.exp(-x / 2)
        obj1 = libopt.K - 2 * cp.sum(term1 + term2 + term3)
        return obj1

    def f2(p,p_pre,zeta):
        term1 = cp.sqrt(cp.sum(cp.multiply(zeta, cp.square(p_pre))))
        v = 2 * cp.multiply(zeta, p_pre)
        obj2 = term1 + 0.5* cp.inv_pos(term1) * (v @ (p - p_pre))
        return obj2

    p_i = p_init
    zeta_i = zeta_init
    for i in range(imax):
        print(f'This is outer iteration of power assignment:{i}')
        p_pre = p_i
        zeta = zeta_i
        for j in range(jmax):
            p = cp.Variable(libopt.K)
            p_reshaped = cp.reshape(p, (libopt.K, 1))
            p_broadcasted = p_reshaped @ np.ones((1, libopt.M))

            k=0
            u=u_init
            l=l_init
            max_iter = 1000
            while u - l > epsilon and k < max_iter:
                k=k+1
                x = (l + u) / 2
                constraints = [f2(p, p_pre, zeta) - x * f1(A, h_norm_squared, p_broadcasted) <= 0,
                               cp.sum(p) == P0,
                               p >= 0]
                problem = cp.Problem(cp.Minimize(0), constraints)
                try:
                    problem.solve(solver=cp.SCS, max_iters=3000)
                except cp.SolverError:
                    print(f"Solver MOSEK failed in iteration {k}. Trying secondary solver (SCS)...")
                    try:
                        problem.solve(solver=cp.SCS)
                    except cp.SolverError:
                        print(f"Secondary solver (SCS) also failed in iteration {i}. Skipping to the next iteration...")
                        continue

                if problem.status == 'optimal':
                    u = x
                    optimal_p_value = p.value
                else:
                    l = x
            p_pre = optimal_p_value
        p_i = p_pre
        epx = rho/2 - 1
        zeta_i = rho /2 * cp.power(cp.square(p_i) + lamb**2, epx)
    p = p_i

    return p,f2(p,p_pre,zeta),f1(A,h_norm_squared,p_broadcasted)

def phase_design(libopt,A,p,theta_init,H_RB,h_UR,h_d,rho,iter_max):
    H_RB_all = np.zeros((libopt.K, libopt.J, libopt.L),dtype=complex)
    h_UR_all = np.zeros((libopt.K, libopt.L),dtype=complex)
    h_d_all = np.zeros((libopt.K, libopt.J),dtype=complex)
    h_d_norm_squared = np.zeros((libopt.K,1),dtype=complex)
    A_k_all = np.zeros((libopt.K, libopt.J, libopt.L),dtype=complex) # in the paper, Ak*theta is RIS-assisted link
    Lambda_k_all = np.zeros((libopt.K, libopt.L+1, libopt.L+1),dtype=complex)
    for k in range(libopt.K):
        assigned_idx = np.where(A[k, :] == 1)[0]
        if len(assigned_idx) == 0:
            continue
        assigned_subband = assigned_idx[0]
        H_RB_all[k] = H_RB[k, assigned_subband, :, :]
        h_UR_all[k] = h_UR[assigned_subband, :, k]
        h_d_all[k] = h_d[assigned_subband, :, k]
        h_d_norm_squared[k] = np.linalg.norm(h_d_all[k]) ** 2
        A_k_all[k] = H_RB_all[k] @ np.diag(h_UR_all[k])
        tmp = h_d_all[k].conj().T @ A_k_all[k]
        tmp=tmp.reshape(1,libopt.L)
        tmp1 = A_k_all[k].conj().T @ A_k_all[k]
        tmp2 = A_k_all[k].conj().T @ h_d_all[k]
        tmp2 = tmp2.reshape(libopt.L,1)
        Lambda_k_all[k] = np.block([[tmp1,tmp2], [tmp,0]])

    def f2(Lambda,X,p,h_norm_squared,sigma):
        obj=0
        K = p.shape[0]
        for k in range(K):
            trace_term = cp.real(cp.trace(Lambda[k] @ X))
            product_1 = trace_term + cp.real(h_norm_squared[k])
            x = 2 * p[k] * product_1 /sigma ** 2
            term1 = 1 / 6 * cp.exp(-2 * x)
            term2 = 1 / 12 * cp.exp(-x)
            term3 = 1 / 4 * cp.exp(-x / 2)
            obj += term1 + term2 + term3
        return obj

    phi = np.append(theta_init,1)
    X_init = np.outer(phi, np.conj(phi))
    for i in range(iter_max):
        eigvals, eigvecs = np.linalg.eig(X_init)
        leading_eigval_index = np.argmax(eigvals)
        v = eigvecs[:, leading_eigval_index]
        sub_grad = np.outer(v, np.conj(v))
        X=cp.Variable((libopt.L+1, libopt.L+1), hermitian=True) # PSD=true
        obj1 = cp.real(f2(Lambda_k_all,X,p,h_d_norm_squared,libopt.sigma))
        I = np.eye(X.shape[0])
        X_real = cp.real(X)
        X_imag = cp.imag(X)
        sub_grad_real = cp.real(sub_grad)
        sub_grad_imag = cp.imag(sub_grad)
        hermitian_mult_real = X_real.T @ (I - sub_grad_real) + X_imag.T @ sub_grad_imag
        hermitian_mult_imag = X_real.T @ sub_grad_imag - X_imag.T @ (I - sub_grad_real)
        obj2 = rho * (cp.trace(hermitian_mult_real) - cp.trace(hermitian_mult_imag))
        objective = obj1 + obj2
        constraints = [X>>0,cp.trace(X) == libopt.L+1]
        problem = cp.Problem(cp.Minimize(objective), constraints)
        solver_failed = False
        try:
            problem.solve(solver=cp.SCS, max_iters=3000)
        except cp.SolverError:
            print("Phase design problem MOSEK failed. Trying SCS...")
            try:
                problem.solve(solver=cp.SCS)
            except cp.SolverError:
                print("Phase design problem Both MOSEK and SCS failed. Stopping optimization...")
                solver_failed = True

        if not solver_failed:
            X_init = X.value
            rank_X = np.linalg.matrix_rank(X_init)
            print(f'Phase rank of X:{rank_X}')
            if problem.status == 'optimal':
                try:
                    np.linalg.cholesky(X_init)
                except np.linalg.LinAlgError:
                    epsilon = 1e-8
                    X_init += epsilon * np.eye(X_init.shape[0])
                try:
                    L = np.linalg.cholesky(X_init)
                except np.linalg.LinAlgError:
                    epsilon = 1e-6
                    X_init = X_init + epsilon * np.eye(X_init.shape[0])
                    try:
                        L = np.linalg.cholesky(X_init)
                    except np.linalg.LinAlgError:
                        print("Cholesky failed, fallback to identity theta")
                        return theta_init
            else:
                print(f'Phase Problem was infeasible.')
                break
        else:
            break

    if problem.status == 'optimal' or solver_failed:
        theta_update = L[0:libopt.L,0]
    else:
        theta_update = theta_init

    return theta_update