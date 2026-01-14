import numpy as np
import cvxpy as cp
import inner_loop as il
import plotting

# Create the A matrix (used to relate Vdq and Idq)
def make_A(R,L,omega):
    return np.array([[-R/L,omega],[-omega,-R/L]])

def make_Z(R,L,omega):
    return np.array([[R,-omega*L],[omega*L,R]])

J = np.array([[0,1],[-1,0]])

### Optimal Controller Helper Functions ###

## Create M_P, M_Q, and M_V2 matrices for the SDP so that Tr(M_x W)=P,Q, or V2 ##
# We are using RMS current and voltages so P=3*VI instead of 3/2, and same for Q. 
def make_M_P(R,Vg):
    first_two_rows = np.hstack((3 * R * np.eye(2), 1.5 * Vg.reshape(2,1)))
    third_row = np.hstack((1.5 * Vg.reshape(1,2), np.array([[0]])))
    return np.vstack((first_two_rows, third_row))  

def make_M_Q(L,Vg,omega):
    JVg = J @ Vg
    first_two_rows = np.hstack((3 * omega * L * np.eye(2), 1.5 * JVg.reshape(2,1)))
    third_row = np.hstack((1.5 * JVg.reshape(1,2), np.array([[0]])))
    return np.vstack((first_two_rows, third_row))  

def make_M_V2(R,L,Vg,omega):
    Z = make_Z(R,L,omega)
    ZTVg = Z.T @ Vg
    first_two_rows = np.hstack( ( (R**2 + (omega*L)**2) * np.eye(2), ZTVg.reshape(2,1) ) )
    third_row = np.hstack((ZTVg.reshape(1,2), np.array([[Vg.T @ Vg]])))
    return np.vstack((first_two_rows, third_row))

## Solve the optimization problem directly (this provides the solution our approach will converge to) ##
def oc_one_step_optimization(ref_1, ref_2, M1, M2, Imag_lim, rho = 0.01):
    W = cp.Variable((3,3), symmetric=True)
    constraints = [W >> 0]

    # CURRENT LIMIT CONSTRAINT: 
    constraints += [W[0,0]+W[1,1] <= Imag_lim**2]
    
    # Matrix/vector consistency constraint:
    constraints += [W[2,2]==1]

    objective = rho * cp.norm(W, "nuc")
    objective += 0.5*cp.square(cp.trace(M1@W) - ref_1)
    objective += 0.5*cp.square(cp.trace(M2@W) - ref_2)

    problem = cp.Problem(cp.Minimize(objective), constraints)

    obj_val = problem.solve(solver=cp.CLARABEL)

    converged = problem.status == cp.OPTIMAL
    if not converged:
        if problem.status == cp.OPTIMAL_INACCURATE:
            print("WARNING - STATUS = OPTIMAL_INACCURATE")
        else:
            raise ValueError(f"Optimization problem did not converge! {problem.status}")
    W_star = W.value
    return W_star, obj_val

## Two-Step Dynamic Strategy (PGD Update and Rank 1 Adjustment) ##
# Projected Gradient Descent Update on W
def pgd_update(Wk,Imag_lim,verbose=False,solver=cp.CLARABEL):
    Wplus = cp.Variable((3,3), symmetric=True)

    constraints = [Wplus >> 0]
    constraints += [Wplus[0,0]+Wplus[1,1] <= Imag_lim**2]
    constraints += [Wplus[2,2]==1]

    objective = cp.norm(Wplus-Wk,'fro')
    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(solver=solver,verbose=verbose)

    converged = problem.status == cp.OPTIMAL
    if not converged:
        raise ValueError(f"Optimization problem did not converge! {problem.status}, {Wk}")
    return Wplus.value

# Rank 1 adjustment finds the vector x that produces the same var1 and var2 values as Wplus
def rank1_adjustment(W,M1,M2):
    # extract parameters from M1, M2 
    alpha = M1[0,0]
    beta = M2[0,0]
    a = 2 * M1[0:2,2]
    b = 2 * M2[0:2,2]
    zeta1 = M1[2,2]
    zeta2 = M2[2,2]

    # define new variables to express quadratic equation 
    ab_inv = np.linalg.inv(np.vstack((a,b)))
    c = ab_inv @ np.array([alpha, beta])
    d = ab_inv @ (np.array([np.trace(M1@W),np.trace(M2@W)]) - np.array([zeta1, zeta2]))

    # setup and solve quadratic equation 
    coef1 = np.inner(c,c)
    coef2 = -(1 + 2*np.inner(d,c))
    coef3 = np.inner(d,d)
    mus = np.roots([coef1, coef2, coef3])

    # Find the two possible Idq solutions
    x = d - mus[0] * c
    x2 = d - mus[1] * c

    # Always the lowest magnitude current solution
    if np.linalg.norm(x2) < np.linalg.norm(x):
        x = x2
    return x, np.outer(np.hstack((x,np.array([1]))),np.hstack((x,np.array([1]))))


# Case 1: Setpoint change without any measurement noise
def oc_setpoint_change(new_setpoints, var_names, x0, R, L, Vg,
                        Imag_lim,omega,
                        rho=0.01,alpha=0.00001, dt=1e-3, 
                        starttime=0.05, endtime=1,
                        units=["W","VAr"],xlim=None,include_dq=False): 
    # var_names is  a 2 element list which must have elements = "P", "Q" or r"$V^2$"
    ts = np.arange(0, endtime, dt)

    # calculate matrices for SDP
    M_dict = {"P":[make_M_P,[R,Vg]], "Q":[make_M_Q,[L,Vg,omega]], r"$V^2$":[make_M_V2,[R,L,Vg,omega]]}
    M1 = M_dict[var_names[0]][0](*M_dict[var_names[0]][1])
    M2 = M_dict[var_names[1]][0](*M_dict[var_names[1]][1])
    A = make_A(R,L,omega=omega)

    # calculate W0 from x0
    outer_prod_vector = np.hstack((x0,np.array([1]))) 
    W0 = np.outer(outer_prod_vector,outer_prod_vector)
    Ws = [W0]

    # Create arrays to store Idq, Vdq, var1 and var2 over time
    Idq_of_t = np.zeros((2,len(ts))) 
    Vdq_of_t = np.zeros((2,len(ts))) 
    var1_of_t = np.zeros(len(ts)) 
    var2_of_t = np.zeros(len(ts))

    # Initialize for time 0
    Idq_of_t[:,0] = x0
    Vdq_of_t[:,0] = Vg - L*A@x0
    var1_of_t[0] = np.trace(M1 @ W0)
    var2_of_t[0] = np.trace(M2 @ W0)
    
    i = 1
    setpoint1 = var1_of_t[0]
    setpoint2 = var2_of_t[0]
    for t in ts[1:]:
        if t > starttime:
            setpoint1 = new_setpoints[0]
            setpoint2 = new_setpoints[1]

        # (Unconstrained) Gradient Descent Update
        grad_f = (var1_of_t[i-1] - setpoint1) * M1
        grad_f += (var2_of_t[i-1] - setpoint2) * M2
        grad_f += rho * np.eye(3)
        Wplus = Ws[i-1] - alpha * grad_f

        # Projected Gradient Descent Update (project gradient update back onto feasible set)
        try:
            Wplus = pgd_update(Wplus,Imag_lim=Imag_lim)
        except:
            Wplus = pgd_update(Wplus,Imag_lim=Imag_lim,solver=cp.SCS)

        # Rank 1 Adjustment
        x, Wplus = rank1_adjustment(Wplus, M1, M2)
        
        Idq_of_t[:,i] = x
        Vdq_of_t[:,i] = Vg - L*A@Idq_of_t[:,i]
        var1_of_t[i] = np.trace(M1 @ Wplus) 
        var2_of_t[i] = np.trace(M2 @ Wplus)

        Ws.append(Wplus)
        i += 1

        # ADD PLOTTING
    
    return ts, Idq_of_t,Vdq_of_t,var1_of_t,var2_of_t

# Case 1 with Inner Loop Dynamics: Setpoint change without any measurement noise
def oc_setpoint_change_with_il(new_setpoints, var_names, x0, R, L, Vg,
                        Imag_lim,omega,
                        rho=0.01,alpha=0.00001, dt=1e-2, 
                        starttime=0.05, endtime=1,
                        units=["W","VAr"],xlim=None,include_dq=False): 
    # var_names is  a 2 element list which must have elements = "P", "Q" or r"$V^2$"
    ts = np.arange(0, endtime, dt)

    # calculate matrices for SDP
    M_dict = {"P":[make_M_P,[R,Vg]], "Q":[make_M_Q,[L,Vg,omega]], r"$V^2$":[make_M_V2,[R,L,Vg,omega]]}
    M1 = M_dict[var_names[0]][0](*M_dict[var_names[0]][1])
    M2 = M_dict[var_names[1]][0](*M_dict[var_names[1]][1])
    A = make_A(R,L,omega=omega)

    # calculate W0 from x0
    outer_prod_vector = np.hstack((x0,np.array([1]))) 
    W0 = np.outer(outer_prod_vector,outer_prod_vector)
    Ws = [W0]

    # Create arrays to store Idq, Vdq, var1 and var2 over time
    Idq_of_t = np.zeros((2,len(ts))) 
    Vdq_of_t = np.zeros((2,len(ts))) 
    var1_of_t = np.zeros(len(ts)) 
    var2_of_t = np.zeros(len(ts))

    # Initialize for time 0
    Idq_of_t[:,0] = x0
    Vdq_of_t[:,0] = Vg - L*A@x0
    var1_of_t[0] = np.trace(M1 @ W0)
    var2_of_t[0] = np.trace(M2 @ W0)
    
    i = 1
    setpoint1 = var1_of_t[0]
    setpoint2 = var2_of_t[0]
    for t in ts[1:]:
        if t > starttime:
            setpoint1 = new_setpoints[0]
            setpoint2 = new_setpoints[1]

        # (Unconstrained) Gradient Descent Update
        grad_f = (var1_of_t[i-1] - setpoint1) * M1
        grad_f += (var2_of_t[i-1] - setpoint2) * M2
        grad_f += rho * np.eye(3)
        Wplus = Ws[i-1] - alpha * grad_f

        # Projected Gradient Descent Update (project gradient update back onto feasible set)
        try:
            Wplus = pgd_update(Wplus,Imag_lim=Imag_lim)
        except:
            Wplus = pgd_update(Wplus,Imag_lim=Imag_lim,solver=cp.SCS)

        # Rank 1 Adjustment
        x, Wplus = rank1_adjustment(Wplus, M1, M2)
        
        Idq_of_t[:,i] = x
        Vdq_of_t[:,i] = Vg - L*A@Idq_of_t[:,i]
        var1_of_t[i] = np.trace(M1 @ Wplus) 
        var2_of_t[i] = np.trace(M2 @ Wplus)

        Ws.append(Wplus)
        i += 1

        # ADD PLOTTING
    
    return ts, Idq_of_t,Vdq_of_t,var1_of_t,var2_of_t



