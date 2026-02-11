import numpy as np
import cvxpy as cp
from . import inner_loop as il

# Create the A matrix (used to relate Vdq and Idq)
def make_A(R,L,omega):
    return np.array([[-R/L,omega],[-omega,-R/L]])

def make_Z(R,L,omega):
    return np.array([[R,-omega*L],[omega*L,R]])

J = np.array([[0,1],[-1,0]])

### Optimal Controller Helper Functions ###

## Create M_P, M_Q, and M_V2 matrices for the SDP so that Tr(M_x W)=P,Q, or V2 ##
# We are using RMS current and voltages so P=3*VI instead of 3/2, and same for Q. 
# When in per unit, just becomes P=VI
def make_M_P(R,Vg,perunit=False):
    first_two_rows = np.hstack((3 * R * np.eye(2), 1.5 * Vg.reshape(2,1)))
    third_row = np.hstack((1.5 * Vg.reshape(1,2), np.array([[0]])))
    if perunit:
        return np.vstack((first_two_rows, third_row)) / 3
    return np.vstack((first_two_rows, third_row))  

def make_M_Q(L,Vg,omega,perunit=False):
    # if in per unit, set L = x/omega_0 and omega=omega_0 
    JVg = J @ Vg
    first_two_rows = np.hstack((3 * omega * L * np.eye(2), 1.5 * JVg.reshape(2,1)))
    third_row = np.hstack((1.5 * JVg.reshape(1,2), np.array([[0]])))
    if perunit:
        return np.vstack((first_two_rows, third_row))/3
    return np.vstack((first_two_rows, third_row))  

def make_M_V2(R,L,Vg,omega):
    # if in per unit, set L = x/omega_0 and omega=omega_0
    Z = make_Z(R,L,omega)
    ZTVg = Z.T @ Vg
    first_two_rows = np.hstack( ( (R**2 + (omega*L)**2) * np.eye(2), ZTVg.reshape(2,1) ) )
    third_row = np.hstack((ZTVg.reshape(1,2), np.array([[Vg.T @ Vg]])))
    return np.vstack((first_two_rows, third_row))

## Solve the optimization problem directly (this provides the solution our approach will converge to) ##
def oc_one_step_optimization(ref_1, ref_2, M1, M2, Imag_lim, rho = 0.001):
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

    # Always choose the lowest magnitude current solution
    if np.linalg.norm(x2) < np.linalg.norm(x):
        x = x2
    return x, np.outer(np.hstack((x,np.array([1]))),np.hstack((x,np.array([1]))))

# Single Iteration of Control Algorithm
def oc_single_iteration(setpoints, var_names, x0, R, L, Vg0,
                        Imag_lim, omega,
                        rho=0.001, alpha=0.00001,
                        perunit=False):
    # var_names is  a 2 element list which must have elements = "P", "Q" or r"$V^2$"
    # if in per-unit, set L = x/omega_0 and omega=omega_0

    # calculate matrices for SDP
    M_dict = {"P":[make_M_P,[R,Vg0,perunit]], "Q":[make_M_Q,[L,Vg0,omega,perunit]], r"$V^2$":[make_M_V2,[R,L,Vg0,omega]]}
    M1 = M_dict[var_names[0]][0](*M_dict[var_names[0]][1])
    M2 = M_dict[var_names[1]][0](*M_dict[var_names[1]][1])

    # calculate W0 from x0
    outer_prod_vector = np.hstack((x0,np.array([1]))) 
    W0 = np.outer(outer_prod_vector,outer_prod_vector)

    # Initialize for time 0
    var1_0 = np.trace(M1 @ W0)
    var2_0 = np.trace(M2 @ W0)

    # (Unconstrained) Gradient Descent Update
    grad_f = (var1_0 - setpoints[0]) * M1
    grad_f += (var2_0 - setpoints[1]) * M2
    grad_f += rho * np.eye(3)
    Wplus = W0 - alpha * grad_f

    # Projected Gradient Descent Update (project gradient update back onto feasible set)
    try:
        Wplus = pgd_update(Wplus,Imag_lim=Imag_lim)
    except:
        # try another solver if it fails
        Wplus = pgd_update(Wplus,Imag_lim=Imag_lim,solver=cp.SCS)

    # Rank 1 Adjustment
    xplus, Wplus = rank1_adjustment(Wplus, M1, M2)
    return xplus

# Case 1: Setpoint change without any measurement noise
def oc_setpoint_change(new_setpoints, var_names, x0, R, L, Vg,
                        Imag_lim,omega,
                        rho=0.001,alpha=0.00001, dt=1e-3, 
                        starttime=0.05, endtime=1, perunit=False): 
    # var_names is  a 2 element list which must have elements = "P", "Q" or r"$V^2$"
    ts = np.arange(0, endtime, dt)

    # calculate matrices for SDP
    M_dict = {"P":[make_M_P,[R,Vg,perunit]], "Q":[make_M_Q,[L,Vg,omega,perunit]], r"$V^2$":[make_M_V2,[R,L,Vg,omega]]}
    M1 = M_dict[var_names[0]][0](*M_dict[var_names[0]][1])
    M2 = M_dict[var_names[1]][0](*M_dict[var_names[1]][1])
    Zeq = make_Z(R, L, omega=omega)

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
    Vdq_of_t[:,0] = Vg + Zeq@x0
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
            # try another solver if it fails
            Wplus = pgd_update(Wplus,Imag_lim=Imag_lim,solver=cp.SCS)

        # Rank 1 Adjustment
        x, Wplus = rank1_adjustment(Wplus, M1, M2)
        
        Idq_of_t[:,i] = x
        Vdq_of_t[:,i] = Vg + Zeq@Idq_of_t[:,i]
        var1_of_t[i] = np.trace(M1 @ Wplus) 
        var2_of_t[i] = np.trace(M2 @ Wplus)

        Ws.append(Wplus)
        i += 1
    
    return ts, Idq_of_t,Vdq_of_t,var1_of_t,var2_of_t

# More inner loop helper functions

def calculate_P(idq_of_t, vdq_of_t, perunit=False):
    if perunit:
        return np.sum(idq_of_t * vdq_of_t, axis=0)
    return 3 * np.sum(idq_of_t * vdq_of_t, axis=0)

def calculate_Q(idq_of_t, vdq_of_t, perunit=False):
    if perunit:
        return np.sum(idq_of_t * (J @ vdq_of_t), axis=0)
    return 3 * np.sum(idq_of_t * (J @ vdq_of_t), axis=0)

def calculate_V2(idq_of_t, vdq_of_t, perunit=False):
    return np.sum(vdq_of_t*vdq_of_t, axis=0)

var_map = {"P":calculate_P, "Q":calculate_Q, r"$V^2$":calculate_V2}
    
def calculate_var(idq_of_t, vdq_of_t, var_name, perunit=False):
    func = var_map[var_name]
    return func(idq_of_t, vdq_of_t, perunit=perunit)

# Case 1 with Inner Loop Dynamics: Setpoint change without any measurement noise
def oc_setpoint_change_with_il(new_setpoints, var_names, x0, 
                               r_i_val, l_i_val, r_g_val, l_g_val, c_val, Vg,
                               Imag_lim,omega,
                               rho=0.001,alpha=0.00001, dt=1e-2, 
                               starttime=0.05, endtime=1,
                               freq_cur=10000, freq_vol=2000, perunit=False): 
    # var_names is  a 2 element list which must have elements = "P", "Q" or r"$V^2$"
    R = r_i_val + r_g_val # for steady-state, outer loop calculations
    L = l_i_val + l_g_val # for steady-state, outer loop calculations

    outer_ts = np.arange(0, endtime, dt)
    full_ts = np.zeros(1)

    # calculate matrices for SDP
    M_dict = {"P":[make_M_P,[R,Vg,perunit]], "Q":[make_M_Q,[L,Vg,omega,perunit]], r"$V^2$":[make_M_V2,[R,L,Vg,omega]]}
    M1 = M_dict[var_names[0]][0](*M_dict[var_names[0]][1])
    M2 = M_dict[var_names[1]][0](*M_dict[var_names[1]][1])
    A = make_A(R,L,omega)

    # calculate W0 from x0
    outer_prod_vector = np.hstack((x0,np.array([1]))) 
    W0 = np.outer(outer_prod_vector,outer_prod_vector)
    Ws = [W0]

    # Create arrays to store Idq, Vdq, var1 and var2 over time
    Idq_of_t = np.zeros((2,1)) 
    Vdq_of_t = np.zeros((2,1)) 
    var1_of_t = np.zeros(1) 
    var2_of_t = np.zeros(1)

    # Initialize for time 0
    Idq_of_t[:,0] = x0
    Vdq_of_t[:,0] = Vg - L*A@x0
    var1_of_t[0] = np.trace(M1 @ W0)
    var2_of_t[0] = np.trace(M2 @ W0)
    
    setpoint1 = var1_of_t[0]
    setpoint2 = var2_of_t[0]
    final_state = np.array([])
    for t in outer_ts[1:]:
        t0 = t - dt
        if t > starttime:
            setpoint1 = new_setpoints[0]
            setpoint2 = new_setpoints[1]

        # (Unconstrained) Gradient Descent Update
        grad_f = (var1_of_t[-1] - setpoint1) * M1
        grad_f += (var2_of_t[-1] - setpoint2) * M2
        grad_f += rho * np.eye(3)
        Wplus = Ws[-1] - alpha * grad_f

        # Projected Gradient Descent Update (project gradient update back onto feasible set)
        try:
            Wplus = pgd_update(Wplus,Imag_lim=Imag_lim)
        except:
            # try another solver if it fails
            Wplus = pgd_update(Wplus,Imag_lim=Imag_lim,solver=cp.SCS)

        # Rank 1 Adjustment
        old_x = Idq_of_t[:,-1]
        new_x, Wplus = rank1_adjustment(Wplus, M1, M2)

        # Give new desired x to inner loop
        inner_ts, i_i_of_t, v_i_of_t, final_state = il.run_inner_loop(old_x, new_x, final_state, 
                                                                      Vg,r_i_val, r_g_val, l_i_val, l_g_val, c_val,
                                                                      f_nom = omega/(2*np.pi), freq_cur=freq_cur, 
                                                                      freq_vol=freq_vol, delta_t_outer_loop = dt)
        
        # Calculate var1 and var2
        var1_inner = calculate_var(i_i_of_t, v_i_of_t, var_name=var_names[0], perunit=perunit)
        var2_inner = calculate_var(i_i_of_t, v_i_of_t, var_name=var_names[1], perunit=perunit)

        # Append to full_ts, Idq, Vdq, var1, and var2 vectors
        full_ts = np.hstack([full_ts, t0+inner_ts])
        Idq_of_t = np.hstack([Idq_of_t, i_i_of_t])
        Vdq_of_t = np.hstack([Vdq_of_t, v_i_of_t])
        var1_of_t = np.hstack([var1_of_t, var1_inner])
        var2_of_t = np.hstack([var2_of_t, var2_inner])

        Ws.append(Wplus)
    
    return full_ts, Idq_of_t,Vdq_of_t,var1_of_t,var2_of_t


# Case 2: Grid voltage change with measurement noise
def oc_Edq_change(Vg_newmag, var_names, x0,R,L,Vg0_mag, 
                  Imag_lim,omega,variance_size = 0.1,
                  rho=0.001,alpha=0.00001, dt=1e-3, 
                  starttime = 0.05, endtime=1, perunit=False): 
    
    ts = np.arange(0, endtime, dt)

    # calculate matrices for SDP
    Vg0 = np.array([Vg0_mag,0])
    M_dict = {"P":[make_M_P,[R,Vg0,perunit]], "Q":[make_M_Q,[L,Vg0,omega,perunit]], r"$V^2$":[make_M_V2,[R,L,Vg0,omega]]}
    M1 = M_dict[var_names[0]][0](*M_dict[var_names[0]][1])
    M2 = M_dict[var_names[1]][0](*M_dict[var_names[1]][1])
    Zeq = make_Z(R,L,omega=omega)

    # calculate W0
    outer_prod_vector = np.hstack((x0,np.array([1]))) 
    W0 = np.outer(outer_prod_vector,outer_prod_vector)
    Ws = [W0]

    # Create arrays to store Idq, Vdq, var1, var2, and Vg over time
    Idq_of_t = np.zeros((2,len(ts)))
    Vdq_of_t = np.zeros((2,len(ts)))
    var1_of_t = np.zeros(len(ts))
    var2_of_t = np.zeros(len(ts))
    Vg_of_t = np.zeros((2,len(ts)))

    # Initialize for time 0
    Idq_of_t[:,0] = x0
    Vdq_of_t[:,0] = Vg0 + Zeq@x0
    setpoint1 = np.trace(M1 @ W0)
    setpoint2 = np.trace(M2 @ W0) 
    var1_of_t[0] = setpoint1
    var2_of_t[0] = setpoint2
    Vg_of_t[:,0] = Vg0

    i = 1
    # Define extra index to help in determining the amount of noise in Vg measurement at any given time
    s = 1.0

    for t in ts[1:]:
        # Update Vg
        if t < starttime:
            # no noise here - already settled
            Vg = Vg0
            Vg_actual = Vg0
        else:
            # measurement noise begins as Vg changes
            variance = variance_size * Vg_newmag / s # noise has decaying variance over time
            Vg = np.array([Vg_newmag,0]) + np.random.normal(0,np.sqrt(variance),2)
            s += 1.0
            Vg_actual = np.array([Vg_newmag,0])

        Vg_of_t[:,i] = Vg
        M_dict = {"P":[make_M_P,[R,Vg,perunit]], "Q":[make_M_Q,[L,Vg,omega,perunit]], r"$V^2$":[make_M_V2,[R,L,Vg,omega]]}
        M_actual_dict = {"P":[make_M_P,[R,Vg_actual,perunit]], "Q":[make_M_Q,[L,Vg_actual,omega,perunit]], r"$V^2$":[make_M_V2,[R,L,Vg_actual,omega]]}

        M1 = M_dict[var_names[0]][0](*M_dict[var_names[0]][1])
        M2 = M_dict[var_names[1]][0](*M_dict[var_names[1]][1])

        M1_actual = M_actual_dict[var_names[0]][0](*M_actual_dict[var_names[0]][1])
        M2_actual = M_actual_dict[var_names[1]][0](*M_actual_dict[var_names[1]][1])

        # Gradient Descent Update
        grad_f = (var1_of_t[i-1] - setpoint1) * M1 # Here P and Q are measured without voltage noise but the gradient M1/M2 is noisy
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
        Vdq_of_t[:,i] = Vg + Zeq@Idq_of_t[:,i]
        var1_of_t[i] = np.trace(M1_actual @ Wplus) # P and Q are measured without noise so use M1 actual and M2 actual
        var2_of_t[i] = np.trace(M2_actual @ Wplus)

        Ws.append(Wplus)
        i += 1

    return ts, Idq_of_t,Vdq_of_t,var1_of_t,var2_of_t, Vg_of_t

