import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy.physics.vector import dynamicsymbols
from sympy.physics.mechanics import init_vprinting
import scipy.integrate as spi
from scipy.integrate import solve_ivp 

# Initialize pretty printing with MathJax
init_vprinting(use_latex="mathjax")


########################## DEFINE DYNAMICS ##########################
def display_alg_eq(lhs_exprs, rhs_exprs, eval=False, alist=True):
  if alist:
    return sp.Eq(sp.Matrix([[lhs] for lhs in lhs_exprs]), sp.Matrix([[rhs] for rhs in rhs_exprs]), evaluate=eval)
  else:
    return sp.Eq(lhs_exprs, rhs_exprs)
def display_diff_eq(states, dynamics, eval=False):
  return sp.Eq(sp.diff(sp.Matrix([[state] for state in states])), sp.Matrix([[dyn] for dyn in dynamics]), evaluate=eval)

state_dynamics = {} # key = variable, value = time derivative
algebraic_expressions = {} # key = variable, value = expresssion. For substitution in state_dynamics

""" Parameter and Variable Definitions """
# lower-case = variable, Upper case = expression

# Input (reference signal from outer loop)
v_c_d_ref, v_c_q_ref = sp.var("V_{c\mathrm{d}}^*, V_{c\mathrm{q}}^*") # reference capacitor voltage

# States
i_i_d, i_i_q = dynamicsymbols("I_{i\mathrm{d}}, I_{i\mathrm{q}}") # inverter/filter inductor current
v_c_d, v_c_q = dynamicsymbols("V_{c\mathrm{d}}, V_{c\mathrm{q}}") # capacitor voltage
i_g_d, i_g_q = dynamicsymbols("I_{g\mathrm{d}}, I_{g\mathrm{q}}") # grid side inductor current

phi_d, phi_q = dynamicsymbols("\Phi_\mathrm{d}, \Phi_\mathrm{q}") # voltage controller
gamma_d, gamma_q = dynamicsymbols("\Gamma_\mathrm{d}, \Gamma_\mathrm{q}") # current controller

# Algebraic Variables
v_i_d, v_i_q = dynamicsymbols("V_{i\mathrm{d}}, V_{i\mathrm{q}}") # inverter voltage (assumed equal to current controller reference)
i_i_d_ref, i_i_q_ref = dynamicsymbols("I_{i\mathrm{d}}^*, I_{i\mathrm{q}}^*") # reference inverter/filter inductor current from voltage controller
i_c_d, i_c_q = dynamicsymbols("i_{c\mathrm{d}}, i_{c\mathrm{q}}") # capacitor current
v_lg_d, v_lg_q = dynamicsymbols("v_{Lg\mathrm{d}}, v_{Lg\mathrm{q}}") # grid-side inductor voltage
v_li_d, v_li_q = dynamicsymbols("v_{Li\mathrm{d}}, v_{Li\mathrm{q}}") # inverter-side inductor voltage

# Parameters
omega, e_d, e_q = sp.var("\omega, E_\mathrm{d}, E_\mathrm{q}") # frequency and stiff grid voltage
r_i, l_i, c = sp.var("R_i, L_i, C") # inverter RLC filter parameters
r_g, l_g = sp.var("R_g, L_g") # grid RL branch parameters
k_pv, k_iv = sp.var("k_{pv},k_{iv}") # voltage controller PI gains
k_pi, k_ii = sp.var("k_{pi},k_{ii}") # current controller PI gains
J = sp.Matrix([[0,1],[-1,0]]) # clockwise 90 degree rotation matrix

# Collect dq components into vectors
v_c_dq_ref = sp.Matrix([v_c_d_ref, v_c_q_ref])

i_i_dq = sp.Matrix([i_i_d, i_i_q])
v_c_dq = sp.Matrix([v_c_d, v_c_q ])
i_g_dq = sp.Matrix([i_g_d, i_g_q])
phi_dq = sp.Matrix([phi_d, phi_q])
gamma_dq = sp.Matrix([gamma_d, gamma_q])

v_i_dq = sp.Matrix([v_i_d, v_i_q])
i_i_dq_ref = sp.Matrix([i_i_d_ref, i_i_q_ref])
i_c_dq = sp.Matrix([i_c_d, i_c_q])
v_lg_dq = sp.Matrix([v_lg_d, v_lg_q])
v_li_dq = sp.Matrix([v_li_d, v_li_q])

e_dq = sp.Matrix([e_d,e_q])


""" Line and Filter Dynamics """

# Update algebraic expressions
V_li_dq = v_i_dq - v_c_dq - r_i * i_i_dq
V_lg_dq = v_c_dq - e_dq - r_g * i_g_dq
I_c_dq = i_i_dq - i_g_dq

Z_g = sp.Matrix([[r_g, -omega * l_g],[omega*l_g, r_g]])
Z_i = sp.Matrix([[r_i, -omega * l_i],[omega*l_i, r_i]])

LCL_alg_lhs = [*v_li_dq, *v_lg_dq, *i_c_dq]
LCL_alg_rhs = [*V_li_dq, *V_lg_dq, *I_c_dq]

algebraic_expressions.update(zip(LCL_alg_lhs, LCL_alg_rhs))

# Update differential equations
I_i_dq_dot = (1/l_i) * v_li_dq + omega * J @ i_i_dq
I_g_dq_dot = (1/l_g) * v_lg_dq + omega * J @ i_g_dq
V_c_dq_dot = (1/c) * i_c_dq + omega * J @ v_c_dq

LCL_diff_lhs = [*i_i_dq, *i_g_dq, *v_c_dq]
LCL_diff_rhs = [*I_i_dq_dot, *I_g_dq_dot, *V_c_dq_dot]

state_dynamics.update(zip(LCL_diff_lhs, LCL_diff_rhs))

""" Voltage Controller Loop Dynamics """

# Update differential equations 
Phi_dq_dot = v_c_dq_ref - v_c_dq

# Update algebraic expressions: VCL outputs I_i_dq ref signal for current controller
I_i_dq_ref = i_g_dq - omega * c * J @ v_c_dq + k_pv * Phi_dq_dot + k_iv * phi_dq

VCL_diff_lhs = [*phi_dq]
VCL_diff_rhs = [*Phi_dq_dot]

VCL_alg_lhs = [*i_i_dq_ref]
VCL_alg_rhs = [*I_i_dq_ref]

state_dynamics.update(zip(VCL_diff_lhs, VCL_diff_rhs))
algebraic_expressions.update(zip(VCL_alg_lhs, VCL_alg_rhs))

""" Current Controller Loop Dynamics """

# Update differential equations 
Gamma_dq_dot = i_i_dq_ref - i_i_dq

# Update algebraic expressions: CCL outputs V_i_dq ref signal that PWM automatically realizes -> V_i_dq
V_i_dq = v_c_dq - omega * l_i * J @ i_i_dq + k_pi * Gamma_dq_dot + k_ii * gamma_dq

CCL_diff_lhs = [*gamma_dq]
CCL_diff_rhs = [*Gamma_dq_dot]

CCL_alg_lhs = [*v_i_dq]
CCL_alg_rhs = [*V_i_dq]

state_dynamics.update(zip(CCL_diff_lhs, CCL_diff_rhs))
algebraic_expressions.update(zip(CCL_alg_lhs, CCL_alg_rhs))

# Put it all together
x = sp.Matrix([[i_i_d],[i_i_q],[i_g_d],[i_g_q],[v_c_d],[v_c_q],[phi_d],[phi_q],[gamma_d],[gamma_q]])
x_dot = sp.Matrix([[state_dynamics[state]] for state in x]).subs(algebraic_expressions).subs(algebraic_expressions).subs(algebraic_expressions)

########################## SOLVE DYNAMICS ##########################
def run_inner_loop(i_i_dq_val_0, i_i_dq_ref_val_new, e_dq_val, 
                   r_i_val, r_g_val, l_i_val, l_g_val, c_val, 
                   f_nom = 60, freq_cur=10000, freq_vol=2000, delta_t_outer_loop = 0.01):
    ''' 
    Function that takes current at time 0, desired new current setpoint (after gradient step), 
    and system parameters and simulates inner loop control to achieve new setpoint.
    Outputs: inverter current and voltage over time within delta_t of outer loop
    '''
    # Process parameters
    t = sp.Symbol('t')

    omega_val = 2 * np.pi * f_nom

    lcl_subs = {r_i:r_i_val, r_g:r_g_val, l_i:l_i_val, l_g:l_g_val, c:c_val,
                omega: omega_val, e_d: e_dq_val[0], e_q: e_dq_val[1]}

    omega_cur = 2*np.pi*freq_cur
    omega_vol = 2*np.pi*freq_vol

    k_pv_val = 2*omega_vol*c_val
    k_iv_val = 2*k_pv_val*omega_vol**2 / omega_cur

    k_pi_val = l_i_val * omega_cur
    k_ii_val = r_i_val * omega_cur
  
    inner_control_subs = {k_pv:k_pv_val, k_iv:k_iv_val, k_pi:k_pi_val, k_ii:k_ii_val}

    # Combine all parameters
    sim_numeric_subs = {**lcl_subs, **inner_control_subs}

    # Handle initial conditions and new desired setpoint
    # Convert i_i_dq_val_0 to v_c_dq_ref_val_old
    v_c_dq_ref_val_old = e_dq_val.T + Z_g.subs(lcl_subs) @ i_i_dq_val_0 

    # Convert v_i_dq_ref_val_new to v_c_dq_ref_val_new
    v_c_dq_ref_val_new = e_dq_val.T + Z_g.subs(lcl_subs) @ i_i_dq_ref_val_new

    # Calculate full state x at initial condition
    x0_guess = [*i_i_dq_val_0,*i_i_dq_val_0,*v_c_dq_ref_val_old,0,0,0,0]
    x_dot_num = x_dot.subs(sim_numeric_subs).evalf()
    f_t0 = x_dot_num.subs({v_c_d_ref:v_c_dq_ref_val_old[0], v_c_q_ref:v_c_dq_ref_val_old[1]}).evalf()
    x0 = np.array(sp.nsolve(f_t0,x,x0_guess)).astype(np.float64).flatten()

    # Solve IVP
    f_inputs = [t,x,v_c_d_ref, v_c_q_ref]
    lambdify_x_dot = sp.lambdify(f_inputs, x_dot_num)

    def f_x_dot(t, x, v_c_d_ref, v_c_q_ref):
        return np.array(lambdify_x_dot(t, x, v_c_d_ref, v_c_q_ref),dtype=float).flatten()

    sol = solve_ivp(f_x_dot,[0,delta_t_outer_loop],x0, 
                    args=(v_c_dq_ref_val_new[0],v_c_dq_ref_val_new[1]))

    f_v_i_dq = sp.lambdify(x, V_i_dq.subs(algebraic_expressions).subs(sim_numeric_subs).subs({v_c_d_ref:v_c_dq_ref_val_new[0], v_c_q_ref:v_c_dq_ref_val_new[1]}))
    ts = sol.t
    i_i_of_t = sol.y.T[:,0:2]
    v_i_of_t = [f_v_i_dq(*x_) for x_ in sol.y.T]
    v_i_of_t = np.hstack(v_i_of_t).T

    return (ts, i_i_of_t, v_i_of_t)