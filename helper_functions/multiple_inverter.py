import numpy as np

from . import optimal_controller as oc

# to get test system
from pypower.api import case14, runpf, printpf, makeYbus
from pypower.idx_bus import *
from pypower.idx_brch import *
from pypower.idx_gen import *


### Multi-Inverter Network Helper functions ###
def convert_rx_to_gb(r,x,positive_b_inductor=True):
    # We define default z = r + jx and y = g - jb
    # Therefore x>0 -> b>0
    denom = r**2 + x**2
    if positive_b_inductor:
        return r/denom, x/denom
    else:
        return r/denom, -x/denom

def convert_gb_to_rx(g, b, positive_b_inductor=True):
    denom = g**2 + b**2
    if positive_b_inductor:
        return g/denom, b/denom
    else:
        return g/denom, -b/denom
    
def convert_complexy_to_ydq(complexy):
    g = complexy.real
    b = -complexy.imag
    return np.array([[g, b],[-b,g]])

def convert_complexY_to_Ydq(complexY):
    m,n = complexY.shape
    Ydq = np.zeros((2*m, 2*n))
    for i in range(m):
        for k in range(n):
            Ydq[2*i:2*i + 2, 2*k:2*k + 2] = convert_complexy_to_ydq(complexY[i,k])
    return Ydq

def calculate_P_network(Iinvs, Vinvs, perunit=True):
    n = int(len(Iinvs) / 2)
    if perunit:
        return np.sum(Iinvs.reshape(n,2) * Vinvs.reshape(n,2), axis=1)
    return 3 * np.sum(Iinvs.reshape(n,2) * Vinvs.reshape(n,2), axis=1)

def calculate_Q_network(Iinvs, Vinvs, perunit=True):
    n = int(len(Iinvs) / 2)
    J = np.array([[0,1],[-1,0]])
    if perunit:
        return np.sum(Iinvs.reshape(n,2) * (Vinvs.reshape(n,2) @ J), axis=1)
    return 3 * np.sum(Iinvs.reshape(n,2) * (Vinvs.reshape(n,2) @ J), axis=1)

def create_Ybus(network, pf_result):
    nb = network["bus"].shape[0]
    nbr = network["branch"].shape[0]

    base_mva = network["baseMVA"]

    Ybus = np.zeros((nb,nb),dtype=complex)

    # Add self-admittances (diagonal-only elements)
    bus_shunts = (network["bus"][:,GS] + network["bus"][:,BS]*1j)/base_mva # pypower uses y=g+jb
    bus_loads = (pf_result["bus"][:,PD] - pf_result["bus"][:,QD] * 1j)/ (base_mva * pf_result["bus"][:,VM] ** 2) # Convert PD and QD to per unit then divide by squared Vm

    # Inverter shunts are not included in Ybus to avoid double counting later
    np.fill_diagonal(Ybus, bus_shunts + bus_loads)

    # Add mutual admittances (add to diagonals and subtract from off-diagonals)
    for i in range(nbr):
        # Ignore line charging susceptance
        from_bus = int(network["branch"][i, F_BUS])
        to_bus = int(network["branch"][i,T_BUS])
        g, b = convert_rx_to_gb(network["branch"][i, BR_R], network["branch"][i, BR_X], positive_b_inductor=True) # already in pu
        branch_admittance = g - b * 1j
        # Assign negative branch admittances to off-diagonals
        Ybus[from_bus - 1, to_bus - 1] = -branch_admittance
        Ybus[to_bus - 1, from_bus - 1] = -branch_admittance

        # Add branch admittances to diagonals
        Ybus[from_bus - 1, from_bus - 1] += branch_admittance
        Ybus[to_bus - 1, to_bus - 1] += branch_admittance
    return Ybus

def Kron_reduce(Ybus, slack_idx_0based, inv_idx_0based):
    nb = Ybus.shape[0]
    keep_nodes = np.hstack((slack_idx_0based, inv_idx_0based))
    elim_nodes = np.setdiff1d(range(nb), keep_nodes)
    new_order = np.hstack((keep_nodes, elim_nodes))

    # Swap rows
    Ybus = Ybus[new_order, :]

    # Swap columns
    Ybus = Ybus[:,new_order]

    # Do Kron Reduction
    n_keep = len(keep_nodes)
    K = Ybus[:n_keep, :n_keep]
    L = Ybus[:n_keep, n_keep:]
    LT = Ybus[n_keep:, :n_keep]
    M = Ybus[n_keep:, n_keep:]

    Ybus_KRON = K - L @ np.linalg.inv(M) @ LT

    # Now remove infinite bus from Ybus_Kron
    Y_slack = -Ybus_KRON[0,1:]
    # convert Yslack to dq
    Y_slackdq = convert_complexY_to_Ydq(Y_slack.reshape(len(Y_slack),1))
    Ybus_KRON = Ybus_KRON[1:,1:]
    Ybus_KRONdq = convert_complexY_to_Ydq(Ybus_KRON)

    return Ybus_KRONdq, Y_slackdq

### Dynamic Simulation Functions ###
def initial_setpoints(slack_id, inverter_bus_ids, pf_result, 
                         Ybus_KRONdq, Y_slackdq, filter_rs, filter_xs):
    # Initialize Vg0 from power flow and solve for the rest
    Z_filters = np.diag(filter_rs + filter_xs * 1j)
    Y_filtersdq = convert_complexY_to_Ydq(np.linalg.inv(Z_filters))
    Z_filtersdq = np.linalg.inv(Y_filtersdq)

    Vslack = np.array([pf_result["bus"][slack_id-1,VM],0])
    Vg0_mags = pf_result["bus"][inverter_bus_ids-1,VM] # voltage magnitudes
    Vg0_angles = np.deg2rad(pf_result["bus"][inverter_bus_ids-1,VA]) # pypower angle converted to radians
    Vgs0 = np.vstack((Vg0_mags * np.cos(Vg0_angles),Vg0_mags * np.sin(Vg0_angles))).T.flatten() # in dq
    Iinvs0 = Ybus_KRONdq @ Vgs0 - Y_slackdq @ Vslack # Approximate currents as I = Y_KRON V_pf
    Vinvs0 = Vgs0 + Iinvs0 @ Z_filtersdq.T
    Ps0 = calculate_P_network(Iinvs0, Vinvs0)
    Qs0 = calculate_Q_network(Iinvs0, Vinvs0)

    return Ps0, Qs0, np.linalg.norm(Vinvs0.reshape(len(inverter_bus_ids),2), axis=1)**2

def multi_inv_simulation(slack_id, inverter_bus_ids, inverter_setpoints, inverter_varnames, 
                         inverter_mag_limits, pf_result, 
                         Ybus_KRONdq, Y_slackdq, filter_rs, filter_xs,
                         rho=0.0001, alpha=1, perunit=True,
                         dt=1e-2, starttime = 0.05, returntime = 10, endtime=1):
    base_mva = pf_result['baseMVA']
    
    Zbus_KRONdq = np.linalg.inv(Ybus_KRONdq)

    Z_filters = np.diag(filter_rs + filter_xs * 1j)
    Y_filtersdq = convert_complexY_to_Ydq(np.linalg.inv(Z_filters))
    Z_filtersdq = np.linalg.inv(Y_filtersdq)

    n_keep = len(inverter_bus_ids)
    ts = np.arange(0, endtime, dt)
    omega = 2*np.pi*60 # just used for converting X to L assuming constant impedance

    Vgs = np.zeros((len(ts),n_keep*2))
    Iinvs = np.zeros((len(ts),n_keep*2))
    Vinvs = np.zeros((len(ts),n_keep*2))
    Vslack = np.array([pf_result["bus"][slack_id-1,VM],0])
    Ps = np.zeros((len(ts),n_keep))
    Qs = np.zeros((len(ts),n_keep))

    # Initialize Vg0 from power flow and solve for the rest
    Vg0_mags = pf_result["bus"][inverter_bus_ids-1,VM] # voltage magnitudes
    Vg0_angles = np.deg2rad(pf_result["bus"][inverter_bus_ids-1,VA]) # pypower angle converted to radians
    Vgs[0,:] = np.vstack((Vg0_mags * np.cos(Vg0_angles),Vg0_mags * np.sin(Vg0_angles))).T.flatten() # in dq
    Iinvs[0,:] = Ybus_KRONdq @ Vgs[0,:] - Y_slackdq @ Vslack # Approximate currents as I = Y_KRON V_pf
    Vinvs[0,:] = Vgs[0,:] + Iinvs[0,:] @ Z_filtersdq.T
    Ps[0,:] = calculate_P_network(Iinvs[0,:], Vinvs[0,:],perunit=perunit)
    Qs[0,:] = calculate_Q_network(Iinvs[0,:], Vinvs[0,:],perunit=perunit)

    # Set the initial setpoints equal to the initial pfresult values
    # TODO: CHANGE to a mix of PQ and PV2 inverters
    original_inv_setpoints = np.vstack((Ps[0,:],np.linalg.norm(Vinvs[0,:].reshape(n_keep,2),axis=1)**2)).T
    #gen_mask = np.isin(pf_result["gen"][:,GEN_BUS], inverter_bus_ids)
    #original_inv_setpoints = np.vstack((pf_result["gen"][gen_mask,PG]/base_mva,pf_result["bus"][inverter_bus_ids-1,VM] **2)).T

    for k,t in enumerate(ts[:-1]):
        xpluses = np.zeros(n_keep*2)
        for inv in range(n_keep):
            if starttime < t < returntime:
                xplus = oc.oc_single_iteration(inverter_setpoints[inv,:], inverter_varnames[inv,:], Iinvs[k,2*inv:2*inv+2], 
                                            filter_rs[inv], filter_xs[inv]/omega, Vgs[k,2*inv:2*inv+2], inverter_mag_limits[inv], omega,
                                                rho=rho, alpha=alpha,
                                                perunit=perunit)
            else:
                xplus = oc.oc_single_iteration(original_inv_setpoints[inv,:], inverter_varnames[inv,:], Iinvs[k,2*inv:2*inv+2], 
                                            filter_rs[inv], filter_xs[inv]/omega, Vgs[k,2*inv:2*inv+2], inverter_mag_limits[inv], omega,
                                                rho=rho, alpha=alpha,
                                                perunit=perunit)                
            xpluses[2*inv:2*inv+2] = xplus
        # Calculate new values across network
        Iinvs[k+1, :] = xpluses
        Vgs[k+1, :] = Zbus_KRONdq @ (Iinvs[k+1, :] + Y_slackdq @ Vslack)
        Vinvs[k+1, :] = Vgs[k+1,:] + Iinvs[k+1,:] @ Z_filtersdq.T
        Ps[k+1,:] = calculate_P_network(Iinvs[k+1,:], Vinvs[k+1,:])
        Qs[k+1,:] = calculate_Q_network(Iinvs[k+1,:], Vinvs[k+1,:])
    return ts, Iinvs, Vinvs, Vgs, Ps, Qs


