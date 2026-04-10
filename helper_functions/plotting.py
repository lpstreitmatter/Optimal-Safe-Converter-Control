import numpy as np
import matplotlib.pyplot as plt

J = np.array([[0,1],[-1,0]])

# Create the A matrix (used to relate Vdq and Idq)
def make_A(R,L,omega):
    return np.array([[-R/L,omega],[-omega,-R/L]])

def make_Z(R,L,omega):
    return np.array([[R,-omega*L],[omega*L,R]])

def angle_and_frequency(Vdq_of_t, dt):
    # shape of len(ts) X 2 or len(ts) X even_number
    # returns angle in rad and frequency in rad/s
    if Vdq_of_t.shape[1] == 2:
        Vdq_of_t_complex = Vdq_of_t[:,0] + Vdq_of_t[:,1] * 1j
    else:
        num_inv = int(Vdq_of_t.shape[1] / 2)
        num_ts = Vdq_of_t.shape[0]
        Vdq_of_t = Vdq_of_t.reshape(num_ts, num_inv, 2)
        Vdq_of_t_complex = Vdq_of_t[:,:,0] + Vdq_of_t[:,:,1] * 1j # shape of len(ts) X num_inv
    angle_of_t = np.angle(Vdq_of_t_complex, deg=False)
    frequency_of_t = np.gradient(angle_of_t, dt,axis=0)
    return angle_of_t, frequency_of_t

# Generate the data needed to plot the safe set S
def calculate_P(idq, vdq, pu=False):
    if pu:
        return np.sum(idq*vdq,axis=1)
    else:
        return 3 * np.sum(idq*vdq,axis=1)

def calculate_Q(idq, vdq, pu=False):
    if pu:
        return np.sum(idq * np.dot(vdq,J.T), axis=1)
    else: 
        return 3 * np.sum(idq * np.dot(vdq,J.T), axis=1)

def calculate_V2(idq, vdq,pu=False):
    return np.sum(vdq*vdq, axis=1)

var_map = {"P":calculate_P, "Q":calculate_Q, r"$V^2$":calculate_V2}

def calculate_var(idq, vdq, var_name, pu):
    func = var_map[var_name]
    return func(idq, vdq, pu)

def set_S(var_names,Imag_lim,E,L,R,omega,pu=False):
    unit_circle = np.array([[np.cos(phi), np.sin(phi)] for phi in np.linspace(0., 2.01*np.pi, 100)])
    mags = np.linspace(0.01,Imag_lim,30)
    idq = np.vstack([mag * unit_circle for mag in mags])
    A = make_A(R,L,omega)
    E_dq = np.array([E,0])
    vdq = E_dq - L*np.dot(idq,A.T)
    var1 = calculate_var(idq, vdq, var_names[0], pu)
    var2 = calculate_var(idq, vdq, var_names[1], pu)
    return np.vstack([var1,var2]).T # array the same shape as idq

# Plotting functions

def plot_S(trajectory_data, ref_vals,var_names, 
           Imag_lim,E,L,R,omega, 
           units = ["W","VAr"], ax = None, pu=False, second_trajectory=np.array([]), two_traj_labels=[]):
    # trajectory_data = [data_var1, data_var2]
    # second_trajectory has same format, include if wanting to plot two trajectories side by side
    # two_traj_labels has corresponding labels for the two trajectory datasets
    # ref_vals = [ref_val1, ref_val2]
    # data_labels = [label_var1, label_var2] where e.g. label_var1 = r"P"
    # units = [unit_var1, unit_var2]

    region_data = set_S(var_names,Imag_lim,E,L,R,omega,pu=pu)

    if ax == None:    
        fig,ax = plt.subplots()
    ax.fill(region_data[:,0],region_data[:,1],c="gainsboro")
    if len(trajectory_data):
        ax.scatter(trajectory_data[0,0],trajectory_data[0,1],facecolors='none', edgecolors='k', marker='o',label=r"$t_0$")
        ax.scatter(ref_vals[0],ref_vals[1],c="r", marker="x",s=50, linewidths=1, label=fr"($\overline{{{var_names[0]}}},\overline{{{var_names[1].strip('$')}}}$)")
    
    if len(second_trajectory):
        ax.plot(trajectory_data[:,0], trajectory_data[:,1],c="C0",label=f"{two_traj_labels[0]}")
        ax.scatter(trajectory_data[-1,0],trajectory_data[-1,1],marker="o", c="C0",label=rf"{two_traj_labels[0]} $t_f$")
        ax.plot(second_trajectory[:,0], second_trajectory[:,1],c="C1",label=f"{two_traj_labels[1]}")
        ax.scatter(second_trajectory[-1,0],second_trajectory[-1,1],marker="o", c="C1",label=rf"{two_traj_labels[1]} $t_f$")
        ncol=3
    elif len(trajectory_data):
        ax.plot(trajectory_data[:,0], trajectory_data[:,1], c="C0")
        ax.scatter(trajectory_data[-1,0],trajectory_data[-1,1],marker="o", c="C0",label=r"$t_f$")
        ncol=2
    else:
        ncol=1
    ax.set_xlabel(f"{var_names[0]} {units[0]}")
    ax.set_ylabel(f"{var_names[1]} {units[1]}")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=ncol)
    return

def plot_magnitude_over_time(timestamps, data, current_lim = None, xlim = None, ax = None,include_dq=False,var_name="Current (A)"):
    # timestamps = np array of timestamps
    # data = np 2D array of Idq data
    # current_lim = max allowed current magnitude. If provided, will plot a line for reference
    if ax == None:
        fig, ax = plt.subplots()
    if current_lim:
        ax.plot(timestamps, current_lim * np.ones(len(timestamps)),"r--",linewidth=1,label="Device Limit")
    ax.plot(timestamps, np.linalg.norm(data,axis=0),linewidth=1,label=f"{var_name.split()[0]} Magnitude")
    ax.set_ylim(bottom=0, top=1.05*np.max(np.linalg.norm(data,axis=0)))
    if include_dq:
        ax.plot(timestamps, data[0,:],label=r"$\mathrm{d}-axis$")
        ax.plot(timestamps, data[1,:],label=r"$\mathrm{q}-axis$")
        ax.set_ylim(bottom=min(0,1.05*np.min(data)))
    ax.set_ylabel(var_name)
    ax.set_xlabel("Time(s)")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    if xlim:
        ax.set_xlim(xlim)
    return

def plot_multi_inverters(timestamps, data, ylabel, inv_or_bus, xlim=None, inv_labels = [], ax=None, show_legend=True, **kwargs):
    # timestamps = np array of timestamps
    # data = np array of shape num_timestamps x num_inv
    if ax == None:
        fig, ax = plt.subplots()
    
    num_inv = data.shape[1]
    if len(inv_labels) == 0:
        inv_labels = np.arange(1, num_inv+1)
    
    for inv in range(num_inv):
        ax.plot(timestamps, data[:,inv], label=f"{inv_or_bus} {inv_labels[inv]}", **kwargs)
    
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Time (s)")
    if show_legend:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=num_inv)
    if xlim:
        ax.set_xlim(xlim)
    return