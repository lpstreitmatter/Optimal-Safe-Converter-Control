import numpy as np
from . import optimal_controller as oc

### Helper Functions ###
J = np.array([[0,1],[-1,0]])

# Rotation Functions
def rotate_grid_frame_to_device_frame(vector, delta):
    # delta := device angle - grid angle
    return np.array([[np.cos(delta), np.sin(delta)],[-np.sin(delta),np.cos(delta)]]) @ vector
def rotate_device_frame_to_grid_frame(vector, delta):
    # delta := device angle - grid angle
    return np.array([[np.cos(delta), -np.sin(delta)],[np.sin(delta),np.cos(delta)]]) @ vector


### PQ Droop Control ###

# Case 1: Setpoint change without measurement noise -> setpoint_change=True
# Case 2: Grid voltage change with measurement noise -> setpoint_change=False
def pq_droop_disturbance(new_setpoints, x0, R, L, Vg, 
                       Imag_lim, omega_nom, m_p, m_q, omega_c, 
                       dt = 1e-3, starttime = 0.05, endtime=1, perunit=True, setpoint_change=True):
    
    # new_setpoints is a list with [new_ref_p, new_ref_q] or the new [Vg_d, Vg_q] after a disturbance (note phase disturbance not implemented so Vg_q should be zero)
    # x0 and Vg are in the grid reference frame

    if perunit:
        power_multiplier=1
    else:
        power_multiplier=3
    
    Zeq = oc.make_Z(R,L,omega_nom)
    Yeq = np.linalg.inv(Zeq)
    Vg_mag = np.linalg.norm(Vg)

    ts = np.arange(0, endtime, dt)
    # Create time vectors indexed as Array[d or q, time]
    Idq_of_t = np.zeros((2,len(ts))) # Idq here is in device frame but will be switched to grid frame at the end
    Vdqref_of_t = np.zeros((2,len(ts))) # Also in device frame, this is the Vdq droop tells inverter to have pre-saturation
    Vmag_of_t = np.zeros(len(ts)) # Array of all the voltage magnitudes droop controller outputs
    omega_of_t = np.zeros(len(ts)) # Array of all the voltage frequencies droop controller outputs
    delta_of_t = np.zeros(len(ts)) # Angle difference between inverter voltage and grid voltage over time
    P_of_t = np.zeros(len(ts)) # Active power output over time
    Q_of_t = np.zeros(len(ts)) # Reactive power output over time
    Pbar_of_t = np.zeros(len(ts)) # Filtered active power output (for droop to track)
    Qbar_of_t = np.zeros(len(ts)) # Filtered reactive power ouptut (for droop to track)

    # Calculate initial values
    Vdq0 = Vg + Zeq@x0 # Vdq of inverter in grid dq frame at t0
    v_nom = np.linalg.norm(Vdq0)
    delta0 = np.angle(Vdq0[0] + Vdq0[1]*1j) - np.angle(Vg[0] + Vg[1]*1j)
    P0 = power_multiplier * x0 @ Vdq0
    Q0 = power_multiplier * x0 @ J @ Vdq0
    
    Idq_of_t[:,0] = rotate_grid_frame_to_device_frame(x0,delta0) # device frame
    Vmag_of_t[0] = v_nom
    omega_of_t[0] = omega_nom
    delta_of_t[0] = delta0
    P_of_t[0] = P0
    Q_of_t[0] = Q0
    Pbar_of_t[0] = P_of_t[0]
    Qbar_of_t[0] = Q_of_t[0]

    i = 1

    for t in ts[1:]:
        # Discretized angle update
        delta_plus = delta_of_t[i-1] + dt * (omega_of_t[i-1] - omega_nom)

        # Droop control updates voltage magnitude and frequency
        if t < starttime: 
            Vmag_plus = v_nom - m_q * (Qbar_of_t[i-1] - Q0)
            omega_plus = omega_nom - m_p*(Pbar_of_t[i-1] - P0)
        else:
            if setpoint_change:
                Vmag_plus = v_nom - m_q * (Qbar_of_t[i-1] - new_setpoints[1])
                omega_plus = omega_nom - m_p*(Pbar_of_t[i-1] - new_setpoints[0])
            else: # signals voltage step change instead 
                Vg = np.array(new_setpoints)
                Vg_mag = np.linalg.norm(Vg)
                v_nom = Vg_mag
            

        # Update current according to new voltage and frequency setpoints
        Idq_plus = Yeq @ np.array([Vmag_plus - Vg_mag * np.cos(delta_plus),Vg_mag * np.sin(delta_plus)])
        i_norm = np.linalg.norm(Idq_plus) # determine current magnitude for later use
        Pplus = power_multiplier * Idq_plus[0] * Vmag_plus
        Qplus = -power_multiplier * Idq_plus[1] * Vmag_plus
        Pbarplus = Pbar_of_t[i-1] + dt * omega_c * (Pplus - Pbar_of_t[i-1])
        Qbarplus = Qbar_of_t[i-1] + dt * omega_c * (Qplus - Qbar_of_t[i-1])

        # Now saturate current and recalculate actual inverter voltage that results
        if i_norm > Imag_lim:
            Idq_plus = Idq_plus * Imag_lim / i_norm # saturate current
            V_actual = np.array([Vg_mag * np.cos(delta_plus),-Vg_mag * np.sin(delta_plus)]) + Zeq @ Idq_plus # recalculate Vdq
            Pplus = power_multiplier * Idq_plus @ V_actual # recalculate P, Q and filtered P, Q
            Qplus = power_multiplier * Idq_plus @ J @ V_actual
            Pbarplus = Pbar_of_t[i-1] + dt * omega_c * (Pplus - Pbar_of_t[i-1])
            Qbarplus = Qbar_of_t[i-1] + dt * omega_c * (Qplus - Qbar_of_t[i-1])

        Idq_of_t[:,i] = Idq_plus
        Vmag_of_t[i] = Vmag_plus
        delta_of_t[i] = delta_plus
        omega_of_t[i] = omega_plus
        P_of_t[i] = Pplus 
        Q_of_t[i] = Qplus 
        Pbar_of_t[i] = Pbarplus
        Qbar_of_t[i] = Qbarplus 

        i += 1

    # Convert Idq and Vdqref back to grid frame and calculate actual Vdq using Idq and Vg (all in grid frame)
    for i,t in enumerate(Idq_of_t[0,:]):
        Idq_of_t[:,i] = rotate_device_frame_to_grid_frame(Idq_of_t[:,i],delta_of_t[i])
        Vdqref_device = np.array([Vmag_of_t[i],0])
        Vdqref_of_t[:,i] = rotate_device_frame_to_grid_frame(Vdqref_device,delta_of_t[i])
    Vdqactual_of_t = Vg.reshape(2,1) + Zeq @ Idq_of_t

    return ts, Idq_of_t, Vdqref_of_t, Vdqactual_of_t, P_of_t, Q_of_t, Vmag_of_t, delta_of_t



### PV^2 Droop Control ###

# Case 1: Setpoint change without measurement noise -> setpoint_change=True
# Case 2: Grid voltage change with measurement noise -> setpoint_change=False
def pv2_droop_disturbance(new_setpoints, x0, R, L, Vg, 
                       Imag_lim, omega_nom, omega_c, m_p, m_v2,  
                       dt = 1e-3, starttime = 0.05, endtime=1, perunit=True, setpoint_change=True):
    
    # new_setpoints is a list with [new_ref_p, new_ref_q] or the new [Vg_d, Vg_q] after a disturbance (note phase disturbance not implemented so Vg_q should be zero)
    # x0 and Vg are in the grid reference frame    
    if perunit:
        power_multiplier=1
    else:
        power_multiplier=3

    Zeq = oc.make_Z(R,L,omega_nom)
    Yeq = np.linalg.inv(Zeq)
    Vg_mag = np.linalg.norm(Vg)

    ts = np.arange(0, endtime, dt)
    # Create time vectors indexed as Array[d or q, time]
    Idq_of_t = np.zeros((2,len(ts))) # Idq here is in device frame but will be switched to grid frame at the end
    Vdqref_of_t = np.zeros((2,len(ts))) # Also in device frame, this is the Vdq droop tells inverter to have pre-saturation
    Vmag_of_t = np.zeros(len(ts)) # Array of all the voltage magnitudes droop controller outputs
    omega_of_t = np.zeros(len(ts)) # Array of all the voltage frequencies droop controller outputs
    delta_of_t = np.zeros(len(ts)) # Angle difference between inverter voltage and grid voltage over time
    P_of_t = np.zeros(len(ts)) # Active power output over time
    V2_of_t = np.zeros(len(ts)) # Reactive power output over time
    Pbar_of_t = np.zeros(len(ts)) # Filtered active power output (for droop to track)
    V2bar_of_t = np.zeros(len(ts)) # Filtered square voltage mag output (for droop to track)

    # Calculate initial values
    Vdq0 = Vg + Zeq@x0 # Vdq of inverter in grid dq frame at t0
    v_nom = np.linalg.norm(Vdq0)
    delta0 = np.angle(Vdq0[0] + Vdq0[1]*1j) - np.angle(Vg[0] + Vg[1]*1j)
    P0 = power_multiplier * x0 @ Vdq0
    V20 = Vdq0 @ Vdq0
    
    Idq_of_t[:,0] = rotate_grid_frame_to_device_frame(x0,delta0) # device frame
    Vmag_of_t[0] = v_nom
    omega_of_t[0] = omega_nom
    delta_of_t[0] = delta0
    P_of_t[0] = P0
    V2_of_t[0] = V20
    Pbar_of_t[0] = P_of_t[0]
    V2bar_of_t[0] = V2_of_t[0]

    i = 1

    for t in ts[1:]:
        # Discretized angle updates
        delta_plus = delta_of_t[i-1] + dt * (omega_of_t[i-1] - omega_nom)

        # Droop control updates voltage magnitude and frequency
        if t < starttime: 
            Vmag_plus = np.sqrt(v_nom**2 - m_v2 * (V2bar_of_t[i-1] - V20))
            omega_plus = omega_nom - m_p*(Pbar_of_t[i-1] - P0)
        else:
            if setpoint_change:
                Vmag_plus = np.sqrt(v_nom**2 - m_v2 * (V2bar_of_t[i-1] - new_setpoints[1]))
                omega_plus = omega_nom - m_p*(Pbar_of_t[i-1] - new_setpoints[0])
            else: # signals voltage step change instead 
                Vg = np.array(new_setpoints)
                Vg_mag = np.linalg.norm(Vg)
                v_nom = Vg_mag
            

        # Update current according to new voltage and frequency setpoints
        Idq_plus = Yeq @ np.array([Vmag_plus - Vg_mag * np.cos(delta_plus),Vg_mag * np.sin(delta_plus)])
        i_norm = np.linalg.norm(Idq_plus) # determine current magnitude for later use
        Pplus = power_multiplier * Idq_plus[0] * Vmag_plus
        V2plus = Vmag_plus ** 2
        Pbarplus = Pbar_of_t[i-1] + dt * omega_c * (Pplus - Pbar_of_t[i-1])
        V2barplus = V2bar_of_t[i-1] + dt * omega_c * (V2plus - V2bar_of_t[i-1])

        # Now saturate current and recalculate actual inverter voltage that results
        if i_norm > Imag_lim:
            Idq_plus = Idq_plus * Imag_lim / i_norm # saturate current
            V_actual = np.array([Vg_mag * np.cos(delta_plus),-Vg_mag * np.sin(delta_plus)]) + Zeq @ Idq_plus # recalculate Vdq
            Pplus = power_multiplier * Idq_plus @ V_actual # recalculate P, Q and filtered P, Q
            V2plus = V_actual @ V_actual
            Pbarplus = Pbar_of_t[i-1] + dt * omega_c * (Pplus - Pbar_of_t[i-1])
            V2barplus = V2bar_of_t[i-1] + dt * omega_c * (V2plus - V2bar_of_t[i-1])

        Idq_of_t[:,i] = Idq_plus
        Vmag_of_t[i] = Vmag_plus
        delta_of_t[i] = delta_plus
        omega_of_t[i] = omega_plus
        P_of_t[i] = Pplus 
        V2_of_t[i] = V2plus 
        Pbar_of_t[i] = Pbarplus
        V2bar_of_t[i] = V2barplus

        i += 1

    # Convert Idq and Vdqref back to grid frame and calculate actual Vdq using Idq and Vg (all in grid frame)
    for i,t in enumerate(Idq_of_t[0,:]):
        Idq_of_t[:,i] = rotate_device_frame_to_grid_frame(Idq_of_t[:,i],delta_of_t[i])
        Vdqref_device = np.array([Vmag_of_t[i],0])
        Vdqref_of_t[:,i] = rotate_device_frame_to_grid_frame(Vdqref_device,delta_of_t[i])
    Vdqactual_of_t = Vg.reshape(2,1) + Zeq @ Idq_of_t

    return ts, Idq_of_t, Vdqref_of_t, Vdqactual_of_t, P_of_t, V2_of_t, Vmag_of_t, delta_of_t



