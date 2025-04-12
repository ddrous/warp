#%%
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib as mpl
import seaborn as sb
sb.set_theme(context='poster', 
             style='ticks',
             font='sans-serif', 
             font_scale=1, 
             color_codes=True, 
             rc={"lines.linewidth": 1})
mpl.rcParams['savefig.facecolor'] = 'w'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['savefig.bbox'] = 'tight'


# Define the mass-spring-damper system
def mass_spring_damper_system(t, y, m=1.0, k=10.0, c=0.5):
    """
    Mass-spring-damper system.

    Parameters:
    - t: time (not used explicitly but required by solve_ivp)
    - y: state vector [position, velocity]
    - m: mass
    - k: spring constant
    - c: damping coefficient
    
    Returns:
    - dydt: derivatives [velocity, acceleration]
    """
    x, v = y
    # From F = ma: m*a + c*v + k*x = 0
    # a = -(c/m)*v - (k/m)*x
    acceleration = -(c/m)*v - (k/m)*x
    return [v, acceleration]

# Function to generate trajectories
def generate_trajectories(batch_size=64, time_steps=1000, t_span=[0, 10], 
                          x_0=-1, v_0=1,
                          m_range=(0.5, 1.5), k_range=(5, 15), c_range=(0.1, 1.0),
                          ):
    """
    Generate trajectories for different initial conditions.
    
    Parameters:
    - batch_size: number of different physical parameter sets
    - time_steps: number of time steps in each trajectory
    - t_span: time span [t_start, t_end]
    - x_0: initial position
    - v_0: initial velocity
    - m_range: range for mass
    - k_range: range for spring constant
    - c_range: range for damping coefficient

    Returns:
    - trajectories: numpy array of shape (batch_size, time_steps, 2)
    - t: time points
    """


    # Create time points
    t = np.linspace(t_span[0], t_span[1], time_steps)
    
    # Initialize trajectories array
    trajectories = np.zeros((batch_size, time_steps, 2))
    
    # Generate random initial condition

    initial_ms = np.random.uniform(m_range[0], m_range[1], batch_size)
    initial_ks = np.random.uniform(k_range[0], k_range[1], batch_size)
    initial_cs = np.random.uniform(c_range[0], c_range[1], batch_size)

    # Solve ODEs for each initial condition
    for i in range(batch_size):
        y0 = [x_0, v_0]  # Initial state [position, velocity]
        m = initial_ms[i]
        k = initial_ks[i]
        c = initial_cs[i]

        # Solve the ODE
        sol = solve_ivp(
            lambda t, y: mass_spring_damper_system(t, y, m, k, c),
            t_span,
            y0,
            method='RK45',
            t_eval=t
        )

        # Store the trajectory
        trajectories[i, :, 0] = sol.y[0]  # Position
        trajectories[i, :, 1] = sol.y[1]  # Velocity

    return trajectories, t

# Plot and visualize a single trajectory
def visualize_trajectory(trajectory, t, save_gif=False, filename="mass_spring_damper.gif", fps=30):
    """
    Visualize a trajectory and optionally save as a GIF.
    
    Parameters:
    - trajectory: single trajectory of shape (time_steps, 2)
    - t: time points
    - save_gif: flag to save animation as GIF
    - filename: name of the GIF file
    - fps: frames per second for the GIF
    """
    # Create phase space plot (position vs. velocity)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(t, trajectory[:, 0], label='Position')
    plt.plot(t, trajectory[:, 1], label='Velocity')
    plt.xlabel('Time')
    plt.ylabel('State')
    plt.title('Position and Velocity vs Time')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(trajectory[:, 0], trajectory[:, 1])
    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.title('Phase Space')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    if save_gif:
        # Create animation
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim([min(trajectory[:, 0]) - 0.5, max(trajectory[:, 0]) + 0.5])
        ax.set_ylim([min(trajectory[:, 1]) - 0.5, max(trajectory[:, 1]) + 0.5])
        ax.set_xlabel('Position')
        ax.set_ylabel('Velocity')
        ax.set_title('Mass-Spring-Damper System Phase Space')
        ax.grid(True)
        
        line, = ax.plot([], [], 'r-', lw=2)
        point, = ax.plot([], [], 'bo', ms=6)
        
        def init():
            line.set_data([], [])
            point.set_data([], [])
            return line, point
        
        def animate(i):
            line.set_data(trajectory[:i+1, 0], trajectory[:i+1, 1])
            point.set_data(trajectory[i, 0], trajectory[i, 1])
            return line, point
        
        anim = FuncAnimation(fig, animate, frames=len(t), 
                             init_func=init, blit=True, interval=1000/fps)

        # Save animation
        anim.save(filename, writer='pillow', fps=fps)
        plt.close()
        print(f"GIF saved as {filename}")

# Main function
def main():
    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate trajectories
    trajectories, t = generate_trajectories(batch_size=20480, time_steps=256, t_span=[0, 1], x_0=1., v_0=-0.0,
                                                m_range=(0.02, 0.03), k_range=(5, 6), c_range=(0.1, 0.2))

    print(f"Generated trajectories shape: {trajectories.shape}")

    # Visualize one trajectory (e.g., the first one)
    visualize_trajectory(trajectories[0], t, save_gif=False)
    
    # Optional: Save the trajectories for ML training
    np.save("train.npy", trajectories)
    print("Trajectories saved to train.npy")

    # Plot some statistics
    plt.figure(figsize=(12, 6))
    
    # Plot mean and std of positions
    plt.subplot(1, 2, 1)
    mean_pos = np.mean(trajectories[:, :, 0], axis=0)
    std_pos = np.std(trajectories[:, :, 0], axis=0)
    plt.plot(t, mean_pos, 'b-', label='Mean Position')
    plt.fill_between(t, mean_pos - std_pos, mean_pos + std_pos, color='b', alpha=0.2, label='±1 Std Dev')
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.title('Mean Position with Standard Deviation')
    plt.legend()
    plt.grid(True)

    # Plot mean and std of velocities
    plt.subplot(1, 2, 2)
    mean_vel = np.mean(trajectories[:, :, 1], axis=0)
    std_vel = np.std(trajectories[:, :, 1], axis=0)
    plt.plot(t, mean_vel, 'r-', label='Mean Velocity')
    plt.fill_between(t, mean_vel - std_vel, mean_vel + std_vel, color='r', alpha=0.2, label='±1 Std Dev')
    plt.xlabel('Time')
    plt.ylabel('Velocity')
    plt.title('Mean Velocity with Standard Deviation')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
