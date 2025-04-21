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


# Define the lotka-volterra system
def lotka_volterra(t, y, a=1.0, b=0.5, c=0.5, d=0.1):
    """
    Lotka-Volterra equations for predator-prey dynamics.

    Parameters:
    - t: time (not used explicitly but required by solve_ivp)
    - y: state vector [prey, predator]
    - a: growth rate of prey
    - b: rate of predation
    - c: growth rate of predator
    - d: rate of death of predator
    
    Returns:
    - dydt: derivatives [dprey/dt, dpredator/dt]
    """
    prey, predator = y
    dprey = a * prey - b * prey * predator
    dpredator = c * prey * predator - d * predator
    return [dprey, dpredator]

# Function to generate trajectories
def generate_trajectories(batch_size=64, time_steps=1000, t_span=[0, 1], x_0=1., y_0=1.,
                          a_range=(0.5, 1.5), b_range=(0.1, 0.5), c_range=(0.1, 0.5), d_range=(0.01, 0.2)):
    """
    Generate trajectories for different initial conditions.
    
    Parameters:
    - batch_size: number of different physical parameter sets
    - time_steps: number of time steps in each trajectory
    - t_span: time span [t_start, t_end]
    - x_0: initial condition for prey   
    - y_0: initial condition for predator
    - a_range: range for prey growth rate
    - b_range: range for predation rate
    - c_range: range for predator growth rate
    - d_range: range for predator death rate

    Returns:
    - trajectories: numpy array of shape (batch_size, time_steps, 2)
    - t: time points
    """


    # Create time points
    t = np.linspace(t_span[0], t_span[1], time_steps)
    
    # Initialize trajectories array
    trajectories = np.zeros((batch_size, time_steps, 2))
    
    initial_as = np.random.uniform(a_range[0], a_range[1], batch_size)
    initial_bs = np.random.uniform(b_range[0], b_range[1], batch_size)
    initial_cs = np.random.uniform(c_range[0], c_range[1], batch_size)
    initial_ds = np.random.uniform(d_range[0], d_range[1], batch_size)

    # Solve ODEs for each initial condition
    for i in range(batch_size):
        # Randomly select parameters for this trajectory
        # a = 3e+1
        # b = 1e+2
        # c = 1e+2
        # d = 3e+1

        a = initial_as[i]
        b = initial_bs[i]
        c = initial_cs[i]
        d = initial_ds[i]

        # Solve the ODE
        sol = solve_ivp(
            lambda t, y: lotka_volterra(t, y, a, b, c, d),
            t_span,
            [x_0, y_0],
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
    train = True

    # Generate trajectories
    if train:
        trajectories, t = generate_trajectories(batch_size=15000, time_steps=256, t_span=[0., 1.], x_0=1., y_0=1.,
                                                a_range=(20, 50), b_range=(80, 120), 
                                                c_range=(80, 120), d_range=(20, 50))
    else:
        trajectories, t = generate_trajectories(batch_size=1500*2, time_steps=256, t_span=[0., 1.], x_0=1., y_0=1.,
                                                a_range=(10, 60), b_range=(70, 130), 
                                                c_range=(70, 130), d_range=(10, 60))

    print("Generated trajectories:")

    print(f"Generated trajectories shape: {trajectories.shape}")

    # Find the maximum and minimum values for normalization
    max_val = np.max(trajectories)

    # Randomly pick a time step, and replace all values after that time step with the maximum value
    random_time_step = np.random.randint(32, 64, size=trajectories.shape[0])
    # trajectories[:, random_time_step:, :] = -max_val
    # trajectories[random_time_step]
    clipped_trajs = trajectories.copy()
    spacing = 50
    for i in range(trajectories.shape[0]):
        # clipped_trajs[i, random_time_step[i]:, :] = -max_val
        clipped_trajs[i, random_time_step[i]:, :] = 0.

        trajectories[i, random_time_step[i]:, :] = 0.
        ## The original trajectory is modified as well. Its start is remembered and copied again and agian with spacing
        core_cycle = trajectories[i, :random_time_step[i], :]
        for j in range(0, trajectories.shape[1], random_time_step[i]+spacing):
            # if j+spacing+random_time_step[i] > trajectories.shape[1]:
            if j+random_time_step[i] > trajectories.shape[1]:
                len_cycle = trajectories.shape[1] - j
            else:
                len_cycle = random_time_step[i]
            if len_cycle < 0:
                break
            ## Copy the core cycle to the end of the trajectory
            # trajectories[i, j+spacing:j+spacing+len_cycle, :] = core_cycle[:len_cycle, :]
            trajectories[i, j:j+len_cycle, :] = core_cycle[:len_cycle, :]

    print(f"Random time step: {random_time_step}")

    # Visualize one trajectory (e.g., the first one)
    visualize_trajectory(clipped_trajs[16], t, save_gif=False)
    visualize_trajectory(trajectories[16], t, save_gif=False)

    # Optional: Save the trajectories for ML training
    save_file = "train.npz" if train else "test.npz"
    np.savez(save_file, full=trajectories, clipped=clipped_trajs)
    print(f"Clipped and full trajectories saved to {save_file}")

    # Plot some statistics
    plt.figure(figsize=(12, 6))

    # Plot mean and std of positions
    plt.subplot(1, 2, 1)
    mean_pos = np.mean(trajectories[:, :, 0], axis=0)
    std_pos = np.std(trajectories[:, :, 0], axis=0)
    plt.plot(t, mean_pos, 'b-', label='Mean Prey')
    plt.fill_between(t, mean_pos - std_pos, mean_pos + std_pos, color='b', alpha=0.2, label='±1 Std Dev')
    plt.xlabel('Time')
    plt.ylabel('Preys')
    plt.title('Mean and Standard Deviation')
    plt.legend()
    plt.grid(True)

    # Plot mean and std of velocities
    plt.subplot(1, 2, 2)
    mean_vel = np.mean(trajectories[:, :, 1], axis=0)
    std_vel = np.std(trajectories[:, :, 1], axis=0)
    plt.plot(t, mean_vel, 'r-', label='Mean Predators')
    plt.fill_between(t, mean_vel - std_vel, mean_vel + std_vel, color='r', alpha=0.2, label='±1 Std Dev')
    plt.xlabel('Time')
    plt.ylabel('Predators')
    plt.title('Mean and Standard Deviation')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
