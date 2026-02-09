import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from scipy.ndimage import gaussian_filter, median_filter, gaussian_filter1d


def plot_trial(trial_path, plot_channels = False):

    # === Open csv into Pd ===
    trial_df = pd.read_csv(trial_path)

    time = trial_df['timestamp_vector'].values
    tof = trial_df['tof'].values

    scA = trial_df['scA'].values #/ 10
    scB = trial_df['scB'].values #/ 10
    scC = trial_df['scC'].values #/ 10


    # # --- Air Pressure Signals Check
    # Aire Pressure Lower threshold
    pr_dn_thr = 220
    if (scA < pr_dn_thr).any() or (scB < pr_dn_thr).any() or (scC < pr_dn_thr).any():
        trial_n = trial_path.split('trial_')[1]
        trial_n = trial_n.split('_downsampled')[0]
        print(f'trial {trial_n} with pressure issues')

        plot_channels = True

        if (scA < pr_dn_thr).any():
            # mask = trial_df['scA'] < pr_dn_thr
            trial_df['scA'] = trial_df[['scB', 'scC']].mean(axis=1)

            scA = trial_df['scA'].values #/ 10
            plot_channels = True

        if (scB < pr_dn_thr).any():
            # mask = trial_df['scB'] < pr_dn_thr
            trial_df['scB'] = trial_df[['scA', 'scC']].mean(axis=1)

            scB = trial_df['scB'].values #/ 10
            plot_channels = True


    # Air Pressure Upper Threshold
    pr_up_thr = 1100
    if (scA > pr_up_thr).any() or (scB > pr_up_thr).any() or (scC > pr_up_thr).any():
        trial_n = trial_path.split('trial_')[1]
        trial_n = trial_n.split('_downsampled')[0]
        print(f'trial {trial_n} with pressure issues')
                

        if (scB > pr_up_thr).any():
            mask = trial_df['scB'] > pr_up_thr
            trial_df.loc[mask, 'scB'] = trial_df.loc[mask, ['scA', 'scC']].mean(axis=1)

            scB = trial_df['scB'].values #/ 10
            plot_channels = True
        
        if (scC > pr_up_thr).any():
            mask = trial_df['scC'] > pr_up_thr
            trial_df.loc[mask, 'scC'] = trial_df.loc[mask, ['scA', 'scB']].mean(axis=1)

            scC = trial_df['scC'].values #/ 10
            plot_channels = True


    fx = trial_df['_wrench._force._x'].values
    fy = trial_df['_wrench._force._y'].values
    fz = trial_df['_wrench._force._z'].values

    # -- Force Singularity Check
    # if (trial_df['_wrench._force._x'].abs() < 1e-4).any(): 
    #     trial_n = trial_path.split('trial_')[1]
    #     trial_n = trial_n.split('_downsampled')[0]
    #     print(f'trial {trial_n} with singularity issues')

    #     plot_channels = True
    

    forces = np.vstack((fx, fy, fz)).T  # shape: (N, 3)
    fnet = np.linalg.norm(forces, axis=1)

    idx_max_fnet = np.argmax(fnet)
    time_max_fnet = time[idx_max_fnet]

    idx_tof_thr = np.where(tof < 50)[0]
    time_tof_thr = time[idx_tof_thr[0]]

    try:
        idx_sca_thr = np.where(scA < 600)[0]
        time_sca_thr = time[idx_sca_thr][0]
    except IndexError:
        time_sca_thr = 0


    try:
        idx_scb_thr = np.where(scB < 600)[0]
        time_scb_thr = time[idx_scb_thr][0]
    except IndexError:
        time_scb_thr = 0


    try:
        idx_scc_thr = np.where(scC < 600)[0]
        time_scc_thr = time[idx_scc_thr][0]
    except IndexError:
        time_scc_thr = 0



    cupslist = [time_sca_thr, time_scb_thr, time_scc_thr]
    sorted_cups = sorted(cupslist)

    second_cup = sorted_cups[1]

    approach_start = time_tof_thr - 7
    approach_end = time_tof_thr + 0.5

    contact_start = time_tof_thr
    contact_end = second_cup + 0.5

    pick_start = second_cup
    pick_end = time_max_fnet + 1.5


    

    if plot_channels: 
        fig, ax_tof = plt.subplots(figsize=(10, 5))

        # === Shade background for phases ===
        ax_tof.axvspan(approach_start, approach_end, color='tab:blue', alpha=0.1, label='Approach')
        ax_tof.axvspan(contact_start, contact_end, color='tab:orange', alpha=0.1, label='Contact')
        ax_tof.axvspan(pick_start, pick_end, color='tab:red', alpha=0.1, label='Pick')

        # === ToF axis (left) ===
        color_tof = 'tab:blue'
        ax_tof.plot(time, tof, color=color_tof, label='ToF', linewidth=2.0)
        ax_tof.set_xlabel('Elapsed time [s]')
        ax_tof.set_ylabel('ToF [mm]', color=color_tof)
        ax_tof.tick_params(axis='y', colors=color_tof)
        ax_tof.set_ylim(0, 320)

        # --- ToF threshold ---
        ax_tof.axhline(50, color=color_tof, linestyle=':', linewidth=1)
        ax_tof.text(
            time[10], 50,
            'ToF approach threshold= 50 mm',
            color=color_tof,
            va='bottom', ha='left',
            fontsize=12
        )

        # === Suction cups axis (right 1) ===
        ax_sc = ax_tof.twinx()
        color_sc = 'tab:orange'
        ax_sc.plot(time, scA, '--', color=color_sc, label='SC A', linewidth=2.0)
        ax_sc.plot(time, scB, '--', color=color_sc, alpha=0.7, label='SC B', linewidth=2.0)
        ax_sc.plot(time, scC, '--', color=color_sc, alpha=0.4, label='SC C', linewidth=2.0)
        ax_sc.set_ylabel('Air pressure [hPa]', color=color_sc)
        ax_sc.tick_params(axis='y', colors=color_sc)
        ax_sc.set_ylim(0, 1500)

        # --- Pressure threshold ---
        ax_sc.axhline(600, color=color_sc, linestyle=':', linewidth=1)
        ax_sc.text(
            time[10], 600,
            'Air Pressure contact threshold= 600',
            color=color_sc,
            va='bottom', ha='left',
            fontsize=12
        )

        # === Force axis (right 2) ===
        ax_force = ax_tof.twinx()
        ax_force.spines['right'].set_position(('axes', 1.15))
        color_force = 'tab:red'
        ax_force.plot(time, fnet, ':', color=color_force, label='Net force', linewidth=2.0)
        ax_force.set_ylabel('Force [N]', color=color_force)
        ax_force.tick_params(axis='y', colors=color_force)
        ax_force.set_ylim(0, 20)

        # === Legend (data only) ===
        lines = (
                ax_tof.get_lines()
                + ax_sc.get_lines()
                + ax_force.get_lines()
        )
        labels = [l.get_label() for l in lines]
        ax_tof.legend(lines, labels, loc='upper right', fontsize=12)
        ax_tof.set_xlim(left=0)
        ax_tof.set_xlim(right=max(time))

        # === Add labels for shaded regions ===
        y_text = ax_tof.get_ylim()[1] * 0.99  # slightly below top of ToF axis
        ax_tof.text((approach_start + approach_end) / 2, y_text, 'Approach',
                    color='tab:blue', ha='center', va='top', fontsize=12, fontweight='bold')
        ax_tof.text((contact_start + contact_end) / 2, y_text, 'Contact',
                    color='tab:orange', ha='center', va='top', fontsize=12, fontweight='bold')
        ax_tof.text((pick_start + pick_end) / 2, y_text, 'Pick',
                    color='tab:red', ha='center', va='top', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.title(trial_path)
        plt.show()


def trial_phases_durations(trial_path):

    # === Open csv into Pd ===
    trial_df = pd.read_csv(trial_path)

    time = trial_df['timestamp_vector'].values
    tof = trial_df['tof'].values

    scA = trial_df['scA'].values  # / 10
    scB = trial_df['scB'].values  # / 10
    scC = trial_df['scC'].values  # / 10

    fx = trial_df['_wrench._force._x'].values
    fy = trial_df['_wrench._force._y'].values
    fz = trial_df['_wrench._force._z'].values

    forces = np.vstack((fx, fy, fz)).T  # shape: (N, 3)
    fnet = np.linalg.norm(forces, axis=1)

    idx_max_fnet = np.argmax(fnet)
    time_max_fnet = time[idx_max_fnet]

    # --- Approach ---
    idx_tof_thr = np.where(tof < 50)[0]
    time_tof_thr = time[idx_tof_thr[0]]

    approach_start = max(time_tof_thr - 9.0, 0)
    approach_end = time_tof_thr + 0.5
    approach_duration = approach_end - approach_start

    # --- Contact ---
    cupslist = []
    try:
        idx_sca_thr = np.where(scA < 600)[0]
        time_sca_thr = time[idx_sca_thr][0]
        cupslist.append(time_sca_thr)
    except IndexError:
        pass

    try:
        idx_scb_thr = np.where(scB < 600)[0]
        time_scb_thr = time[idx_scb_thr][0]
        cupslist.append(time_scb_thr)
    except IndexError:
        pass

    try:
        idx_scc_thr = np.where(scC < 600)[0]
        time_scc_thr = time[idx_scc_thr][0]
        cupslist.append(time_scc_thr)
    except IndexError:
        pass

    if len(cupslist) >1:
        sorted_cups = sorted(cupslist)
        second_cup = sorted_cups[1]

        contact_start = time_tof_thr
        contact_end = second_cup + 0.5
        contact_duration = contact_end - contact_start

        pick_start = second_cup
        pick_end = time_max_fnet + 1.5
        pick_duration = pick_end - pick_start
        if pick_duration < 0:
            pick_duration = float('nan')

    else:
        contact_duration = float('nan')
        pick_duration = float('nan')

    return approach_duration, contact_duration, pick_duration


def plot_batch_trials():

    # --- Use LaTeX for all text ---
    # Do NOT use usetex
    plt.rcParams.update({
        "font.family": "serif",
        "axes.labelsize": 15,
        "axes.titlesize": 10,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 15
    })

    # === Build Path ===
    # base_folder = os.path.join(r'D:',
    #                            'DATA',
    #                            '03_IL_preprocessed_(transformed_to_eef)',
    #                            'experiment_1_(pull)')
    
    base_folder = os.path.join('/media/alejo/IL_data',
                               'DATA',
                               '03_IL_preprocessed_(transformed_to_eef)',
                               'experiment_1_(pull)')
                            #    'only_human_demos/with_palm_cam')



    # csv_files = [
    #     os.path.join(base_folder, f)
    #     for f in os.listdir(base_folder)
    #     if f.endswith('.csv')
    # ]
    
    # for trial in csv_files:
    #     # print(trial)
    #     plot_trial(trial)

    # ============= UNCOMMENT THIS TO PLOT THESE SPECIFIC TRIALS =============
    # cool_trials = [38, 70, 33, 105, 185, 75, 60, 71]
    cool_trials = [92]
    for trial in cool_trials:
        trial_name = 'trial_' + str(trial) + '_downsampled_aligned_data_transformed.csv'
        trial_path = os.path.join(base_folder, trial_name)
    
        plot_trial(trial_path)


def phases_stats_from_reading_entire_trial():

    # === Build Path ===
    base_folder_1 = os.path.join(r'D:',
                               'DATA',
                               '03_IL_preprocessed_(transformed_to_eef)',
                               'experiment_1_(pull)')
   

    base_folder_2 = os.path.join(r'D:',
                               'DATA',
                               '03_IL_preprocessed_(transformed_to_eef)',
                               'only_human_demos',
                               'with_palm_cam')

    base_folder_1 = '/home/alejo/Documents/DATA/03_IL_preprocessed_(transformed_to_eef)/experiment_1_(pull)'
    base_folder_2 = '/home/alejo/Documents/DATA/03_IL_preprocessed_(transformed_to_eef)/only_human_demos/with_palm_cam'


    base_folders = [base_folder_1, base_folder_2]

    approach_durations = []
    contact_durations = []
    pick_durations = []

    ctr=0
    for base_folder in base_folders:

        csv_files = [
            os.path.join(base_folder, f)
            for f in os.listdir(base_folder)
            if f.endswith('.csv')
        ]
        for trial in csv_files:
            approach, contact, pick = trial_phases_durations(trial)
            approach_durations.append(approach)
            contact_durations.append(contact)
            pick_durations.append(pick)

            ctr += 1
    print('Trials:', ctr)

    approach_durations.extend(np.ravel(approach))
    contact_durations.extend(np.ravel(contact))
    pick_durations.extend(np.ravel(pick))

    # Plot boxplots
    data = [approach_durations, contact_durations, pick_durations]
    labels = ['Approach', 'Contact', 'Pick']
    colors = ['skyblue', 'lightgreen', 'lightcoral']

    # === Create subplots ===
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

    for ax, d, label, color in zip(axes, data, labels, colors):
        # Make sure d is a 1D numpy array
        d = np.array(d).ravel()

        # Remove NaNs for plotting
        d_clean = d[~np.isnan(d)]

        # Plot boxplot
        box = ax.boxplot(d_clean, patch_artist=True, showmeans=True, meanline=True)
        for patch in box['boxes']:
            patch.set_facecolor(color)

        # Compute stats ignoring NaNs
        mean_val = np.nanmean(d)
        std_val = np.nanstd(d)
        print(f'{label}, mean:{mean_val}, std:{std_val}')

        # Text above box
        ax.text(1, mean_val + 0.05 * np.nanmax(d), f"μ={mean_val:.2f}\nσ={std_val:.2f}",
                ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_title(label)
        ax.set_xticks([])  # remove x-ticks

    axes[0].set_ylabel('Duration [s]')
    plt.suptitle('Durations per phase')
    plt.tight_layout()
    plt.show()


def phases_stats_from_last_row():

    # Approahc Path:
    base_folder = '/home/alejo/Documents/DATA/04_IL_preprocessed_(cropped_per_phase)/experiment_1_(pull)/'

    phases = ['phase_1_approach', 'phase_2_contact', 'phase_3_pick']


    for phase in phases:

        csv_files = [
                os.path.join(base_folder,phase, f)
                for f in os.listdir(os.path.join(base_folder,phase,))
                if f.endswith('.csv')
        ]

        phase_times =[]
        for trial in csv_files:

            trial_df = pd.read_csv(trial)
            time = trial_df['timestamp_vector'].values

            elapsed_time = time[-1] - time[0]
            phase_times.append(elapsed_time)

        print(f'{phase} mean Time: ', np.mean(phase_times), np.std(phase_times))


def compare_losses_plots():

    base = '/home/alejo/Documents/DATA/06_IL_learning/experiment_1_(pull)'   

    phases = ['phase_1_approach']    
    inputs_list = [['tof__inhand_cam_features__apple_prior']]    

    phases = ['phase_2_contact'] 
    inputs_list = [['tof__air_pressure__wrench__apple_prior', 'tof__air_pressure__apple_prior']]    

    # phases = ['phase_3_pick']
    # inputs_list = [['tof__air_pressure__wrench__apple_prior',
    #                 'tof__air_pressure__wrench',
    #                 'tof__wrench__apple_prior'
    #                 ]]

    
    for phase, inputs in zip(phases, inputs_list):       

        for input in inputs:

            plt.figure(figsize=(10, 5))

            npz_folder = os.path.join(base, phase, '0_timesteps', input)

            # Find npz files
            npz_files = [
                os.path.join(npz_folder, f)
                for f in os.listdir(os.path.join(npz_folder))
                if f.endswith('.npz')
            ]

            for file in npz_files:
                
                data = np.load(file)

                train_loss = data['train_losses']
                val_loss = data['val_losses']

                filtered_train_loss = gaussian_filter(train_loss, 2)
                filtered_val_loss = gaussian_filter(val_loss, 2)

                x_tr = len(filtered_train_loss) - 1
                y_tr = filtered_train_loss[-1]

                x_val = len(filtered_val_loss) - 1
                y_val = filtered_val_loss[-1]

                
                name = file.split(input)[1]
                name = name.split('_lstm')[0]

                cmap = plt.cm.tab10   # or viridis, plasma, tab20
                color = cmap(np.random.rand())
                
                # plt.figure(figsize=(10, 5))

                # plt.plot(filtered_train_loss, label=f'Tr Loss {name}', color=color, linestyle='--')
                plt.plot(filtered_val_loss, label=f'Val Loss {name}', color=color)
                # plt.plot(train_loss, color=color, alpha=0.4)
                plt.plot(val_loss, color=color, alpha=0.4)
                plt.text(
                    x_tr,
                    y_tr,
                    f'Tr {name}',
                    color=color,
                    fontsize=9,
                    va='center'
                )

                plt.text(
                    x_val,
                    y_val,
                    f'Val {name}',
                    color=color,
                    fontsize=9,
                    va='center'
                )
                

            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title(f'Training and Validation Loss Over Time\n{phase}\n{input}')
            plt.ylim([0,1])
            plt.legend()
            plt.grid(True)
        
    plt.show()
            
            
    




if __name__ == '__main__':

    
    # plot_batch_trials()
   
    # phases_stats_from_reading_entire_trial()
    phases_stats_from_last_row()
    # compare_losses_plots()