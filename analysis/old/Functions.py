# Filter on MC and BS events
def select_MC_and_BS_events(cut_off,
                            pid_x_pred,
                            MC_pid,
                            MC_events,
                            BS_events,
                            df_BS,
                            df_MC,
                            printing=True):
    import pandas as pd
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    df_BS = df_BS[df_BS[pid_x_pred] > cut_off]

    if not isinstance(MC_pid, (list, tuple, set)):
        MC_pid = [MC_pid]

    df_MC = df_MC[df_MC["pid"].isin(MC_pid) & (df_MC[pid_x_pred] > cut_off)]

    print("Available MC after cut:", len(df_MC))
    print("Available BS after cut:", len(df_BS))


    # Take first 5 events from both MC and BS
    df_BS = df_BS.head(BS_events)
    df_MC = df_MC.head(MC_events)

    # Create event lists
    bs_event_list = df_BS["event_no"].tolist()
    mc_event_list = df_MC["event_no"].tolist()

    #prepares indexing for the db dataframes
    df_bs_events = pd.DataFrame({
    "event_no": bs_event_list,
    "is_MC": False
    })

    df_mc_events = pd.DataFrame({
    "event_no": mc_event_list,
    "is_MC": True
    })

    df_all_events = pd.concat([df_bs_events, df_mc_events], ignore_index=True)

    if printing:
        print(f"Total events with {pid_x_pred} > {cut_off}:", len(df_BS))
        print(f"Total events with {pid_x_pred} > {cut_off} and pid = {MC_pid}:", len(df_MC))

        print(f"\nSelected {len(mc_event_list)} events of MC, and {len(bs_event_list)} events of BS:")
        print("")
        print("5 first BS events:")
        print(
        pd.concat(
            [
                df_BS.head().reset_index(drop=True),
                df_bs_events.head().reset_index(drop=True)
            ],
            axis=1
        ))

        print("")
        print("5 first MC events:")
        print(
        pd.concat(
            [
                df_MC.head().reset_index(drop=True),
                df_mc_events.head().reset_index(drop=True)
            ],
            axis=1
        ))


    return df_all_events













def data_frame_features(df_all_events):
    import sqlite3
    import numpy as np
    import pandas as pd
    import time

    DB_MC = "/groups/icecube/holgerkc/Thesis_Analysis/old/MC_pulsemap_muon_noise_neutrino_02_02_2026.db"
    DB_BS = "file:/lustre/hpc/project/icecube/Burnsample/databases/IC86.22/burnsample_IC8622_merged.db?mode=ro&immutable=1"

    # We use SplitInIcePulses, in the spirit of Niels.
    MC_PULSE_TABLE = "SplitInIcePulses"
    BS_PULSE_TABLE = "SplitInIcePulses"

    rows = []
    t0 = time.time()

    # one connection per DB (faster than reconnecting each event)
    conn_mc = sqlite3.connect(DB_MC, timeout=10)
    cur_mc = conn_mc.cursor() # a cursor is used to do SQL-requests

    conn_bs = sqlite3.connect(DB_BS, timeout=10, uri=True)
    cur_bs = conn_bs.cursor()

    try:
        for i, (event_no, is_MC) in enumerate(df_all_events[["event_no", "is_MC"]].itertuples(index=False), 1): # This loops over all (event_no, is_MC) one row at the time. i is just a count.

            if bool(is_MC): # if is_MC = True
                cur = cur_mc
                table = MC_PULSE_TABLE
            else:           # if is_MC = False
                cur = cur_bs
                table = BS_PULSE_TABLE

            cur.execute(f"""
                SELECT charge, dom_time, dom_z
                FROM {table}
                WHERE event_no = ?
            """, (int(event_no),))  # takes charge, dom_time and dom_z for the given event_no from the respective table

            pulses = cur.fetchall() # sets up pulses as [(charge, dom_time, dom_z),
                                    #                    (charge, dom_time, dom_z),
                                    #                    ...]
            if not pulses:
                continue # if there are no pulses for this event, skip this iteration of the loop and move on to the next event

            arr = np.asarray(pulses, dtype=float) # converts the list of tuples (pulses) into a numpy array for easier processing.
            q = arr[:, 0] # column 0 = charge
            t = arr[:, 1] # column 1 = dom_time
            z = arr[:, 2] # column 2 = dom_z


            # Input variables for our ML!
            Q_tot = q.sum()
            if Q_tot <= 0:
                continue # if the total charge is zero or negative, skip this event.
            N_hits = len(q)
            dt = t.max() - t.min()
            dz = z.max() - z.min()
            z_cw = (q * z).sum() / Q_tot #Charge-weighted mean of the z positions of the pulses, e.g. if it was through going muon, z_cw would be somewhere in the middle of the detector, while for a non through going it would be closer to where it stopped.

            # We simply build our model on only 4 variables, total charge, number of hits, time duration and z extension of the event. We could of course add more variables, but let's keep it simple for now.

            # Saves a dictionary with the variables and labels for this event and appends it to the list of rows. That is, every row is a event in rows.
            rows.append({
                "event_no": int(event_no),
                "is_MC": bool(is_MC),
                "y": int(bool(is_MC)),     # 0/1 bool label for the BDT, where 1 = MC and 0 = BS
                "Q_tot": float(Q_tot),
                "N_hits": int(N_hits),
                "dt": float(dt),
                "dz": float(dz),
                "z_cw": float(z_cw),
            })

            #This is just for our sanity, to track the progress of the code.
            if i % 200 == 0:
                elapsed = time.time() - t0
                print(f"processed {i}/{len(df_all_events)} | {i/elapsed:.1f} events/s | rows kept: {len(rows)}")

    finally: # closes database connections after the loop is done, even if there was an error
        conn_mc.close()
        conn_bs.close()

    #Final dataframe with all the variables and labels for our BDT. Each row is an event.
    df_features = pd.DataFrame(rows)  #converts to pandas df

    #prints first 5 rows to see how the dataframe looks like.
    print("\nFeature table head:")
    print(df_features.head(5).to_string(index=False))

    # checking the MC/BS distribution is correct.
    print("\nLabel counts in df_features (y = is_MC):")
    print(df_features["y"].value_counts())

    return df_features

















def dogshit():
    print("This is a placeholder function. Please replace it with actual code.")