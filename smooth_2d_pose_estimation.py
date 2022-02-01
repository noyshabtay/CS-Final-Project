import pandas as pd
import os
from scipy import signal
from csv import reader
import matplotlib.pyplot as plt


def plot_leds_locations(df, keys_dict):
    fleftx = df[keys_dict['FLX']]
    flefty = df[keys_dict['FLY']]
    frightx = df[keys_dict['FRX']]
    frighty = df[keys_dict['FRY']]
    bleftx = df[keys_dict['BLX']]
    blefty = df[keys_dict['BLY']]
    brightx = df[keys_dict['BRX']]
    brighty = df[keys_dict['BRY']]
    time = list(range(len(fleftx)))
    # plot
    plt.figure(figsize=(8,8))
    #plt.gcf().autofmt_xdate()

    plt.plot(time, fleftx, label='fl-x')
    plt.plot(time,flefty, label='fl-y')

    plt.plot(time,frightx, label='fr-x')
    plt.plot(time,frighty, label='fr-y')

    plt.plot(time,bleftx, label='bl-x')
    plt.plot(time,blefty, label='bl-y')

    plt.plot(time,brightx, label='br-x')
    plt.plot(time,brighty, label='br-y')

    plt.legend(loc='upper left')
    plt.show()


def extract_header(csvs_dir, file):
    in_fle = os.path.join(csvs_dir, file)
    header_lines = []
    with open(in_fle, 'r') as read_obj:
        # pass the file object to reader() to get the reader object
        csv_reader = reader(read_obj)
        # Iterate over each row in the csv using reader object
        i = 0
        for row in csv_reader:
            i += 1
            if i > 3:
                continue

            header_lines.append(row)
    return header_lines


def apply_filter(x, chunck_maximal_window, window):
    window = min(window, chunck_maximal_window)
    if 'x' in x.name or 'y' in x.name:
        return signal.savgol_filter(x, window, 1)
    return x


def filter_chunck(chunck, filter_name, window=13):
    half_len = int(len(chunck) / 2)
    chunck_maximal_window = half_len if half_len % 2 == 1 else (half_len - 1)
    if chunck_maximal_window < 2:
        return chunck
    if filter_name == "savgol":
        chunck = chunck.apply(lambda x: apply_filter(x, chunck_maximal_window, window))
    else:
        chunck = chunck.apply(lambda x: signal.medfilt(x, (window,)) if ('x' in x.name or 'y' in x.name) else x)
    return chunck

def write_df_to_csv(headers, df,csv_file, csvs_dir):
    csv_name = csv_file.split('.')[0]
    print(csv_name)
    with open(f'{csvs_dir}/{csv_name}_filtered.csv', 'w', newline='') as out:
        for row in headers:
            out.write(','.join(row)+'\n')
        df.to_csv(out, index=False, header=False)

    print('output:',f'{csvs_dir}/{csv_name}_filtered.csv')


def write_filtered_data(csvs_dir, csv_file):
    df = pd.read_csv(os.path.join(csvs_dir, csv_file), header=2).astype(float)
    header_lines = extract_header(csvs_dir, csv_file)
    keys = list(df.keys())
    keys_dict = {'FLX': keys[1], 'FLY': keys[2], 'FLL': keys[3],
                 'FRX': keys[4], 'FRY': keys[5], 'FRL': keys[6],
                 'BLX': keys[7], 'BLY': keys[8], 'BLL': keys[9],
                 'BRX': keys[10], 'BRY': keys[11], 'BRL': keys[12]}
    plot_leds_locations(df, keys_dict)

    likelihood = 0.95
    led_missing_chuncks = df.loc[(df[keys_dict['FLL']] < likelihood) | (df[keys_dict['FRL']] < likelihood)
                                 | (df[keys_dict['BLL']] < likelihood) | (df[keys_dict['BRL']] < likelihood)]
    led_missing_chuncks_index = led_missing_chuncks.index

    filter_name = "savgol"
    frames = []
    last = -1
    for i in range(len(led_missing_chuncks)):
        if led_missing_chuncks_index[i] - last > 1:
            chunck = df[last + 1:led_missing_chuncks_index[i] - 1]
            chunck = filter_chunck(chunck, filter_name, window=11)
            frames.append(chunck)
        last = led_missing_chuncks_index[i]

    chunck = df[last + 1:]
    chunck = filter_chunck(chunck, filter_name, window=11)
    frames.append(chunck)

    only_likely_filtered_result = pd.concat(frames)
    only_likely_filtered_result = only_likely_filtered_result.sort_index()

    led_missing_chuncks = pd.DataFrame(led_missing_chuncks, index=led_missing_chuncks_index)
    frames.append(led_missing_chuncks)
    filtered_result = pd.concat(frames)
    filtered_result = filtered_result.sort_index()

    plot_leds_locations(only_likely_filtered_result, keys_dict)
    write_df_to_csv(header_lines, filtered_result, csv_file, csvs_dir)
