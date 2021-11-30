from glob import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from modules.filters import butter_highpass_filter, butter_bandpass_filter


def pre_process():
    debug = False
    use_freq_resp = True
    remove_dc = True
    sampling_rate = 1000  # In Hz
    length_keep = 3500
    num_samples_to_keep_before = 1500
    num_samples_to_keep_after = 1500
    extra_signal_number = (4 - 1) * 2 + (5 - 1) * 10 + (6 - 1) * 115 + (14 - 1) + (24 - 1) + (26 - 1) + (30 - 1) + (
            48 - 1)  # too mussy, should count manually

    dir_name = 'D://Xin Zhang/Sensor/conbination/NewData/All'
    dir_to_save = 'D://Xin Zhang/Sensor/conbination/processed_data/'
    file_name_to_save = "All_data_twoinputs"
    file_list, all_data, _ = read_files(dir_name, debug)
    single_signals, single_time, new_file_list, all_classes = cut_and_class(all_data, file_list, extra_signal_number,
                                                                            sampling_rate, length_keep,
                                                                            num_samples_to_keep_before,
                                                                            num_samples_to_keep_after, debug)
    final_array = filter_and_fft(single_signals, single_time, all_classes, new_file_list, sampling_rate, remove_dc,
                                 use_freq_resp, debug)

    # if(not remove_dc):
    #     file_name_to_save = file_name_to_save + "_original"
    # elif(use_freq_resp):
    #     file_name_to_save = file_name_to_save + "_freq_resp"

    with open(dir_to_save + file_name_to_save + '.npy', 'wb') as f:
        np.save(f, final_array)
    print("Saved to:", dir_to_save + file_name_to_save + '.npy')


def read_files(dir_name, debug):
    # read all txt files
    file_list = [y for x in os.walk(dir_name) for y in glob(os.path.join(x[0], '*.txt'))]
    print('{} files in total'.format(len(file_list)))

    # check the number of multi files
    num_multi_files = 0
    for i in range(len(file_list)):
        if 'Multi' in file_list[i]:
            num_multi_files += 1
    print('The number of multi-files is ', num_multi_files)

    # to dataframe
    all_data = []
    for j in file_list:
        temp_df = pd.read_csv(j, sep="\t")
        for x in temp_df.columns:
            temp_df[x] = pd.to_numeric(temp_df[x], downcast="float")
        all_data.append(temp_df)

    # plot all signals & point out all the peaks, if necessary
    if debug:
        ctr = 0
        for j in range(len(all_data)):
            if 'Multi' in file_list[j]:
                if 'trough' in file_list[j]:
                    peak_index, _ = find_peaks(-all_data[j]['Current (A)'], distance=3500)
                    peaks = peak_index / 1000
                elif 'peak' in file_list[j]:
                    peak_index, _ = find_peaks(all_data[j]['Current (A)'], distance=3500)
                    peaks = peak_index / 1000

                plt.plot(all_data[j]['Elapsed Time (s)'], all_data[j]['Current (A)'])
                plt.scatter(peaks, all_data[j]['Current (A)'][peak_index], c='red')

                plt.title(file_list[ctr].rsplit('/', 1)[1].rsplit('\\', 2)[2])
                plt.xlabel("Time (s)")
                plt.ylabel("Current (A)")
                plt.pause(0.0001)
                ctr += 1
            else:
                plt.plot(all_data[j]['Elapsed Time (s)'], all_data[j]['Current (A)'])
                plt.title(file_list[ctr].rsplit('/', 1)[1].rsplit('\\', 2)[2])
                plt.xlabel("Time (s)")
                plt.ylabel("Current (A)")
                plt.pause(0.0001)
                ctr += 1

    return file_list, all_data, num_multi_files


def cut_and_class(data, file_list, extra_signal_number, sampling_rate, length_keep, num_keep_before, num_keep_after,
                  debug):
    num_files = len(data)
    total_num_samples_to_keep = num_keep_before + num_keep_after

    new_file_list = [[] for _ in range(num_files + extra_signal_number)]
    reserved_samples = [[] for _ in range(num_files + extra_signal_number)]
    reserved_time = [[] for _ in range(num_files + extra_signal_number)]
    time_to_keep = np.linspace(0, total_num_samples_to_keep, num=total_num_samples_to_keep)

    class_all = [[] for _ in range(num_files + extra_signal_number)]

    class1 = ['speak-yes_Multi_peak', 'speak-yes_Multi_trough']
    class2 = ['speak-no_Multi_trough', 'speak-no_Multi_peak']
    class3 = ['speak-one', 'speak-one_Multi_trough']
    class4 = ['speak-two', 'speak-two_Multi_peak', 'speak-two_Multi_trough']

    class5 = ['move-shaking']
    class6 = ['move-nodding_Multi_trough_30s_5s per file', 'move-nodding_Multi_peak_30s_5s per file']
    class7 = ['move-stretch_Multi_trough']

    class8 = ['combine-nodding-no_Multi_trough']
    class9 = ['combine-nodding-yes']
    class10 = ['combine-shaking-no']
    class11 = ['combine-shaking-yes_Multi_trough']

    x = 0
    for i in range(num_files):
        class_match = False

        temp_data = data[i]
        actual_name = file_list[i].rsplit('/', 1)[1].rsplit('.', 1)[0].rsplit('\\', 2)[1]

        # deal with Multi =================================================================================
        if 'Multi' in file_list[i]:
            if 'trough' in file_list[i]:
                peak_index, _ = find_peaks(-temp_data['Current (A)'], distance=3500)
                if 'No_4' in file_list[i]:
                    temp_six_peak = temp_data['Current (A)'][peak_index].nsmallest(4).index
                if 'No_5' in file_list[i]:
                    temp_six_peak = temp_data['Current (A)'][peak_index].nsmallest(5).index
                if 'No_6' in file_list[i]:
                    temp_six_peak = temp_data['Current (A)'][peak_index].nsmallest(6).index
                if 'No_14' in file_list[i]:
                    temp_six_peak = temp_data['Current (A)'][peak_index].nsmallest(14).index
                if 'No_24' in file_list[i]:
                    temp_six_peak = temp_data['Current (A)'][peak_index].nsmallest(24).index
                if 'No_26' in file_list[i]:
                    temp_six_peak = temp_data['Current (A)'][peak_index].nsmallest(26).index
                if 'No_30' in file_list[i]:
                    temp_six_peak = temp_data['Current (A)'][peak_index].nsmallest(30).index
                if 'No_48' in file_list[i]:
                    temp_six_peak = temp_data['Current (A)'][peak_index].nsmallest(48).index

            if 'peak' in file_list[i]:
                peak_index, _ = find_peaks(temp_data['Current (A)'], distance=3500)
                temp_six_peak = temp_data['Current (A)'][peak_index].nlargest(6).index

            for k in range(len(temp_six_peak)):
                if temp_six_peak[k] - num_keep_before < 0:
                    print('Too close to the start in', file_list[i].rsplit('/', 1)[1])

                    fix = num_keep_before - temp_six_peak[k]
                    samples_to_keep = pd.Series(temp_data['Current (A)'][0])
                    for q in range(fix - 1):
                        samples_to_keep = samples_to_keep.append(pd.Series(temp_data['Current (A)'][0]),
                                                                 ignore_index=True)
                    samples_to_keep = samples_to_keep.append(
                        temp_data['Current (A)'][0:temp_six_peak[k] + num_keep_after], ignore_index=True)

                    print('fixed (start) length is', len(samples_to_keep))
                    print('================================================')

                elif temp_six_peak[k] + num_keep_after > len(temp_data['Current (A)']):
                    print('Too close to the end in', file_list[i].rsplit('/', 1)[1])

                    fix = temp_six_peak[k] + num_keep_after - len(temp_data['Current (A)'])
                    samples_to_keep = temp_data['Current (A)'][
                                      temp_six_peak[k] - num_keep_before:len(temp_data['Current (A)']) - 1]
                    for q in range(fix + 1):
                        samples_to_keep = samples_to_keep.append(
                            pd.Series(temp_data['Current (A)'][len(temp_data['Current (A)']) - 1]), ignore_index=True)

                    print('fixed (end) length is', len(samples_to_keep))
                    print('================================================')

                else:
                    samples_to_keep = temp_data['Current (A)'][
                                      temp_six_peak[k] - num_keep_before:temp_six_peak[
                                                                             k] + num_keep_after]

                reserved_samples[x].append(samples_to_keep)
                reserved_time[x].append(time_to_keep)

                new_file_list[x] = file_list[i]

                # distinguish classes =============================================================================
                if any(temp_name in actual_name for temp_name in class1):
                    class_all[x] = 0
                    class_match = True

                if any(temp_name in actual_name for temp_name in class2):
                    class_all[x] = 1
                    class_match = True

                if any(temp_name in actual_name for temp_name in class3):
                    class_all[x] = 2
                    class_match = True

                if any(temp_name in actual_name for temp_name in class4):
                    class_all[x] = 3
                    class_match = True

                if any(temp_name in actual_name for temp_name in class5):
                    class_all[x] = 4
                    class_match = True

                if any(temp_name in actual_name for temp_name in class6):
                    class_all[x] = 5
                    class_match = True

                if any(temp_name in actual_name for temp_name in class7):
                    class_all[x] = 6
                    class_match = True

                if any(temp_name in actual_name for temp_name in class8):
                    class_all[x] = 7
                    class_match = True

                if any(temp_name in actual_name for temp_name in class9):
                    class_all[x] = 8
                    class_match = True

                if any(temp_name in actual_name for temp_name in class10):
                    class_all[x] = 9
                    class_match = True

                if any(temp_name in actual_name for temp_name in class11):
                    class_all[x] = 10
                    class_match = True

                if not class_match:
                    print("Could not find any class matches for ", actual_name)

                x += 1

        # deal with single=================================================================================================
        else:
            new_file_list[x] = file_list[i]

            # euqal signal length =============================================================================
            indices_to_keep = temp_data['Current (A)'].idxmin()

            if indices_to_keep - num_keep_before < 0:
                print('Too close to the start in', file_list[i].rsplit('/', 1)[1])

                fix = num_keep_before - indices_to_keep
                samples_to_keep = pd.Series(temp_data['Current (A)'][0])
                for q in range(fix - 1):
                    samples_to_keep = samples_to_keep.append(pd.Series(temp_data['Current (A)'][0]), ignore_index=True)
                samples_to_keep = samples_to_keep.append(
                    temp_data['Current (A)'][0:indices_to_keep + num_keep_after], ignore_index=True)

                print('fixed (start) length is', len(samples_to_keep))
                print('================================================')

            elif indices_to_keep + num_keep_after > len(temp_data['Current (A)']):
                print('Too close to the end in', file_list[i].rsplit('/', 1)[1])

                fix = indices_to_keep + num_keep_after - len(temp_data['Current (A)'])
                samples_to_keep = temp_data['Current (A)'][
                                  indices_to_keep - num_keep_before:len(temp_data['Current (A)']) - 1]
                for q in range(fix + 1):
                    samples_to_keep = samples_to_keep.append(
                        pd.Series(temp_data['Current (A)'][len(temp_data['Current (A)']) - 1]), ignore_index=True)

                print('fixed (end) length is', len(samples_to_keep))
                print('================================================')

            else:
                samples_to_keep = temp_data['Current (A)'][
                                  indices_to_keep - num_keep_before:indices_to_keep + num_keep_after]

            reserved_samples[x].append(samples_to_keep)
            reserved_time[x].append(time_to_keep)

            # distinguish classes =============================================================================
            if any(temp_name in actual_name for temp_name in class1):
                class_all[x] = 0
                class_match = True

            if any(temp_name in actual_name for temp_name in class2):
                class_all[x] = 1
                class_match = True

            if any(temp_name in actual_name for temp_name in class3):
                class_all[x] = 2
                class_match = True

            if any(temp_name in actual_name for temp_name in class4):
                class_all[x] = 3
                class_match = True

            if any(temp_name in actual_name for temp_name in class5):
                class_all[x] = 4
                class_match = True

            if any(temp_name in actual_name for temp_name in class6):
                class_all[x] = 5
                class_match = True

            if any(temp_name in actual_name for temp_name in class7):
                class_all[x] = 6
                class_match = True

            if any(temp_name in actual_name for temp_name in class8):
                class_all[x] = 7
                class_match = True

            if any(temp_name in actual_name for temp_name in class9):
                class_all[x] = 8
                class_match = True

            if any(temp_name in actual_name for temp_name in class10):
                class_all[x] = 9
                class_match = True

            if any(temp_name in actual_name for temp_name in class11):
                class_all[x] = 10
                class_match = True

            if not class_match:
                print("Could not find any class matches for ", actual_name)

            x += 1

    # plot each single signal if necessary
    if debug:
        for i in range(len(reserved_samples)):
            for j in range(len(reserved_samples[i])):
                if len(reserved_samples[i][j]) == total_num_samples_to_keep:
                    plt.plot(reserved_time[i][j], reserved_samples[i][j])
                    plt.xlabel("Time (s)")
                    plt.ylabel("Current (A)")
                    plt.title(new_file_list[i].rsplit('/', 1)[1].rsplit('\\', 2)[2])
                    plt.pause(0.000001)
                else:
                    print('Not enough points in', new_file_list[i].rsplit('/', 1)[1])

    return reserved_samples, reserved_time, new_file_list, class_all


def filter_and_fft(all_samples, all_time, all_classes, file_list, sampling_rate, remove_dc, use_freq_resp, debug):
    overall_array = np.array([])
    temp_array = []

    for i in range(len(all_samples)):
        for j in range(len(all_samples[i])):
            normed_wave = all_samples[i][j] / all_samples[i][j].max()
            if remove_dc:
                filtered_wave = butter_bandpass_filter(normed_wave, 1, 300, sampling_rate, 2)
                # filtered_wave = butter_highpass_filter(normed_wave, 0.01, sampling_rate, 2)
                if debug:
                    plt.plot(all_time[i][j], filtered_wave)
                    plt.title(file_list[i].rsplit('/', 1)[1].rsplit('\\', 2)[2])
                    plt.xlabel("Time (s)")
                    plt.ylabel("filtered_wave")
                    plt.pause(0.000001)
                if use_freq_resp:
                    temp_data = np.fft.fft(normed_wave).real
                    temp_array = np.expand_dims(
                        np.concatenate([all_samples[i][j], temp_data, np.array([all_classes[i]])]), 0)
                    if debug:
                        plt.plot(all_time[i][j], temp_data)
                        plt.ylim((-0.0005, 0.0005))
                        plt.ylabel("FFT")
                        plt.title(file_list[i].rsplit('/', 1)[1].rsplit('\\', 2)[2])
                        plt.pause(0.000001)

                else:
                    temp_array = np.expand_dims(np.concatenate([filtered_wave, np.array([all_classes[i]])]), 0)
                    if debug:
                        plt.plot(all_time[i][j], filtered_wave)
                        plt.ylim((-0.00025, 0.00025))
                        plt.ylabel("current")
                        plt.xlabel("time")
                        plt.title(file_list[i].rsplit('/', 1)[1].rsplit('\\', 2)[2])
                        plt.pause(0.000001)
            else:
                temp_array = np.expand_dims(np.concatenate([all_samples[i][j], np.array([all_classes[i]])]), 0)
                if debug:
                    plt.plot(all_time[i][j], all_samples[i][j])
                    plt.ylabel("current")
                    plt.xlabel("time")
                    plt.title(file_list[i].rsplit('/', 1)[1].rsplit('\\', 2)[2])
                    plt.pause(0.000001)

        if len(overall_array) == 0:
            overall_array = temp_array
        else:
            overall_array = np.concatenate([overall_array, temp_array], 0)

    print(overall_array.shape)
    print('{} signals in total'.format(overall_array.shape[0]))

    return overall_array


if __name__ == '__main__':
    pre_process()
