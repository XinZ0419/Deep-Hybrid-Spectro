import os
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from modules.filters import butter_bandpass_filter


def read_files(opt):
    # read all txt files
    file_list = [y for x in os.walk(opt.raw_data_file) for y in glob(os.path.join(x[0], '*.txt'))]
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
        if j.find('Multi') == -1:
            temp_df.drop(temp_df.head(150).index, inplace=True)
        all_data.append(temp_df)

    # plot all signals & point out all the peaks, if necessary
    if opt.debug:
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
                plt.savefig(opt.processed_data_folder+file_list[ctr].rsplit('/', 1)[1].rsplit('\\', 2)[2]+'.jpg')
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


def cut_and_class(opt, data, file_list):
    num_files = len(data)
    total_num_samples_to_keep = opt.num_samples_to_keep_before + opt.num_samples_to_keep_after
    new_file_list = [[] for _ in range(num_files + opt.extra_signal_number)]
    reserved_samples = [[] for _ in range(num_files + opt.extra_signal_number)]
    reserved_time = [[] for _ in range(num_files + opt.extra_signal_number)]
    time_to_keep = np.linspace(0, total_num_samples_to_keep, num=total_num_samples_to_keep)

    class_all = [[] for _ in range(num_files + opt.extra_signal_number)]
    partic_all = [[] for _ in range(num_files + opt.extra_signal_number)]
    pc_all = [[] for _ in range(num_files + opt.extra_signal_number)]

    class0 = ['speak-yes_Multi_peak', 'speak-yes_Multi_trough', 'Yes_Multi_peak', 'Yes_Multi_trough', 'speak-yes', 'yes', 'Yes']
    class1 = ['speak-no_Multi_trough', 'speak-no_Multi_peak', 'No_Multi_peak', 'No_Multi_trough', 'speak-no', 'no']
    class2 = ['speak-one', 'speak-one_Multi_trough', 'One_Multi_trough', 'one', 'One', 'SPEAK-ONE']
    class3 = ['speak-two', 'speak-two_Multi_peak', 'speak-two_Multi_trough', 'Two_Multi_trough', 'Two_Multi_peak', 'two', 'Two', 'SPEAK-TWO']

    class4 = ['move-shaking', 'shaking_Multi_trough', 'shaking', 'Shaking', 'MOVE-SHAKING']
    class5 = ['move-nodding', 'move-nodding_Multi_trough_30s_5s per file', 'move-nodding_Multi_peak_30s_5s per file',
              'nodding_Multi_peak', 'nodding_Multi_trough', 'nodding', 'Nodding']
    class6 = ['move-stretch', 'move-stretch_Multi_trough', 'stretch']

    class7 = ['combine-nodding-no', 'combine-nodding-no_Multi_trough', 'nodding_no', 'Nodding_No', 'No_Nodding']
    class8 = ['combine-nodding-yes', 'nodding_yes', 'SPEAK-YES-NODING', 'Yes_nodding']
    class9 = ['combine-shaking-no', 'shaking_no', 'SPEAK-NO-SHAKING', 'No_Shaking']
    class10 = ['combine-shaking-yes', 'combine-shaking-yes_Multi_trough', 'shaking_yes', 'Shaking_Yes', 'Yes_Shaking']

    x = 0
    for i in range(num_files):
        class_match = False

        temp_data = data[i]
        actual_name = file_list[i].rsplit('/', 1)[1].rsplit('.', 1)[0].rsplit('\\', 2)[2]
        partic_name = file_list[i].rsplit('/', 1)[1].rsplit('.', 1)[0].rsplit('\\', 2)[1]
        # deal with Multi =================================================================================
        if 'Multi' in file_list[i]:
            if 'trough' in file_list[i]:
                peak_index, _ = find_peaks(-temp_data['Current (A)'], distance=3500)
                if '14in1' in file_list[i]:
                    temp_six_peak = temp_data['Current (A)'][peak_index].nsmallest(14).index
                elif '24in1' in file_list[i]:
                    temp_six_peak = temp_data['Current (A)'][peak_index].nsmallest(24).index
                elif '26in1' in file_list[i]:
                    temp_six_peak = temp_data['Current (A)'][peak_index].nsmallest(26).index
                elif '30in1' in file_list[i]:
                    temp_six_peak = temp_data['Current (A)'][peak_index].nsmallest(30).index
                elif '48in1' in file_list[i]:
                    temp_six_peak = temp_data['Current (A)'][peak_index].nsmallest(48).index
                elif '2in1' in file_list[i]:
                    temp_six_peak = temp_data['Current (A)'][peak_index].nsmallest(2).index
                elif '3in1' in file_list[i]:
                    temp_six_peak = temp_data['Current (A)'][peak_index].nsmallest(3).index
                elif '4in1' in file_list[i]:
                    temp_six_peak = temp_data['Current (A)'][peak_index].nsmallest(4).index
                elif '5in1' in file_list[i]:
                    temp_six_peak = temp_data['Current (A)'][peak_index].nsmallest(5).index
                elif '6in1' in file_list[i]:
                    temp_six_peak = temp_data['Current (A)'][peak_index].nsmallest(6).index
                else:
                    raise TypeError('miss number_in_1 in trough')

            elif 'peak' in file_list[i]:
                peak_index, _ = find_peaks(temp_data['Current (A)'], distance=3500)
                if '2in1' in file_list[i]:
                    temp_six_peak = temp_data['Current (A)'][peak_index].nlargest(2).index
                if '3in1' in file_list[i]:
                    temp_six_peak = temp_data['Current (A)'][peak_index].nlargest(3).index
                elif '4in1' in file_list[i]:
                    temp_six_peak = temp_data['Current (A)'][peak_index].nlargest(4).index
                elif '5in1' in file_list[i]:
                    temp_six_peak = temp_data['Current (A)'][peak_index].nlargest(5).index
                elif '6in1' in file_list[i]:
                    temp_six_peak = temp_data['Current (A)'][peak_index].nlargest(6).index
                else:
                    raise TypeError('miss number_in_1 in peak')
            else:
                raise TypeError('miss trough or peak')

            for k in range(len(temp_six_peak)):
                if temp_six_peak[k] - opt.num_samples_to_keep_before < 0:
                    print('Too close to the start in', file_list[i].rsplit('/', 1)[1])

                    fix = opt.num_samples_to_keep_before - temp_six_peak[k]
                    samples_to_keep = pd.Series(temp_data['Current (A)'][0])
                    for q in range(fix - 1):
                        samples_to_keep = samples_to_keep.append(pd.Series(temp_data['Current (A)'][0]),
                                                                 ignore_index=True)
                    samples_to_keep = samples_to_keep.append(
                        temp_data['Current (A)'][0:temp_six_peak[k] + opt.num_samples_to_keep_after], ignore_index=True)

                    print('fixed (start) length is', len(samples_to_keep))
                    print('================================================')

                elif temp_six_peak[k] + opt.num_samples_to_keep_after > len(temp_data['Current (A)']):
                    print('Too close to the end in', file_list[i].rsplit('/', 1)[1])

                    fix = temp_six_peak[k] + opt.num_samples_to_keep_after - len(temp_data['Current (A)'])
                    samples_to_keep = temp_data['Current (A)'][
                                      temp_six_peak[k] - opt.num_samples_to_keep_before:len(temp_data['Current (A)']) - 1]
                    for q in range(fix + 1):
                        samples_to_keep = samples_to_keep.append(
                            pd.Series(temp_data['Current (A)'][len(temp_data['Current (A)']) - 1]), ignore_index=True)

                    print('fixed (end) length is', len(samples_to_keep))
                    print('================================================')

                else:
                    samples_to_keep = temp_data['Current (A)'][
                                      temp_six_peak[k] - opt.num_samples_to_keep_before:temp_six_peak[
                                                                             k] + opt.num_samples_to_keep_after]
                reserved_samples[x].append(samples_to_keep)
                reserved_time[x].append(time_to_keep)

                new_file_list[x] = file_list[i]

                # distinguish classes =============================================================================
                if any(temp_name in actual_name for temp_name in class2):
                    class_all[x] = 2
                    class_match = True
                elif any(temp_name in actual_name for temp_name in class3):
                    class_all[x] = 3
                    class_match = True
                elif any(temp_name in actual_name for temp_name in class6):
                    class_all[x] = 6
                    class_match = True
                elif 'Nodding' in actual_name:
                    if 'Yes_Nodding' in actual_name:
                        class_all[x] = 8
                        class_match = True
                    elif 'No_Nodding' in actual_name:
                        class_all[x] = 7
                        class_match = True
                    else:
                        class_all[x] = 5
                        class_match = True
                elif 'Shaking' in actual_name:
                    if 'Yes_Shaking' in actual_name:
                        class_all[x] = 10
                        class_match = True
                    elif 'No_Shaking' in actual_name:
                        class_all[x] = 9
                        class_match = True
                    else:
                        class_all[x] = 4
                        class_match = True
                elif 'no' in actual_name:
                    class_all[x] = 1
                    class_match = True
                elif 'yes' in actual_name:
                    class_all[x] = 0
                    class_match = True
                else:
                    print("Could not find any class matches for ", actual_name)
                    raise TypeError('Please check the class!')

                if opt.partic:
                    if '-1' in partic_name:
                        partic_all[x] = 1
                    elif '-2' in partic_name:
                        partic_all[x] = 2
                    else:
                        raise TypeError('cannot find participate number')
                    pc_all[x] = int(str(partic_all[x]) + str(class_all[x]))
                else:
                    partic_all[x] = 0
                    pc_all[x] = 0

                x += 1

        # deal with single=================================================================================================
        else:
            new_file_list[x] = file_list[i]

            # euqal signal length =============================================================================
            indices_to_keep = temp_data['Current (A)'].idxmin()

            if indices_to_keep - opt.num_samples_to_keep_before < 0:
                print('Too close to the start in', file_list[i].rsplit('/', 1)[1])

                fix = opt.num_samples_to_keep_before - indices_to_keep
                samples_to_keep = pd.Series(temp_data['Current (A)'][0])
                for q in range(fix - 1):
                    samples_to_keep = samples_to_keep.append(pd.Series(temp_data['Current (A)'][0]), ignore_index=True)
                samples_to_keep = samples_to_keep.append(
                    temp_data['Current (A)'][0:indices_to_keep + opt.num_samples_to_keep_after], ignore_index=True)

                print('fixed (start) length is', len(samples_to_keep))
                print('================================================')

            elif indices_to_keep + opt.num_samples_to_keep_after > len(temp_data['Current (A)']):
                print('Too close to the end in', file_list[i].rsplit('/', 1)[1])

                fix = indices_to_keep + opt.num_samples_to_keep_after - len(temp_data['Current (A)'])
                samples_to_keep = temp_data['Current (A)'][
                                  indices_to_keep - opt.num_samples_to_keep_before:len(temp_data['Current (A)']) - 1]
                for q in range(fix + 1):
                    samples_to_keep = samples_to_keep.append(
                        pd.Series(temp_data['Current (A)'][len(temp_data['Current (A)']) - 1]), ignore_index=True)

                print('fixed (end) length is', len(samples_to_keep))
                print('================================================')

            else:
                samples_to_keep = temp_data['Current (A)'][
                                  indices_to_keep - opt.num_samples_to_keep_before:indices_to_keep + opt.num_samples_to_keep_after]
            reserved_samples[x].append(samples_to_keep)
            reserved_time[x].append(time_to_keep)

            # distinguish classes =============================================================================
            if any(temp_name in actual_name for temp_name in class2):
                class_all[x] = 2
                class_match = True
            elif any(temp_name in actual_name for temp_name in class3):
                class_all[x] = 3
                class_match = True
            elif any(temp_name in actual_name for temp_name in class6):
                class_all[x] = 6
                class_match = True
            elif 'Nodding' in actual_name:
                if 'Yes_Nodding' in actual_name:
                    class_all[x] = 8
                    class_match = True
                elif 'No_Nodding' in actual_name:
                    class_all[x] = 7
                    class_match = True
                else:
                    class_all[x] = 5
                    class_match = True
            elif 'Shaking' in actual_name:
                if 'Yes_Shaking' in actual_name:
                    class_all[x] = 10
                    class_match = True
                elif 'No_Shaking' in actual_name:
                    class_all[x] = 9
                    class_match = True
                else:
                    class_all[x] = 4
                    class_match = True
            elif 'no' in actual_name:
                class_all[x] = 1
                class_match = True
            elif 'yes' in actual_name:
                class_all[x] = 0
                class_match = True
            else:
                print("Could not find any class matches for ", actual_name)
                raise TypeError('Please check the class!')

            if opt.partic:
                if '-1' in partic_name:
                    partic_all[x] = 1
                elif '-2' in partic_name:
                    partic_all[x] = 2
                else:
                    raise TypeError('cannot find participate number')
                pc_all[x] = int(str(partic_all[x]) + str(class_all[x]))
            else:
                partic_all[x] = 0
                pc_all[x] = 0

            x += 1

    # plot each single signal if necessary
    if opt.debug:
        for i in range(len(reserved_samples)):
            for j in range(len(reserved_samples[i])):
                if len(reserved_samples[i][j]) == total_num_samples_to_keep:
                    plt.plot(reserved_time[i][j], reserved_samples[i][j])
                    plt.xlabel("Time (s)")
                    plt.ylabel("Current (A)")
                    plt.title(new_file_list[i].rsplit('/', 1)[1].rsplit('\\', 2)[2])
                    # plt.savefig('plot_signal/' + str(i) + new_file_list[i].rsplit('/', 1)[1].rsplit('\\', 2)[2] + '.jpg')
                    plt.pause(0.000001)
                else:
                    print('Not enough points in', new_file_list[i].rsplit('/', 1)[1])

    return reserved_samples, reserved_time, new_file_list, class_all, partic_all, pc_all


def filter_and_fft(opt, all_samples, all_time, all_classes, all_partic, all_pc, file_list):
    overall_array = np.array([])
    temp_array = []

    for i in range(len(all_samples)):
        for j in range(len(all_samples[i])):
            normed_wave = all_samples[i][j] / all_samples[i][j].max()
            if opt.remove_dc:
                filtered_wave = butter_bandpass_filter(normed_wave, 1, 499, opt.sampling_rate, 2)
            else:
                filtered_wave = normed_wave
            temp_array = np.expand_dims(np.concatenate([filtered_wave, np.array([all_classes[i]]), np.array([all_partic[i]]), np.array([all_pc[i]])]), 0)
            if opt.debug:
                plt.plot(all_time[i][j], filtered_wave)
                plt.title(file_list[i])
                plt.xlabel("Time (s)")
                plt.ylabel("filtered_wave")
                plt.pause(0.000001)

        if len(overall_array) == 0:
            overall_array = temp_array
        else:
            overall_array = np.concatenate([overall_array, temp_array], 0)
    print(overall_array.shape)
    print('{} signals in total'.format(overall_array.shape[0]))

    return overall_array


if __name__ == '__main__':
    pass
