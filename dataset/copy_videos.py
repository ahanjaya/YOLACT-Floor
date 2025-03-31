#! /usr/bin/env python3

import os

sub_dir   = "/media/ahan/Data/4. Research/4. Thormang3/3. Data/7. Wolf Walk/raw/"
n_sub_dir = len(os.walk(sub_dir).__next__()[1])
print('Total folder: {}'.format(n_sub_dir))

for n_folder in range (n_sub_dir):
    # robot frame
    # file_name = "/media/ahan/Data/4.\ Research/4.\ Thormang3/3.\ Data/7.\ Wolf\ Walk/raw/{0}/wolf_robot_cam-{0}.avi".format(n_folder)
    # command   = "cp {} robot_video/".format(file_name)

    # tipod frame
    file_name = "/media/ahan/Data/4.\ Research/4.\ Thormang3/3.\ Data/7.\ Wolf\ Walk/raw/{0}/wolf_tripod_cam-{0}.avi".format(n_folder)
    command   = "cp {} tripod_video/".format(file_name)

    try:
        os.system(command)
        print('Succesfull copy: {}'.format(file_name))
    except:
        continue

    # break