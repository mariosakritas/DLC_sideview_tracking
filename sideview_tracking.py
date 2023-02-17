#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pupil tracking using deeplabcut
https://github.com/AlexEMG/DeepLabCut
requires the mousecam package:
https://github.com/arnefmeyer/mousecam
"""

from __future__ import print_function

import click
import os.path as op
import os
import numpy as np
import yaml
import traceback
import pandas as pd
from pathlib import Path
import shutil
import glob
import json
import socket
import time
import pdb
import math

import deeplabcut
from mousecam.util.roirect import selectROI
from mousecam.io.video import get_first_frame
from mousecam.util.system import makedirs_save

def get_project_path(working_directory, project, experimenter):

    from datetime import datetime as dt

    date = dt.today().strftime('%Y-%m-%d')
    if working_directory is None:
        working_directory = '.'
    wd = Path(working_directory).resolve()
    project_name = '{pn}-{exp}-{date}'.format(
            pn=project,
            exp=experimenter,
            date=date)
    project_path = wd / project_name

    return project_path


def convert_h264_to_mp4(video_file, output_file=None, gamma=None):

    param_file = op.splitext(video_file)[0] + '.npz'
    #pdb.set_trace()
    if op.exists(param_file):
        pp = np.load(param_file)

        fps = int(pp['framerate'])
    else:
        param_file = op.splitext(video_file)[0] + '_params.json'
        with open(param_file, 'r') as f:
            dd = json.load(f)
        fps = int(dd['framerate'])

    if output_file is None:
        output_file = op.splitext(video_file)[0] + ".mp4"

    cmd = "MP4Box -fps {} -add {} {}".format(
            fps, video_file, output_file)
    os.system(cmd)

    if gamma is None:
        # try to read gamma value from file

        gamma_file = op.splitext(video_file)[0] + '_gamma.json'
        if op.exists(gamma_file):
            # NOTE: preview of gamma correction via
            # ffplay -vf eq=gamma=2 rpi_camera_2.mp4
            with open(gamma_file, 'r') as f:
                gg = json.load(f)
                gamma = gg['gamma']

    if gamma is not None and gamma != 1:
        #pdb.set_trace()
        # perform gamma correction (using ffmpeg video filter)
        temp_file = op.splitext(output_file)[0] + '_gamma_corrected.mp4'
        cmd = 'ffmpeg -i {} -vf eq=gamma={:.2f} -c:v libx264 -crf 20 {}'.format(output_file, gamma, temp_file)
        print("performing gamma correction:", cmd)
        os.system(cmd)
        os.remove(output_file)
        os.rename(temp_file, output_file)

    return output_file


def crop_mp4_file(video_file, bbox):

    x, y, w, h = bbox

    cropped_file = op.splitext(video_file)[0] + '_cropped.mp4'
    cmd = 'ffmpeg -i {} -filter:v "crop={}:{}:{}:{}" {}'.format(video_file, w, h, x, y, cropped_file)
    print(cmd)
    os.system(cmd)

    return cropped_file


def find_video_files(path, pattern='*.h264'):

    video_files = []
    for root, dirs, files in os.walk(path):
        ff = glob.glob(op.join(root, pattern))
        if len(ff) > 0:
            video_files.extend(ff)

    return video_files


def exclude_video_files(video_files, files_to_exclude):
    # exclude files based on names (not including file extension)

    valid_files = []
    for vf in video_files:
        fn = op.splitext(op.split(vf)[1])[0]
        if fn not in files_to_exclude and '_DRC' not in vf:
            valid_files.append(vf)
        else:
            print("excluding file", vf)

    return valid_files


def ignore_videos_with_existing_tracking_files(video_files):

    valid_video_files = []
    for vf in video_files:

        path, f = op.split(vf)
        found = glob.glob(op.join(path, '{}DLC_resnet50_all_mice_side_view_tracking*.h5'.format(op.splitext(f)[0])))
        if len(found) == 0:
            valid_video_files.append(vf)
        else:
            print("Ignoring video file: {}".format(vf))

    return valid_video_files


def convert_dlc_to_npy_files(path,
                             overwrite=False):
    """convert output of deeplabcut (*.h5) to numpy files with pupil data and video time stamps"""

    import pandas as pd
    for root, dirs, _ in os.walk(path):

        # get all DLC tracking files
        #CHANGE THIS PATTEERN ACCORDING TO THE NETWORK WHOSEE PREEDICTIONS YOU WANT TO USE
        pattern = 'rpi_camera_*DLC_resnet50_all_mice_side_view_trackingJul21*.h5'
        dlc_files = glob.glob(op.join(root, pattern))

        # assign DLC files to videos (a video can have multiple DLC files)
        videos = {}
        for f in dlc_files:

            fn = op.splitext(op.split(f)[1])[0]
            if 'croppedDeepCut' in fn:
                index = fn.find('croppedDeepCut')
            else:
                index = fn.find('DLC')
            name = fn[:index]
            if name.endswith('_'):
                name = name[:-1]

            if name in videos:
                videos[name].append(f)
            else:
                videos[name] = [f]

        for name in videos:
            print('name= ', name)
            #pdb.set_trace()
            data = {}
            n_obs = -1
            video_ts = None
            video_params = {}
            for f in videos[name]:

                print('f= ', f)

                # read data as pandas dataframe
                df = pd.read_hdf(f)
                levels = df.columns.levels
                table = df[levels[0].item()]
                cols = list(table)

                # get body parts
                body_parts = list(set([c[0] for c in cols]))

                # bounding box?
                bbox_file = op.join(op.split(f)[0], name + '_bbox.npy')
                if op.exists(bbox_file):
                    bbox = np.load(bbox_file)
                    x0, y0 = bbox[:2]
                else:
                    x0 = 0
                    y0 = 0
                    bbox = None

                for k in body_parts:
                    print("  found body part:", k)

                    xyl = np.asarray(table[k])
                    n_obs = max(n_obs, xyl.shape[0])
                    data[k] = {'xy': xyl[:, :2] + np.asarray([x0, y0]),
                               'likelihood': xyl[:, -1]}

                if video_ts is None:

                    # load video file time stamps and parameters from camera file
                    ts_file = op.join(op.split(f)[0], '{}.npz'.format(name))
                    dd = np.load(ts_file)
                    print("  timestamp file:", ts_file)

                    video_ts = dd['timestamps']

                    width = 640
                    height = 480
                    if 'width' in dd:
                        width = int(dd['width'])
                    if 'height' in dd:
                        height = int(dd['height'])

                    video_params = {'width': width,
                                    'height': height,
                                    'framerate': float(dd['framerate']),
                                    'bbox': bbox}

            side_view_file = op.join(root, '{}_side_view_data.npy'.format(name))
            if not op.exists(side_view_file) or overwrite:

                # there might be more frames than frame timestamps (as OE is stopping acquisition before the RPi)
                n = min(len(video_ts), n_obs)
                for k in data:
                    for kk in data[k]:
                        if data[k][kk].ndim == 1:
                            data[k][kk] = data[k][kk][:n]
                        else:
                            data[k][kk] = data[k][kk][:n, :]

                # save to new file
                dd = {'timestamps': video_ts[:n]}
                dd.update(**video_params)
                dd.update(**data)

                print("  saving data to file:", side_view_file)
                np.save(side_view_file, dd)


def compute_2D_movement(path, overwrite=False,
                              outlier_threshold=[15., 8., 8.]):

    import cv2
    import matplotlib.pyplot as plt
    import tqdm

    for root, dirs, _ in os.walk(path):

        files = glob.glob(op.join(root, 'rpi_camera_*_side_view_data.npy'))

        for f in files:
            dd = np.load(f,
                         encoding='latin1',
                         allow_pickle=True).item()
            if '2Dmov' not in dd or overwrite:
                print('computing 2D movement for: ', f)
                # try to fit ellipse for each frame
                ts = dd['timestamps']
                bodyparts = ['pina', 'nose', 'whisker_pad']

                for i, bodypart in enumerate(bodyparts):
                    bb = dd[bodypart]
                    xy = bb['xy']
                    mask = bb['likelihood'] < 0.99
                    xy[mask] = np.nan
                    dx = np.diff(xy[:,0])
                    dy = np.diff(xy[:,1])
                    dist = np.sqrt(dx ** 2 + dy ** 2)
                 # # detect outliers (using maximum jump criterion)
                    ddist = np.diff(dist)
                    ind = np.where(np.abs(ddist) >= outlier_threshold[i])[0]
                    dist[ind] = np.nan
                    bb['dist'] = dist

                dd = dict(**dd)
                dd['ts_dist'] = ts[1:]
                np.save(f, dd)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
@click.group()
def cli():
    pass


@click.command(name='create-project')
@click.argument('data_path', nargs=1)
@click.argument('video_files', nargs=-1)
@click.option('--recursive', '-r', is_flag=True)
@click.option('--name', '-n', default=None)
@click.option('--experimenter', '-e', default='marios')
@click.option('--frames', '-n', default=20,
              help='# frames to annotate per video file (default: 20)')
@click.option('--markers', '-m', default=3)
@click.option('--exclude', '-E', multiple=True, default=[])
def cli_create_project(data_path=None,
                       video_files=None,
                       recursive=False,
                       name=None,
                       experimenter=None,
                       frames=20,
                       markers=4,
                       exclude=[]):

    data_path = op.expanduser(data_path)
    files_to_exclude = list(exclude)

    if name is None:
        name = 'markers_{}'.format(time.strftime('%Y-%m-%d_%H-%M-%S'))

    # get video files
    if recursive:
        path = op.expanduser(video_files[0])
        video_files = exclude_video_files(find_video_files(path), files_to_exclude)

    # convert to mp4
    converted_files = []
    mp4_files = []
    for i, vf in enumerate(video_files):

        vf = op.abspath(vf)

        # convert h264 file to mp4; append recording name (=suffix). This allows for using the same video file name
        # (e.g., using the same camera for different recordings).
        path, name_ext = op.split(vf)
        file_name = op.splitext(name_ext)[0]
        rec_name = path.split(op.sep)[-1]

        mp4_file = op.join(path, file_name + '_' + rec_name + '.mp4')
        if not op.exists(mp4_file):
            mp4_file = convert_h264_to_mp4(vf,
                                           output_file=mp4_file)
            converted_files.append(mp4_file)

        mp4_files.append(mp4_file)

    config_path = deeplabcut.create_new_project(name, experimenter, mp4_files,
                                                working_directory=data_path,
                                                copy_videos=False)

    # save file paths to project directory
    with open(op.join(op.split(config_path)[0], 'project_settings.json'), 'w') as f:
        json.dump({'video_files': video_files,
                   'converted_files': converted_files,
                   'project_name': name,
                   'experimenter': experimenter,
                   'computer': socket.gethostname()},
                  f,
                  indent=4)

    # set number of frames to label (same for each video file)
    cfg = deeplabcut.utils.read_config(config_path)
    cfg['numframes2pick'] = frames
    cfg['bodyparts'] = ['nose'] + ['pina'] + ['whisker_pad']
    deeplabcut.utils.write_config(config_path, cfg)

    deeplabcut.extract_frames(config_path, 'automatic', 'kmeans',
                              crop='GUI',
                              userfeedback=False)
    deeplabcut.label_frames(config_path)
    deeplabcut.check_labels(config_path)
    deeplabcut.create_training_dataset(config_path, num_shuffles=1)

    # delete all converted mp4 files
    for f in converted_files:
        os.remove(f)


cli.add_command(cli_create_project)


@click.command(name='train')
@click.argument('project_path', nargs=1)
@click.option('--save-iters', '-s', default=50000)
def cli_train(project_path=None, save_iters=50000):
    # train deep net on data generated via "cli_create_project"

    config_path = op.join(op.expanduser(project_path), 'config.yaml')

    try:
        deeplabcut.train_network(config_path,
                                 saveiters=save_iters)

    except KeyboardInterrupt:
        pass

    finally:
        deeplabcut.evaluate_network(config_path,
                                    plotting=True)


cli.add_command(cli_train)


@click.command(name='analyze')
@click.argument('project_path', nargs=1)
@click.argument('video_files', nargs=-1)
@click.option('--create-videos', '-c', is_flag=True)
@click.option('--recursive', '-r', is_flag=True)
@click.option('--crop', '-C', is_flag=True)
@click.option('--exclude', '-E', multiple=True, default=[])
@click.option('--overwrite', '-o', is_flag=True)
@click.option('--gamma', '-g', default=None, type=float)
def cli_analyze(project_path=None,
                video_files=[],
                create_videos=False,
                recursive=False,
                crop=False,
                exclude=[],
                overwrite=False,
                gamma=None):
    # extract posture using a trained deep net (see "cli_train")

    config_path = op.join(op.expanduser(project_path), 'config.yaml')

    files_to_exclude = list(exclude)

    if recursive:
        path = op.expanduser(video_files[0])
        video_files = exclude_video_files(find_video_files(path), files_to_exclude)

    if not overwrite:
        video_files = ignore_videos_with_existing_tracking_files(video_files)

    if crop:

        # try to load project settings file
        settings_file = op.join(project_path, 'project_settings.json')
        if op.exists(settings_file):
            with open(settings_file, 'r') as f:
                settings = json.load(f)
        else:
            settings = None

        bboxes = {}
        for f in video_files:
            if settings is not None and f in settings['video_files']:
                index = settings['video_files'].index(f)
                bbox = settings['bboxes'][index]
            elif op.exists(op.splitext(f)[0] + '_bbox.npy'):
                bbox = np.load(op.splitext(f)[0] + '_bbox.npy')
            else:
                bbox = selectROI(get_first_frame(f))
            bboxes[f] = bbox

    print("analyzing video files:")
    for vf in video_files:

        print("  ", vf)

        ext = op.splitext(vf)[1]
        remove_file = False
        if ext == '.mp4':
            mp4_file = vf
        else:
            try:
                mp4_file = op.splitext(vf)[0] + '.mp4'
                if not op.exists(mp4_file):
                    mp4_file = convert_h264_to_mp4(vf,
                                                   gamma=gamma)
                    remove_file = True
            except:
                print("This video f60ile or its config file is missing!")

        try:
            if crop and bboxes[vf] is not None:

                print("  cropping video file to:", bboxes[vf])
                cropped_file = crop_mp4_file(mp4_file, bboxes[vf])
                if remove_file:
                    os.remove(mp4_file)
                mp4_file = cropped_file

                bbox_file = op.splitext(vf)[0] + '_bbox.npy'
                np.save(bbox_file, bboxes[vf])
                print("  saved bbox to file:", bbox_file)

            deeplabcut.analyze_videos(config_path, [mp4_file],
                                      save_as_csv=True)

            if create_videos:
                print("CREATING LABELED VIDEOS")
                deeplabcut.create_labeled_video(config_path, [mp4_file])

            if remove_file:
                os.remove(mp4_file)
        except:
            print("This video file or its config file is missing!")


cli.add_command(cli_analyze)


@click.command(name='repack')
@click.argument('data_path')
@click.option('--overwrite', '-o', is_flag=True)
def cli_repack(data_path=None, **kwargs):
    # convert deeplabcut tracking files (pandas *.h5) to npy format
    #input is one animal's database

    convert_dlc_to_npy_files(data_path, **kwargs)

cli.add_command(cli_repack)

@click.command(name='compute-2D-movement')
@click.argument('path')
@click.option('--overwrite', '-o', is_flag=True)
def cli_compute_2D_movement(path, **kwargs):
    # fit ellipse to pupil data files

    compute_2D_movement(path, **kwargs)


cli.add_command(cli_compute_2D_movement)

@click.command(name='merge-projects')
@click.argument('project_paths', nargs=-1)
@click.argument('new_path', nargs=1)
def cli_merge_projects(project_paths=None, new_path=None):
    # merge two projects (generated via "cli_create_project")

    # ----- create directory structure -----
    if op.exists(new_path):
        shutil.rmtree(new_path)

    os.makedirs(new_path)

    for d in ['dlc-models', 'labeled-data', 'training-datasets', 'videos']:
        p = op.join(new_path, d)
        if not op.exists(p):
            os.makedirs(p)

    new_config_path = op.join(op.expanduser(new_path), 'config.yaml')
    project_name = new_path.split(op.sep)[-1]

    # ----- copy data from other projects -----
    for project_path in project_paths:

        print(20*"-", project_path, 20*"-")

        # load config
        config_path = op.join(op.expanduser(project_path), 'config.yaml')
        config = deeplabcut.utils.read_config(config_path)

        if not op.exists(new_config_path):
            config['Task'] = project_name
            config['project_path'] = new_path
            deeplabcut.utils.write_config(new_config_path, config)
        else:
            new_config = deeplabcut.utils.read_config(new_config_path)
            new_config['video_sets'].update(config['video_sets'])
            deeplabcut.utils.write_config(new_config_path, new_config)

        # copy videos
        video_files = glob.glob(op.join(project_path, 'videos', '*.mp4'))
        for vf in video_files:
            print("  copying video file:", vf)
            shutil.copy(vf, op.join(new_path, 'videos'))

        # copy labeled training sets
        data_path = op.join(project_path, 'labeled-data')
        for d in os.listdir(data_path):
            print("  copying labeled data:", d)
            shutil.copytree(op.join(data_path, d), op.join(new_path, 'labeled-data', d))

    deeplabcut.create_training_dataset(new_config_path, num_shuffles=1)


cli.add_command(cli_merge_projects)


@click.command(name='compute-brightness')
@click.argument('path')
@click.option('--exclude', '-E', multiple=True, default=[])
@click.option('--output', '-o', type=click.Path(), default=None)
@click.option('--overwrite', '-O', is_flag=True)
def cli_compute_brightness(path,
                           exclude=[],
                           output=None,
                           overwrite=False):
    # compute brightness of videos (to identify very dark/bright videos)

    import imageio
    import cv2

    # parse output file
    if output is not None:

        output = op.realpath(op.expanduser(output))

        base, ext = op.splitext(output)
        if len(ext) == 0:
            # output is directory
            output_file = op.join(output, 'videos_brightness.csv')
        else:
            # output is file
            output_file = output
            output = op.split(output)[0]

        makedirs_save(output)

        if not op.exists(output_file) or overwrite:
            # write header (comment)
            with open(output_file, 'w') as f:
                f.write('# video file, mean, std\n')

    # get (and potentially exclude) video files
    path = op.realpath(op.expanduser(path))
    files_to_exclude = list(exclude)
    video_files = exclude_video_files(find_video_files(path), files_to_exclude)

    for vf in video_files:

        try:
            mean_frames = []
            std_frames = []
            with imageio.get_reader(vf, 'ffmpeg') as reader:

                for frame in reader:

                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    mean_frames.append(np.mean(frame))
                    std_frames.append(np.std(frame))

            print(50*'-')
            print(vf)
            print("  {:.2f} +- {:.2f}".format(
                np.mean(mean_frames),
                np.mean(std_frames)))

            if output is not None:
                # slightly inefficient but makes sure to save all data
                # until an error occurs or the user presses ctrl+c

                with open(output_file, 'a') as f:

                    f.write('{},{:.2f},{:.2f}\n'.format(
                        vf,
                        np.mean(mean_frames),
                        np.mean(std_frames)))

        except KeyboardInterrupt:
            break

        except BaseException:
            traceback.print_exc()

    if output is not None:
        # print all data in sorted order

        with open(output_file, 'r') as f:
            lines = f.readlines()

        # remove header
        if lines[0].startswith('#'):
            lines = lines[1:]

        # remove duplicate entries
        lines = list(set(lines))

        files = [ll.split(',')[0] for ll in lines]
        means = np.asarray([float(ll.split(',')[1]) for ll in lines])
        stds = np.asarray([float(ll.split(',')[2]) for ll in lines])
        order = np.argsort(means)  # low to high

        # writed sorted video data to new csv file
        output_file_sorted = op.splitext(output_file)[0] + '_sorted.csv'
        with open(output_file_sorted, 'w') as s:

            for index in order:

                print("{} -- {}".format(index+1, files[index]))
                print("  {:.2f} +- {:.2f}".format(means[index], stds[index]))

                s.write('{},{:.2f},{:.2f}\n'.format(
                    files[index],
                    means[index],
                    stds[index]))


cli.add_command(cli_compute_brightness)


@click.command(name='analyze-adjust-gamma')
@click.argument('project_path', nargs=1)
@click.argument('brightness_file', nargs=1)
@click.option('--create-videos', '-c', is_flag=True)
@click.option('--overwrite', '-O', is_flag=True)
@click.option('--crop', '-C', is_flag=True)
@click.option('--brightness-min', '-b', default=0)
@click.option('--brightness-max', '-B', default=80.)
@click.option('--gamma', '-g', default=3.0)
def cli_analyze_adjust_gamma(project_path=None,
                             brightness_file=None,
                             create_videos=False,
                             overwrite=False,
                             crop=False,
                             brightness_min=None,
                             brightness_max=None,
                             gamma=None):
    # re-analyze files with low brightness using a user-specified gamma value

    # read files from (sorted) file created using "cli_compute_brightness"
    with open(brightness_file, 'r') as f:
        lines = f.readlines()

    # remove duplicate entries
    lines = list(set(lines))

    files = [ll.split(',')[0] for ll in lines]
    means = np.asarray([float(ll.split(',')[1]) for ll in lines])

    # row-wise index numbers of file names in the csv file where the values fall within this threshold
    ind = np.where(np.logical_and(means >= brightness_min,
                                  means <= brightness_max))[0]

    video_files = []
    for index in ind:

        f = files[index]
        m = means[index]

        # ignore all Go/No-go paradigm files; name structure:
        # database/M10234/2019_04_15/screening/2019-04-15_12-08-39_NoiseBurst/rpi_camera_2.h264
        parts = f.split(op.sep)
        session_name = parts[-3]
        seg_name = parts[-2]
        if 'gonogo' in session_name:
            print("ignoring go/nogo recording:", f)
        elif '_DRC' in seg_name:
            print("ignoring DRC recording:", f)

        else:
            # check for existing tracking files
            base = op.splitext(f)[0]
            tracking_files = glob.glob(base + 'rpi_camera_*DLC_resnet50_all_mice_side_view_tracking*')

            if len(tracking_files) == 0:
                video_files.append(f)

            elif overwrite:
                # delete tracking files
                for tf in tracking_files:
                    print("  removing existing tracking file", tf)
                    os.remove(tf)

                video_files.append(f)

            else:
                print("video already has some tracking data:", f)

    video_files = sorted(video_files)

    print("analyzing video files:")
    for f in video_files:
        print("  ", f)

    cli_analyze.callback(project_path=project_path,
                         video_files=video_files,
                         create_videos=create_videos,
                         recursive=False,
                         crop=crop,
                         overwrite=True,
                         gamma=gamma)


cli.add_command(cli_analyze_adjust_gamma)


if __name__ == '__main__':
    cli()