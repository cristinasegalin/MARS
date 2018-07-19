from __future__ import print_function
import os
import sys
import scipy.io as sp
from MARS_pose_machinery import *
import warnings
import multiprocessing as mp
import logging
warnings.filterwarnings('ignore')
logging.getLogger("tensorflow").setLevel(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

sys.path.append('./')

import MARS_output_format as mof
from seqIo import *


def extract_pose_wrapper(video_fullpath, view, doOverwrite, progress_bar_signal='', verbose=0, output_suffix=''):
    video_path = os.path.dirname(video_fullpath)
    video_name = os.path.basename(video_fullpath)
    output_folder = mof.get_mouse_output_dir(dir_output_should_be_in=video_path, video_name=video_name,
                                         output_suffix = output_suffix)
    extract_pose(video_fullpath= video_fullpath,
                 output_folder=output_folder,
                 output_suffix=output_suffix,
                 view = view,
                 doOverwrite=doOverwrite,
                 progress_bar_signal=progress_bar_signal,
                 verbose = verbose
                 )
    return


def extract_pose(video_fullpath, output_folder, output_suffix, view,
                       doOverwrite, progress_bar_signal,
                       verbose=0):

    pose_basename = mof.get_pose_no_ext(video_fullpath=video_fullpath,
                                    output_folder=output_folder,
                                    view=view,
                                    output_suffix=output_suffix)
    video_name = os.path.basename(video_fullpath)

    pose_mat_name = pose_basename  + '.mat'
    
    # Makes the output directory, if it doesn't exist.
    mof.getdir(output_folder)
    
    ext = video_name[-3:]

    already_extracted_msg = (
        '1 - Pose already extracted. Change your settings to override, if you still want to extract the pose.')
    
    if not (ext in video_name):
        print("File type unsupported! Aborted.")
        return
    
    try:
        if verbose:
            print('1 - Extracting pose')
    
        if (not os.path.exists(pose_mat_name)) | (doOverwrite):
    
            if ext == 'seq':
                sr_top = seqIo_reader(video_fullpath)
                NUM_FRAMES = sr_top.header['numFrames']
                IM_H = sr_top.header['height']
                IM_W = sr_top.header['width']
                sr_top.buildSeekTable()
            elif ext in ['avi','mpg']:
                vc = cv2.VideoCapture(video_fullpath)
                if vc.isOpened():
                    rval = True
                else:
                    rval = False
                    print('video not readable')
                fps = vc.get(cv2.cv.CV_CAP_PROP_FPS)
                if np.isnan(fps): fps = 30.
                NUM_FRAMES = int(vc.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
                IM_H = vc.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
                IM_W = vc.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)

            if view == 'front':
                NUM_PARTS = 11
            elif view == 'top':
                NUM_PARTS = 7
            else:
                raise ValueError('Invalid view type!')
                return
    
            # print 'Processing video for detection and pose ...'
            DET_IM_SIZE = 299
            POSE_IM_SIZE = 256
    
            # print("Creating pool...")
            if mp.cpu_count() < 8:
                workers_to_use = mp.cpu_count()
            else:
                workers_to_use = 8
            pool = mp.Pool(workers_to_use)
            manager = mp.Manager()
            maxsize = 1
            # print("Pool Created. \nCreating Queues")
    
            # create managed queues
            q_start_to_predet = manager.Queue(maxsize)
            q_predet_to_det = manager.Queue(maxsize)
            q_predet_to_prehm = manager.Queue(maxsize)
            q_det_to_postdet = manager.Queue(maxsize)
            q_postdet_to_prehm = manager.Queue(maxsize)
            q_prehm_to_hm_IMG = manager.Queue(maxsize)
            q_prehm_to_posthm_BBOX = manager.Queue(maxsize)
            q_hm_to_posthm_HM = manager.Queue(maxsize)
            # q_posthm_to_end = manager.Queue(maxsize)
            # print("Queues Created. \nStarting Pools")
    
    
            try:

                results_predet = pool.apply_async(pre_det,
                                                  (q_start_to_predet,
                                                   q_predet_to_det, q_predet_to_prehm,
                                                    IM_H, IM_W))

                results_det = pool.apply_async(run_det,
                                               (q_predet_to_det,
                                                q_det_to_postdet,
                                                view))

                results_postdet = pool.apply_async(post_det,
                                                   (q_det_to_postdet,
                                                    q_postdet_to_prehm))

                results_prehm = pool.apply_async(pre_hm,
                                                 (q_postdet_to_prehm, q_predet_to_prehm,
                                                  q_prehm_to_hm_IMG, q_prehm_to_posthm_BBOX,
                                                  IM_W, IM_H))

                results_hm = pool.apply_async(run_hm,
                                              (q_prehm_to_hm_IMG,
                                               q_hm_to_posthm_HM,
                                               view))

                results_posthm = pool.apply_async(post_hm,
                                                    (q_hm_to_posthm_HM,
                                                    q_prehm_to_posthm_BBOX,
                                                    IM_W, IM_H,
                                                    NUM_PARTS,
                                                    POSE_IM_SIZE,
                                                    NUM_FRAMES,
                                                    pose_basename))
            except Exception as e:
                print("Error starting Pools:")
                print(e)
                raise(e)

            if progress_bar_signal:
                # Update the progress bar with the number of total frames it will be processing.
                progress_bar_signal.emit(0, NUM_FRAMES)

            for f in xrange(NUM_FRAMES):
                if ext == 'seq':
                    img = sr_top.getFrame(f)[0]
                elif ext in ['avi','mpg']:
                    _, img = vc.read()
                    img = img.astype(np.float32)
                q_start_to_predet.put(img)
                if progress_bar_signal:
                    progress_bar_signal.emit(f, 0)
    
            # Push through the poison pill.
            q_start_to_predet.put(get_poison_pill())
    
            # print("Pools Started...")
            pool.close()
            pool.join()
    
            # print("Pools Finished. \n Saving...")
            top_pose_frames = results_posthm.get()
    
            top_pose_frames['keypoints'] = np.array(top_pose_frames['keypoints'])
            top_pose_frames['scores'] = np.array(top_pose_frames['scores'])
    
            top_pose_frames['bbox'] = np.array(top_pose_frames['bbox'])
            top_pose_frames['bscores'] = np.array(top_pose_frames['bscores'])
    
            sp.savemat(pose_mat_name, top_pose_frames)
    
            # print("Saved.")
            # print 'Pose Extracted'
            if ext == 'seq':
                sr_top.close()
            elif ext in ['avi','mpg']:
                vc.release()  # sr_front.close()
            return
        else:
            if verbose:
                print(already_extracted_msg)
            return
    except Exception as e:
        print(e)
        raise(e)
    return