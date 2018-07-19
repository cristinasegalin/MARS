import tensorflow as tf
import numpy as np
from scipy.misc import imresize
import json
import multiprocessing as mp
import cv2
import progressbar


class ImportGraphDetection():
    """ Convenience class for setting up the detector and using it."""
    def __init__(self, quant_model):
        # Read the graph protocol buffer (.pb) file and parse it to retrieve the unserialized graph definition.
        with tf.gfile.GFile(quant_model, 'rb') as f:
            self.graph_def = tf.GraphDef()
            self.graph_def.ParseFromString(f.read())

        # Load the graph definition (stored in graph_def) into a live graph.
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.import_graph_def(self.graph_def, name="")

        # Configure the settings for our Tensorflow session.
        sess_config = tf.ConfigProto(
            log_device_placement=False,
            allow_soft_placement=True,
            # gpu_options=tf.GPUOptions(allow_growth=True))
            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=.30))

        # Create the Session we will use to execute the model.
        self.sess = tf.Session(graph=self.graph, config=sess_config)

        # Give object access to the input and output nodes.
        self.input_op = self.graph.get_operation_by_name('images')
        self.input_tensor = self.input_op.outputs[0]
        self.output_op_loc = self.graph.get_operation_by_name('predicted_locations')
        self.output_tensor_loc = self.output_op_loc.outputs[0]
        self.output_op_conf = self.graph.get_operation_by_name('Multibox/Sigmoid')
        self.output_tensor_conf = self.output_op_conf.outputs[0]

    def run(self, input_image):
        ''' This method is what actually runs an image through the Multibox network.'''
        return self.sess.run([self.output_tensor_loc, self.output_tensor_conf], {self.input_tensor: input_image})


class ImportGraphPose():
    """ Convenience class for setting up the pose estimator and using it."""
    def __init__(self, quant_model):
        # Read the graph protbuf (.pb) file and parse it to retrieve the unserialized graph definition.
        with tf.gfile.GFile(quant_model, 'rb') as f:
            self.graph_def = tf.GraphDef()
            self.graph_def.ParseFromString(f.read())

        # Load the graph definition (stored in graph_def) into a live graph.
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.import_graph_def(self.graph_def, name="")

        # Create the tf.session we will use to execute the model.
        sess_config = tf.ConfigProto(
            log_device_placement=False,
            allow_soft_placement=True,
            # gpu_options=tf.GPUOptions(allow_growth=True))
            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=.45))

        self.sess = tf.Session(graph=self.graph, config=sess_config)

        # access to input and output nodes
        self.input_op = self.graph.get_operation_by_name('images')
        self.input_tensor = self.input_op.outputs[0]
        self.output_op_heatmaps = self.graph.get_operation_by_name('HourGlass/Conv_30/BiasAdd')
        self.output_tensor_heatmaps = self.output_op_heatmaps.outputs[0]

    def run(self, cropped_images):
        """ This method is what actually runs an image through the stacked hourglass network."""
        return self.sess.run([self.output_tensor_heatmaps], {self.input_tensor: cropped_images})


def pre_process_image(image,DET_IM_SIZE):
    """ Takes a u8int image and prepares it for detection. """
    # Resize the image to the size the detector takes.
    prep_image = imresize(image, [DET_IM_SIZE, DET_IM_SIZE])
    # Convert the image to a float.
    prep_image = prep_image.astype(np.float32)
    # Shift the image values from [0, 256] to [-1, 1].
    prep_image = (prep_image - 128.) / 128.
    # Flatten the array.
    prep_image = prep_image.ravel()    
    # Add an additional dimension to stack images on.
    return np.expand_dims(prep_image, 0)


def post_process_detection(locations, confidences):
    """ Takes Multibox predictions and their confidences scores, chooses the best one, and returns a stretched version.
    """
    # pred_locs: [x1,y1,x2,y2] in normalized coordinates
    pred_locs = np.clip(locations[0], 0., 1.)

    # First, we want to filter the proposals that are not in the square.
    filtered_bboxes = []
    filtered_confs = []
    for bbox, conf in zip(pred_locs, confidences[0]):
        if bbox[0] < 0.: continue
        if bbox[1] < 0.: continue
        if bbox[2] > 1.: continue
        if bbox[3] > 1.: continue
        filtered_bboxes.append(bbox)
        filtered_confs.append(conf)

    # Convert these from lists to numpy arrays.
    filtered_bboxes = np.array(filtered_bboxes)
    filtered_confs = np.array(filtered_confs)

    # Now, take the bounding box we are most confident in. If it's above 0.005 confidence, stretch and return it.
    # Otherwise, just return an empty list and 0 confidence.
    if filtered_bboxes.shape[0] != 0:
        sorted_idxs = np.argsort(filtered_confs.ravel())[::-1]
        filtered_bboxes = filtered_bboxes[sorted_idxs]
        filtered_confs = filtered_confs[sorted_idxs]
        bbox_to_keep = filtered_bboxes[0].ravel()
        conf_to_keep = float(np.asscalar(filtered_confs[0]))
        # are we enough confident?
        if conf_to_keep > .005:
            # Unpack the bbox values.
            xmin, ymin, xmax, ymax = bbox_to_keep

            # Whether we use constant (vs width-based) stretch.
            useConstant = 0

            # Set the constant stretch amount.
            stretch_const = 0.06

            # Set the fractional stretch factor.
            stretch_factor = 0.10

            if useConstant:
                stretch_constx = stretch_const
                stretch_consty = stretch_const
            else:
                stretch_constx = (xmax-xmin)*stretch_factor #  of the width
                stretch_consty = (ymax-ymin)*stretch_factor

            # Calculate the amount to stretch the x by.
            x_stretch = np.minimum(xmin, abs(1-xmax))
            x_stretch = np.minimum(x_stretch, stretch_constx)

            # Calculate the amount to stretch the y by.
            y_stretch = np.minimum(ymin, abs(1-ymax))
            y_stretch = np.minimum(y_stretch, stretch_consty)

            # Adjust the bounding box accordingly.
            xmin -= x_stretch
            xmax += x_stretch
            ymin -= y_stretch
            ymax += y_stretch
            return [xmin, ymin, xmax, ymax], conf_to_keep
        else:
            # No good proposals, return substantial nothing.
            return [], 0.
    else:
        # No proposals, return nothing.
        return [], 0.


def extract_resize_crop_bboxes(bboxes, IM_W, IM_H, image):
    """ Resizes the bbox, and crops the image accordingly. Returns the cropped image."""
    # Define the image input size. TODO: Make this an input to the function.
    POSE_IM_SIZE = 256
    # Prepare a placeholder for the images.
    prepped_images = np.zeros((0, POSE_IM_SIZE, POSE_IM_SIZE, 3), dtype=np.uint8)
    # Scale normalized coordinates to image coordinates.
    scaled_bboxes = np.round(bboxes * np.array([IM_W, IM_H, IM_W, IM_H])).astype(int)

    # Extract the image using the bbox, then resize it to square (distorts aspect ratio).
    for i, bbox in enumerate(scaled_bboxes):
        # Unpack the bbox.
        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox

        # Crop the image.
        bbox_image = image[bbox_y1:bbox_y2, bbox_x1:bbox_x2]

        # Resize the image to the pose input size.
        im = imresize(bbox_image, (POSE_IM_SIZE, POSE_IM_SIZE))

       # Get a new 0th-dimension, to make the image(s) stackable.
        im = np.expand_dims(im, 0)

        # Concatenate the image to the stack.
        prepped_images = np.concatenate([prepped_images, im])

    # Now convert the image to a float and rescale between -1 and 1.
    prepped_images = prepped_images.astype(np.uint8)
    prepped_images = prepped_images.astype(np.float32)
    prepped_images = np.subtract(prepped_images, 128.)
    prepped_images = np.divide(prepped_images, 128.)
    return prepped_images


def post_proc_heatmaps(predicted_heatmaps, bboxes, IM_W, IM_H, NUM_PARTS, POSE_IM_SIZE):
    """ Postprocesses the heatmaps generated by the SHG. Returns the keypoints and their associated scores in a list."""
    keypoints_res = []
    scores =[]
    # For each stack in the batch, extract out the bboxes from the heatmaps,
    # then, for each heatmap in the stack, extract out argmax point. This is the estimated keypoint.
    for b in range(len(predicted_heatmaps[0])):
        # Get the stack of heatmaps.
        heatmaps = predicted_heatmaps[0][b]

        # Clip the heatmaps.
        heatmaps = np.clip(heatmaps, 0., 1.)

        # Resize them to square.
        resized_heatmaps = cv2.resize(heatmaps, (POSE_IM_SIZE, POSE_IM_SIZE), interpolation=cv2.INTER_LINEAR)

        # Unpack the bboxes and rescale them from norm coordinates to image coordinates.
        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bboxes[b]
        bbox_w = (bbox_x2 - bbox_x1) * IM_W
        bbox_h = (bbox_y2 - bbox_y1) * IM_H

        # Now resize the heatmaps to the original bbox size.
        rescaled_heatmaps = cv2.resize(resized_heatmaps, (int(np.round(bbox_w)), int(np.round(bbox_h))),
                                       interpolation=cv2.INTER_LINEAR)


        keypoints = np.zeros((NUM_PARTS, 3))
        # For each part-heatmap, extract out the keypoint, then place it in the original image's coordinates.
        for j in range(NUM_PARTS):
            # Get the current heatmap.
            hm = rescaled_heatmaps[:, :, j]
            score = float(np.max(hm))

            # Extract out the keypoint.
            x, y = np.array(np.unravel_index(np.argmax(hm), hm.shape)[::-1])

            # Place it in the original image's coordinates.
            imx = x + bbox_x1 * IM_W
            imy = y + bbox_y1 * IM_H

            # Store it.
            keypoints[j, :] = [imx, imy, score]

        # Store the x's, y's, and scores in lists.
        xs = keypoints[:, 0]
        ys = keypoints[:, 1]
        ss = keypoints[:, 2]
        keypoints_res.append([xs.tolist(), ys.tolist()])
        scores.append(ss.tolist())
    return keypoints_res, scores


# The global poison pill that we pass between processes. TODO: Could probably have this encapsulated in a fxn?
POISON_PILL = "STOP"

def get_poison_pill():
    """ Just in case we need to access the poison pill from outside this module."""
    return POISON_PILL


def pre_det(q_in, q_out_predet, q_out_raw, IM_TOP_H, IM_TOP_W):
    """ Worker function that preprocesses raw images for input to the detection network.
    q_in: from frame feeding loop in the main function.

    q_out_predet: to det, the detection function.
    q_out_raw: to pre_hm, the pose estimation pre-processing function. """
    try:
        frame_num = 0
        # TODO: Make these parameters inputs to the function.
        DET_IM_SIZE = 299
        POSE_IM_SIZE = 256
        while True:
            # Load in the raw image.
            top_image = q_in.get()

            # Check if we got the poison pill --if so, shut down.
            if top_image == POISON_PILL:
                q_out_predet.put(POISON_PILL)
                q_out_raw.put(POISON_PILL)
                return

            # Pre-process the image.
            if len(top_image.shape) != 3:
                # If the image isn't 3d, then make it 3d by adding a new dimension. (The third dimension is color.)
                new_im = np.empty((IM_TOP_H, IM_TOP_W, 3), dtype=np.uint8)
                new_im[:, :, :] = top_image[:, :, np.newaxis]
                # Convert the image to a float as well.
                top_image = new_im.astype(np.float32)

            # Preprocess the image.
            top_input_image = pre_process_image(top_image, DET_IM_SIZE)

            # Send the altered output image to the detection network, and the raw image to the pre-pose estimation fxn.
            q_out_predet.put(top_input_image)
            q_out_raw.put(top_image)
            frame_num += 1
    except Exception as e:
        print("error predet")
        print(e)
        raise(e)


def run_det(q_in,q_out, view):
    """ Worker function that houses and runs the bounding box detection network.
    q_in: from pre-det, the detection pre-processing function.

    q_out: to post-det, the detection post-processing function."""
    try:
        # Decide on which view to use.
        if view == 'front':
            QUANT_DET_B_PATH = 'models/optimized_model_front_detection_black_0.pb'
            QUANT_DET_W_PATH = 'models/optimized_model_front_detection_white_0.pb'
        else:
            QUANT_DET_B_PATH = 'models/optimized_model_top_detection_black_0.pb'
            QUANT_DET_W_PATH = 'models/optimized_model_top_detection_white_0.pb'

        # Import the detection networks.
        det_black = ImportGraphDetection(QUANT_DET_B_PATH)
        det_white = ImportGraphDetection(QUANT_DET_W_PATH)
        while True:
            # Get the input image.
            input_image = q_in.get()

            # Check if we got the poison pill --if so, shut down.
            if input_image == POISON_PILL:
                q_out.put(POISON_PILL)
                return

            # Run the detection networks.
            locations_b, confidences_b = det_black.run(input_image)
            locations_w, confidences_w = det_white.run(input_image)

            # Package the output up.
            det_b = [locations_b, confidences_b]
            det_w = [locations_w, confidences_w]
            det_out = [det_b, det_w]

            # Send the output to the post-detection processing worker.
            q_out.put(det_out)
    except Exception as e:
        print("error det")
        print(e)
        raise e


def post_det(q_in, q_out):
    """ Worker function that processes the detection output, so that we know which portion of the image to crop for
        pose estimation.
    q_in: from det, the detection network.

    q_out: to pre_hm, the pose estimation pre-processing function."""
    try:
        # Initialize the bounding boxes.
        det_prev_ok_loc_b = np.array([1e-2, 1e-2, 2e-2, 2e-2]);
        det_prev_ok_conf_b = 0.0001
        det_prev_ok_loc_w = np.array([1e-2, 1e-2, 2e-2, 2e-2]);
        det_prev_ok_conf_w = 0.0001

        ARBITRARY_CONFIDENCE_THRESHOLD = 0.005

        while True:
            # Load in the detection output.
            det_output = q_in.get()

            # Check if we got the poison pill --if so, shut down.
            if det_output == POISON_PILL:
                q_out.put(POISON_PILL)
                return

            # Unpack the detection output.
            det_b, det_w = det_output
            locations_b, confidences_b = det_b

            # Post-process the detection.
            det_bbox_to_pass_b, det_conf_to_pass_b = post_process_detection(locations_b, confidences_b)

            # If the confidence is high enough, use our new bounding box. Otherwise. use the old one.
            if det_conf_to_pass_b > ARBITRARY_CONFIDENCE_THRESHOLD:
                det_prev_ok_loc_b = det_bbox_to_pass_b
                det_prev_ok_conf_b = det_conf_to_pass_b
            else:
                det_bbox_to_pass_b = det_prev_ok_loc_b
                det_conf_to_pass_b = det_prev_ok_conf_b

            # Do the same thing with the white mouse.
            locations_w, confidences_w = det_w
            det_bbox_to_pass_w, det_conf_to_pass_w = post_process_detection(locations_w, confidences_w)
            if det_conf_to_pass_w > ARBITRARY_CONFIDENCE_THRESHOLD:
                det_prev_ok_loc_w = det_bbox_to_pass_w
                det_prev_ok_conf_w = det_conf_to_pass_w
            else:
                det_bbox_to_pass_w = det_prev_ok_loc_w
                det_conf_to_pass_w = det_prev_ok_conf_w

            # Bundle everything together.
            det_bbox = [det_bbox_to_pass_b, det_bbox_to_pass_w]
            det_conf = [det_conf_to_pass_b, det_conf_to_pass_w]

            det_output = [det_bbox, det_conf]

            # Send it to for pose estimation pre-processing.
            q_out.put(det_output)

    except Exception as e:
        print("error postdet")
        print(e)
        raise(e)


def pre_hm(q_in_det, q_in_image, q_out_img, q_out_bbox, IM_W, IM_H):
    """ Worker function that pre-processes an image for pose-estimation. Includes cropping and resizing it.
    q_in_det: from post_det, contains the post-processed bounding boxes.
    q_in_image: from pre_det, contains the raw image that needs to be cropped.

    q_out_img: to hm, the pose estimator.
    q_out_bbox: to post_hm, in order to reconstruct the keypoints' true locations."""
    try:
        while True:
            # Collect the detection output.
            det_out = q_in_det.get()

            # Check if we got the poison pill --if so, shut down.
            if det_out == POISON_PILL:
                q_out_img.put(POISON_PILL)
                q_out_bbox.put(POISON_PILL)
                return

            # Unpack the detection output.
            bboxes, confs = det_out

            # Collect the raw image.
            raw_image = q_in_image.get()

            # Convert the bounding boxes to an np.array
            bboxes = np.array(bboxes)

            # Prepare the images for pose estimation, by extracting the proper cropped region.
            prepped_images = extract_resize_crop_bboxes(bboxes, IM_W, IM_H, raw_image)

            q_out_img.put(prepped_images)
            q_out_bbox.put([bboxes,confs])
    except Exception as e:
        print("error prehm")
        print( e)
        raise(e)


def run_hm(q_in_img, q_out_hm, view):
    """ Worker function that performs the pose-estimation.
    q_in_img: from pre_hm, contains the images cropped and resized for inference.

    q_out_hm: to post_hm, contains the heatmaps."""
    try:
        # Figure out which view we're using.
        if view == 'front':
            QUANT_POSE = 'models/model_pose_front.pb'
        else:
            QUANT_POSE = 'models/model_pose_top.pb'

        # Import the pose model.
        pose_model = ImportGraphPose(QUANT_POSE)

        while True:
            # Collect the prepared images for inference.
            prepped_images = q_in_img.get()

            # Check if we got the poison pill --if so, shut down.
            if prepped_images == POISON_PILL:
                q_out_hm.put(POISON_PILL)
                return

            # Run the pose estimation network.
            predicted_heatmaps = pose_model.run(prepped_images)

            # Send the heatmaps out for post-processing.
            q_out_hm.put(predicted_heatmaps)
    except Exception as e:
        print("error hm")
        print(e)
        raise(e)


def post_hm(q_in_hm, q_in_bbox,
            IM_W, IM_H, NUM_PARTS, POSE_IM_SIZE, NUM_FRAMES, POSE_BASENAME):
    """ Worker function that processes the heatmaps for their keypoints, and saves them.
    q_in_hm: from hm, contains the unprocessed heatmaps.
    q_in_bbox: from prehm, contains the bounding boxes used to generate a given heatmap.
    """

    try:
        # Set a placeholder dictionary for our outputs.
        pose_frames = {'scores': [], 'keypoints': [],'bbox':[],'bscores':[]}
        # Set up the command-line progress bar.
        bar = progressbar.ProgressBar(widgets=['Pose ', progressbar.Percentage(), ' -- ',
                                               progressbar.FormatLabel('frame %(value)d'), '/',
                                               progressbar.FormatLabel('%(max)d'), ' [', progressbar.Timer(), '] ',
                                               progressbar.Bar(), ' (', progressbar.ETA(), ') '], maxval=NUM_FRAMES)
        # Start the progress bar.
        bar.start()

        #TODO: Should probably be an input?
        SAVE_EVERY_X_FRAMES = 2000

        # Initialize the current frame number.
        current_frame_num = 0
        while True:
            # Collect the predicted heatmaps.
            predicted_heatmaps = q_in_hm.get()

            # Collect the predicted bounding boxes.
            bboxes = q_in_bbox.get()

            # Check if we got the poison pill --if so, shut down.
            if bboxes == POISON_PILL:
                bar.finish()
                return pose_frames

            # Post-process the heatmaps to get the keypoints out.
            keypoints_res, scores = post_proc_heatmaps(predicted_heatmaps, bboxes[0], IM_W, IM_H,
                                                               NUM_PARTS, POSE_IM_SIZE)

            # Store the fresh keypoints, keypoint-scores, bounding boxes, and bounding box confidences.
            pose_frames['keypoints'].append(keypoints_res)
            pose_frames['scores'].append(scores)
            pose_frames['bbox'].append(bboxes[0].tolist())
            pose_frames['bscores'].append(bboxes[1])

            # Update our progress bar.
            bar.update(current_frame_num)

            # If we've reached a certain point, save a checkpoint for our pose output.
            if (current_frame_num % SAVE_EVERY_X_FRAMES == 0 or current_frame_num == NUM_FRAMES - 1) and current_frame_num > 1:
                with open(POSE_BASENAME + '.json', 'wb') as fp:
                    json.dump(pose_frames, fp)

            # Increment the frame_number.
            current_frame_num += 1
    except Exception as e:
        print("Error in post_hm")
        print(e)
        raise(e)
