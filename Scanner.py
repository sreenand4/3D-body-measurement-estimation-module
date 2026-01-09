import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass
from typing import List, Tuple, Dict
import os

class Scanner:
    def __init__(self):
        # Init MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        # Init MediaPipe drawing
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Initialize measurements
        self.measurements = {}  # Initialize as empty dictionary
        self.reference_height = None
        self.image_shape = None  
        self.scaling_factors = {}
        self.height_pixels = {}
        self.segmentation_maps = {}
        self.landmark_maps = {}
        self.chest_circumference = None
        self.waist_circumference = None
        self.hip_circumference = None
        self.gender = None
    
    # Euclidean distance
    def _calculate_distance(self, p1, p2):
        # Handle MediaPipe landmarks (have .x and .y attributes)
        if hasattr(p1, 'x') and hasattr(p1, 'y'):
            return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
        # Handle tuples (p1[0], p1[1] format)
        else:
            return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    # Find midpoint between two points
    def _find_midpoint(self, point1, point2):
        return ((point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2)  
    
    # Ramanujan's approximation for ellipse circumference
    def _calculate_ellipse_circumference(self, a: float, b: float) -> float:
        h = ((a - b) / (a + b))**2
        return np.pi * (a + b) * (1 + (3 * h)/(10 + np.sqrt(4 - 3 * h)))
    
    def _process_single_image(self, image, view_name=""):
        # preprocess image
        self.image_shape = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # get pose landmarks and segmentation mask
        pose_results = self.pose.process(image_rgb)
        pose_landmarks = pose_results.pose_landmarks
        segmentation_mask = pose_results.segmentation_mask

        # store results
        self.landmark_maps[view_name] = pose_landmarks
        self.segmentation_maps[view_name] = segmentation_mask

        return image_rgb, pose_landmarks, segmentation_mask
    
    def _get_scaling_factors(self, front_image, side_image, back_image, reference_height):
        front_rgb, front_landmarks, front_seg_map = self._process_single_image(front_image, "front")
        front_height_pixels = self._calculate_height_pixels(front_rgb, front_landmarks, front_seg_map, "front")
        front_scaling_factor = reference_height / front_height_pixels

        side_rgb, side_landmarks, side_seg_map = self._process_single_image(side_image, "side")
        side_height_pixels = self._calculate_height_pixels(side_rgb, side_landmarks, side_seg_map, "side")
        side_scaling_factor = reference_height / side_height_pixels

        back_rgb, back_landmarks, back_seg_map = self._process_single_image(back_image, "back")
        back_height_pixels = self._calculate_height_pixels(back_rgb, back_landmarks, back_seg_map, "back")
        back_scaling_factor = reference_height / back_height_pixels

        self.scaling_factors = {
            "front": front_scaling_factor, 
            "side": side_scaling_factor, 
            "back": back_scaling_factor
        }
        return self.scaling_factors
    
    def _calculate_height_pixels(self, rgb_image, landmarks, seg_map, view_name):
        image_height, image_width = rgb_image.shape[:2]

        # extract coords forkey landmarks
        left_shoulder = (int(landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x * image_width), 
                         int(landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y * image_height))
        right_shoulder = (int(landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x * image_width), 
                          int(landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y * image_height))
        left_heel = (int(landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HEEL].x * image_width), 
                     int(landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HEEL].y * image_height))
        right_heel = (int(landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HEEL].x * image_width), 
                      int(landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HEEL].y * image_height))
        if (view_name == "front"):
            left_foot_index = (int(landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x * image_width), 
                               int(landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y * image_height))
            right_foot_index = (int(landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x * image_width), 
                                int(landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y * image_height))
            # for front view, we use the foot index as the heel because the heel is not visible
            left_heel = left_foot_index
            right_heel = right_foot_index
        
        # find midpoints
        shoulder_midpoint = self._find_midpoint(left_shoulder, right_shoulder)
        heel_midpoint = self._find_midpoint(left_heel, right_heel)

        # get the point of top of head
        top_of_head = self._find_top_of_head(rgb_image, seg_map, shoulder_midpoint)
        
        # calculate seperate distances
        shoulder_to_heel_distance = self._calculate_distance(shoulder_midpoint, heel_midpoint)
        
        shoulder_to_top_distance = self._calculate_distance(shoulder_midpoint, top_of_head)
        if (view_name == "side"):
            # if its a side view, we only consider LEFT key landmarks
            total_height_pixels = ((left_heel[1] - left_shoulder[1]) + (left_shoulder[1] - top_of_head[1])) * 1.03 # 1.03 is a correction factor
        else:
            total_height_pixels = (shoulder_to_heel_distance + shoulder_to_top_distance) * 1.03 # 1.03 is a correction factor

        self.height_pixels[view_name] = total_height_pixels
        return total_height_pixels
    
    def _find_top_of_head(self, rgb_image, seg_map, shoulder_midpoint):
        """Find the highest point of a person's head using contour-based approach"""
        # Create a binary mask for the person
        binary_mask = (seg_map > 0.1).astype(np.uint8) * 255
        shoulder_x = int(shoulder_midpoint[0])
        # Search upward along the vertical line from shoulder to find the highest point of the person
        top_point = None
        for y in range(int(shoulder_midpoint[1]), 0, -1):
            if binary_mask[y, shoulder_x] > 0:
                top_point = (shoulder_x, y)
            else:
                break
        
        if top_point is None:
            raise ValueError("Could not find top of head in segmentation mask")
        
        return top_point
    
    def process_images(self, front_image, side_image, back_image, reference_height, gender):
        # clear all measurements and reset all variables
        self.gender = gender
        self.segmentation_maps.clear()
        self.landmark_maps.clear()
        self.measurements.clear()
        self.chest_circumference = None
        self.waist_circumference = None
        self.hip_circumference = None
        self.scaling_factors.clear()
        self.height_pixels.clear()

        scaling_factors = self._get_scaling_factors(front_image, side_image, back_image, reference_height)

        # calc chest circumference
        chest_circumference = self._calculate_chest_circumference(front_image, side_image, back_image, scaling_factors)
        self.measurements['chest_circumference'] = chest_circumference
        self.chest_circumference = chest_circumference
        print(f"Chest circumference: {chest_circumference} cm")

        # calc waist circumference
        waist_circumference = self._calculate_waist_circumference(front_image, side_image, back_image, scaling_factors)
        self.measurements['waist_circumference'] = waist_circumference
        self.waist_circumference = waist_circumference
        print(f"Waist circumference: {waist_circumference} cm")

        # calc hip circumference
        hip_circumference = self._calculate_hip_circumference(front_image, side_image, back_image, scaling_factors)
        self.measurements['hip_circumference'] = hip_circumference
        self.hip_circumference = hip_circumference
        print(f"Hip circumference: {hip_circumference} cm")

        return self.measurements
        

    def _calculate_chest_circumference(self, front_image, side_image, back_image, scaling_factors):
        height, width = front_image.shape[:2]

        # front view chest measurement
        front_landmarks = self.landmark_maps['front']
        left_shoulder = (int(front_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x * width), 
                         int(front_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y * height))
        right_shoulder = (int(front_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x * width), 
                          int(front_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y * height))
        left_hip = (int(front_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].x * width), 
                    int(front_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].y * height))
        right_hip = (int(front_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP].x * width), 
                     int(front_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP].y * height))
        
        # calculate the front chest line (15-20% down from shoulder)
        front_shoulder_to_hip_distance = self._calculate_distance(left_shoulder, left_hip)
        front_chest_y = left_shoulder[1] + (front_shoulder_to_hip_distance * 0.17)
        # find left and right chest line points
        front_left_chest_point = (left_shoulder[0], front_chest_y)
        front_right_chest_point = (right_shoulder[0], front_chest_y)
        front_chest_width = abs(front_left_chest_point[0] - front_right_chest_point[0])

        # back view chest measurement
        back_landmarks = self.landmark_maps['back']
        back_left_shoulder = (int(back_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x * width), 
                              int(back_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y * height))
        back_right_shoulder = (int(back_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x * width), 
                               int(back_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y * height))
        back_left_hip = (int(back_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].x * width), 
                         int(back_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].y * height))
        back_right_hip = (int(back_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP].x * width), 
                          int(back_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP].y * height))
        
        # calculate the back chest line (15-20% down from shoulder)
        back_shoulder_to_hip_distance = self._calculate_distance(back_left_shoulder, back_left_hip)
        back_chest_y = back_left_shoulder[1] + (back_shoulder_to_hip_distance * 0.17)
        # find left and right chest line points
        back_left_chest_point = (back_left_shoulder[0], back_chest_y)
        back_right_chest_point = (back_right_shoulder[0], back_chest_y)
        back_chest_width = abs(back_left_chest_point[0] - back_right_chest_point[0])

        # side view chest measurement
        side_landmarks = self.landmark_maps['side']
        side_left_shoulder = (int(side_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x * width), 
                              int(side_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y * height))
        side_left_hip = (int(side_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].x * width), 
                         int(side_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].y * height))
        
        # calculate the side chest line (15-20% down from shoulder)
        side_shoulder_to_hip_distance = self._calculate_distance(side_left_shoulder, side_left_hip)
        side_chest_y = side_left_shoulder[1] + (side_shoulder_to_hip_distance * 0.17)
        
        # Get segmentation mask for side view
        side_seg_map = self.segmentation_maps['side']
        side_binary_mask = (side_seg_map > 0.1).astype(np.uint8) * 255
        
        # Get actual dimensions of the side image
        side_height, side_width = side_binary_mask.shape
        
        # Ensure chest_y is within bounds
        side_chest_y = max(0, min(int(side_chest_y), side_height - 1))
        
        # Find the width of the person at bust level in side view
        side_chest_width = 0
        for x in range(side_width):
            if side_binary_mask[side_chest_y, x] > 0:
                side_chest_width += 1
        
        # Convert all measurements to centimeters using respective scaling factors
        front_chest_width_cm = front_chest_width * scaling_factors['front']
        back_chest_width_cm = back_chest_width * scaling_factors['back']
        side_chest_width_cm = side_chest_width * scaling_factors['side']

        # Calculate avg chest width
        avg_chest_width_cm = (front_chest_width_cm + back_chest_width_cm) / 2
        print(f"---Avg chest width: {avg_chest_width_cm} cm")
        print(f"---Side chest width: {side_chest_width_cm} cm")
        chest_circumference = self._calculate_ellipse_circumference(avg_chest_width_cm/2, side_chest_width_cm/2)

        if (self.gender == "F"):
            chest_circumference = chest_circumference * 1.05 # 5% correction factor for female

        return chest_circumference
    
    def _calculate_waist_circumference(self, front_image, side_image, back_image, scaling_factors):
        height, width = front_image.shape[:2]

        # front view waist measurement
        front_landmarks = self.landmark_maps['front']
        left_pinky = (int(front_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_PINKY].x * width), 
                      int(front_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_PINKY].y * height))
        right_pinky = (int(front_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_PINKY].x * width), 
                       int(front_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_PINKY].y * height))
        
        # calculate distance between pinkys
        pinky_distance = self._calculate_distance(left_pinky, right_pinky)
        front_waist_width = pinky_distance * .98 # 98% is a correction factor

        # back view waist measurement
        back_landmarks = self.landmark_maps['back']
        left_thumb = (int(back_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_THUMB].x * width), 
                      int(back_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_THUMB].y * height))
        right_thumb = (int(back_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_THUMB].x * width), 
                       int(back_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_THUMB].y * height))

        # calculate distance between thumbs
        thumb_distance = self._calculate_distance(left_thumb, right_thumb)
        back_waist_width = thumb_distance * .98 # 98% is a correction factor

        # side view waist measurement
        side_landmarks = self.landmark_maps['side']
        side_left_hip = (int(side_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].x * width), 
                         int(side_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].y * height))
        side_right_hip = (int(side_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP].x * width), 
                          int(side_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP].y * height))
        mid_hip_point = self._find_midpoint(side_left_hip, side_right_hip)
        
        waist_y = mid_hip_point[1] * .80 # we are estimating waist level at 20% above hip
        waist_x = mid_hip_point[0]

        # Get segmentation mask for side view
        side_seg_map = self.segmentation_maps['side']
        side_binary_mask = (side_seg_map > 0.1).astype(np.uint8) * 255
        
        # Get actual dimensions of the side image
        side_height, side_width = side_binary_mask.shape
        
        # Ensure waist_y is within bounds
        waist_y = int(max(0, min(waist_y, side_height - 1)))
        waist_x = int(max(0, min(waist_x, side_width - 1)))

        # Find the width of the person at waist level in side view by expanding from the center
        side_waist_width = 0
        if side_binary_mask[waist_y, waist_x] > 0:
            side_waist_width = 1
            # Expand to the right
            for x in range(waist_x + 1, side_width):
                if side_binary_mask[waist_y, x] > 0:
                    side_waist_width += 1
                else:
                    break
            # Expand to the left
            for x in range(waist_x - 1, -1, -1):
                if side_binary_mask[waist_y, x] > 0:
                    side_waist_width += 1
                else:
                    break
        
        # Convert all measurements to centimeters using respective scaling factors
        front_waist_width_cm = front_waist_width * scaling_factors['front']
        back_waist_width_cm = back_waist_width * scaling_factors['back']
        side_waist_width_cm = side_waist_width * scaling_factors['side']

        # Calculate avg waist width and circumference
        avg_waist_width_cm = (front_waist_width_cm + back_waist_width_cm) / 2
        print(f"---Avg waist width: {avg_waist_width_cm} cm")
        print(f"---waist depth: {side_waist_width_cm} cm")
        waist_circumference = self._calculate_ellipse_circumference(avg_waist_width_cm / 2, side_waist_width_cm / 2) * .95 # tightness correction factor

        return waist_circumference
    
    def _calculate_hip_circumference(self, front_image, side_image, back_image, scaling_factors):
        height, width = front_image.shape[:2]

        # front view hip measurement
        front_landmarks = self.landmark_maps['front']
        left_hip = (int(front_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].x * width), 
                    int(front_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].y * height))
        right_hip = (int(front_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP].x * width), 
                     int(front_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP].y * height))
        front_hip_midpoint = self._find_midpoint(left_hip, right_hip)
        front_hip_y = front_hip_midpoint[1]
        front_hip_x = front_hip_midpoint[0]
        front_seg_map = self.segmentation_maps['front']
        front_binary_mask = (front_seg_map > 0.1).astype(np.uint8) * 255
        front_height, front_width = front_binary_mask.shape
        front_hip_y = int(max(0, min(front_hip_y, front_height - 1)))
        front_hip_x = int(max(0, min(front_hip_x, front_width - 1)))
        front_hip_width = 0
        if front_binary_mask[front_hip_y, front_hip_x] > 0:
            front_hip_width = 1
            # Expand to the right
            for x in range(front_hip_x + 1, front_width):
                if front_binary_mask[front_hip_y, x] > 0:
                    front_hip_width += 1
                else:
                    break
            for x in range(front_hip_x - 1, -1, -1):
                if front_binary_mask[front_hip_y, x] > 0:
                    front_hip_width += 1
                else:
                    break
        front_hip_width_cm = front_hip_width * scaling_factors['front']

        # back view hip measurement
        back_landmarks = self.landmark_maps['back']
        back_left_hip = (int(back_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].x * width), 
                         int(back_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].y * height))
        back_right_hip = (int(back_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP].x * width), 
                          int(back_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP].y * height))
        back_hip_midpoint = self._find_midpoint(back_left_hip, back_right_hip)
        back_hip_y = back_hip_midpoint[1]
        back_hip_x = back_hip_midpoint[0]
        back_seg_map = self.segmentation_maps['back']
        back_binary_mask = (back_seg_map > 0.1).astype(np.uint8) * 255
        back_height, back_width = back_binary_mask.shape
        back_hip_y = int(max(0, min(back_hip_y, back_height - 1)))
        back_hip_x = int(max(0, min(back_hip_x, back_width - 1)))
        back_hip_width = 0
        if back_binary_mask[back_hip_y, back_hip_x] > 0:
            back_hip_width = 1
            # Expand to the right
            for x in range(back_hip_x + 1, back_width):
                if back_binary_mask[back_hip_y, x] > 0:
                    back_hip_width += 1
                else:
                    break
            for x in range(back_hip_x - 1, -1, -1):
                if back_binary_mask[back_hip_y, x] > 0:
                    back_hip_width += 1
                else:
                    break
        back_hip_width_cm = back_hip_width * scaling_factors['back']

        # side view hip measurement
        side_landmarks = self.landmark_maps['side']
        side_left_hip = (int(side_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].x * width), 
                         int(side_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].y * height))
        side_right_hip = (int(side_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP].x * width), 
                          int(side_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP].y * height))
        side_hip_midpoint = self._find_midpoint(side_left_hip, side_right_hip)
        side_hip_y = side_hip_midpoint[1]
        side_hip_x = side_hip_midpoint[0]

        # Get segmentation mask for side view
        side_seg_map = self.segmentation_maps['side']
        side_binary_mask = (side_seg_map > 0.1).astype(np.uint8) * 255
        # Get actual dimensions of the side image
        side_height, side_width = side_binary_mask.shape

        # Ensure side_hip_coords are within bounds
        side_hip_y = int(max(0, min(side_hip_y, side_height - 1)))
        side_hip_x = int(max(0, min(side_hip_x, side_width - 1)))

        # Find the width of the person at hip level in side view by expanding from the center
        side_hip_width = 0
        if side_binary_mask[side_hip_y, side_hip_x] > 0:
            side_hip_width = 1
            # Expand to the right
            for x in range(side_hip_x + 1, side_width):
                if side_binary_mask[side_hip_y, x] > 0:
                    side_hip_width += 1
                else:
                    break
            # Expand to the left
            for x in range(side_hip_x - 1, -1, -1):
                if side_binary_mask[side_hip_y, x] > 0:
                    side_hip_width += 1
                else:
                    break
        
        side_hip_width_cm = side_hip_width * scaling_factors['side'] * .85 # tightness correction factor

        # Calculate avg hip width and circumference
        avg_hip_width_cm = (front_hip_width_cm + back_hip_width_cm) / 2
        print(f"---Avg hip width: {avg_hip_width_cm} cm")
        print(f"---Hip depth: {side_hip_width_cm} cm")
        hip_circumference = self._calculate_ellipse_circumference(avg_hip_width_cm / 2, side_hip_width_cm / 2) * .95 # tightness correction factor

        return hip_circumference