import numpy as np
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

# MediaPipe Pose Landmark Indices
NOSE = 0
LEFT_EYE = 2
RIGHT_EYE = 5
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32

# Side pairs used for left/right consistency checks
SIDE_PAIRS = [
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_ELBOW, RIGHT_ELBOW),
    (LEFT_WRIST, RIGHT_WRIST),
    (LEFT_HIP, RIGHT_HIP),
    (LEFT_KNEE, RIGHT_KNEE),
    (LEFT_ANKLE, RIGHT_ANKLE),
    (LEFT_HEEL, RIGHT_HEEL),
    (LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX),
]

# Config from Sprint Analysis Reference
VISIBILITY_THRESHOLD = 0.4
FOOT_VISIBILITY_THRESHOLD = 0.6
MAX_SWAP_DISTANCE_SQ = 0.01           
MAX_SWAP_DISTANCE_SQ_DISTAL = 0.002
SWAP_MARGIN = 1e-4
SWAP_COST_RATIO = 0.1
LEG_LENGTH_MAX_SCALE = 1.02
LEG_LENGTH_MAX_DIFF_RATIO = 0.05
MAX_JOINT_DELTA = 0.5  # Max normalized movement per frame

def _squared_distance_2d(a, b):
    dx = a.x - b.x
    dy = a.y - b.y
    return dx * dx + dy * dy

def _segment_length(a, b):
    return np.sqrt(_squared_distance_2d(a, b))

class PoseStabilizer:
    """
    Handles temporal smoothing and side-consistency enforcement to prevent
    left/right swapping and jitter.
    """
    def __init__(self, smoothing_alpha=0.5, conf_thresh=0.3):
        self.smoothing_alpha = smoothing_alpha # Not used directly if we use the reference's SMOOTHING_ALPHA logic which was 1 (no smoothing) but we want some smoothing. 
                                              # User pipeline had 0.5. Reference had 1.0 (disabled) but relied on velocity clamp.
                                              # We will combine them: Velocity Clamp then EMA.
        self.conf_thresh = conf_thresh
        self.prev_landmarks = None # Map of idx -> NormalizedLandmark (smoothed/corrected)
        self.prev_raw_landmarks = None # For EMA history if needed, but reference uses prev_smoothed.

    def process(self, current_landmarks_list):
        """
        Consumes a NormalizedLandmarkList, applies consistency checks and smoothing,
        returns a new NormalizedLandmarkList.
        """
        if not current_landmarks_list:
            return None

        # Convert to dict for easier manipulation, similar to reference
        current_map = {}
        for idx, lm in enumerate(current_landmarks_list.landmark):
            # Create a copy to avoid mutating original immediately
            new_lm = landmark_pb2.NormalizedLandmark()
            new_lm.CopyFrom(lm)
            current_map[idx] = new_lm

        # 1. Enforce Side Consistency
        if self.prev_landmarks:
            self._enforce_side_consistency(current_map)

        # 2. Smooth and Clamp
        smoothed_map = self._smooth_landmarks(current_map)
        
        # Update history
        self.prev_landmarks = smoothed_map

        # Reconstruct list
        smoothed_list = landmark_pb2.NormalizedLandmarkList()
        # Sort by index to maintain order 0..32
        for i in range(33):
            if i in smoothed_map:
                smoothed_list.landmark.append(smoothed_map[i])
            else:
                # Should not happen if input was full
                smoothed_list.landmark.append(landmark_pb2.NormalizedLandmark())
        
        return smoothed_list

    def _enforce_side_consistency(self, current_landmarks):
        """
        Detects likely left/right swaps and corrects them in-place.
        """
        prev = self.prev_landmarks
        
        # Precompute previous leg lengths
        prev_left_hip = prev.get(LEFT_HIP)
        prev_right_hip = prev.get(RIGHT_HIP)
        prev_left_ankle = prev.get(LEFT_ANKLE)
        prev_right_ankle = prev.get(RIGHT_ANKLE)

        prev_left_leg_len = _segment_length(prev_left_hip, prev_left_ankle) if (prev_left_hip and prev_left_ankle) else None
        prev_right_leg_len = _segment_length(prev_right_hip, prev_right_ankle) if (prev_right_hip and prev_right_ankle) else None

        for left_idx, right_idx in SIDE_PAIRS:
            # Skip knees/elbows as per reference (let MP handle them to avoid over-correction)
            if left_idx in [LEFT_KNEE, LEFT_ELBOW]:
                continue

            is_distal = left_idx in [LEFT_ANKLE, LEFT_HEEL, LEFT_FOOT_INDEX]
            
            thresh = FOOT_VISIBILITY_THRESHOLD if is_distal else VISIBILITY_THRESHOLD
            max_swap_sq = MAX_SWAP_DISTANCE_SQ_DISTAL if is_distal else MAX_SWAP_DISTANCE_SQ

            left_cur = current_landmarks.get(left_idx)
            right_cur = current_landmarks.get(right_idx)
            left_prev = prev.get(left_idx)
            right_prev = prev.get(right_idx)

            if not (left_cur and right_cur and left_prev and right_prev):
                continue

            if (left_cur.visibility < thresh or right_cur.visibility < thresh or 
                left_prev.visibility < thresh or right_prev.visibility < thresh):
                continue

            # Calculate costs
            orig_cost = _squared_distance_2d(left_cur, left_prev) + _squared_distance_2d(right_cur, right_prev)
            swapped_cost = _squared_distance_2d(left_cur, right_prev) + _squared_distance_2d(right_cur, left_prev)

            # Leg length check for distal joints
            if is_distal and prev_left_leg_len and prev_right_leg_len:
                cur_left_hip = current_landmarks.get(LEFT_HIP)
                cur_right_hip = current_landmarks.get(RIGHT_HIP)
                
                # Check lengths with potential swap
                # If we swap ankles, the "new" left ankle is the current right ankle
                swapped_left_len = _segment_length(cur_left_hip, right_cur) 
                swapped_right_len = _segment_length(cur_right_hip, left_cur)
                
                # Validation logic
                valid_swap = True
                
                # 1. Max Growth
                if swapped_left_len > prev_left_leg_len * LEG_LENGTH_MAX_SCALE: valid_swap = False
                if swapped_right_len > prev_right_leg_len * LEG_LENGTH_MAX_SCALE: valid_swap = False
                
                # 2. Asymmetry
                max_len = max(swapped_left_len, swapped_right_len)
                min_len = min(swapped_left_len, swapped_right_len)
                if max_len > 0 and (max_len - min_len) / max_len > LEG_LENGTH_MAX_DIFF_RATIO:
                    valid_swap = False
                    
                if not valid_swap:
                    continue

            # Final cost check
            if (swapped_cost + SWAP_MARGIN < orig_cost * SWAP_COST_RATIO and 
                swapped_cost < max_swap_sq and 
                orig_cost > 0.001):
                
                # Perform Swap
                # modifying current_landmarks objects in place? No, swapping the references in the dict
                current_landmarks[left_idx], current_landmarks[right_idx] = current_landmarks[right_idx], current_landmarks[left_idx]

    def _smooth_landmarks(self, current_landmarks):
        """
        Applies velocity clamping and exponential smoothing.
        """
        if not self.prev_landmarks:
            return current_landmarks

        smoothed = {}
        max_delta_sq = MAX_JOINT_DELTA * MAX_JOINT_DELTA
        
        # We use the previous smoothed landmarks (self.prev_landmarks)
        
        for idx, cur in current_landmarks.items():
            prev = self.prev_landmarks.get(idx)
            
            if not prev:
                smoothed[idx] = cur
                continue
                
            # Velocity Clamp (from reference)
            dx = cur.x - prev.x
            dy = cur.y - prev.y
            # Note: Reference only checks x/y for clamping, z is often less reliable? 
            # Reference: delta_sq = dx*dx + dy*dy
            delta_sq = dx*dx + dy*dy
            
            clamped_x, clamped_y = cur.x, cur.y
            
            if delta_sq > max_delta_sq:
                scale = MAX_JOINT_DELTA / np.sqrt(delta_sq)
                clamped_x = prev.x + dx * scale
                clamped_y = prev.y + dy * scale
                # We interpret "clamping" as modifying the current target before smoothing
            
            # Exponential Smoothing (EMA)
            # Reference used SMOOTHING_ALPHA = 1 (no smoothing) but implied ability to smooth.
            # User's current pipeline uses 0.5. Let's stick to user's 0.5 or a blend.
            # Actually, reference says "SMOOTHING_ALPHA = 1 means no temporal averaging".
            # The user explicitly asked to "compare my pipeline to that of the sprint analysis... what can we improve?".
            # The sprint analysis uses NO EMA (alpha=1), only Velocity Clamping.
            # However, the user's current pipeline uses EMA=0.5.
            # Compounding both might be too laggy. 
            # But the user's pipeline currently lacks velocity clamping.
            # Let's enforce Velocity Clamping, then apply the user's preferred alpha (0.5).
            
            alpha = self.smoothing_alpha
            
            final_x = alpha * clamped_x + (1 - alpha) * prev.x
            final_y = alpha * clamped_y + (1 - alpha) * prev.y
            final_z = alpha * cur.z + (1 - alpha) * prev.z # Z usually not clamped 
            
            new_lm = landmark_pb2.NormalizedLandmark()
            new_lm.x = final_x
            new_lm.y = final_y
            new_lm.z = final_z
            new_lm.visibility = cur.visibility
            new_lm.presence = cur.presence
            
            smoothed[idx] = new_lm
            
        return smoothed
