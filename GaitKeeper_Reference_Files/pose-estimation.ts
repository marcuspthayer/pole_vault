/**
 * Pose estimation utilities using TensorFlow.js and MediaPipe
 * This runs in the browser for client-side processing
 */

import * as poseDetection from '@tensorflow-models/pose-detection';
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-webgl';

export interface Keypoint {
  x: number;
  y: number;
  score?: number;
  name?: string;
}

export interface Pose {
  keypoints: Keypoint[];
  score?: number;
}

export interface PoseFrame {
  timestamp: number;
  pose: Pose | null;
}

let detector: poseDetection.PoseDetector | null = null;
let poseHistory: Pose[] = []; // Store recent poses for smoothing
let isBlazePose: boolean = true; // Track which model is loaded

export function getIsBlazePose(): boolean {
  return isBlazePose;
}

// Keypoint index mapping for different models
export const KEYPOINT_INDICES = {
  blazePose: {
    leftShoulder: 11,
    rightShoulder: 12,
    leftHip: 23,
    rightHip: 24,
    leftKnee: 25,
    rightKnee: 26,
    leftAnkle: 27,
    rightAnkle: 28,
    leftHeel: 29,
    rightHeel: 30,
    leftToe: 31,
    rightToe: 32,
  },
  moveNet: {
    leftShoulder: 5,
    rightShoulder: 6,
    leftHip: 11,
    rightHip: 12,
    leftKnee: 13,
    rightKnee: 14,
    leftAnkle: 15,
    rightAnkle: 16,
    leftHeel: -1, // Not available
    rightHeel: -1, // Not available
    leftToe: -1, // Not available
    rightToe: -1, // Not available
  }
};

export function getKeypointIndices() {
  return isBlazePose ? KEYPOINT_INDICES.blazePose : KEYPOINT_INDICES.moveNet;
}

export async function initializePoseDetector(): Promise<poseDetection.PoseDetector> {
  if (detector) {
    // Reset history when reusing detector
    poseHistory = [];
    return detector;
  }

  // Set backend to WebGL explicitly (WebGPU may not be available)
  await tf.setBackend('webgl');
  await tf.ready();
  
  try {
    // Try BlazePose with MediaPipe runtime (more stable than tfjs runtime)
    console.log('Loading BlazePose with MediaPipe runtime...');
    const model = poseDetection.SupportedModels.BlazePose;
    detector = await poseDetection.createDetector(model, {
      runtime: 'mediapipe',
      solutionPath: 'https://cdn.jsdelivr.net/npm/@mediapipe/pose',
      modelType: 'lite',
    });
    isBlazePose = true;
    console.log('BlazePose (MediaPipe runtime) loaded successfully');
  } catch (error) {
    // Fallback to MoveNet if BlazePose fails
    console.warn('BlazePose failed, using MoveNet:', error);
    const model = poseDetection.SupportedModels.MoveNet;
    detector = await poseDetection.createDetector(model, {
      modelType: poseDetection.movenet.modelType.SINGLEPOSE_THUNDER,
    });
    isBlazePose = false;
    console.log('MoveNet model loaded successfully');
  }

  return detector;
}

export async function detectPoseInFrame(
  video: HTMLVideoElement,
  detector: poseDetection.PoseDetector
): Promise<Pose | null> {
  try {
    const poses = await detector.estimatePoses(video);
    if (!poses || poses.length === 0) {
      return null;
    }
    
    const currentPose = poses[0];
    
    // Apply temporal smoothing to reduce jitter and lag
    const smoothedPose = smoothPose(currentPose);
    
    // Update history
    poseHistory.push(currentPose);
    if (poseHistory.length > 5) {
      poseHistory.shift(); // Keep only last 5 frames
    }
    
    return smoothedPose;
  } catch (error) {
    console.error('Error detecting pose:', error);
    return null;
  }
}

// Unused - kept for potential future use
export function filterOutliers(pose: Pose): Pose {
  return pose;
}

function smoothPose(currentPose: Pose): Pose {
  if (poseHistory.length === 0) {
    return currentPose;
  }
  
  // Use exponential moving average for smooth transitions
  const alpha = 0.5; // Smoothing factor (0 = all history, 1 = no smoothing)
  
  const smoothedKeypoints = currentPose.keypoints.map((kp, index) => {
    // Get corresponding keypoint from most recent history
    const prevKp = poseHistory[poseHistory.length - 1]?.keypoints[index];
    
    if (!prevKp || !kp.score || kp.score < 0.3) {
      return kp;
    }
    
    // Apply exponential moving average
    return {
      ...kp,
      x: alpha * kp.x + (1 - alpha) * prevKp.x,
      y: alpha * kp.y + (1 - alpha) * prevKp.y,
    };
  });
  
  return {
    ...currentPose,
    keypoints: smoothedKeypoints,
  };
}

export function drawPose(
  ctx: CanvasRenderingContext2D,
  pose: Pose,
  width: number,
  height: number,
  options?: {
    strikes?: { 
      left?: { x: number; y: number; age: number }; 
      right?: { x: number; y: number; age: number } 
    };
  }
): void {
  if (!pose || !pose.keypoints) {
    console.warn('No pose or keypoints to draw');
    return;
  }
  
  // Color palette per body region
  const colors = {
    torso: '#3b82f6',       // blue
    leftArm: '#10b981',     // green
    rightArm: '#10b981',
    leftLeg: '#f59e0b',     // amber
    rightLeg: '#ef4444',    // red
    jointsFill: '#111827',  // near-black
    jointsStroke: '#ffffff' // white outline
  } as const;

  // Helper to draw a colored segment if both endpoints are confident
  const drawSegment = (aIdx: number, bIdx: number, color: string, widthPx = 3) => {
    const a = pose.keypoints[aIdx];
    const b = pose.keypoints[bIdx];
    if (!a || !b) return;
    if ((a.score ?? 0) < 0.3 || (b.score ?? 0) < 0.3) return;
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.strokeStyle = color;
    ctx.lineWidth = widthPx;
    ctx.lineCap = 'round';
    ctx.stroke();
  };

  const idx = getKeypointIndices();
  
  // Torso
  drawSegment(idx.leftShoulder, idx.rightShoulder, colors.torso);   // shoulders
  drawSegment(idx.leftShoulder, idx.leftHip, colors.torso);   // left shoulder to left hip
  drawSegment(idx.rightShoulder, idx.rightHip, colors.torso);   // right shoulder to right hip
  drawSegment(idx.leftHip, idx.rightHip, colors.torso);   // hips

  // Arms (BlazePose: elbow=13/14, wrist=15/16)
  drawSegment(idx.leftShoulder, 13, colors.leftArm);  // L shoulder-elbow
  drawSegment(13, 15, colors.leftArm);  // L elbow-wrist
  drawSegment(idx.rightShoulder, 14, colors.rightArm); // R shoulder-elbow
  drawSegment(14, 16, colors.rightArm); // R elbow-wrist

  // Legs
  drawSegment(idx.leftHip, idx.leftKnee, colors.leftLeg);  // L hip-knee
  drawSegment(idx.leftKnee, idx.leftAnkle, colors.leftLeg);  // L knee-ankle
  drawSegment(idx.rightHip, idx.rightKnee, colors.rightLeg); // R hip-knee
  drawSegment(idx.rightKnee, idx.rightAnkle, colors.rightLeg); // R knee-ankle
  
  // Feet (only if heel/toe available - BlazePose)
  if (idx.leftHeel >= 0 && idx.leftToe >= 0) {
    drawSegment(idx.leftAnkle, idx.leftHeel, colors.leftLeg);  // L ankle-heel
    drawSegment(idx.leftAnkle, idx.leftToe, colors.leftLeg);  // L ankle-toe
  }
  if (idx.rightHeel >= 0 && idx.rightToe >= 0) {
    drawSegment(idx.rightAnkle, idx.rightHeel, colors.rightLeg); // R ankle-heel
    drawSegment(idx.rightAnkle, idx.rightToe, colors.rightLeg); // R ankle-toe
  }

  // Draw joints bigger and with outline
  pose.keypoints.forEach((kp) => {
    if (!kp) return;
    if ((kp.score ?? 0) < 0.3) return;
    const r = 6; // larger radius
    ctx.beginPath();
    ctx.arc(kp.x, kp.y, r, 0, 2 * Math.PI);
    ctx.fillStyle = colors.jointsFill;
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = colors.jointsStroke;
    ctx.stroke();
  });

  // Draw animated foot-strike arrows that fade out
  const drawStrikeArrow = (x: number, y: number, age: number, color: string) => {
    // age is in milliseconds since strike detected
    const maxAge = 300; // Fade out over 300ms
    const alpha = Math.max(0, 1 - (age / maxAge));
    
    if (alpha <= 0) return;
    
    // Arrow parameters
    const arrowHeight = 40;
    const arrowWidth = 20;
    const tipY = y + 10; // Point slightly below foot
    
    ctx.save();
    ctx.globalAlpha = alpha;
    
    // Draw arrow shaft
    ctx.beginPath();
    ctx.moveTo(x, tipY - arrowHeight);
    ctx.lineTo(x, tipY - 10);
    ctx.strokeStyle = color;
    ctx.lineWidth = 4;
    ctx.lineCap = 'round';
    ctx.stroke();
    
    // Draw arrow head (triangle)
    ctx.beginPath();
    ctx.moveTo(x, tipY); // tip
    ctx.lineTo(x - arrowWidth / 2, tipY - 15); // left
    ctx.lineTo(x + arrowWidth / 2, tipY - 15); // right
    ctx.closePath();
    ctx.fillStyle = color;
    ctx.fill();
    
    // Add a subtle glow effect
    ctx.shadowColor = color;
    ctx.shadowBlur = 10;
    ctx.fill();
    
    ctx.restore();
  };
  
  // Draw strike indicators
  if (options?.strikes?.left) {
    const strike = options.strikes.left;
    drawStrikeArrow(strike.x, strike.y, strike.age, '#f59e0b'); // amber for left
  }
  if (options?.strikes?.right) {
    const strike = options.strikes.right;
    drawStrikeArrow(strike.x, strike.y, strike.age, '#ef4444'); // red for right
  }
}

function getSkeletonConnections(): [number, number][] {
  // BlazePose (MediaPipe) keypoint indices
  return [
    // Face
    [0, 2],   // nose to left eye
    [0, 5],   // nose to right eye
    [2, 7],   // left eye to left ear
    [5, 8],   // right eye to right ear
    [9, 10],  // mouth
    
    // Torso
    [11, 12], // shoulders
    [11, 23], // left shoulder to left hip
    [12, 24], // right shoulder to right hip
    [23, 24], // hips
    
    // Left arm
    [11, 13], // left shoulder to left elbow
    [13, 15], // left elbow to left wrist
    
    // Right arm
    [12, 14], // right shoulder to right elbow
    [14, 16], // right elbow to right wrist
    
    // Left leg
    [23, 25], // left hip to left knee
    [25, 27], // left knee to left ankle
    [27, 29], // left ankle to left heel
    [27, 31], // left ankle to left toe
    [29, 31], // left heel to left toe (foot)
    
    // Right leg
    [24, 26], // right hip to right knee
    [26, 28], // right knee to right ankle
    [28, 30], // right ankle to right heel
    [28, 32], // right ankle to right toe
    [30, 32], // right heel to right toe (foot)
  ];
}

// BlazePose 33 keypoints (MediaPipe Pose)
export const KEYPOINT_NAMES = [
  'nose',              // 0
  'left_eye_inner',    // 1
  'left_eye',          // 2
  'left_eye_outer',    // 3
  'right_eye_inner',   // 4
  'right_eye',         // 5
  'right_eye_outer',   // 6
  'left_ear',          // 7
  'right_ear',         // 8
  'mouth_left',        // 9
  'mouth_right',       // 10
  'left_shoulder',     // 11
  'right_shoulder',    // 12
  'left_elbow',        // 13
  'right_elbow',       // 14
  'left_wrist',        // 15
  'right_wrist',       // 16
  'left_pinky',        // 17
  'right_pinky',       // 18
  'left_index',        // 19
  'right_index',       // 20
  'left_thumb',        // 21
  'right_thumb',       // 22
  'left_hip',          // 23
  'right_hip',         // 24
  'left_knee',         // 25
  'right_knee',        // 26
  'left_ankle',        // 27
  'right_ankle',       // 28
  'left_heel',         // 29
  'right_heel',        // 30
  'left_foot_index',   // 31 (toe)
  'right_foot_index',  // 32 (toe)
];
