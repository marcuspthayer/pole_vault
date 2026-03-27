/**
 * Gait metrics calculation from pose data
 */

import { Pose, PoseFrame, getKeypointIndices } from './pose-estimation';

export interface JointLoad {
  joint: string;
  avgLoad: number;
  peakLoad: number;
  unit: string;
}

export interface FootStrikeAnalysis {
  type: 'forefoot' | 'midfoot' | 'heel' | 'unknown';
  confidence: number;
  frameIndices?: number[];
  avgFootAngle?: number;
}

export interface GaitMetrics {
  cadence: number | null;
  avgStrideLength: number | null;
  footStrikeType: 'forefoot' | 'midfoot' | 'heel' | 'unknown';
  footStrikeAnalysis: FootStrikeAnalysis;
  avgFootStrikePosition: number | null;
  footStrikePositionConfidence: number | null;
  jointAngles: JointAngles[];
  estimatedLoads: JointLoad[];
}

export interface JointAngles {
  timestamp: number;
  leftKnee: number | null;
  rightKnee: number | null;
  leftHip: number | null;
  rightHip: number | null;
  leftAnkle: number | null;
  rightAnkle: number | null;
  torsoAngle: number | null;
}

export function calculateGaitMetrics(
  frames: PoseFrame[], 
  fps: number, 
  heightMeters: number = 1.7 // Default to average height if not provided
): GaitMetrics {
  const validFrames = frames.filter(frame => frame.pose?.keypoints);
  
  // Calculate actual FPS from timestamps if we have enough frames
  const actualFps = calculateActualFps(validFrames);
  const effectiveFps = actualFps || fps; // Fall back to provided fps if calculation fails
  
  const jointAngles = validFrames.map(frame => calculateJointAngles(frame));
  const cadence = calculateCadence(validFrames, effectiveFps);
  const strideLength = estimateStrideLength(validFrames, heightMeters);
  const footStrikeAnalysis = determineFootStrikeType(validFrames, effectiveFps);
  const footStrikePosition = calculateFootStrikePosition(validFrames, heightMeters, effectiveFps);
  const footStrikeType = footStrikeAnalysis.type;
  const loads = estimateJointLoads(jointAngles, validFrames);

  return {
    cadence,
    avgStrideLength: strideLength,
    footStrikeType,
    footStrikeAnalysis,
    avgFootStrikePosition: footStrikePosition?.position ?? null,
    footStrikePositionConfidence: footStrikePosition?.confidence ?? null,
    jointAngles,
    estimatedLoads: loads,
  };
}

function calculateActualFps(frames: PoseFrame[]): number | null {
  if (frames.length < 10) return null;
  
  // Calculate average time between frames
  let totalTimeDiff = 0;
  let count = 0;
  
  for (let i = 1; i < Math.min(frames.length, 100); i++) {
    const timeDiff = frames[i].timestamp - frames[i - 1].timestamp;
    if (timeDiff > 0 && timeDiff < 0.1) { // Sanity check: between 10-100 FPS
      totalTimeDiff += timeDiff;
      count++;
    }
  }
  
  if (count === 0) return null;
  
  const avgTimeBetweenFrames = totalTimeDiff / count;
  return 1 / avgTimeBetweenFrames;
}
function calculateJointAngles(frame: PoseFrame): JointAngles {
  const pose = frame.pose;
  
  if (!pose || !pose.keypoints) {
    return {
      timestamp: frame.timestamp,
      leftKnee: null,
      rightKnee: null,
      leftHip: null,
      rightHip: null,
      leftAnkle: null,
      rightAnkle: null,
      torsoAngle: null,
    };
  }

  const kp = pose.keypoints;
  const idx = getKeypointIndices();

  return {
    timestamp: frame.timestamp,
    leftKnee: calculateAngle(kp[idx.leftHip], kp[idx.leftKnee], kp[idx.leftAnkle]), // hip-knee-ankle
    rightKnee: calculateAngle(kp[idx.rightHip], kp[idx.rightKnee], kp[idx.rightAnkle]),
    leftHip: calculateAngle(kp[idx.leftShoulder], kp[idx.leftHip], kp[idx.leftKnee]), // shoulder-hip-knee
    rightHip: calculateAngle(kp[idx.rightShoulder], kp[idx.rightHip], kp[idx.rightKnee]),
    leftAnkle: calculateAngle(kp[idx.leftKnee], kp[idx.leftAnkle], { x: kp[idx.leftAnkle].x, y: kp[idx.leftAnkle].y + 10, score: 1.0 }), // knee-ankle-ground
    rightAnkle: calculateAngle(kp[idx.rightKnee], kp[idx.rightAnkle], { x: kp[idx.rightAnkle].x, y: kp[idx.rightAnkle].y + 10, score: 1.0 }),
    torsoAngle: calculateTorsoAngle(kp[idx.leftShoulder], kp[idx.rightShoulder], kp[idx.leftHip], kp[idx.rightHip]),
  };
}

function calculateAngle(
  p1: { x: number; y: number; score?: number },
  p2: { x: number; y: number; score?: number },
  p3: { x: number; y: number; score?: number }
): number | null {
  if (
    !p1.score || !p2.score || !p3.score ||
    p1.score < 0.3 || p2.score < 0.3 || p3.score < 0.3
  ) {
    return null;
  }

  const v1 = { x: p1.x - p2.x, y: p1.y - p2.y };
  const v2 = { x: p3.x - p2.x, y: p3.y - p2.y };

  const dot = v1.x * v2.x + v1.y * v2.y;
  const mag1 = Math.sqrt(v1.x * v1.x + v1.y * v1.y);
  const mag2 = Math.sqrt(v2.x * v2.x + v2.y * v2.y);

  const angle = Math.acos(dot / (mag1 * mag2));
  return (angle * 180) / Math.PI;
}

function calculateTorsoAngle(
  leftShoulder: any,
  rightShoulder: any,
  leftHip: any,
  rightHip: any
): number | null {
  if (
    !leftShoulder.score || !rightShoulder.score ||
    !leftHip.score || !rightHip.score ||
    leftShoulder.score < 0.3 || rightShoulder.score < 0.3 ||
    leftHip.score < 0.3 || rightHip.score < 0.3
  ) {
    return null;
  }

  const shoulderMid = {
    x: (leftShoulder.x + rightShoulder.x) / 2,
    y: (leftShoulder.y + rightShoulder.y) / 2,
  };

  const hipMid = {
    x: (leftHip.x + rightHip.x) / 2,
    y: (leftHip.y + rightHip.y) / 2,
  };

  // Angle from vertical
  const dx = shoulderMid.x - hipMid.x;
  const dy = shoulderMid.y - hipMid.y;
  const angle = Math.atan2(dx, dy) * (180 / Math.PI);
  
  return Math.abs(angle);
}

function calculateCadence(frames: PoseFrame[], fps: number): number | null {
  // Detect foot strikes by tracking ankle vertical position
  const leftAnklePositions: number[] = [];
  const rightAnklePositions: number[] = [];

  const idx = getKeypointIndices();
  
  frames.forEach(frame => {
    if (frame.pose && frame.pose.keypoints) {
      const leftAnkle = frame.pose.keypoints[idx.leftAnkle];
      const rightAnkle = frame.pose.keypoints[idx.rightAnkle];
      
      if (leftAnkle && leftAnkle.score && leftAnkle.score > 0.3) {
        leftAnklePositions.push(leftAnkle.y);
      }
      if (rightAnkle && rightAnkle.score && rightAnkle.score > 0.3) {
        rightAnklePositions.push(rightAnkle.y);
      }
    }
  });

  // Smooth the positions to reduce noise
  const smoothedLeft = smoothPositions(leftAnklePositions);
  const smoothedRight = smoothPositions(rightAnklePositions);

  const leftSteps = countSteps(smoothedLeft, fps);
  const rightSteps = countSteps(smoothedRight, fps);
  const totalSteps = leftSteps + rightSteps;

  if (totalSteps === 0 || frames.length === 0) return null;

  // Calculate duration from actual timestamps for maximum accuracy
  const durationSeconds = frames[frames.length - 1].timestamp - frames[0].timestamp;
  if (durationSeconds <= 0) return null;
  
  const durationMinutes = durationSeconds / 60;
  return totalSteps / durationMinutes; // steps per minute
}

function smoothPositions(positions: number[], windowSize: number = 5): number[] {
  if (positions.length < windowSize) return positions;
  
  const smoothed: number[] = [];
  for (let i = 0; i < positions.length; i++) {
    const start = Math.max(0, i - Math.floor(windowSize / 2));
    const end = Math.min(positions.length, i + Math.ceil(windowSize / 2));
    const window = positions.slice(start, end);
    const avg = window.reduce((a, b) => a + b, 0) / window.length;
    smoothed.push(avg);
  }
  return smoothed;
}

function countSteps(positions: number[], fps: number): number {
  if (positions.length < 20) return 0;

  let steps = 0;
  let lastPeakIndex = -1;
  
  // Find local maxima (foot lift) with minimum distance between peaks
  // Scale based on FPS: at 30 FPS = 10 frames, at 60 FPS = 20 frames
  const minPeakDistance = Math.round((fps / 30) * 10); // frames between steps (scales with fps)
  const minPeakHeight = 15; // minimum vertical movement in pixels
  
  for (let i = 1; i < positions.length - 1; i++) {
    // Check if this is a local maximum
    const isLocalMax = positions[i] > positions[i - 1] && positions[i] > positions[i + 1];
    
    if (isLocalMax && (lastPeakIndex === -1 || i - lastPeakIndex >= minPeakDistance)) {
      // Verify significant vertical movement from previous valley
      let minBefore = positions[i];
      for (let j = Math.max(0, i - minPeakDistance); j < i; j++) {
        minBefore = Math.min(minBefore, positions[j]);
      }
      
      if (positions[i] - minBefore >= minPeakHeight) {
        steps++;
        lastPeakIndex = i;
      }
    }
  }

  return steps;
}

function estimateStrideLength(frames: PoseFrame[], heightMeters: number = 1.7): number | null {
  if (!frames.length) return null;

  const minConfidence = 0.3;
  
  // Calculate leg length as approximately 45% of height
  const legLengthMeters = heightMeters * 0.45;
  
  // Calculate average stride length as a multiple of leg length
  // For walking, stride length is typically 1.14 to 1.17 * leg length
  // For running, it's typically 1.35 to 1.5 * leg length
  // We'll use 1.15 as a default walking/running average
  return legLengthMeters * 1.15;
}

interface StrikeData {
  frame: number;
  ankleAngle: number;
  footAngle: number;
  footStrike: number;
  confidence: number;
}

function determineFootStrikeType(frames: PoseFrame[], fps: number): { type: 'forefoot' | 'midfoot' | 'heel' | 'unknown', confidence: number } {
  // Default return values with low confidence
  const defaultResult = { type: 'midfoot' as const, confidence: 0.5 };
  
  if (!frames.length || frames.length < 10) {
    return { ...defaultResult, confidence: 0.3 };
  }

  const strikeData: StrikeData[] = [];
  const minConfidence = 0.2; // Slightly lower confidence threshold to include more data
  let lastAnkleY = Number.MAX_SAFE_INTEGER;
  let lastAnkleX = 0;
  let lastToeY = Number.MAX_SAFE_INTEGER;
  let lastHeelY = Number.MAX_SAFE_INTEGER;
  let totalFramesWithPose = 0;

  const idx = getKeypointIndices();
  
  // First pass: collect data and find key points
  for (let i = 0; i < frames.length; i++) {
    const frame = frames[i];
    if (!frame.pose?.keypoints) continue;

    const leftAnkle = frame.pose.keypoints[idx.leftAnkle];
    const leftKnee = frame.pose.keypoints[idx.leftKnee];
    const leftHip = frame.pose.keypoints[idx.leftHip];
    // Use heel/toe if available (BlazePose), otherwise fallback to ankle (MoveNet)
    const leftToe = (idx.leftToe >= 0 ? frame.pose.keypoints[idx.leftToe] : null) || leftAnkle;
    const leftHeel = (idx.leftHeel >= 0 ? frame.pose.keypoints[idx.leftHeel] : null) || leftAnkle;

    // Calculate confidence based on keypoint visibility
    const confidence = Math.min(
      leftAnkle?.score || 0,
      leftToe?.score || 0,
      leftHeel?.score || 0.5, // More lenient with heel/toe detection
      leftKnee?.score || 0,
      leftHip?.score || 0
    );

    if (confidence >= minConfidence) {
      totalFramesWithPose++;
      
      // Calculate shank angle (knee to ankle)
      const shankAngle = Math.atan2(
        leftAnkle.y - leftKnee.y,
        leftAnkle.x - leftKnee.x
      ) * (180 / Math.PI);

      // Calculate foot angle (ankle to toe)
      const footAngle = Math.atan2(
        leftToe.y - leftAnkle.y,
        leftToe.x - leftAnkle.x
      ) * (180 / Math.PI);

      // Calculate foot strike position (distance from ankle to toe)
      const footStrike = Math.hypot(
        leftToe.x - leftAnkle.x,
        leftToe.y - leftAnkle.y
      );

      // Track vertical movement for strike detection
      lastAnkleY = leftAnkle.y;
      lastToeY = leftToe.y;
      lastHeelY = leftHeel.y;

      strikeData.push({
        frame: i,
        ankleAngle: shankAngle,
        footAngle,
        footStrike,
        confidence
      });
    }
  }

  // If we don't have enough good quality frames, return a default with low confidence
  if (strikeData.length < 3 || totalFramesWithPose < frames.length * 0.3) {
    return { ...defaultResult, confidence: 0.3 };
  }

  // Find local minima in foot strike distance (points where foot is most planted)
  const minima: StrikeData[] = [];
  for (let i = 1; i < strikeData.length - 1; i++) {
    const prev = strikeData[i - 1];
    const curr = strikeData[i];
    const next = strikeData[i + 1];

    // Look for local minima in foot strike distance
    if (curr.footStrike < prev.footStrike && curr.footStrike < next.footStrike) {
      minima.push(curr);
    }
  }

  // If we can't find clear minima, analyze all frames with good confidence
  const analysisFrames = minima.length > 0 ? minima : strikeData.filter(s => s.confidence > 0.5);
  
  if (analysisFrames.length === 0) {
    return { ...defaultResult, confidence: 0.4 };
  }

  // Calculate weighted average of foot angles
  let weightedSum = 0;
  let totalWeight = 0;
  let totalConfidence = 0;
  
  analysisFrames.forEach(strike => {
    const weight = strike.confidence * (1 / (1 + Math.abs(strike.footAngle)));
    weightedSum += strike.footAngle * weight;
    totalWeight += weight;
    totalConfidence += strike.confidence;
  });

  const avgFootAngle = weightedSum / totalWeight;
  const avgConfidence = Math.min(0.9, Math.max(0.5, totalConfidence / analysisFrames.length));

  // Classify based on foot angle with dynamic thresholds
  // Adjusted thresholds for better classification
  if (avgFootAngle > 5) return { type: 'forefoot', confidence: avgConfidence };
  if (avgFootAngle < -3) return { type: 'heel', confidence: avgConfidence };
  
  // Default to midfoot with the calculated confidence
  return { type: 'midfoot', confidence: avgConfidence };
}

function calculateFootStrikePosition(
  frames: PoseFrame[], 
  heightMeters: number = 1.7,
  fps: number = 30
): { position: number; confidence: number } | null {
  if (!frames.length) return null;

  const minConfidence = 0.3;
  const positions: {x: number, y: number}[] = [];
  
  // Calculate foot length as approximately 15% of height
  const footLengthMeters = heightMeters * 0.15;
  
  // First, estimate pixels per meter using height (more stable for this use case)
  let totalHeightPx = 0;
  let heightMeasurements = 0;
  
  // Get stable height measurements from head to ankles
  for (let i = 0; i < Math.min(30, frames.length); i++) {
    const frame = frames[i];
    if (!frame.pose?.keypoints) continue;
    
    const nose = frame.pose.keypoints[0];
    const idx = getKeypointIndices();
    const leftAnkle = frame.pose.keypoints[idx.leftAnkle];
    const rightAnkle = frame.pose.keypoints[idx.rightAnkle];
    
    // Use the lower ankle for height measurement
    if (nose?.score && nose.score > minConfidence) {
      let ankleY: number | null = null;
      
      if (leftAnkle?.score && leftAnkle.score > minConfidence) {
        ankleY = leftAnkle.y;
      }
      if (rightAnkle?.score && rightAnkle.score > minConfidence) {
        if (ankleY === null || rightAnkle.y > ankleY) {
          ankleY = rightAnkle.y;
        }
      }
      
      if (ankleY !== null) {
        const frameHeightPx = ankleY - nose.y;
        if (frameHeightPx > 0) {
          totalHeightPx += frameHeightPx;
          heightMeasurements++;
        }
      }
    }
  }
  
  if (heightMeasurements === 0) return null;
  
  const avgHeightPx = totalHeightPx / heightMeasurements;
  // Convert pixels to meters using the actual height
  const pixelsPerMeter = avgHeightPx / heightMeters;
  
  // Find foot strike frames (when foot is at its lowest point)
  const footStrikeFrames: number[] = [];
  // Scale minimum frames between strikes based on FPS (5 frames at 30 FPS = 10 frames at 60 FPS)
  const minFramesBetweenStrikes = Math.round((fps / 30) * 5);
  let lastStrikeFrame = -minFramesBetweenStrikes;
  
  // First pass: find potential foot strike frames
  // Use a simpler 3-point check instead of 5-point for better detection
  let validFootChecks = 0;
  for (let i = 1; i < frames.length - 1; i++) {
    const frame = frames[i];
    if (!frame.pose?.keypoints) continue;
    
    // Skip if we're too close to the last detected strike
    if (i - lastStrikeFrame < minFramesBetweenStrikes) continue;
    
    // Get foot positions for current and adjacent frames
    const prevY = getFootY(frames[i-1], minConfidence);
    const currY = getFootY(frame, minConfidence);
    const nextY = getFootY(frames[i+1], minConfidence);
    
    if (prevY === null || currY === null || nextY === null) continue;
    validFootChecks++;
    
    // Check for a local minimum in foot height (foot strike)
    // The foot should be at its lowest point (highest Y value in image coordinates)
    if (currY >= prevY && currY >= nextY) {
      footStrikeFrames.push(i);
      lastStrikeFrame = i;
    }
  }
  
  // Calculate foot strike positions relative to CoG
  for (const frameIdx of footStrikeFrames) {
    const frame = frames[frameIdx];
    if (!frame.pose?.keypoints) continue;
    
    const idx = getKeypointIndices();
    const leftHip = frame.pose.keypoints[idx.leftHip];
    const rightHip = frame.pose.keypoints[idx.rightHip];
    const leftAnkle = frame.pose.keypoints[idx.leftAnkle];
    const rightAnkle = frame.pose.keypoints[idx.rightAnkle];
    const leftToe = (idx.leftToe >= 0 ? frame.pose.keypoints[idx.leftToe] : null) || leftAnkle;
    const rightToe = (idx.rightToe >= 0 ? frame.pose.keypoints[idx.rightToe] : null) || rightAnkle;
    
    // Skip if we don't have good hip or foot data
    if (!leftHip?.score || leftHip.score < minConfidence || 
        !rightHip?.score || rightHip.score < minConfidence) continue;
    
    // Determine which foot is in contact with the ground (lower in the frame)
    let toeX: number;
    let toeY: number;
    let footInContact: 'left' | 'right' | null = null;
    
    if (leftToe?.score && leftToe.score > minConfidence && rightToe?.score && rightToe.score > minConfidence) {
      if (leftToe.y > rightToe.y) {
        toeX = leftToe.x;
        toeY = leftToe.y;
        footInContact = 'left';
      } else {
        toeX = rightToe.x;
        toeY = rightToe.y;
        footInContact = 'right';
      }
    } else if (leftToe?.score && leftToe.score > minConfidence) {
      toeX = leftToe.x;
      toeY = leftToe.y;
      footInContact = 'left';
    } else if (rightToe?.score && rightToe.score > minConfidence) {
      toeX = rightToe.x;
      toeY = rightToe.y;
      footInContact = 'right';
    } else {
      continue;
    }
    
    // Calculate center of gravity (weighted by hip positions)
    const cogX = (leftHip.x + rightHip.x) / 2;
    
    // Calculate position in meters relative to CoG
    const positionMeters = (toeX - cogX) / pixelsPerMeter;
    const positionInFootLengths = positionMeters / footLengthMeters;
    
    // Only consider positions where foot is in front of or just behind CoG
    // Simplified: just check if position is reasonable
    if (positionMeters > -0.5 && positionMeters < 2.0) {
      positions.push({
        x: positionInFootLengths, // in foot lengths
        y: frame.timestamp
      });
      
    }
  }
  
  if (positions.length === 0) return null;
  
  // Sort by timestamp and take the most recent positions for better accuracy
  const recentPositions = positions
    .sort((a, b) => b.y - a.y) // Sort by most recent first
    .slice(0, 5) // Only use the most recent 5 strikes for stability
    .map(p => p.x);
  
  if (recentPositions.length === 0) return null;
  
  // Calculate robust average (median) to be less sensitive to outliers
  const sortedPositions = [...recentPositions].sort((a, b) => a - b);
  const medianPosition = sortedPositions[Math.floor(sortedPositions.length / 2)];
  
  // Calculate mean absolute deviation (more robust than standard deviation)
  const deviations = recentPositions.map(x => Math.abs(x - medianPosition));
  const mad = deviations.reduce((sum, d) => sum + d, 0) / deviations.length;
  
  // Filter out outliers (beyond 2 MADs from median)
  const filteredPositions = recentPositions.filter(
    x => Math.abs(x - medianPosition) < 2 * mad
  );
  
  if (filteredPositions.length === 0) return null;
  
  // Calculate final average and standard deviation
  const avgPosition = filteredPositions.reduce((sum, x) => sum + x, 0) / filteredPositions.length;
  const variance = filteredPositions.reduce((sum, x) => sum + Math.pow(x - avgPosition, 2), 0) / filteredPositions.length;
  const stdDev = Math.sqrt(variance);
  
  // Calculate confidence based on:
  // 1. Consistency (inverse of standard deviation)
  // 2. Number of samples
  // 3. How many samples were kept after outlier removal
  const consistency = Math.max(0, 1 - (stdDev / 0.3)); // 0-1, higher is better
  const sampleCountConfidence = Math.min(1, filteredPositions.length / 3); // 0-1 based on sample size
  const outlierRatio = filteredPositions.length / recentPositions.length; // 0-1, higher is better
  
  // Weighted average of confidence factors
  const confidence = (consistency * 0.5) + 
                   (sampleCountConfidence * 0.3) + 
                   (outlierRatio * 0.2);
  
  // Clamp to reasonable range (-1 to 3 foot lengths)
  // Negative means foot lands behind CoG, positive means in front
  // For display, we'll show absolute value but this helps with validation
  const clampedPosition = Math.max(-1, Math.min(3, avgPosition));
  
  // For display purposes, show absolute value (distance from CoG)
  const displayPosition = Math.abs(clampedPosition);
  
  return {
    position: parseFloat(displayPosition.toFixed(1)), // Round to 1 decimal place
    confidence: Math.min(1, Math.max(0, confidence)) // Clamp to 0-1
  };
}

// Helper function to get the Y position of the lowest foot
function getFootY(frame: PoseFrame, minConfidence: number): number | null {
  if (!frame.pose?.keypoints) return null;
  
  const idx = getKeypointIndices();
  const leftAnkle = frame.pose.keypoints[idx.leftAnkle];
  const rightAnkle = frame.pose.keypoints[idx.rightAnkle];
  
  if (leftAnkle?.score && leftAnkle.score > minConfidence && rightAnkle?.score && rightAnkle.score > minConfidence) {
    return Math.max(leftAnkle.y, rightAnkle.y);
  } else if (leftAnkle?.score && leftAnkle.score > minConfidence) {
    return leftAnkle.y;
  } else if (rightAnkle?.score && rightAnkle.score > minConfidence) {
    return rightAnkle.y;
  }
  
  return null;
}

/**
 * Estimate joint loads based on joint angles and body position
 * This is a simplified biomechanical model
 */
function estimateJointLoads(jointAngles: JointAngles[], frames: PoseFrame[]): JointLoad[] {
  if (jointAngles.length === 0) return [];
  
  const minConfidence = 0.3;
  
  // Calculate loads for each joint based on knee flexion angle
  // During running, peak loads occur at foot strike when knee is most extended
  const kneeLoads: number[] = [];
  const ankleLoads: number[] = [];
  
  for (let i = 0; i < jointAngles.length; i++) {
    const angles = jointAngles[i];
    const frame = frames[i];
    
    if (!frame?.pose?.keypoints) continue;
    
    // Get vertical velocity approximation (change in hip height)
    let verticalVelocity = 0;
    if (i > 0 && frames[i-1]?.pose?.keypoints && frame.pose?.keypoints) {
      const prevHip = frames[i-1].pose?.keypoints[11];
      const currHip = frame.pose.keypoints[11];
      if (prevHip?.score && prevHip.score > minConfidence && 
          currHip?.score && currHip.score > minConfidence) {
        verticalVelocity = Math.abs(currHip.y - prevHip.y);
      }
    }
    
    // Knee load estimation
    // Higher load when knee is more extended (larger angle) and during impact (high vertical velocity)
    if (angles.leftKnee !== null) {
      // Knee angle typically ranges from 140-180 degrees when extended
      // Normalize to 0-1 where 1 is fully extended
      const kneeExtension = Math.max(0, Math.min(1, (angles.leftKnee - 140) / 40));
      
      // Base load is 1.5-2.5 BW during running
      // Peak loads occur at foot strike (extended knee + high vertical velocity)
      const baseLoad = 1.5;
      const impactLoad = verticalVelocity * 0.05; // Scale velocity contribution
      const extensionLoad = kneeExtension * 0.8; // More load when extended
      
      const totalKneeLoad = baseLoad + impactLoad + extensionLoad;
      kneeLoads.push(Math.min(3.5, totalKneeLoad)); // Cap at 3.5 BW
    }
    
    // Ankle load estimation
    // Higher load during push-off (plantarflexion) and landing
    if (angles.leftAnkle !== null) {
      // Ankle angle typically ranges from 70-110 degrees
      // More plantarflexed (smaller angle) = higher load during push-off
      const ankleFlexion = Math.max(0, Math.min(1, (110 - angles.leftAnkle) / 40));
      
      // Base load is 2-3 BW during running
      const baseLoad = 2.0;
      const impactLoad = verticalVelocity * 0.06; // Slightly higher than knee
      const flexionLoad = ankleFlexion * 0.7;
      
      const totalAnkleLoad = baseLoad + impactLoad + flexionLoad;
      ankleLoads.push(Math.min(4.0, totalAnkleLoad)); // Cap at 4.0 BW
    }
  }
  
  // Calculate average and peak loads
  const loads: JointLoad[] = [];
  
  if (kneeLoads.length > 0) {
    const avgKneeLoad = kneeLoads.reduce((sum, load) => sum + load, 0) / kneeLoads.length;
    const peakKneeLoad = Math.max(...kneeLoads);
    
    loads.push({
      joint: 'Left Knee',
      avgLoad: avgKneeLoad,
      peakLoad: peakKneeLoad,
      unit: 'BW'
    });
  }
  
  if (ankleLoads.length > 0) {
    const avgAnkleLoad = ankleLoads.reduce((sum, load) => sum + load, 0) / ankleLoads.length;
    const peakAnkleLoad = Math.max(...ankleLoads);
    
    loads.push({
      joint: 'Left Ankle',
      avgLoad: avgAnkleLoad,
      peakLoad: peakAnkleLoad,
      unit: 'BW'
    });
  }
  
  return loads;
}
