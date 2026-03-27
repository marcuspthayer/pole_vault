/**
 * Simple 2D Kalman Filter for tracking ankle position and velocity
 * State: [x, y, vx, vy] - position and velocity in 2D
 */

export class KalmanFilter2D {
  // State vector: [x, y, vx, vy]
  private state: [number, number, number, number];
  
  // State covariance matrix (4x4) - simplified as diagonal
  private P: number[];
  
  // Process noise (how much we trust the model)
  private Q: number;
  
  // Measurement noise (how much we trust the measurements)
  private R: number;
  
  // Time of last update
  private lastTime: number;
  
  constructor(initialX: number, initialY: number, processNoise = 0.1, measurementNoise = 5.0) {
    this.state = [initialX, initialY, 0, 0]; // Start with zero velocity
    this.P = [10, 10, 10, 10]; // Initial uncertainty (diagonal only for simplicity)
    this.Q = processNoise;
    this.R = measurementNoise;
    this.lastTime = Date.now();
  }
  
  /**
   * Predict next state based on constant velocity model
   */
  predict(currentTime: number): void {
    const dt = Math.min((currentTime - this.lastTime) / 1000, 0.1); // Cap at 100ms
    
    // State transition: x = x + vx*dt, y = y + vy*dt, vx = vx, vy = vy
    this.state[0] += this.state[2] * dt;
    this.state[1] += this.state[3] * dt;
    
    // Increase uncertainty due to process noise
    this.P[0] += this.Q * dt;
    this.P[1] += this.Q * dt;
    this.P[2] += this.Q * dt * 0.1; // Velocity uncertainty grows slower
    this.P[3] += this.Q * dt * 0.1;
  }
  
  /**
   * Update state with new measurement
   */
  update(measuredX: number, measuredY: number, currentTime: number): void {
    // Predict first
    this.predict(currentTime);
    
    const dt = Math.min((currentTime - this.lastTime) / 1000, 0.1);
    this.lastTime = currentTime;
    
    // Kalman gain for position (simplified)
    const Kx = this.P[0] / (this.P[0] + this.R);
    const Ky = this.P[1] / (this.P[1] + this.R);
    
    // Innovation (difference between measurement and prediction)
    const innovationX = measuredX - this.state[0];
    const innovationY = measuredY - this.state[1];
    
    // Update position
    this.state[0] += Kx * innovationX;
    this.state[1] += Ky * innovationY;
    
    // Update velocity estimate based on position change
    if (dt > 0.001) {
      const measuredVx = innovationX / dt;
      const measuredVy = innovationY / dt;
      
      // Blend measured velocity with predicted velocity
      const alpha = 0.3; // How much to trust new velocity measurement
      this.state[2] = (1 - alpha) * this.state[2] + alpha * measuredVx;
      this.state[3] = (1 - alpha) * this.state[3] + alpha * measuredVy;
    }
    
    // Update uncertainty (reduce after measurement)
    this.P[0] *= (1 - Kx);
    this.P[1] *= (1 - Ky);
  }
  
  /**
   * Get current state estimate
   */
  getState(): { x: number; y: number; vx: number; vy: number } {
    return {
      x: this.state[0],
      y: this.state[1],
      vx: this.state[2],
      vy: this.state[3],
    };
  }
  
  /**
   * Detect if foot is landing based on filtered state
   * Returns true when foot is moving down and decelerating near ground
   */
  detectLanding(groundY: number, threshold = 20): boolean {
    const { y, vy } = this.getState();
    
    // Check if:
    // 1. Near ground (within threshold pixels)
    // 2. Moving downward (vy > 0, since Y increases downward)
    // 3. Velocity is significant but not too fast (actual landing phase)
    const nearGround = Math.abs(y - groundY) < threshold;
    const movingDown = vy > 5; // pixels/second, moving down
    const notTooFast = vy < 200; // Not in free fall
    
    return nearGround && movingDown && notTooFast;
  }
  
  /**
   * Check if foot has left the ground (toe-off)
   */
  detectToeOff(groundY: number, threshold = 15): boolean {
    const { y, vy } = this.getState();
    
    // Moving up and away from ground
    const movingUp = vy < -5;
    const aboveGround = y < groundY - threshold;
    
    return movingUp || aboveGround;
  }
}
