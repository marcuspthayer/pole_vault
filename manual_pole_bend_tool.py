import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

window_name = "Pole Bend Annotator"

# Globals used by mouse callback
current_points = []        # points in ORIGINAL frame coordinates
max_points_this_step = 0   # how many clicks allowed
current_step = 0           # 0=full length, 1=plant orientation, 2=max bend
display_scale = 1.0        # scale factor from original -> displayed frame


def mouse_callback(event, x, y, flags, param):
    """
    Mouse callback: x, y come in DISPLAY coordinates.
    We convert to ORIGINAL frame coordinates using display_scale.
    """
    global current_points, max_points_this_step, display_scale

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(current_points) < max_points_this_step:
            # map display coords back to original coords
            x_orig = int(x / display_scale)
            y_orig = int(y / display_scale)
            current_points.append((x_orig, y_orig))


def draw_points_and_lines(display_frame, pts_orig, color=(0, 255, 0)):
    """
    Draw clicked points/line on the DISPLAY-SIZED frame.
    pts_orig are in ORIGINAL coordinates, so we scale them.
    """
    vis = display_frame.copy()
    for (xo, yo) in pts_orig:
        xd = int(xo * display_scale)
        yd = int(yo * display_scale)
        cv2.circle(vis, (xd, yd), 5, color, -1)
    if len(pts_orig) == 2:
        x1 = int(pts_orig[0][0] * display_scale)
        y1 = int(pts_orig[0][1] * display_scale)
        x2 = int(pts_orig[1][0] * display_scale)
        y2 = int(pts_orig[1][1] * display_scale)
        cv2.line(vis, (x1, y1), (x2, y2), color, 2)
    return vis


def distance(p1, p2):
    return float(np.linalg.norm(np.array(p1, dtype=float) - np.array(p2, dtype=float)))


def pick_video_file_and_screen():
    """
    Open a file dialog and return (video_path, screen_w, screen_h),
    or (None, None, None) if canceled.
    """
    root = tk.Tk()
    root.withdraw()  # hide the root window

    # Get screen size
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()

    filetypes = [
        ("Video files", "*.mp4 *.avi *.mov *.mkv *.m4v"),
        ("All files", "*.*"),
    ]

    path = filedialog.askopenfilename(
        title="Select a pole vault video",
        filetypes=filetypes
    )
    root.destroy()

    if not path:
        return None, None, None
    return path, screen_w, screen_h


def main():
    global current_points, max_points_this_step, current_step, display_scale

    # --- File picker + screen size ---
    video_path, screen_w, screen_h = pick_video_file_and_screen()
    if video_path is None:
        print("No video selected. Exiting.")
        return
    print("Selected video:", video_path)
    print(f"Screen size: {screen_w} x {screen_h}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video:", video_path)
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Loaded video with", total_frames, "frames")

    # Start in the middle of the video
    frame_idx = total_frames // 2
    print(f"Starting annotation at frame {frame_idx} (about halfway).")

    # Read a frame at the start index to decide scaling
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame0 = cap.read()
    if not ret:
        print("Error: could not read initial frame at index", frame_idx)
        cap.release()
        return

    h0, w0 = frame0.shape[:2]

    # Make the window up to ~75% of the screen size
    screen_factor = 0.75
    max_disp_w = int(screen_w * screen_factor)
    max_disp_h = int(screen_h * screen_factor)

    sx = max_disp_w / w0
    sy = max_disp_h / h0

    # Allow upscaling OR downscaling so the video actually fills ~75% of the screen
    display_scale = min(sx, sy)

    print(f"Original frame size: {w0} x {h0}")
    print(f"Target display size: ~{int(w0 * display_scale)} x {int(h0 * display_scale)}")
    print(f"Display scale:       {display_scale:.3f}")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # allow manual resize
    cv2.setMouseCallback(window_name, mouse_callback)

    # Explicitly set initial window size to match our target scaled frame
    initial_w = int(w0 * display_scale)
    initial_h = int(h0 * display_scale)
    cv2.resizeWindow(window_name, initial_w, initial_h)

    # Data collected across steps (all in ORIGINAL coords)
    full_length_points = []       # [p0, p1] before plant (straight pole)
    step1_frame_idx = None        # frame index for straight pole

    plant_tip_point = None        # (x,y) estimated tip at plant (occluded)
    plant_top_point = None        # (x,y) top point used at plant
    plant_other_point = None      # (x,y) second point at plant
    plant_frame_idx = None        # frame index for plant

    maxbend_top_point = None      # (x,y) at max bend
    maxbend_frame_idx = None      # frame index for max bend

    # Step labels & required clicks
    step_descriptions = [
        "STEP 1/3: Pre-plant frame. Click TWO points on the straight pole (tip & top, any order).",
        "STEP 2/3: Plant frame. Click TWO points on the visible pole (one near top, one lower).",
        "STEP 3/3: Max bend frame. Click the TOP of the pole (1 click)."
    ]
    # Step 1: 2 clicks (full length)
    # Step 2: 2 clicks (top + another point to define direction)
    # Step 3: 1 click (top at max bend)
    step_required_clicks = [2, 2, 1]

    # Main loop over steps
    for step in range(3):
        current_step = step
        max_points_this_step = step_required_clicks[step]
        current_points = []
        print(step_descriptions[step])
        print("Controls: n=next, p=prev, c=clear clicks, space=confirm step, q=quit")

        while True:
            # Clamp frame index
            frame_idx = max(0, min(frame_idx, total_frames - 1))
            # Seek to frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                print("Error: could not read frame", frame_idx)
                break

            # Resize frame for display
            if display_scale != 1.0:
                display = cv2.resize(
                    frame, None,
                    fx=display_scale, fy=display_scale,
                    interpolation=cv2.INTER_AREA
                )
            else:
                display = frame.copy()

            # Draw current step's points/line (current_points are ORIGINAL coords)
            display = draw_points_and_lines(display, current_points, color=(0, 255, 0))

            # Overlay text
            text = f"Step {step+1}/3 | Frame {frame_idx} | Clicks: {len(current_points)}/{max_points_this_step}"
            cv2.putText(display, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 255), 2)
            cv2.imshow(window_name, display)

            key = cv2.waitKey(30) & 0xFF

            if key == ord('q'):
                print("Quitting early.")
                cap.release()
                cv2.destroyAllWindows()
                return

            elif key == ord('n'):
                frame_idx += 1

            elif key == ord('p'):
                frame_idx -= 1

            elif key == ord('c'):
                current_points = []

            elif key == 32:  # spacebar to confirm
                if len(current_points) < max_points_this_step:
                    print("Not enough points yet, need", max_points_this_step)
                else:
                    # Save the points for this step
                    if step == 0:
                        # Full length, 2 points anywhere on straight pole.
                        # We'll just use the distance between them as L_full.
                        full_length_points = current_points.copy()
                        step1_frame_idx = frame_idx
                        print("Saved pre-plant straight pole points:", full_length_points)
                        print("Step 1 frame index:", step1_frame_idx)

                    elif step == 1:
                        # Plant frame: two points on visible pole.
                        # Use them to define direction and then extend by L_full.
                        pA, pB = current_points
                        # Decide which is top: smaller y is higher in the image
                        if pA[1] < pB[1]:
                            plant_top_point = pA
                            plant_other_point = pB
                        else:
                            plant_top_point = pB
                            plant_other_point = pA

                        plant_frame_idx = frame_idx

                        print("Plant frame points (original):", current_points)
                        print("Interpreted plant TOP point:", plant_top_point)
                        print("Interpreted plant SECOND point:", plant_other_point)
                        print("Plant frame index:", plant_frame_idx)

                        # Compute L_full from step 1
                        if len(full_length_points) != 2:
                            print("Error: Full-length points missing; cannot estimate plant tip.")
                        else:
                            q1, q2 = full_length_points
                            L_full = distance(q1, q2)
                            print(f"Computed straight pole length L_full = {L_full:.2f} px")

                            # Direction from top -> other
                            vec = np.array(plant_other_point, dtype=float) - np.array(plant_top_point, dtype=float)
                            norm = np.linalg.norm(vec)
                            if norm < 1e-6:
                                print("Error: Plant points too close together to define a direction.")
                                plant_tip_point = None
                            else:
                                direction = vec / norm
                                # Extrapolate along this direction by full pole length
                                tip_est = np.array(plant_top_point, dtype=float) + direction * L_full
                                plant_tip_point = (float(tip_est[0]), float(tip_est[1]))
                                print("Estimated occluded plant tip position:", plant_tip_point)

                    elif step == 2:
                        # Max bend: single click for pole top
                        maxbend_top_point = current_points[0]
                        maxbend_frame_idx = frame_idx
                        print("Saved max-bend top point:", maxbend_top_point)
                        print("Max-bend frame index:", maxbend_frame_idx)

                    break  # move to next step

    cap.release()
    cv2.destroyAllWindows()

    # Compute lengths and bend percentage + save annotated frames
    if (
        len(full_length_points) == 2
        and plant_tip_point is not None
        and maxbend_top_point is not None
        and step1_frame_idx is not None
        and plant_frame_idx is not None
        and maxbend_frame_idx is not None
    ):
        # Straight pole chord length (pre-plant)
        q1, q2 = full_length_points
        L_full = distance(q1, q2)

        # Bent chord uses plant tip estimate + new top at max bend
        L_bent = distance(plant_tip_point, maxbend_top_point)

        bend_fraction = 1.0 - (L_bent / L_full)
        bend_percent = bend_fraction * 100.0

        print("\n=== Results ===")
        print(f"Straight chord length L_full: {L_full:.2f} px")
        print(f"Max-bend chord length L_bent: {L_bent:.2f} px")
        print(f"Max bend (shrink) fraction:   {bend_fraction:.4f}")
        print(f"Max bend (shrink) percent:    {bend_percent:.2f}%")
        print(f"Estimated plant tip (occluded) at: {plant_tip_point}")

        # ---------- Save STEP 1 annotated frame ----------
        cap1 = cv2.VideoCapture(video_path)
        cap1.set(cv2.CAP_PROP_POS_FRAMES, step1_frame_idx)
        ret1, frame_step1 = cap1.read()
        cap1.release()
        if ret1:
            p1 = (int(round(full_length_points[0][0])), int(round(full_length_points[0][1])))
            p2 = (int(round(full_length_points[1][0])), int(round(full_length_points[1][1])))
            ann1 = frame_step1.copy()
            cv2.line(ann1, p1, p2, (0, 0, 255), 2)
            cv2.circle(ann1, p1, 6, (255, 0, 0), -1)
            cv2.circle(ann1, p2, 6, (0, 255, 0), -1)

            text1 = "Step 1: Straight pole reference"
            text2 = f"L_full = {L_full:.1f} px"
            cv2.putText(ann1, text1, (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.putText(ann1, text2, (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            out1 = "step1_straight_pole.png"
            cv2.imwrite(out1, ann1)
            print(f"Saved Step 1 annotated frame as {out1}")

        # ---------- Save STEP 2 (plant) annotated frame ----------
        cap2 = cv2.VideoCapture(video_path)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, plant_frame_idx)
        ret2, frame_step2 = cap2.read()
        cap2.release()
        if ret2:
            ann2 = frame_step2.copy()
            top_px = (int(round(plant_top_point[0])), int(round(plant_top_point[1])))
            other_px = (int(round(plant_other_point[0])), int(round(plant_other_point[1])))
            tip_px = (int(round(plant_tip_point[0])), int(round(plant_tip_point[1])))

            # Draw the visible segment the user clicked
            cv2.line(ann2, top_px, other_px, (0, 255, 0), 2)
            cv2.circle(ann2, top_px, 6, (0, 255, 0), -1)   # top
            cv2.circle(ann2, other_px, 6, (0, 255, 255), -1)  # second point

            # Draw the full projected line from top to estimated tip
            cv2.line(ann2, top_px, tip_px, (0, 0, 255), 2)
            cv2.circle(ann2, tip_px, 6, (255, 0, 0), -1)  # projected tip

            text1 = "Step 2: Plant frame with projected hidden tip"
            text2 = f"L_full = {L_full:.1f} px"
            cv2.putText(ann2, text1, (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.putText(ann2, text2, (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            out2 = "step2_plant_projection.png"
            cv2.imwrite(out2, ann2)
            print(f"Saved Step 2 annotated frame as {out2}")

        # ---------- Save STEP 3 (max bend) annotated frame ----------
        cap3 = cv2.VideoCapture(video_path)
        cap3.set(cv2.CAP_PROP_POS_FRAMES, maxbend_frame_idx)
        ret3, frame_mb = cap3.read()
        cap3.release()

        if ret3:
            # Convert points to int pixel coords
            tip_px = (int(round(plant_tip_point[0])), int(round(plant_tip_point[1])))
            top_px = (int(round(maxbend_top_point[0])), int(round(maxbend_top_point[1])))

            annotated = frame_mb.copy()
            cv2.line(annotated, tip_px, top_px, (0, 0, 255), 2)
            cv2.circle(annotated, tip_px, 6, (255, 0, 0), -1)  # tip: blue
            cv2.circle(annotated, top_px, 6, (0, 255, 0), -1)  # top: green

            text1 = f"Step 3: Max bend = {bend_percent:.2f}%"
            text2 = f"L_full={L_full:.1f}px, L_bent={L_bent:.1f}px"
            cv2.putText(annotated, text1, (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.putText(annotated, text2, (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            out3 = "step3_maxbend_annotated.png"
            cv2.imwrite(out3, annotated)
            print(f"Saved Step 3 annotated frame as {out3}")

            # Show the max-bend annotated frame
            if display_scale != 1.0:
                annotated_disp = cv2.resize(
                    annotated, None,
                    fx=display_scale, fy=display_scale,
                    interpolation=cv2.INTER_AREA
                )
            else:
                annotated_disp = annotated

            cv2.imshow("Max Bend Annotated", annotated_disp)
            print("Press any key in the window to close the annotated view.")
            cv2.waitKey(0)
            try:
                cv2.destroyWindow("Max Bend Annotated")
            except cv2.error:
                # Window might already be closed; ignore the error
                pass


    else:
        print("Missing points or frames; could not compute bend or save all annotated frames.")


if __name__ == "__main__":
    main()
