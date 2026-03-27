import json
import numpy as np
import pandas as pd
import sqlite3

from src.app.streamlit.display.message import display_error
from src.db.os.database_paths import join_paths, path_exists, list_directory, splitext, basename
from src.db.sqlite.analysis_data.angle_data_database import get_all_angle_data, save_angle_data
from src.db.sqlite.analysis_data.position_data_database import select_all_position_data
from src.db.sqlite.analysis_data.speed_data_database import query_all_speed_data
from src.db.sqlite.user_database_access import connect_to_db
from src.utils.calculate.calculate import calculate_hip_angle
from src.utils.calculate.distance import pixels_to_cm
from src.utils.extract.extract import extract_plateau_widths_second


# def apsprint(fn, destination_folder, fps, gender, age, height_feet, height_inches, weight):
#     """
#     Analyze sprint kinematics from video data, calculating stride, joint angles, and ground contact metrics.

#     Args:
#         fn (str): Filename prefix for input and output files.
#         destination_folder (str): Path to store output files and database.
#         fps (int): Frames per second of the video (120 or 240).
#         gender (str): Athlete's gender ('male' or 'female') for optimal value selection.
#         age (int): Athlete's age (unused in calculations).
#         height_feet (int): Athlete's height in feet (unused in calculations).
#         height_inches (int): Athlete's height in inches (unused in calculations).
#         weight (float): Athlete's weight (unused in calculations).

#     Returns:
#         None: Saves results to a database and generates an HTML plot.
#     """
#     # Connect to the SQLite database
#     conn = connect_to_db(join_paths(destination_folder, f'{fn}_video.db'))
    
    
#     # Read the data from the database into a pandas DataFrame
#     df1 = select_all_position_data(conn)
#     df2 = get_all_angle_data(conn)
#     df3 = query_all_speed_data(conn)

#     meters_per_pixel = df3['meters_per_pixel'].mean()

#     df = pd.concat([df1, df2], axis=1)
#     df = df.loc[:,~df.columns.duplicated()]

#     # df = df.filter(regex='^(?!Unnamed:).*')
#     # df = df[['frame', 'time'] + [col for col in df.columns if col not in ['frame', 'time']]]

#     if fps == 240:
#         fps = fps 
#         derivative_threshold = 0.01
#         min_plateau_width = 24       
#         max_plateau_width = 50
#         frames_before = 0
#         frames_after = 0
#         fps_multiplier = 2
#         y_threshold = 0.78
#     elif fps == 120: 
#         fps = fps
#         derivative_threshold = 0.01
#         min_plateau_width = 12
#         max_plateau_width = 25
#         frames_before = 0
#         frames_after = 0
#         fps_multiplier = 2
#         y_threshold = 0.78

        
        

#     df['Ltoe_y_derivative'] = np.gradient(df['Ltoe_y'])

#     def extract_plateau_widths(df, derivative_column, y_column, frames_before, frames_after, min_plateau_width, max_plateau_width, derivative_threshold, y_threshold):
#         plateau_indices = np.where(
#             (df[derivative_column] >= -derivative_threshold) &
#             (df[derivative_column] <= derivative_threshold)
#         )[0]
#         plateau_start = None
#         plateau_end = None
#         plateau_widths = []

#         for index in plateau_indices:
#             if plateau_start is None:
#                 plateau_start = index - frames_before
#                 plateau_end = index + frames_after
#             elif index != plateau_end + 1:
#                 plateau_width = plateau_end - plateau_start
#                 if plateau_width >= min_plateau_width and \
#                     plateau_width <= max_plateau_width and \
#                     df[y_column].iloc[plateau_start] and df[y_column].iloc[plateau_end] > y_threshold:
#                     if plateau_end + 1 < len(df):
#                         if df[y_column].iloc[plateau_end] > df[y_column].iloc[plateau_end + 1]:
#                             plateau_widths.append((plateau_start, plateau_end))
#                     else:
#                         plateau_widths.append((plateau_start, plateau_end))
#                         print(f"Warning: Plateau ends at the last frame (index {plateau_end})")
#                 plateau_start = index - frames_before
#                 plateau_end = index + frames_after
#             else:
#                 plateau_end = index + frames_after

#         if plateau_start is not None:
#             plateau_width = plateau_end - plateau_start
#             if plateau_width >= min_plateau_width and \
#                 plateau_width <= max_plateau_width and \
#                 df[y_column].iloc[plateau_start] and df[y_column].iloc[plateau_end] > y_threshold:
#                 if plateau_end + 1 < len(df):
#                     if df[y_column].iloc[plateau_end] > df[y_column].iloc[plateau_end + 1]:
#                         plateau_widths.append((plateau_start, plateau_end))
#                 else:
#                     plateau_widths.append((plateau_start, plateau_end))
#                     print(f"Warning: Plateau ends at the last frame (index {plateau_end})")

#         return plateau_widths

#     manual_entry_json = join_paths(destination_folder, f'{fn}_manual_entry.json')
#     # st.write(f"Manual Entry JSON: {manual_entry_json}")
#     if path_exists(manual_entry_json):
#         with open(manual_entry_json, 'r') as json_file:
#             manual_entry = json.load(json_file)
#             plateau_widths = list(zip(manual_entry['starts'], manual_entry['ends']))
#     else:
#         plateau_widths = extract_plateau_widths(
#                 df = df, 
#                 derivative_column = 'Ltoe_y_derivative', 
#                 y_column = 'Ltoe_y', 
#                 frames_before = frames_before, 
#                 frames_after = frames_after, 
#                 min_plateau_width = min_plateau_width, 
#                 max_plateau_width = max_plateau_width, 
#                 derivative_threshold = derivative_threshold, 
#                 y_threshold = y_threshold
#             )   
    
#     # Output the results
#     num_plateaus = len(plateau_widths)
#     print(f"Number of plateaus extracted: {num_plateaus}")
#     if plateau_widths:
#         print("Plateau widths (start, end):")
#         for start, end in plateau_widths:
#             print(f"Start: {start}, End: {end}")

#     ### Filter and not continue if less than 2 plateaus are found ###
#     if num_plateaus < 2:
#         pass
#     else:
#         # Calculate 'snds' and 'st_rt'
#         df['steps'] = np.nan

#         for i in range(len(plateau_widths) - 1):
#             plateau_start = plateau_widths[i][0]
#             next_plateau_start = plateau_widths[i + 1][0]
#             df.loc[plateau_start:next_plateau_start - 1, 'steps'] = i + 1

#         df['steps'].ffill(inplace=True)

#         #calculate frames in each step
#         df['step_frames'] = np.nan

#         for i in range(len(plateau_widths) - 1):
#             plateau_start = plateau_widths[i][0]
#             next_plateau_start = plateau_widths[i + 1][0]
#             step_frames = next_plateau_start - plateau_start
#             df.loc[plateau_start:next_plateau_start - 1, 'step_frames'] = step_frames

#         df['step_frames'].ffill(inplace=True)
        
#         df['seconds'] = df['step_frames'] / fps
#         df['st_rt'] = (1 / df['seconds'])*fps_multiplier

#         for i in range(1, len(df)):
#             if abs(df.loc[i, 'st_rt'] - df.loc[i - 1, 'st_rt']) > 0.2:
#                 df.loc[i, 'st_rt'] = df.loc[i - 1, 'st_rt']  

#         ### Filter and not continue if the avg st_rt is less than 2.5 ###
#         if df['st_rt'].mean() < 2.5:
#             pass
#         else:
#             import plotly.graph_objects as go

#             # Create a figure with secondary y-axis
#             fig = go.Figure()

#             # First subplot for Ltoe_y position
#             fig.add_trace(go.Scatter(
#                 x=df['frame'],
#                 y=df["Ltoe_y"],
#                 mode='lines+markers',
#                 name='Ltoe_y',
#                 line=dict(dash='dash')
#             ))

#             fig.add_trace(go.Scatter(
#                 x=df['frame'],
#                 y=df["Ltoe_y_derivative"],
#                 mode='lines',
#                 name='Ltoe_y_derivative',
#                 line=dict(color='green')
#             ))

#             # Add stride rate trace
#             fig.add_trace(go.Scatter(
#                 x=df['frame'],
#                 y=df['st_rt'],
#                 mode='markers',
#                 name='Stride Rate',
#                 yaxis='y2',  # Use secondary y-axis
#                 marker=dict(
#                     color='red',
#                     size=2,
#                     symbol='circle'
#                 ),
#             ))

#             # Add plateau regions for Ltoe_y
#             for start, end in plateau_widths:
#                 fig.add_shape(type='rect',
#                             x0=df['frame'].iloc[start],
#                             x1=df['frame'].iloc[end],
#                             y0=-0.1,
#                             y1=1,
#                             fillcolor='red',
#                             opacity=0.5,
#                             line_width=0)

#             # Update layout for the figure
#             fig.update_layout(
#                 title=f'{fn} - Ground Contact Analysis',
#                 xaxis_title='Frames',
#                 yaxis_title='Ltoe_y Position',
#                 yaxis=dict(range=[-0.01, 0.01]),
#                 yaxis2=dict(
#                     title='Stride Rate',
#                     overlaying='y',
#                     side='right'
#                 ),
#                 template='plotly_white'
#             )

#             # Save the figure as an interactive HTML file
#             fig.write_html(join_paths(destination_folder, f'{fn}_ground_contact_regions.html'))

#             # Show the figure in a notebook or web app
#             # fig.show()


#         # Calculate gct for each step
#             if plateau_widths:  # Check if plateau_widths is not empty
#                 gct_frames = [end - start for start, end in plateau_widths]
                
#                 # Initialize gct_frames column
#                 df['gct_frames'] = np.nan
                
#                 for i, gct in enumerate(gct_frames, start=1):
#                     if i >= 1:
#                         try:
#                             start, end = plateau_widths[i-1]
#                             # Create the mask using comparison operators
#                             # mask = (df['frame'] >= start) & (df['frame'] <= end) ###########
#                             mask = df['frame'].astype(float) >= start
#                             mask &= df['frame'].astype(float) <= end
#                             # Assign values using the mask
#                             df.loc[mask, 'gct_frames'] = gct
#                         except Exception as e:
#                             print(f"Error processing plateau {i}: {e}")
#                             continue
                
#                 # Forward fill and calculate final values
#                 df['gct_frames'] = df['gct_frames'].ffill()
#                 df['gct'] = (df['gct_frames']/fps).round(4)
#             else:
#                 print("No plateau regions found")
#                 df['gct_frames'] = 0
#                 df['gct'] = 0
#             #forward fill ground contact time
#             unique_steps = df['steps'].unique()

#             for step in unique_steps:
#                 first_non_nan_index = df.loc[df['steps'] == step, 'gct_frames'].first_valid_index()
#                 if first_non_nan_index is not None:
#                     df.loc[df['steps'] == step, 'gct_frames'] = df.loc[first_non_nan_index, 'gct_frames']

#             for step in unique_steps:
#                 first_non_nan_index = df.loc[df['steps'] == step, 'gct'].first_valid_index()
#                 if first_non_nan_index is not None:
#                     df.loc[df['steps'] == step, 'gct'] = df.loc[first_non_nan_index, 'gct']


#             def calculate_hip_angle(knee_coord, hip_coord, side):
#                 # Convert coordinates to numpy arrays
#                 knee_coord = np.array(knee_coord)
#                 hip_coord = np.array(hip_coord)
                
#                 # Define the vertical vector (assuming positive y-axis is vertical)
#                 vertical_vector = np.array([0, 1])
                
#                 # Calculate the vector from knee to hip
#                 knee_to_hip_vector = hip_coord - knee_coord
                
#                 # Normalize the vectors
#                 knee_to_hip_vector_normalized = knee_to_hip_vector / np.linalg.norm(knee_to_hip_vector)
#                 vertical_vector_normalized = vertical_vector / np.linalg.norm(vertical_vector)
                
#                 # Calculate the angle using the dot product
#                 dot_product = np.dot(knee_to_hip_vector_normalized, vertical_vector_normalized)
#                 angle_radians = np.arccos(dot_product)
#                 angle_degrees = np.degrees(angle_radians)
                
#                 # Calculate the hip angle as 360 minus the angle
#                 # hip_angle = 360 - angle_degrees
#                 # hip_angle = angle_degrees + 180
#                 hip_angle = angle_degrees
                
#                 return hip_angle
            
#             left_hip_angles = []
#             right_hip_angles = []

#             # Calculate hip angles
#             for index, row in df.iterrows():
#                     left_hip_coord = np.array([row['Lhip_x'], row['Lhip_y']])
#                     right_hip_coord = np.array([row['Rhip_x'], row['Rhip_y']])
#                     left_knee_coord = np.array([row['Lknee_x'], row['Lknee_y']])
#                     right_knee_coord = np.array([row['Rknee_x'], row['Rknee_y']])
                
#                     left_hip_angle = calculate_hip_angle(left_hip_coord, left_knee_coord, 'left')
#                     right_hip_angle = calculate_hip_angle(right_hip_coord, right_knee_coord, 'right')
                
                
#                     left_hip_angles.append(left_hip_angle)
#                     right_hip_angles.append(right_hip_angle)
            
#             # Assign q angles to DataFrame
#             df['Left_Hip_Angle'] = left_hip_angles
#             df['Right_Hip_Angle'] = right_hip_angles

#             # pull max hip angles during each step
#             df['max_hip_angle'] = np.nan

#             for step in unique_steps:
#                 max_hip_angle = df.loc[df['steps'] == step, 'Left_Hip_Angle'].max()
#                 df.loc[df['steps'] == step, 'max_hip_angle'] = max_hip_angle

#             df['max_hip_angle'].ffill(inplace=True)
            

#             df['Left_Knee_Angle'] = 180 - df['Left_Knee_Angle']
#             df['Right_Knee_Angle'] = 180 - df['Right_Knee_Angle']

#             # pull min knee angle during each step
#             df['min_knee_angle'] = np.nan

#             for step in unique_steps:
#                 min_knee_angle = df.loc[df['steps'] == step, 'Left_Knee_Angle'].max()
#                 df.loc[df['steps'] == step, 'min_knee_angle'] = min_knee_angle

#             df['min_knee_angle'].ffill(inplace=True)

#             #pull max knee angle for each step
#             df['max_knee_angle'] = np.nan

#             for step in unique_steps:
#                 max_knee_angle = df.loc[df['steps'] == step, 'Left_Knee_Angle'].min()
#                 df.loc[df['steps'] == step, 'max_knee_angle'] = max_knee_angle

#             df['max_knee_angle'].ffill(inplace=True)

#                 # Calculate trunk angles
#             df['trunk_angle'] = np.nan

#             for index, row in df.iterrows():
#                 # Calculate midpoints on the z-axis
#                 shoulder_midpoint_x = (row['Lshldr_x'] + row['Rshldr_x']) / 2
#                 shoulder_midpoint_y = (row['Lshldr_y'] + row['Rshldr_y']) / 2
            

#                 hip_midpoint_x = (row['Lhip_x'] + row['Rhip_x']) / 2
#                 hip_midpoint_y = (row['Lhip_y'] + row['Rhip_y']) / 2
            

#                 hip_coord = np.array([hip_midpoint_x, hip_midpoint_y])
#                 shoulder_coord = np.array([shoulder_midpoint_x, shoulder_midpoint_y])  

#                 # Create a vector between the midpoints
#                 hip_shoulder_vector = hip_coord - shoulder_coord

#                 # Define the vertical vector
#                 vertical_vector = np.array([0, -1])  # Ensure the vector points upwards

#                 # Calculate the angle between the hip_shoulder_vector and the vertical_vector
#                 angle_radians = np.arctan2(np.linalg.norm(np.cross(hip_shoulder_vector, vertical_vector)),
#                                         np.dot(hip_shoulder_vector, vertical_vector))
#                 angle_degrees = np.degrees(angle_radians)

#                 df.at[index, 'trunk_angle'] = angle_degrees

        
#             def plot_trunk_angle(df, start_index):
#                 import plotly.graph_objects as go

#                 # Create a Plotly figure
#                 fig = go.Figure()

#                 # Use the start index to get the row
#                 row = df.loc[start_index]
            
#                 shoulder_midpoint_x = (row['Lshldr_x'] + row['Rshldr_x']) / 2
#                 shoulder_midpoint_y = (row['Lshldr_y'] + row['Rshldr_y']) / 2
            
#                 hip_midpoint_x = (row['Lhip_x'] + row['Rhip_x']) / 2
#                 hip_midpoint_y = (row['Lhip_y'] + row['Rhip_y']) / 2
            
#                 hip_coord = np.array([hip_midpoint_x, hip_midpoint_y])
#                 shoulder_coord = np.array([shoulder_midpoint_x, shoulder_midpoint_y])  

#                 # Create a vector between the midpoints
#                 hip_shoulder_vector = hip_coord - shoulder_coord
#                 vertical_vector = [0, -1]

#                 # Add traces for the hip-shoulder vector
#                 fig.add_trace(go.Scatter(
#                     x=[shoulder_midpoint_x, hip_midpoint_x], 
#                     y=[shoulder_midpoint_y, hip_midpoint_y],
#                     mode='lines',
#                     line=dict(dash='dash', color='red'),
#                     name='Hip-Shoulder Vector'
#                 ))

#                 # Add traces for the vertical vector
#                 fig.add_trace(go.Scatter(
#                     x=[hip_midpoint_x, hip_midpoint_x], 
#                     y=[hip_midpoint_y, hip_midpoint_y + 1],  # Assuming a unit vector for vertical
#                     mode='lines',
#                     line=dict(dash='dot', color='blue'),
#                     name='Vertical Vector'
#                 ))

#                 # Add scatter points for midpoints
#                 fig.add_trace(go.Scatter(
#                     x=[shoulder_midpoint_x], 
#                     y=[shoulder_midpoint_y],
#                     mode='markers',
#                     marker=dict(color='green'),
#                     name='Shoulder Midpoint'
#                 ))

#                 fig.add_trace(go.Scatter(
#                     x=[hip_midpoint_x], 
#                     y=[hip_midpoint_y],
#                     mode='markers',
#                     marker=dict(color='magenta'),
#                     name='Hip Midpoint'
#                 ))

#                 # Calculate angle
#                 angle_radians = np.arctan2(np.linalg.norm(np.cross(hip_shoulder_vector, vertical_vector)),
#                                             np.dot(hip_shoulder_vector, vertical_vector))
#                 angle_degrees = np.degrees(angle_radians)

#                 # Update layout for each frame
#                 fig.update_layout(
#                     title=f'Trunk Angle: {angle_degrees:.2f} degrees',
#                     xaxis=dict(range=[min(hip_midpoint_x, shoulder_midpoint_x) - 1, max(hip_midpoint_x, shoulder_midpoint_x) + 1], title='X'),
#                     yaxis=dict(range=[min(hip_midpoint_y, shoulder_midpoint_y) - 1, max(hip_midpoint_y, shoulder_midpoint_y) + 1], title='Y'),
#                     template='plotly_white'
#                 )

#                 # Save the figure as an interactive HTML file
#                 fig.write_html('trunk_angle.html')

#                 # Show the figure
#                 fig.show()

#             # for start, _ in plateau_widths:
#             #     plot_trunk_angle(df, start)


#             for start, _ in plateau_widths:
#                 mask = df['frame'].astype(float) == float(start)
#                 if any(mask):
#                     touch_down_trunk_angle = df.loc[mask, 'trunk_angle'].iloc[0]
#                     df.loc[mask, 'touchdown_trunk_angle'] = touch_down_trunk_angle
#                     # Debug: Check the touchdown trunk angle
#                     print(f"Touchdown trunk angle at frame {start}: {touch_down_trunk_angle}")
#                 else:
#                     print(f"No matching frame found for start: {start}")

#             for step in unique_steps:
#                 step_mask = df['steps'] == step
#                 first_value = df.loc[step_mask, 'touchdown_trunk_angle'].first_valid_index()
#                 if first_value is not None:
#                     df.loc[step_mask, 'touchdown_trunk_angle'] = df.loc[first_value, 'touchdown_trunk_angle']
#                 else:
#                     print(f"No valid touchdown trunk angle found for step: {step}")
        
#             # st.dataframe(df['touchdown_trunk_angle'])
        
#             def pixels_to_cm(pixel_distance, meters_per_pixel):
#                 # Convert to real-world distance
#                 euclidean_distance_m = pixel_distance * meters_per_pixel
#                 euclidean_distance_cm = (euclidean_distance_m * 100)*100 #accounts for the zoom preprocessing

#                 return euclidean_distance_cm


#             df['touchdown_knee_sep'] = np.nan  # Initialize the column with NaN values

#             for start, _ in plateau_widths:  # Assuming plateau_widths is defined and contains touchdown frames
#                 touchdown_frame = df.loc[start, 'frame']
#                 if touchdown_frame < len(df):
#                     Rknee_x = df.loc[df['frame'] == touchdown_frame, 'Rknee_x'].iloc[0]
#                     Lknee_x = df.loc[df['frame'] == touchdown_frame, 'Lknee_x'].iloc[0]
#                     pixel_distance = abs(Rknee_x - Lknee_x)
#                     real_distance = pixels_to_cm(pixel_distance, meters_per_pixel)
#                     df.loc[df['frame'] == touchdown_frame, 'touchdown_knee_sep'] = real_distance

#             df['touchdown_knee_sep'].ffill(inplace=True)


#             df['toe_velocity'] = np.nan
#             for i in range(1, len(df)):
#                 # Calculate displacement using only the x-coordinate
#                 frame_before = 2
#                 frame_after = 1
#                 # Add bounds checking to prevent KeyError
#                 if i >= frame_before and i + frame_after < len(df):
#                     displacement = (df.loc[i+frame_after, 'Ltoe_x'] - df.loc[i-frame_before, 'Ltoe_x'])
#                     displacement_cm = pixels_to_cm(displacement, meters_per_pixel)
#                     frame_diff = frame_after + frame_before  # Total frames between measurements
#                     time_diff = frame_diff / fps
#                     df.loc[i, 'toe_velocity'] = (displacement_cm / time_diff)/100 #converts to m/s 

#             df['toe_velocity'].ffill(inplace=True)

#             # Extract toe velocity at the start of each plateau
#             df['touchdown_toe_velocity'] = np.nan
#             for start, _ in plateau_widths:
#                 if start < len(df):
#                     df.loc[start, 'touchdown_toe_velocity'] = df.loc[start, 'toe_velocity']

#             # Forward fill the extracted velocities by unique step
#             unique_steps = df['steps'].unique()
#             for step in unique_steps:
#                 step_mask = df['steps'] == step
#                 first_value = df.loc[step_mask, 'touchdown_toe_velocity'].first_valid_index()
#                 if first_value is not None:
#                     df.loc[step_mask, 'touchdown_toe_velocity'] = df.loc[first_value, 'touchdown_toe_velocity']
        
#             df['touchdown_toe_velocity'].ffill(inplace=True)
#             df['touchdown_toe_velocity'] = df['touchdown_toe_velocity']
#             # if 'touchdown_toe_velocity' in df.columns:
#             #     last_valid_value = None
#             #     for i in range(len(df)):
#             #         if df.loc[i, 'touchdown_toe_velocity'] > 11:
#             #             if last_valid_value is not None:
#             #                 df.loc[i, 'touchdown_toe_velocity'] = last_valid_value
#             #         else:
#             #             last_valid_value = df.loc[i, 'touchdown_toe_velocity']


#             def plot_trunk_angle(df, start_index):
#                 import plotly.graph_objects as go

#                 # Create a Plotly figure
#                 fig = go.Figure()

#                 # Use the start index to get the row
#                 row = df.loc[start_index]
            
#                 shoulder_midpoint_x = (row['Lshldr_x'] + row['Rshldr_x']) / 2
#                 shoulder_midpoint_y = (row['Lshldr_y'] + row['Rshldr_y']) / 2
            
#                 hip_midpoint_x = (row['Lhip_x'] + row['Rhip_x']) / 2
#                 hip_midpoint_y = (row['Lhip_y'] + row['Rhip_y']) / 2
            
#                 hip_coord = np.array([hip_midpoint_x, hip_midpoint_y])
#                 shoulder_coord = np.array([shoulder_midpoint_x, shoulder_midpoint_y])  

#                 # Create a vector between the midpoints
#                 hip_shoulder_vector = hip_coord - shoulder_coord
#                 vertical_vector = [0, -1]

#                 # Add traces for the hip-shoulder vector
#                 fig.add_trace(go.Scatter(
#                     x=[shoulder_midpoint_x, hip_midpoint_x], 
#                     y=[shoulder_midpoint_y, hip_midpoint_y],
#                     mode='lines',
#                     line=dict(dash='dash', color='red'),
#                     name='Hip-Shoulder Vector'
#                 ))

#                 # Add traces for the vertical vector
#                 fig.add_trace(go.Scatter(
#                     x=[hip_midpoint_x, hip_midpoint_x], 
#                     y=[hip_midpoint_y, hip_midpoint_y + 1],  # Assuming a unit vector for vertical
#                     mode='lines',
#                     line=dict(dash='dot', color='blue'),
#                     name='Vertical Vector'
#                 ))

#                 # Add scatter points for midpoints
#                 fig.add_trace(go.Scatter(
#                     x=[shoulder_midpoint_x], 
#                     y=[shoulder_midpoint_y],
#                     mode='markers',
#                     marker=dict(color='green'),
#                     name='Shoulder Midpoint'
#                 ))

#                 fig.add_trace(go.Scatter(
#                     x=[hip_midpoint_x], 
#                     y=[hip_midpoint_y],
#                     mode='markers',
#                     marker=dict(color='magenta'),
#                     name='Hip Midpoint'
#                 ))

#                 # Calculate angle
#                 angle_radians = np.arctan2(np.linalg.norm(np.cross(hip_shoulder_vector, vertical_vector)),
#                                             np.dot(hip_shoulder_vector, vertical_vector))
#                 angle_degrees = np.degrees(angle_radians)

#                 # Update layout for each frame
#                 fig.update_layout(
#                     title=f'Trunk Angle: {angle_degrees:.2f} degrees',
#                     xaxis=dict(range=[min(hip_midpoint_x, shoulder_midpoint_x) - 1, max(hip_midpoint_x, shoulder_midpoint_x) + 1], title='X'),
#                     yaxis=dict(range=[min(hip_midpoint_y, shoulder_midpoint_y) - 1, max(hip_midpoint_y, shoulder_midpoint_y) + 1], title='Y'),
#                     template='plotly_white'
#                 )

#                 # Save the figure as an interactive HTML file
#                 fig.write_html('trunk_angle.html')

#                 # Show the figure
#                 fig.show()

#             # for start, _ in plateau_widths:
#             #     plot_trunk_angle(df, start)


#             for start, _ in plateau_widths:
#                 mask = df['frame'].astype(float) == float(start)
#                 if any(mask):
#                     touch_down_trunk_angle = df.loc[mask, 'trunk_angle'].iloc[0]
#                     df.loc[mask, 'touchdown_trunk_angle'] = touch_down_trunk_angle
#                     # Debug: Check the touchdown trunk angle
#                     print(f"Touchdown trunk angle at frame {start}: {touch_down_trunk_angle}")
#                 else:
#                     print(f"No matching frame found for start: {start}")

#             for step in unique_steps:
#                 step_mask = df['steps'] == step
#                 first_value = df.loc[step_mask, 'touchdown_trunk_angle'].first_valid_index()
#                 if first_value is not None:
#                     df.loc[step_mask, 'touchdown_trunk_angle'] = df.loc[first_value, 'touchdown_trunk_angle']
#                 else:
#                     print(f"No valid touchdown trunk angle found for step: {step}")
        
        

#             # Prep variables for Alpha trainer
        
        
#             # Define optimal values and bounds based on gender
#             if gender == 'male':
#                 optimal_values = {
#                     'stride_rate': {'optimal': 4.76, 'lower': 4.61, 'upper': 4.76},
#                     'gct': {'optimal': 0.087, 'lower': 0.087, 'upper': 0.094},
#                     'max_hip_angle': {'optimal': 79, 'lower': 74, 'upper':  79},
#                     'max_knee_angle': {'optimal': 47, 'lower': 44, 'upper': 47},
#                     'touchdown_trunk_angle': {'optimal': 170, 'lower': 165, 'upper': 170},
#                     'touchdown_knee_sep': {'optimal': 1, 'lower': 1, 'upper': 3},
#                     'touchdown_toe_velocity': {'optimal': 7.6, 'lower': 6.8, 'upper': 7.6}
#                 }
#             else:  # female
#                 optimal_values = {
#                     'stride_rate': {'optimal': 4.85, 'lower': 4.70, 'upper': 4.85},
#                     'gct': {'optimal': 0.083, 'lower': 0.083, 'upper': 0.090},
#                     'max_hip_angle': {'optimal': 83, 'lower': 80, 'upper': 83},
#                     'max_knee_angle': {'optimal': 47, 'lower': 44, 'upper': 47},
#                     'touchdown_trunk_angle': {'optimal': 176, 'lower': 173, 'upper': 176},
#                     'touchdown_knee_sep': {'optimal': 1, 'lower': 1, 'upper': 3},
#                     'touchdown_toe_velocity': {'optimal': 7.6, 'lower': 6.8, 'upper': 7.6}
#                 }
        
#             # Calculate means for each step
#             try:
#                 grouped_means = df.groupby('steps').mean()
#             except Exception as e:
#                 display_error(f"Error calculating means: {str(e)}")
#                 return
        
#             # Get values for each metric
#             try:
#                 st_rt_values = grouped_means['st_rt'].tolist()
#                 gct_values = grouped_means['gct'].tolist()
#                 max_hip_angle_values = grouped_means['max_hip_angle'].tolist()
#                 max_knee_angle_values = grouped_means['max_knee_angle'].tolist()
#                 touchdown_trunk_angle_values = grouped_means['touchdown_trunk_angle'].tolist()
#                 touchdown_knee_sep_values = grouped_means['touchdown_knee_sep'].tolist()
#                 touchdown_toe_velocity_values = grouped_means['touchdown_toe_velocity'].tolist()
#             except Exception as e:
#                 display_error(f"Error getting metric values: {str(e)}")
#                 return
        
#             # Calculate averages with error handling
#             try:
#                 avg_st_rt = sum(st_rt_values) / len(st_rt_values)
#                 avg_gct = sum(gct_values) / len(gct_values)
#                 avg_max_hip_angle = sum(max_hip_angle_values) / len(max_hip_angle_values)
#                 avg_max_knee_angle = sum(max_knee_angle_values) / len(max_knee_angle_values)
#                 avg_touchdown_trunk_angle = sum(touchdown_trunk_angle_values) / len(touchdown_trunk_angle_values)
#                 avg_touchdown_knee_sep = sum(touchdown_knee_sep_values) / len(touchdown_knee_sep_values)
#                 avg_touchdown_toe_velocity = sum(touchdown_toe_velocity_values) / len(touchdown_toe_velocity_values)
#             except ZeroDivisionError:
#                 pass
#                 # st.error("No valid steps detected in the analysis")
#                 return
        
#             # Add optimal values and bounds to df
#             df['opt_st_rt'] = optimal_values['stride_rate']['optimal']
#             df['opt_gct'] = optimal_values['gct']['optimal']
#             df['opt_max_hip_angle'] = optimal_values['max_hip_angle']['optimal']
#             df['opt_max_knee_angle'] = optimal_values['max_knee_angle']['optimal']
#             df['opt_touchdown_trunk_angle'] = optimal_values['touchdown_trunk_angle']['optimal']
#             df['opt_touchdown_knee_sep'] = optimal_values['touchdown_knee_sep']['optimal']
#             df['opt_touchdown_toe_velocity'] = optimal_values['touchdown_toe_velocity']['optimal']

#             # Add upper bounds
#             df['upper_st_rt'] = optimal_values['stride_rate']['upper']
#             df['upper_gct'] = optimal_values['gct']['upper']
#             df['upper_max_hip_angle'] = optimal_values['max_hip_angle']['upper']
#             df['upper_max_knee_angle'] = optimal_values['max_knee_angle']['upper']
#             df['upper_touchdown_trunk_angle'] = optimal_values['touchdown_trunk_angle']['upper']
#             df['upper_touchdown_knee_sep'] = optimal_values['touchdown_knee_sep']['upper']
#             df['upper_touchdown_toe_velocity'] = optimal_values['touchdown_toe_velocity']['upper']

#             # Add lower bounds
#             df['lower_st_rt'] = optimal_values['stride_rate']['lower']
#             df['lower_gct'] = optimal_values['gct']['lower']
#             df['lower_max_hip_angle'] = optimal_values['max_hip_angle']['lower']
#             df['lower_max_knee_angle'] = optimal_values['max_knee_angle']['lower']
#             df['lower_touchdown_trunk_angle'] = optimal_values['touchdown_trunk_angle']['lower']
#             df['lower_touchdown_knee_sep'] = optimal_values['touchdown_knee_sep']['lower']
#             df['lower_touchdown_toe_velocity'] = optimal_values['touchdown_toe_velocity']['lower']

#             # Add averages
#             df['avg_st_rt'] = avg_st_rt
#             df['avg_gct'] = avg_gct
#             df['avg_max_hip_angle'] = avg_max_hip_angle
#             df['avg_max_knee_angle'] = avg_max_knee_angle
#             df['avg_touchdown_trunk_angle'] = avg_touchdown_trunk_angle
#             df['avg_touchdown_knee_sep'] = avg_touchdown_knee_sep
#             df['avg_touchdown_toe_velocity'] = avg_touchdown_toe_velocity

#             # Connect to the database and save analysis data
#             try:
#                 db_path = join_paths(destination_folder, f'{fn}_video.db')
#                 print(f"Attempting to save to database at: {db_path}")
            
#                 conn = connect_to_db(db_path)
            
#                 # Print DataFrame info before saving
#                 print(f"DataFrame shape before saving: {df.shape}")
            
#                 save_angle_data(df, conn)
            
#                 # Commit the changes
#                 conn.commit()
#                 print("Data successfully saved to database")
            
#             except Exception as e:
#                 print(f"Detailed error while saving analysis data: {str(e)}")  # More detailed error
#                 display_error(f"Error saving analysis data: {str(e)}")
#                 return
#             finally:
#                 conn.close()


def apsprint(fn, destination_folder, fps, gender, age, height_feet, height_inches, weight):
    # Initialize database to ensure all tables exist (including speed_data)
    # Strip _original from fn to ensure consistent database naming (same as max_velocity)
    # This prevents creating multiple databases for the same video
    fn_for_db = fn.replace('_original', '') if '_original' in fn else fn
    print(f"DEBUG apsprint: Using fn_for_db={fn_for_db} (original fn={fn})")
    
    from src.db.sqlite.analysis_data.initialize import initialize_database
    conn = initialize_database(fn_for_db, destination_folder)
    
    
    # Read the data from the database into a pandas DataFrame
    df1 = pd.read_sql_query("SELECT * FROM position_data", conn)
    df2 = pd.read_sql_query("SELECT * FROM angle_data", conn)
    
    # Check and warn if position_data or angle_data is empty
    if df1.empty:
        print("WARNING: position_data table is empty. Cannot process biomechanic analysis without position data.")
        print("         Please ensure apform has been run first to generate position data from MediaPipe processing.")
        conn.close()
        return
    
    if df2.empty:
        print("WARNING: angle_data table is empty. Cannot process biomechanic analysis without angle data.")
        print("         Please ensure apang has been run first to generate angle data.")
        conn.close()
        return
    
    # Try to read speed_data, handle gracefully if table doesn't exist
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='speed_data'")
    if cursor.fetchone():
        df3 = pd.read_sql_query('SELECT * FROM speed_data', conn)
        # Check if meters_per_pixel column exists
        if 'meters_per_pixel' in df3.columns and not df3.empty:
            meters_per_pixel = df3['meters_per_pixel'].mean()
        else:
            print("Warning: meters_per_pixel column not found in speed_data table. Using default value.")
            meters_per_pixel = 0.01  # Default value
    else:
        print("Warning: speed_data table not found. Using default meters_per_pixel value.")
        # Use a default value or calculate from other data if available
        meters_per_pixel = 0.01  # Default value, adjust as needed
        df3 = pd.DataFrame()  # Empty DataFrame

    df = pd.concat([df1, df2], axis=1)
    df = df.loc[:,~df.columns.duplicated()]

    # df = df.filter(regex='^(?!Unnamed:).*')
    # df = df[['frame', 'time'] + [col for col in df.columns if col not in ['frame', 'time']]]

    if fps == 240:
        fps = fps 
        derivative_threshold = 0.01
        min_plateau_width = 24       
        max_plateau_width = 50
        frames_before = 0
        frames_after = 0
        fps_multiplier = 2
        y_threshold = 0.78
    elif fps == 120: 
        fps = fps
        derivative_threshold = 0.01
        min_plateau_width = 12
        max_plateau_width = 25
        frames_before = 0
        frames_after = 0
        fps_multiplier = 2
        y_threshold = 0.78
    else:
        # Default case for other FPS values (e.g., 60)
        fps = fps
        derivative_threshold = 0.01
        min_plateau_width = 6  # Adjusted for 60 fps
        max_plateau_width = 12  # Adjusted for 60 fps
        frames_before = 0
        frames_after = 0
        fps_multiplier = 2
        y_threshold = 0.78

        
        

    if len(df['Ltoe_y']) >= 2:
        df['Ltoe_y_derivative'] = np.gradient(df['Ltoe_y'])
    else:
        df['Ltoe_y_derivative'] = np.zeros(len(df['Ltoe_y']))
    
    if len(df['Rtoe_y']) >= 2:
        df['Rtoe_y_derivative'] = np.gradient(df['Rtoe_y'])
    else:
        df['Rtoe_y_derivative'] = np.zeros(len(df['Rtoe_y']))

    # def extract_plateau_widths(df, derivative_column, y_column, frames_before, frames_after, min_plateau_width, max_plateau_width, derivative_threshold, y_threshold):
    #     plateau_indices = np.where(
    #         (df[derivative_column] >= -derivative_threshold) &
    #         (df[derivative_column] <= derivative_threshold)
    #     )[0]
    #     plateau_start = None
    #     plateau_end = None
    #     plateau_widths = []

    #     for index in plateau_indices:
    #         if plateau_start is None:
    #             plateau_start = index - frames_before
    #             plateau_end = index + frames_after
    #         elif index != plateau_end + 1:
    #             plateau_width = plateau_end - plateau_start
    #             if plateau_width >= min_plateau_width and \
    #                 plateau_width <= max_plateau_width and \
    #                 df[y_column].iloc[plateau_start] and df[y_column].iloc[plateau_end] > y_threshold:
    #                 if plateau_end + 1 < len(df):
    #                     if df[y_column].iloc[plateau_end] > df[y_column].iloc[plateau_end + 1]:
    #                         plateau_widths.append((plateau_start, plateau_end))
    #                 else:
    #                     plateau_widths.append((plateau_start, plateau_end))
    #                     print(f"Warning: Plateau ends at the last frame (index {plateau_end})")
    #             plateau_start = index - frames_before
    #             plateau_end = index + frames_after
    #         else:
    #             plateau_end = index + frames_after

    #     if plateau_start is not None:
    #         plateau_width = plateau_end - plateau_start
    #         if plateau_width >= min_plateau_width and \
    #             plateau_width <= max_plateau_width and \
    #             df[y_column].iloc[plateau_start] and df[y_column].iloc[plateau_end] > y_threshold:
    #             if plateau_end + 1 < len(df):
    #                 if df[y_column].iloc[plateau_end] > df[y_column].iloc[plateau_end + 1]:
    #                     plateau_widths.append((plateau_start, plateau_end))
    #             else:
    #                 plateau_widths.append((plateau_start, plateau_end))
    #                 print(f"Warning: Plateau ends at the last frame (index {plateau_end})")

    #     return plateau_widths


   

    # manual_entry_json = os.path.join(destination_folder, 'manual_entry.json')
    # # st.write(f"Manual Entry JSON: {manual_entry_json}")
    # if os.path.exists(manual_entry_json):
    #     with open(manual_entry_json, 'r') as json_file:
    #         manual_entry = json.load(json_file)
    #         # Use both starts and ends from the JSON file
    #         plateau_widths = list(zip(manual_entry['starts'], manual_entry['ends']))
    #         st.write(f"Manual Entry Plateau Widths: {plateau_widths}")   
    # else:
    #     plateau_widths = extract_plateau_widths(
    #             df = df, 
    #             derivative_column = 'Ltoe_y_derivative', 
    #             y_column = 'Ltoe_y', 
    #             frames_before = frames_before, 
    #             frames_after = frames_after, 
    #             min_plateau_width = min_plateau_width, 
    #             max_plateau_width = max_plateau_width, 
    #             derivative_threshold = derivative_threshold, 
    #             y_threshold = y_threshold
    #         )


    # Remove extension from fn to match manual entry filename format
    # Manual entry system uses processed video filename (e.g., "Copy of IMG_6644_zoom_processed.mp4" -> "Copy of IMG_6644_zoom_manual_entry.json")
    fn_base, _ = splitext(fn)
    
    # Try multiple variations of the manual entry filename
    # 1. Try with _zoom suffix (if processed video was zoomed)
    # 2. Try with _processed suffix (if processed video exists)
    # 3. Try base filename only
    manual_entry_variations = [
        join_paths(destination_folder, f'{fn_base}_zoom_manual_entry.json'),
        join_paths(destination_folder, f'{fn_base}_processed_manual_entry.json'),
        join_paths(destination_folder, f'{fn_base}_manual_entry.json'),
    ]
    
    # Also try finding any manual_entry.json file in the folder
    import glob
    all_manual_entries = glob.glob(join_paths(destination_folder, '*_manual_entry.json'))
    
    manual_entry_json = None
    has_manual_entry = False
    
    # First try the variations
    for variation in manual_entry_variations:
        if path_exists(variation):
            manual_entry_json = variation
            has_manual_entry = True
            break
    
    # If not found, try any manual_entry.json file
    if not has_manual_entry and all_manual_entries:
        manual_entry_json = all_manual_entries[0]
        has_manual_entry = True
        print(f"DEBUG: Found manual entry file with different naming: {manual_entry_json}")
    
    print(f"DEBUG: Looking for manual entry file: {manual_entry_variations[0]}, exists: {has_manual_entry}")
    if has_manual_entry:
        print(f"DEBUG: Using manual entry file: {manual_entry_json}")
    # st.write(f"Manual Entry JSON: {manual_entry_json}")
    if has_manual_entry:
        with open(manual_entry_json, 'r') as json_file:
            manual_entry = json.load(json_file)
            # Handle both old format (single leg) and new format (dual leg)
            if 'Left' in manual_entry and 'Right' in manual_entry:
                plateau_widths_left = list(zip(manual_entry['Left']['starts'], manual_entry['Left']['ends']))
                plateau_widths_right = list(zip(manual_entry['Right']['starts'], manual_entry['Right']['ends']))
            else:
                # Legacy format - use for both legs
                plateau_widths = list(zip(manual_entry['starts'], manual_entry['ends']))
                plateau_widths_left = plateau_widths
                plateau_widths_right = plateau_widths
        print(f"Using manual entry data: Left={len(plateau_widths_left)} plateaus, Right={len(plateau_widths_right)} plateaus")
    else:
        plateau_widths_left = extract_plateau_widths_second(
            df = df, 
            derivative_column = 'Ltoe_y_derivative', 
            y_column = 'Ltoe_y', 
            frames_before = frames_before, 
            frames_after = frames_after, 
            min_plateau_width = min_plateau_width, 
            max_plateau_width = max_plateau_width, 
            derivative_threshold = derivative_threshold, 
            y_threshold = y_threshold
        )
        plateau_widths_right = extract_plateau_widths_second(
            df = df, 
            derivative_column = 'Rtoe_y_derivative', 
            y_column = 'Rtoe_y', 
            frames_before = frames_before, 
            frames_after = frames_after, 
            min_plateau_width = min_plateau_width, 
            max_plateau_width = max_plateau_width, 
            derivative_threshold = derivative_threshold, 
            y_threshold = y_threshold
        )
    
    # Output the results
    num_plateaus_left = len(plateau_widths_left)
    num_plateaus_right = len(plateau_widths_right)
    print(f"Number of left leg plateaus extracted: {num_plateaus_left}")
    print(f"Number of right leg plateaus extracted: {num_plateaus_right}")
    if plateau_widths_left:
        print("Left leg plateau widths (start, end):")
        for start, end in plateau_widths_left:
            print(f"Start: {start}, End: {end}")
    if plateau_widths_right:
        print("Right leg plateau widths (start, end):")
        for start, end in plateau_widths_right:
            print(f"Start: {start}, End: {end}")

    ### Filter and not continue if less than 2 plateaus are found for either leg AND no manual entry exists ###
    ### If manual entry exists, use it regardless of the number of plateaus ###
    if num_plateaus_left < 2 and num_plateaus_right < 2 and not has_manual_entry:
        print("Warning: Less than 2 plateaus found for both legs and no manual entry data available. Skipping analysis.")
        pass
    else:
        # Calculate steps for both legs
        df['steps'] = np.nan
        df['steps_left'] = np.nan
        df['steps_right'] = np.nan

        # Left leg steps - use manual entry if available, otherwise require >= 2 plateaus
        if has_manual_entry and num_plateaus_left >= 1:
            # Use manual entry even with 1 plateau
            if num_plateaus_left >= 2:
                for i in range(len(plateau_widths_left) - 1):
                    plateau_start = plateau_widths_left[i][0]
                    next_plateau_start = plateau_widths_left[i + 1][0]
                    df.loc[plateau_start:next_plateau_start - 1, 'steps_left'] = i + 1
                df['steps_left'].ffill(inplace=True)
            else:
                # Single plateau - assign step 1 to that plateau region
                if num_plateaus_left == 1:
                    start, end = plateau_widths_left[0]
                    df.loc[start:end, 'steps_left'] = 1
                    df['steps_left'].ffill(inplace=True)
        elif num_plateaus_left >= 2:
            for i in range(len(plateau_widths_left) - 1):
                plateau_start = plateau_widths_left[i][0]
                next_plateau_start = plateau_widths_left[i + 1][0]
                df.loc[plateau_start:next_plateau_start - 1, 'steps_left'] = i + 1
            df['steps_left'].ffill(inplace=True)

        # Right leg steps - use manual entry if available, otherwise require >= 2 plateaus
        if has_manual_entry and num_plateaus_right >= 1:
            # Use manual entry even with 1 plateau
            if num_plateaus_right >= 2:
                for i in range(len(plateau_widths_right) - 1):
                    plateau_start = plateau_widths_right[i][0]
                    next_plateau_start = plateau_widths_right[i + 1][0]
                    df.loc[plateau_start:next_plateau_start - 1, 'steps_right'] = i + 1
                df['steps_right'].ffill(inplace=True)
            else:
                # Single plateau - assign step 1 to that plateau region
                if num_plateaus_right == 1:
                    start, end = plateau_widths_right[0]
                    df.loc[start:end, 'steps_right'] = 1
                    df['steps_right'].ffill(inplace=True)
        elif num_plateaus_right >= 2:
            for i in range(len(plateau_widths_right) - 1):
                plateau_start = plateau_widths_right[i][0]
                next_plateau_start = plateau_widths_right[i + 1][0]
                df.loc[plateau_start:next_plateau_start - 1, 'steps_right'] = i + 1
            df['steps_right'].ffill(inplace=True)

        # Legacy steps column (use left leg if available, otherwise right)
        if num_plateaus_left >= 1 and (has_manual_entry or num_plateaus_left >= 2):
            df['steps'] = df['steps_left']
        elif num_plateaus_right >= 1 and (has_manual_entry or num_plateaus_right >= 2):
            df['steps'] = df['steps_right']

        # Calculate frames in each step for both legs
        df['step_frames'] = np.nan
        df['step_frames_left'] = np.nan
        df['step_frames_right'] = np.nan

        # Left leg step frames - handle manual entry with any number of plateaus
        if has_manual_entry and num_plateaus_left >= 1:
            if num_plateaus_left >= 2:
                for i in range(len(plateau_widths_left) - 1):
                    plateau_start = plateau_widths_left[i][0]
                    next_plateau_start = plateau_widths_left[i + 1][0]
                    step_frames = next_plateau_start - plateau_start
                    df.loc[plateau_start:next_plateau_start - 1, 'step_frames_left'] = step_frames
                df['step_frames_left'].ffill(inplace=True)
            else:
                # Single plateau - use a default or calculate from video length
                if num_plateaus_left == 1:
                    start, end = plateau_widths_left[0]
                    # Use the plateau width as step frames
                    step_frames = end - start
                    df.loc[start:end, 'step_frames_left'] = step_frames
                    df['step_frames_left'].ffill(inplace=True)
        elif num_plateaus_left >= 2:
            for i in range(len(plateau_widths_left) - 1):
                plateau_start = plateau_widths_left[i][0]
                next_plateau_start = plateau_widths_left[i + 1][0]
                step_frames = next_plateau_start - plateau_start
                df.loc[plateau_start:next_plateau_start - 1, 'step_frames_left'] = step_frames
            df['step_frames_left'].ffill(inplace=True)

        # Right leg step frames - handle manual entry with any number of plateaus
        if has_manual_entry and num_plateaus_right >= 1:
            if num_plateaus_right >= 2:
                for i in range(len(plateau_widths_right) - 1):
                    plateau_start = plateau_widths_right[i][0]
                    next_plateau_start = plateau_widths_right[i + 1][0]
                    step_frames = next_plateau_start - plateau_start
                    df.loc[plateau_start:next_plateau_start - 1, 'step_frames_right'] = step_frames
                df['step_frames_right'].ffill(inplace=True)
            else:
                # Single plateau - use a default or calculate from video length
                if num_plateaus_right == 1:
                    start, end = plateau_widths_right[0]
                    # Use the plateau width as step frames
                    step_frames = end - start
                    df.loc[start:end, 'step_frames_right'] = step_frames
                    df['step_frames_right'].ffill(inplace=True)
        elif num_plateaus_right >= 2:
            for i in range(len(plateau_widths_right) - 1):
                plateau_start = plateau_widths_right[i][0]
                next_plateau_start = plateau_widths_right[i + 1][0]
                step_frames = next_plateau_start - plateau_start
                df.loc[plateau_start:next_plateau_start - 1, 'step_frames_right'] = step_frames
            df['step_frames_right'].ffill(inplace=True)

        # Legacy step_frames column
        if num_plateaus_left >= 1 and (has_manual_entry or num_plateaus_left >= 2):
            df['step_frames'] = df['step_frames_left']
        elif num_plateaus_right >= 1 and (has_manual_entry or num_plateaus_right >= 2):
            df['step_frames'] = df['step_frames_right']
        
        # Calculate stride rates for both legs
        df['seconds'] = df['step_frames'] / fps
        df['seconds_left'] = df['step_frames_left'] / fps
        df['seconds_right'] = df['step_frames_right'] / fps
        
        df['st_rt'] = (1 / df['seconds'])*fps_multiplier
        df['st_rt_left'] = (1 / df['seconds_left'])*fps_multiplier
        df['st_rt_right'] = (1 / df['seconds_right'])*fps_multiplier

        for i in range(1, len(df)):
            if abs(df.loc[i, 'st_rt'] - df.loc[i - 1, 'st_rt']) > 0.2:
                df.loc[i, 'st_rt'] = df.loc[i - 1, 'st_rt']  

        # Generate plot only if stride rate is reasonable (>= 2.5)
        # But continue with all analysis calculations regardless
        if df['st_rt'].mean() >= 2.5:
            try:
                import plotly.graph_objects as go

                # Create a figure with secondary y-axis
                fig = go.Figure()

                # First subplot for Ltoe_y position
                fig.add_trace(go.Scatter(
                    x=df['frame'],
                    y=df["Ltoe_y"],
                    mode='lines+markers',
                    name='Ltoe_y',
                    line=dict(dash='dash')
                ))

                fig.add_trace(go.Scatter(
                    x=df['frame'],
                    y=df["Ltoe_y_derivative"],
                    mode='lines',
                    name='Ltoe_y_derivative',
                    line=dict(color='green')
                ))

                # Add stride rate trace
                fig.add_trace(go.Scatter(
                    x=df['frame'],
                    y=df['st_rt'],
                    mode='markers',
                    name='Stride Rate',
                    yaxis='y2',  # Use secondary y-axis
                    marker=dict(
                        color='red',
                        size=2,
                        symbol='circle'
                    ),
                ))

                # Add plateau regions for Ltoe_y
                for start, end in plateau_widths_left:
                    fig.add_shape(type='rect',
                                x0=df['frame'].iloc[start],
                                x1=df['frame'].iloc[end],
                                y0=-0.1,
                                y1=1,
                                fillcolor='red',
                                opacity=0.5,
                                line_width=0)

                # Update layout for the figure
                fig.update_layout(
                    title=f'{fn} - Ground Contact Analysis',
                    xaxis_title='Frames',
                    yaxis_title='Ltoe_y Position',
                    yaxis=dict(range=[-0.01, 0.01]),
                    yaxis2=dict(
                        title='Stride Rate',
                        overlaying='y',
                        side='right'
                    ),
                    template='plotly_white'
                )

                # Save the figure as an interactive HTML file
                fig.write_html(join_paths(destination_folder, f'{fn}_ground_contact_regions.html'))
            except Exception as e:
                print(f"Warning: Could not generate ground contact plot: {e}")

        # Calculate gct for each step for both legs
        # Left leg GCT
        if plateau_widths_left:  # Check if plateau_widths_left is not empty
            gct_frames_left = [end - start for start, end in plateau_widths_left]
            
            # Initialize gct_frames_left column
            df['gct_frames_left'] = np.nan
            
            for i, gct in enumerate(gct_frames_left, start=1):
                if i >= 1:
                    try:
                        start, end = plateau_widths_left[i-1]
                        mask = df['frame'].astype(float) >= start
                        mask &= df['frame'].astype(float) <= end
                        df.loc[mask, 'gct_frames_left'] = gct
                    except Exception as e:
                        print(f"Error processing left plateau {i}: {e}")
                        continue
            
            # Forward fill and calculate final values
            df['gct_frames_left'] = df['gct_frames_left'].ffill()
            df['gct_left'] = (df['gct_frames_left']/fps).round(4)
        else:
            print("No left leg plateau regions found")
            df['gct_frames_left'] = 0
            df['gct_left'] = 0

        # Right leg GCT
        if plateau_widths_right:  # Check if plateau_widths_right is not empty
            gct_frames_right = [end - start for start, end in plateau_widths_right]
            
            # Initialize gct_frames_right column
            df['gct_frames_right'] = np.nan
            
            for i, gct in enumerate(gct_frames_right, start=1):
                if i >= 1:
                    try:
                        start, end = plateau_widths_right[i-1]
                        mask = df['frame'].astype(float) >= start
                        mask &= df['frame'].astype(float) <= end
                        df.loc[mask, 'gct_frames_right'] = gct
                    except Exception as e:
                        print(f"Error processing right plateau {i}: {e}")
                        continue
            
            # Forward fill and calculate final values
            df['gct_frames_right'] = df['gct_frames_right'].ffill()
            df['gct_right'] = (df['gct_frames_right']/fps).round(4)
        else:
            print("No right leg plateau regions found")
            df['gct_frames_right'] = 0
            df['gct_right'] = 0

        # Legacy GCT (use left leg if available, otherwise right)
        if plateau_widths_left:
            df['gct_frames'] = df['gct_frames_left']
            df['gct'] = df['gct_left']
        elif plateau_widths_right:
            df['gct_frames'] = df['gct_frames_right']
            df['gct'] = df['gct_right']
        else:
            df['gct_frames'] = 0
            df['gct'] = 0

        # Forward fill ground contact time for both legs
        unique_steps_left = df['steps_left'].unique() if 'steps_left' in df.columns else []
        unique_steps_right = df['steps_right'].unique() if 'steps_right' in df.columns else []
        unique_steps = df['steps'].unique()

        # Forward fill for left leg
        for step in unique_steps_left:
            first_non_nan_index = df.loc[df['steps_left'] == step, 'gct_frames_left'].first_valid_index()
            if first_non_nan_index is not None:
                df.loc[df['steps_left'] == step, 'gct_frames_left'] = df.loc[first_non_nan_index, 'gct_frames_left']

        for step in unique_steps_left:
            first_non_nan_index = df.loc[df['steps_left'] == step, 'gct_left'].first_valid_index()
            if first_non_nan_index is not None:
                df.loc[df['steps_left'] == step, 'gct_left'] = df.loc[first_non_nan_index, 'gct_left']

        # Forward fill for right leg
        for step in unique_steps_right:
            first_non_nan_index = df.loc[df['steps_right'] == step, 'gct_frames_right'].first_valid_index()
            if first_non_nan_index is not None:
                df.loc[df['steps_right'] == step, 'gct_frames_right'] = df.loc[first_non_nan_index, 'gct_frames_right']

        for step in unique_steps_right:
            first_non_nan_index = df.loc[df['steps_right'] == step, 'gct_right'].first_valid_index()
            if first_non_nan_index is not None:
                df.loc[df['steps_right'] == step, 'gct_right'] = df.loc[first_non_nan_index, 'gct_right']

        # Legacy forward fill
        for step in unique_steps:
            first_non_nan_index = df.loc[df['steps'] == step, 'gct_frames'].first_valid_index()
            if first_non_nan_index is not None:
                df.loc[df['steps'] == step, 'gct_frames'] = df.loc[first_non_nan_index, 'gct_frames']

        for step in unique_steps:
            first_non_nan_index = df.loc[df['steps'] == step, 'gct'].first_valid_index()
            if first_non_nan_index is not None:
                df.loc[df['steps'] == step, 'gct'] = df.loc[first_non_nan_index, 'gct']


        # def calculate_hip_angle(knee_coord, hip_coord, side):
        #     # Convert coordinates to numpy arrays
        #     knee_coord = np.array(knee_coord)
        #     hip_coord = np.array(hip_coord)
            
        #     # Define the vertical vector (assuming positive y-axis is vertical)
        #     vertical_vector = np.array([0, 1])
            
        #     # Calculate the vector from knee to hip
        #     knee_to_hip_vector = hip_coord - knee_coord
            
        #     # Normalize the vectors
        #     knee_to_hip_vector_normalized = knee_to_hip_vector / np.linalg.norm(knee_to_hip_vector)
        #     vertical_vector_normalized = vertical_vector / np.linalg.norm(vertical_vector)
            
        #     # Calculate the angle using the dot product
        #     dot_product = np.dot(knee_to_hip_vector_normalized, vertical_vector_normalized)
        #     angle_radians = np.arccos(dot_product)
        #     angle_degrees = np.degrees(angle_radians)
            
        #     # Calculate the hip angle as 360 minus the angle
        #     # hip_angle = 360 - angle_degrees
        #     # hip_angle = angle_degrees + 180
        #     hip_angle = angle_degrees
            
        #     return hip_angle
        
        left_hip_angles = []
        right_hip_angles = []

        # Calculate hip angles
        for index, row in df.iterrows():
            left_hip_coord = np.array([row['Lhip_x'], row['Lhip_y']])
            right_hip_coord = np.array([row['Rhip_x'], row['Rhip_y']])
            left_knee_coord = np.array([row['Lknee_x'], row['Lknee_y']])
            right_knee_coord = np.array([row['Rknee_x'], row['Rknee_y']])
            
            left_hip_angle = calculate_hip_angle(left_hip_coord, left_knee_coord, 'left')
            right_hip_angle = calculate_hip_angle(right_hip_coord, right_knee_coord, 'right')
            
            left_hip_angles.append(left_hip_angle)
            right_hip_angles.append(right_hip_angle)
        
        # Assign q angles to DataFrame
        df['Left_Hip_Angle'] = left_hip_angles
        df['Right_Hip_Angle'] = right_hip_angles

        # Calculate max hip angles for both legs
        df['max_hip_angle'] = np.nan
        df['max_hip_angle_left'] = np.nan
        df['max_hip_angle_right'] = np.nan

        # Left leg max hip angles (filter out invalid values)
        for step in unique_steps_left:
            step_data = df.loc[df['steps_left'] == step, 'Left_Hip_Angle']
            # Filter out NaN and invalid values (outside realistic range)
            valid_data = step_data.dropna()
            valid_data = valid_data[(valid_data >= 200) & (valid_data <= 300)]
            if len(valid_data) > 0:
                max_hip_angle_left = valid_data.max()
                df.loc[df['steps_left'] == step, 'max_hip_angle_left'] = max_hip_angle_left
        df['max_hip_angle_left'].ffill(inplace=True)

        # Right leg max hip angles (filter out invalid values)
        for step in unique_steps_right:
            step_data = df.loc[df['steps_right'] == step, 'Right_Hip_Angle']
            # Filter out NaN and invalid values (outside realistic range)
            valid_data = step_data.dropna()
            valid_data = valid_data[(valid_data >= 200) & (valid_data <= 300)]
            if len(valid_data) > 0:
                max_hip_angle_right = valid_data.max()
                df.loc[df['steps_right'] == step, 'max_hip_angle_right'] = max_hip_angle_right
        df['max_hip_angle_right'].ffill(inplace=True)

        # Legacy max_hip_angle (use left leg if available, otherwise right)
        if len(unique_steps_left) > 0:
            for step in unique_steps:
                step_data = df.loc[df['steps'] == step, 'Left_Hip_Angle']
                # Filter out NaN and invalid values (outside realistic range)
                valid_data = step_data.dropna()
                valid_data = valid_data[(valid_data >= 200) & (valid_data <= 300)]
                if len(valid_data) > 0:
                    max_hip_angle = valid_data.max()
                    df.loc[df['steps'] == step, 'max_hip_angle'] = max_hip_angle
        elif len(unique_steps_right) > 0:
            for step in unique_steps:
                step_data = df.loc[df['steps'] == step, 'Right_Hip_Angle']
                # Filter out NaN and invalid values (outside realistic range)
                valid_data = step_data.dropna()
                valid_data = valid_data[(valid_data >= 200) & (valid_data <= 300)]
                if len(valid_data) > 0:
                    max_hip_angle = valid_data.max()
                    df.loc[df['steps'] == step, 'max_hip_angle'] = max_hip_angle
        df['max_hip_angle'].ffill(inplace=True)
            

        # Calculate knee angles for each frame (ankle-knee-hip joint angle)
        # Simple calculation: angle at knee joint between thigh and shank segments
        print("Calculating knee angles from coordinates...")
        
        def calculate_knee_angle(hip, knee, ankle):
            """Calculate knee joint angle (ankle-knee-hip). Returns ~180° when straight, smaller when bent.
            Uses same vector directions as video display code for consistency."""
            try:
                # Check for NaN or invalid coordinates
                if np.any(np.isnan(hip)) or np.any(np.isnan(knee)) or np.any(np.isnan(ankle)):
                    return np.nan
                if np.any(np.isinf(hip)) or np.any(np.isinf(knee)) or np.any(np.isinf(ankle)):
                    return np.nan
                
                # Use same vector directions as video display code (draw_improved.py) and gather_distance_metrics.py
                # Thigh vector: from hip to knee
                # Shank vector: from knee to ankle
                thigh_vector = knee - hip
                shank_vector = ankle - knee
                
                # Check for zero vectors or very small vectors (unrealistic)
                n1 = np.linalg.norm(thigh_vector)
                n2 = np.linalg.norm(shank_vector)
                if n1 < 1e-6 or n2 < 1e-6:
                    return np.nan
                
                # Check for unrealistic segment lengths (likely bad tracking)
                # Typical thigh and shank should be at least a few pixels - use a very lenient threshold
                # This check is mainly to catch zero-length vectors which are already handled above
                if n1 < 1e-3 or n2 < 1e-3:
                    return np.nan
                
                # Calculate angle using same method as video display code
                # Use arctan2 method for consistency with calculate_angle_between_vectors
                dot = np.dot(thigh_vector, shank_vector)
                cross = np.cross(thigh_vector, shank_vector)
                cross_norm = np.linalg.norm(cross)
                angle = np.degrees(np.arctan2(cross_norm, dot))
                
                # Normalize to 0-180° range (same as calculate_angle_between_vectors with normalize180=False)
                if angle < 0:
                    angle += 360
                if angle > 180:
                    angle = 360 - angle
                
                # Validate the angle is in a reasonable range for knee flexion
                # Knee angles should typically be between ~30° (deep flexion) and ~180° (straight)
                if angle < 0 or angle > 180:
                    return np.nan
                
                # Return the simple internal angle - no switching or supplement
                return angle
            except:
                return np.nan
        
        # Calculate left and right knee angles for each frame
        left_knee_angles = []
        right_knee_angles = []
        
        for idx, row in df.iterrows():
            # Left knee angle
            l_hip = np.array([row.get('Lhip_x', np.nan), row.get('Lhip_y', np.nan)])
            l_knee = np.array([row.get('Lknee_x', np.nan), row.get('Lknee_y', np.nan)])
            l_ankle = np.array([row.get('Lankle_x', np.nan), row.get('Lankle_y', np.nan)])
            left_knee_angles.append(calculate_knee_angle(l_hip, l_knee, l_ankle))
            
            # Right knee angle
            r_hip = np.array([row.get('Rhip_x', np.nan), row.get('Rhip_y', np.nan)])
            r_knee = np.array([row.get('Rknee_x', np.nan), row.get('Rknee_y', np.nan)])
            r_ankle = np.array([row.get('Rankle_x', np.nan), row.get('Rankle_y', np.nan)])
            right_knee_angles.append(calculate_knee_angle(r_hip, r_knee, r_ankle))
        
        df['Left_Knee_Angle'] = left_knee_angles
        df['Right_Knee_Angle'] = right_knee_angles
        
        print(f"Knee angles calculated. Left: {np.nanmin(left_knee_angles):.1f}° - {np.nanmax(left_knee_angles):.1f}°, Right: {np.nanmin(right_knee_angles):.1f}° - {np.nanmax(right_knee_angles):.1f}°")
        
        # Calculate min_knee_angle (maximum flexion) for each step
        # min_knee_angle = smallest angle during step = maximum flexion
        df['min_knee_angle_left'] = np.nan
        df['min_knee_angle_right'] = np.nan
        
        # Left leg: find minimum angle (max flexion) for each step
        # Note: min_knee_angle is calculated from the same Left_Knee_Angle column that's displayed in videos
        filtered_count_left = 0
        for step in unique_steps_left:
            step_mask = df['steps_left'] == step
            step_angles = df.loc[step_mask, 'Left_Knee_Angle'].dropna()
            if len(step_angles) > 0:
                original_count = len(step_angles)
                # Filter out invalid angles: must be >= 35° and <= 180° (filter obviously bad data)
                # Values below 35° are unrealistic for running biomechanics (max flexion should be ~47°)
                step_angles = step_angles[(step_angles >= 35) & (step_angles <= 180)]
                filtered_count_left += (original_count - len(step_angles))
                if len(step_angles) > 0:
                    min_angle = step_angles.min()  # Minimum angle = maximum flexion
                    df.loc[step_mask, 'min_knee_angle_left'] = min_angle
        
        # Right leg: find minimum angle (max flexion) for each step
        # Note: min_knee_angle is calculated from the same Right_Knee_Angle column that's displayed in videos
        filtered_count_right = 0
        for step in unique_steps_right:
            step_mask = df['steps_right'] == step
            step_angles = df.loc[step_mask, 'Right_Knee_Angle'].dropna()
            if len(step_angles) > 0:
                original_count = len(step_angles)
                # Filter out invalid angles: must be >= 35° and <= 180° (filter obviously bad data)
                # Values below 35° are unrealistic for running biomechanics (max flexion should be ~47°)
                step_angles = step_angles[(step_angles >= 35) & (step_angles <= 180)]
                filtered_count_right += (original_count - len(step_angles))
                if len(step_angles) > 0:
                    min_angle = step_angles.min()  # Minimum angle = maximum flexion
                    df.loc[step_mask, 'min_knee_angle_right'] = min_angle
        
        if filtered_count_left > 0 or filtered_count_right > 0:
            print(f"Filtered out {filtered_count_left} left and {filtered_count_right} right knee angles < 35° or > 180°")
        
        # Forward fill to propagate values throughout each step (only within step boundaries)
        for step in unique_steps_left:
            step_mask = df['steps_left'] == step
            if step_mask.any():
                df.loc[step_mask, 'min_knee_angle_left'] = df.loc[step_mask, 'min_knee_angle_left'].ffill()
        
        for step in unique_steps_right:
            step_mask = df['steps_right'] == step
            if step_mask.any():
                df.loc[step_mask, 'min_knee_angle_right'] = df.loc[step_mask, 'min_knee_angle_right'].ffill()
        
        # Legacy: min_knee_angle (use left if available, otherwise right)
        df['min_knee_angle'] = df['min_knee_angle_left'].fillna(df['min_knee_angle_right'])
        
        # Also create max_knee_angle_left and max_knee_angle_right (same as min, for compatibility)
        df['max_knee_angle_left'] = df['min_knee_angle_left']
        df['max_knee_angle_right'] = df['min_knee_angle_right']
        df['max_knee_angle'] = df['min_knee_angle']

            # Calculate trunk angles
        df['trunk_angle'] = np.nan

        for index, row in df.iterrows():
            # Calculate midpoints on the z-axis
            shoulder_midpoint_x = (row['Lshldr_x'] + row['Rshldr_x']) / 2
            shoulder_midpoint_y = (row['Lshldr_y'] + row['Rshldr_y']) / 2
            

            hip_midpoint_x = (row['Lhip_x'] + row['Rhip_x']) / 2
            hip_midpoint_y = (row['Lhip_y'] + row['Rhip_y']) / 2
            

            hip_coord = np.array([hip_midpoint_x, hip_midpoint_y])
            shoulder_coord = np.array([shoulder_midpoint_x, shoulder_midpoint_y])  

            # Create a vector between the midpoints
            hip_shoulder_vector = hip_coord - shoulder_coord

            # Define the vertical vector
            vertical_vector = np.array([0, -1])  # Ensure the vector points upwards

            # Calculate the angle between the hip_shoulder_vector and the vertical_vector
            angle_radians = np.arctan2(np.linalg.norm(np.cross(hip_shoulder_vector, vertical_vector)),
                                    np.dot(hip_shoulder_vector, vertical_vector))
            angle_degrees = np.degrees(angle_radians)

            df.at[index, 'trunk_angle'] = angle_degrees

        # for start, _ in plateau_widths:
        #     plot_trunk_angle(df, start)


        # Calculate touchdown trunk angles for both legs
        df['touchdown_trunk_angle'] = np.nan
        df['touchdown_trunk_angle_left'] = np.nan
        df['touchdown_trunk_angle_right'] = np.nan

        # Left leg touchdown trunk angles
        for start, _ in plateau_widths_left:
            mask = df['frame'].astype(float) == float(start)
            if any(mask):
                touch_down_trunk_angle = df.loc[mask, 'trunk_angle'].iloc[0]
                df.loc[mask, 'touchdown_trunk_angle_left'] = touch_down_trunk_angle
                print(f"Left leg touchdown trunk angle at frame {start}: {touch_down_trunk_angle}")
            else:
                print(f"No matching frame found for left leg start: {start}")

        # Right leg touchdown trunk angles
        for start, _ in plateau_widths_right:
            mask = df['frame'].astype(float) == float(start)
            if any(mask):
                touch_down_trunk_angle = df.loc[mask, 'trunk_angle'].iloc[0]
                df.loc[mask, 'touchdown_trunk_angle_right'] = touch_down_trunk_angle
                print(f"Right leg touchdown trunk angle at frame {start}: {touch_down_trunk_angle}")
            else:
                print(f"No matching frame found for right leg start: {start}")

        # Forward fill for both legs
        for step in unique_steps_left:
            step_mask = df['steps_left'] == step
            first_value = df.loc[step_mask, 'touchdown_trunk_angle_left'].first_valid_index()
            if first_value is not None:
                df.loc[step_mask, 'touchdown_trunk_angle_left'] = df.loc[first_value, 'touchdown_trunk_angle_left']

        for step in unique_steps_right:
            step_mask = df['steps_right'] == step
            first_value = df.loc[step_mask, 'touchdown_trunk_angle_right'].first_valid_index()
            if first_value is not None:
                df.loc[step_mask, 'touchdown_trunk_angle_right'] = df.loc[first_value, 'touchdown_trunk_angle_right']

        # Legacy touchdown_trunk_angle (use left leg if available, otherwise right)
        if len(plateau_widths_left) > 0:
            for start, _ in plateau_widths_left:
                mask = df['frame'].astype(float) == float(start)
                if any(mask):
                    touch_down_trunk_angle = df.loc[mask, 'trunk_angle'].iloc[0]
                    df.loc[mask, 'touchdown_trunk_angle'] = touch_down_trunk_angle
        elif len(plateau_widths_right) > 0:
            for start, _ in plateau_widths_right:
                mask = df['frame'].astype(float) == float(start)
                if any(mask):
                    touch_down_trunk_angle = df.loc[mask, 'trunk_angle'].iloc[0]
                    df.loc[mask, 'touchdown_trunk_angle'] = touch_down_trunk_angle

        for step in unique_steps:
            step_mask = df['steps'] == step
            first_value = df.loc[step_mask, 'touchdown_trunk_angle'].first_valid_index()
            if first_value is not None:
                df.loc[step_mask, 'touchdown_trunk_angle'] = df.loc[first_value, 'touchdown_trunk_angle']
        
        # st.dataframe(df['touchdown_trunk_angle'])
        
        # def pixels_to_cm(pixel_distance, meters_per_pixel):
        #     # Convert to real-world distance
        #     euclidean_distance_m = pixel_distance * meters_per_pixel
        #     euclidean_distance_cm = (euclidean_distance_m * 100)*100 #accounts for the zoom preprocessing

        #     return euclidean_distance_cm


        # Calculate touchdown knee separation for both legs
        df['touchdown_knee_sep'] = np.nan
        df['touchdown_knee_sep_left'] = np.nan
        df['touchdown_knee_sep_right'] = np.nan

        # Left leg touchdown knee separation
        for start, _ in plateau_widths_left:
            # Guard against indices that fall outside the dataframe,
            # which can happen if fewer pose frames are available after reprocessing.
            if start < 0 or start >= len(df):
                continue

            touchdown_frame = df.loc[start, 'frame']
            if touchdown_frame < len(df):
                Rknee_x = df.loc[df['frame'] == touchdown_frame, 'Rknee_x'].iloc[0]
                Lknee_x = df.loc[df['frame'] == touchdown_frame, 'Lknee_x'].iloc[0]
                pixel_distance = abs(Rknee_x - Lknee_x)
                try:
                    real_distance = pixels_to_cm(pixel_distance, meters_per_pixel)
                    df.loc[df['frame'] == touchdown_frame, 'touchdown_knee_sep_left'] = real_distance
                except Exception as e:
                    print(f"Warning: Error calculating knee separation for left leg: {e}")
                    df.loc[df['frame'] == touchdown_frame, 'touchdown_knee_sep_left'] = 0

        # Right leg touchdown knee separation
        for start, _ in plateau_widths_right:
            if start < 0 or start >= len(df):
                continue

            touchdown_frame = df.loc[start, 'frame']
            if touchdown_frame < len(df):
                Rknee_x = df.loc[df['frame'] == touchdown_frame, 'Rknee_x'].iloc[0]
                Lknee_x = df.loc[df['frame'] == touchdown_frame, 'Lknee_x'].iloc[0]
                pixel_distance = abs(Rknee_x - Lknee_x)
                try:
                    real_distance = pixels_to_cm(pixel_distance, meters_per_pixel)
                    df.loc[df['frame'] == touchdown_frame, 'touchdown_knee_sep_right'] = real_distance
                except Exception as e:
                    print(f"Warning: Error calculating knee separation for right leg: {e}")
                    df.loc[df['frame'] == touchdown_frame, 'touchdown_knee_sep_right'] = 0

        # Forward fill for both legs
        df['touchdown_knee_sep_left'].ffill(inplace=True)
        df['touchdown_knee_sep_right'].ffill(inplace=True)

        # Legacy touchdown_knee_sep (use left leg if available, otherwise right)
        if len(plateau_widths_left) > 0:
            for start, _ in plateau_widths_left:
                if start < 0 or start >= len(df):
                    continue

                touchdown_frame = df.loc[start, 'frame']
                if touchdown_frame < len(df):
                    Rknee_x = df.loc[df['frame'] == touchdown_frame, 'Rknee_x'].iloc[0]
                    Lknee_x = df.loc[df['frame'] == touchdown_frame, 'Lknee_x'].iloc[0]
                    pixel_distance = abs(Rknee_x - Lknee_x)
                    try:
                        real_distance = pixels_to_cm(pixel_distance, meters_per_pixel)
                        df.loc[df['frame'] == touchdown_frame, 'touchdown_knee_sep'] = real_distance
                    except Exception as e:
                        print(f"Warning: Error calculating knee separation: {e}")
                        df.loc[df['frame'] == touchdown_frame, 'touchdown_knee_sep'] = 0
        elif len(plateau_widths_right) > 0:
            for start, _ in plateau_widths_right:
                if start < 0 or start >= len(df):
                    continue

                touchdown_frame = df.loc[start, 'frame']
                if touchdown_frame < len(df):
                    Rknee_x = df.loc[df['frame'] == touchdown_frame, 'Rknee_x'].iloc[0]
                    Lknee_x = df.loc[df['frame'] == touchdown_frame, 'Lknee_x'].iloc[0]
                    pixel_distance = abs(Rknee_x - Lknee_x)
                    try:
                        real_distance = pixels_to_cm(pixel_distance, meters_per_pixel)
                        df.loc[df['frame'] == touchdown_frame, 'touchdown_knee_sep'] = real_distance
                    except Exception as e:
                        print(f"Warning: Error calculating knee separation: {e}")
                        df.loc[df['frame'] == touchdown_frame, 'touchdown_knee_sep'] = 0

        df['touchdown_knee_sep'].ffill(inplace=True)

        # def compute_touchdown_knee_sep(df, steps_col='steps', gct_col='gct_frames', conversion_factor=1.0):
        #     df['touchdown_knee_sep'] = np.nan  # Initialize the column with NaN values
        #     if steps_col not in df.columns:
        #         df['steps'] = 1
        #     unique_steps = df[steps_col].dropna().unique()
        #     for step in unique_steps:
        #         step_mask = (df[steps_col] == step)
        #         td_indices = df[step_mask & (df[gct_col] > 0)].index
        #         if len(td_indices) == 0:
        #             continue
        #         td_idx = td_indices[0]
        #         # Retrieve knee x-coordinates at touchdown
        #         Lx = df.at[td_idx, 'Lknee_x']
        #         Rx = df.at[td_idx, 'Rknee_x']
        #         # Compute 2D separation as (Lknee_x - Rknee_x)
        #         sep_2d_pixels = Lx - Rx
        #         sep_2d_m = sep_2d_pixels * conversion_factor
        #         df.loc[step_mask, 'touchdown_knee_sep'] = sep_2d_m  # Assign the value to the entire step
        #     return df['touchdown_knee_sep']  # Return the column

        # df['touchdown_knee_sep'] = compute_touchdown_knee_sep(
        #     df,
        #     steps_col='steps',
        #     gct_col='gct_frames',
        #     conversion_factor=conversion_factor
        # )

        # df['touchdown_knee_sep'].ffill(inplace=True)

        # unique_steps = sorted(df['steps'].unique())
        # velocities_method1 = []
        # for step in unique_steps:
        #     step_df = df[df['steps'] == step].reset_index(drop=True)
        #     if len(step_df) < 2:
        #         continue
        #     # Define peak frame as the frame with maximum Lknee_y
        #     peak_index = step_df['Lknee_y'].idxmax()
        #     ground_strike_index = None
        #     # Find the first frame after the peak where touchdown is indicated by non-NaN touchdown_trunk_angle
        #     for i in range(peak_index + 1, len(step_df)):
        #         if not pd.isna(step_df.loc[i, 'touchdown_trunk_angle']):
        #             ground_strike_index = i
        #             break
        #     velocity = None
        #     if ground_strike_index is not None and ground_strike_index != peak_index:
        #         peak_toe = np.array([step_df.loc[peak_index, 'Ltoe_x'],
        #                              step_df.loc[peak_index, 'Ltoe_y'],
        #                              step_df.loc[peak_index, 'Ltoe_z']])
        #         peak_hip = np.array([step_df.loc[peak_index, 'Lhip_x'],
        #                              step_df.loc[peak_index, 'Lhip_y'],
        #                              step_df.loc[peak_index, 'Lhip_z']])
        #         strike_toe = np.array([step_df.loc[ground_strike_index, 'Ltoe_x'],
        #                                step_df.loc[ground_strike_index, 'Ltoe_y'],
        #                                step_df.loc[ground_strike_index, 'Ltoe_z']])
        #         strike_hip = np.array([step_df.loc[ground_strike_index, 'Lhip_x'],
        #                                step_df.loc[ground_strike_index, 'Lhip_y'],
        #                                step_df.loc[ground_strike_index, 'Lhip_z']])
        #         rel_peak = peak_toe - peak_hip
        #         rel_strike = strike_toe - strike_hip
        #         disp_pixels = rel_strike - rel_peak
        #         disp_meters = np.linalg.norm(disp_pixels * conversion_factor)
        #         time_diff = (ground_strike_index - peak_index) / fps
        #         if time_diff > 0:
        #             velocity = disp_meters / time_diff
        #         df['touchdown_toe_velocity'] = velocity

        # df['touchdown_toe_velocity'].ffill(inplace=True)

        # Calculate Center of Mass (COM) using simplified segment model
        # Based on de Leva (1996) segment model, using available joint positions
        # For sagittal view, we use a weighted average of key segments
        df['COM_x'] = np.nan
        df['COM_y'] = np.nan
        
        for i in range(len(df)):
            row = df.iloc[i]
            com_x_components = []
            com_y_components = []
            weights = []
            
            # Head/neck segment (approximated from shoulder midpoint if available)
            if 'Lshldr_x' in df.columns and 'Rshldr_x' in df.columns:
                if not pd.isna(row['Lshldr_x']) and not pd.isna(row['Rshldr_x']):
                    head_x = (row['Lshldr_x'] + row['Rshldr_x']) / 2
                    # Use average of shoulder y-coordinates, or estimate head position above shoulders
                    if not pd.isna(row.get('Lshldr_y')) and not pd.isna(row.get('Rshldr_y')):
                        head_y = (row['Lshldr_y'] + row['Rshldr_y']) / 2
                    else:
                        head_y = row.get('Lshldr_y', row.get('Rshldr_y', np.nan))
                    if not pd.isna(head_x) and not pd.isna(head_y):
                        com_x_components.append(head_x)
                        com_y_components.append(head_y)
                        weights.append(0.081)  # Head + neck mass proportion (de Leva 1996)
            
            # Trunk segment (approximated from hip midpoint)
            if not pd.isna(row.get('Lhip_x')) and not pd.isna(row.get('Rhip_x')):
                trunk_x = (row['Lhip_x'] + row['Rhip_x']) / 2
                trunk_y = (row.get('Lhip_y', 0) + row.get('Rhip_y', 0)) / 2
                if not pd.isna(trunk_x):
                    com_x_components.append(trunk_x)
                    com_y_components.append(trunk_y)
                    weights.append(0.497)  # Trunk mass proportion (de Leva 1996)
            
            # Left leg segments (thigh, shank, foot)
            if not pd.isna(row.get('Lhip_x')) and not pd.isna(row.get('Lknee_x')):
                thigh_x = (row['Lhip_x'] + row['Lknee_x']) / 2
                thigh_y = (row.get('Lhip_y', 0) + row.get('Lknee_y', 0)) / 2
                if not pd.isna(thigh_x):
                    com_x_components.append(thigh_x)
                    com_y_components.append(thigh_y)
                    weights.append(0.100)  # Thigh mass proportion (de Leva 1996)
            
            if not pd.isna(row.get('Lknee_x')) and not pd.isna(row.get('Lankle_x')):
                shank_x = (row['Lknee_x'] + row['Lankle_x']) / 2
                shank_y = (row.get('Lknee_y', 0) + row.get('Lankle_y', 0)) / 2
                if not pd.isna(shank_x):
                    com_x_components.append(shank_x)
                    com_y_components.append(shank_y)
                    weights.append(0.046)  # Shank mass proportion (de Leva 1996)
            
            if not pd.isna(row.get('Lankle_x')) and not pd.isna(row.get('Ltoe_x')):
                foot_x = (row['Lankle_x'] + row['Ltoe_x']) / 2
                foot_y = (row.get('Lankle_y', 0) + row.get('Ltoe_y', 0)) / 2
                if not pd.isna(foot_x):
                    com_x_components.append(foot_x)
                    com_y_components.append(foot_y)
                    weights.append(0.014)  # Foot mass proportion (de Leva 1996)
            
            # Right leg segments (thigh, shank, foot)
            if not pd.isna(row.get('Rhip_x')) and not pd.isna(row.get('Rknee_x')):
                thigh_x = (row['Rhip_x'] + row['Rknee_x']) / 2
                thigh_y = (row.get('Rhip_y', 0) + row.get('Rknee_y', 0)) / 2
                if not pd.isna(thigh_x):
                    com_x_components.append(thigh_x)
                    com_y_components.append(thigh_y)
                    weights.append(0.100)  # Thigh mass proportion (de Leva 1996)
            
            if not pd.isna(row.get('Rknee_x')) and not pd.isna(row.get('Rankle_x')):
                shank_x = (row['Rknee_x'] + row['Rankle_x']) / 2
                shank_y = (row.get('Rknee_y', 0) + row.get('Rankle_y', 0)) / 2
                if not pd.isna(shank_x):
                    com_x_components.append(shank_x)
                    com_y_components.append(shank_y)
                    weights.append(0.046)  # Shank mass proportion (de Leva 1996)
            
            if not pd.isna(row.get('Rankle_x')) and not pd.isna(row.get('Rtoe_x')):
                foot_x = (row['Rankle_x'] + row['Rtoe_x']) / 2
                foot_y = (row.get('Rankle_y', 0) + row.get('Rtoe_y', 0)) / 2
                if not pd.isna(foot_x):
                    com_x_components.append(foot_x)
                    com_y_components.append(foot_y)
                    weights.append(0.014)  # Foot mass proportion (de Leva 1996)
            
            # Calculate weighted COM if we have components
            if len(com_x_components) > 0 and len(weights) > 0:
                # Normalize weights to sum to 1
                total_weight = sum(weights)
                if total_weight > 0:
                    normalized_weights = [w / total_weight for w in weights]
                    df.loc[i, 'COM_x'] = np.average(com_x_components, weights=normalized_weights)
                    df.loc[i, 'COM_y'] = np.average(com_y_components, weights=normalized_weights)
        
        # Fallback: if COM calculation failed, use hip midpoint
        if df['COM_x'].isna().all():
            print("Warning: COM calculation failed, using hip midpoint as fallback")
            df['COM_x'] = (df['Lhip_x'] + df['Rhip_x']) / 2
            df['COM_y'] = (df['Lhip_y'] + df['Rhip_y']) / 2

        # Calculate toe velocity relative to COM for both legs
        # Improved method: use multiple frame windows and median filtering for robustness
        df['toe_velocity'] = np.nan
        df['toe_velocity_left'] = np.nan
        df['toe_velocity_right'] = np.nan

        # Parameters for velocity calculation
        min_frames = 3  # Minimum frames for velocity calculation
        max_frames = 10  # Maximum frames for velocity calculation
        frames_before_touchdown = 2  # Use velocity from 2 frames before touchdown (more stable)
        max_velocity_mps = 15.0  # Maximum reasonable velocity in m/s (sprint max ~12-13 m/s)
        min_velocity_mps = -5.0  # Minimum reasonable velocity (backward motion limit)

        def calculate_robust_velocity(toe_x_col, com_x_col, df, fps, meters_per_pixel):
            """Calculate velocity using multiple frame windows and median filtering"""
            velocities = []
            
            # Calculate velocity using multiple frame windows (3-7 frames)
            for window in range(min_frames, max_frames + 1):
                window_velocities = np.full(len(df), np.nan)
                
                for i in range(window, len(df)):
                    # Check if both current and past frames have valid data
                    if (pd.notna(df.loc[i, toe_x_col]) and pd.notna(df.loc[i-window, toe_x_col]) and
                        pd.notna(df.loc[i, com_x_col]) and pd.notna(df.loc[i-window, com_x_col])):
                        try:
                            # Calculate toe displacement
                            toe_displacement = df.loc[i, toe_x_col] - df.loc[i-window, toe_x_col]
                            # Calculate COM displacement
                            com_displacement = df.loc[i, com_x_col] - df.loc[i-window, com_x_col]
                            # Relative displacement (toe relative to COM)
                            relative_displacement = toe_displacement - com_displacement
                            # Convert to cm and calculate velocity
                            relative_displacement_cm = pixels_to_cm(relative_displacement, meters_per_pixel)
                            time_diff = window / fps
                            velocity_mps = (relative_displacement_cm / time_diff) / 100  # Convert to m/s
                            
                            # Validate velocity is within reasonable bounds
                            if min_velocity_mps <= velocity_mps <= max_velocity_mps:
                                window_velocities[i] = velocity_mps
                        except Exception:
                            pass  # Skip invalid calculations
                
                velocities.append(window_velocities)
            
            # Use median of all window calculations for robustness (reduces noise)
            if velocities:
                velocity_array = np.array(velocities)
                # Calculate median across windows, ignoring NaN values
                median_velocities = np.nanmedian(velocity_array, axis=0)
                
                # Apply additional smoothing with rolling median (3-frame window)
                smoothed_velocities = pd.Series(median_velocities).rolling(
                    window=3, center=True, min_periods=1
                ).median().values
                
                return smoothed_velocities
            else:
                return np.full(len(df), np.nan)

        # Calculate velocities for both legs
        df['toe_velocity_left'] = calculate_robust_velocity('Ltoe_x', 'COM_x', df, fps, meters_per_pixel)
        df['toe_velocity_right'] = calculate_robust_velocity('Rtoe_x', 'COM_x', df, fps, meters_per_pixel)

        # Legacy toe_velocity (use left leg)
        df['toe_velocity'] = df['toe_velocity_left']

        # Forward fill only valid values (not NaN)
        df['toe_velocity_left'] = df['toe_velocity_left'].ffill()
        df['toe_velocity_right'] = df['toe_velocity_right'].ffill()
        df['toe_velocity'] = df['toe_velocity'].ffill()

        # Extract touchdown toe velocities for both legs
        # Use velocity from a few frames before touchdown for more stable/representative values
        df['touchdown_toe_velocity'] = np.nan
        df['touchdown_toe_velocity_left'] = np.nan
        df['touchdown_toe_velocity_right'] = np.nan

        # Left leg touchdown toe velocity - use velocity from frames_before_touchdown frames before touchdown
        for start, _ in plateau_widths_left:
            if start < len(df):
                # Use velocity from a few frames before touchdown (more stable)
                velocity_frame = max(0, start - frames_before_touchdown)
                
                # Try to get valid velocity from the frame before touchdown
                if pd.notna(df.loc[velocity_frame, 'toe_velocity_left']):
                    velocity_value = df.loc[velocity_frame, 'toe_velocity_left']
                else:
                    # Fallback: find nearest valid velocity within 5 frames
                    search_range = range(max(0, start - 5), min(len(df), start + 1))
                    valid_velocities = []
                    for frame_idx in search_range:
                        if pd.notna(df.loc[frame_idx, 'toe_velocity_left']):
                            valid_velocities.append((abs(frame_idx - start), df.loc[frame_idx, 'toe_velocity_left']))
                    
                    if valid_velocities:
                        # Use closest valid velocity
                        valid_velocities.sort(key=lambda x: x[0])
                        velocity_value = valid_velocities[0][1]
                    else:
                        velocity_value = np.nan
                
                # Only assign if value is valid and within reasonable bounds
                if pd.notna(velocity_value) and min_velocity_mps <= velocity_value <= max_velocity_mps:
                    df.loc[start, 'touchdown_toe_velocity_left'] = velocity_value

        # Right leg touchdown toe velocity - use velocity from frames_before_touchdown frames before touchdown
        for start, _ in plateau_widths_right:
            if start < len(df):
                # Use velocity from a few frames before touchdown (more stable)
                velocity_frame = max(0, start - frames_before_touchdown)
                
                # Try to get valid velocity from the frame before touchdown
                if pd.notna(df.loc[velocity_frame, 'toe_velocity_right']):
                    velocity_value = df.loc[velocity_frame, 'toe_velocity_right']
                else:
                    # Fallback: find nearest valid velocity within 5 frames
                    search_range = range(max(0, start - 5), min(len(df), start + 1))
                    valid_velocities = []
                    for frame_idx in search_range:
                        if pd.notna(df.loc[frame_idx, 'toe_velocity_right']):
                            valid_velocities.append((abs(frame_idx - start), df.loc[frame_idx, 'toe_velocity_right']))
                    
                    if valid_velocities:
                        # Use closest valid velocity
                        valid_velocities.sort(key=lambda x: x[0])
                        velocity_value = valid_velocities[0][1]
                    else:
                        velocity_value = np.nan
                
                # Only assign if value is valid and within reasonable bounds
                if pd.notna(velocity_value) and min_velocity_mps <= velocity_value <= max_velocity_mps:
                    df.loc[start, 'touchdown_toe_velocity_right'] = velocity_value

        # Legacy touchdown_toe_velocity (use left leg if available, otherwise right)
        if len(plateau_widths_left) > 0:
            for start, _ in plateau_widths_left:
                if start < len(df) and pd.notna(df.loc[start, 'touchdown_toe_velocity_left']):
                    df.loc[start, 'touchdown_toe_velocity'] = df.loc[start, 'touchdown_toe_velocity_left']
        elif len(plateau_widths_right) > 0:
            for start, _ in plateau_widths_right:
                if start < len(df) and pd.notna(df.loc[start, 'touchdown_toe_velocity_right']):
                    df.loc[start, 'touchdown_toe_velocity'] = df.loc[start, 'touchdown_toe_velocity_right']

        # Forward fill for both legs (only within each step, don't cross step boundaries)
        for step in unique_steps_left:
            step_mask = df['steps_left'] == step
            first_value_idx = df.loc[step_mask, 'touchdown_toe_velocity_left'].first_valid_index()
            if first_value_idx is not None:
                first_value = df.loc[first_value_idx, 'touchdown_toe_velocity_left']
                if pd.notna(first_value) and min_velocity_mps <= first_value <= max_velocity_mps:
                    df.loc[step_mask, 'touchdown_toe_velocity_left'] = first_value

        for step in unique_steps_right:
            step_mask = df['steps_right'] == step
            first_value_idx = df.loc[step_mask, 'touchdown_toe_velocity_right'].first_valid_index()
            if first_value_idx is not None:
                first_value = df.loc[first_value_idx, 'touchdown_toe_velocity_right']
                if pd.notna(first_value) and min_velocity_mps <= first_value <= max_velocity_mps:
                    df.loc[step_mask, 'touchdown_toe_velocity_right'] = first_value

        # Legacy forward fill
        for step in unique_steps:
            step_mask = df['steps'] == step
            first_value_idx = df.loc[step_mask, 'touchdown_toe_velocity'].first_valid_index()
            if first_value_idx is not None:
                first_value = df.loc[first_value_idx, 'touchdown_toe_velocity']
                if pd.notna(first_value) and min_velocity_mps <= first_value <= max_velocity_mps:
                    df.loc[step_mask, 'touchdown_toe_velocity'] = first_value
        
        # Final forward fill only for remaining NaN values (within step boundaries)
        df['touchdown_toe_velocity_left'] = df['touchdown_toe_velocity_left'].ffill()
        df['touchdown_toe_velocity_right'] = df['touchdown_toe_velocity_right'].ffill()
        df['touchdown_toe_velocity'] = df['touchdown_toe_velocity'].ffill()
        # if 'touchdown_toe_velocity' in df.columns:
        #     last_valid_value = None
        #     for i in range(len(df)):
        #         if df.loc[i, 'touchdown_toe_velocity'] > 11:
        #             if last_valid_value is not None:
        #                 df.loc[i, 'touchdown_toe_velocity'] = last_valid_value
        #         else:
        #             last_valid_value = df.loc[i, 'touchdown_toe_velocity']


        # def plot_trunk_angle(df, start_index):
        #     import plotly.graph_objects as go

        #     # Create a Plotly figure
        #     fig = go.Figure()

        #     # Use the start index to get the row
        #     row = df.loc[start_index]
            
        #     shoulder_midpoint_x = (row['Lshldr_x'] + row['Rshldr_x']) / 2
        #     shoulder_midpoint_y = (row['Lshldr_y'] + row['Rshldr_y']) / 2
            
        #     hip_midpoint_x = (row['Lhip_x'] + row['Rhip_x']) / 2
        #     hip_midpoint_y = (row['Lhip_y'] + row['Rhip_y']) / 2
            
        #     hip_coord = np.array([hip_midpoint_x, hip_midpoint_y])
        #     shoulder_coord = np.array([shoulder_midpoint_x, shoulder_midpoint_y])  

        #     # Create a vector between the midpoints
        #     hip_shoulder_vector = hip_coord - shoulder_coord
        #     vertical_vector = [0, -1]

        #     # Add traces for the hip-shoulder vector
        #     fig.add_trace(go.Scatter(
        #         x=[shoulder_midpoint_x, hip_midpoint_x], 
        #         y=[shoulder_midpoint_y, hip_midpoint_y],
        #         mode='lines',
        #         line=dict(dash='dash', color='red'),
        #         name='Hip-Shoulder Vector'
        #     ))

        #     # Add traces for the vertical vector
        #     fig.add_trace(go.Scatter(
        #         x=[hip_midpoint_x, hip_midpoint_x], 
        #         y=[hip_midpoint_y, hip_midpoint_y + 1],  # Assuming a unit vector for vertical
        #         mode='lines',
        #         line=dict(dash='dot', color='blue'),
        #         name='Vertical Vector'
        #     ))

        #     # Add scatter points for midpoints
        #     fig.add_trace(go.Scatter(
        #         x=[shoulder_midpoint_x], 
        #         y=[shoulder_midpoint_y],
        #         mode='markers',
        #         marker=dict(color='green'),
        #         name='Shoulder Midpoint'
        #     ))

        #     fig.add_trace(go.Scatter(
        #         x=[hip_midpoint_x], 
        #         y=[hip_midpoint_y],
        #         mode='markers',
        #         marker=dict(color='magenta'),
        #         name='Hip Midpoint'
        #     ))

        #     # Calculate angle
        #     angle_radians = np.arctan2(np.linalg.norm(np.cross(hip_shoulder_vector, vertical_vector)),
        #                                 np.dot(hip_shoulder_vector, vertical_vector))
        #     angle_degrees = np.degrees(angle_radians)

        #     # Update layout for each frame
        #     fig.update_layout(
        #         title=f'Trunk Angle: {angle_degrees:.2f} degrees',
        #         xaxis=dict(range=[min(hip_midpoint_x, shoulder_midpoint_x) - 1, max(hip_midpoint_x, shoulder_midpoint_x) + 1], title='X'),
        #         yaxis=dict(range=[min(hip_midpoint_y, shoulder_midpoint_y) - 1, max(hip_midpoint_y, shoulder_midpoint_y) + 1], title='Y'),
        #         template='plotly_white'
        #     )

        #     # Save the figure as an interactive HTML file
        #     fig.write_html('trunk_angle.html')

        #     # Show the figure
        #     fig.show()

        # for start, _ in plateau_widths:
        #     plot_trunk_angle(df, start)


        # Calculate touchdown trunk angle for left leg
        for start, _ in plateau_widths_left:
            mask = df['frame'].astype(float) == float(start)
            if any(mask):
                touch_down_trunk_angle = df.loc[mask, 'trunk_angle'].iloc[0]
                df.loc[mask, 'touchdown_trunk_angle_left'] = touch_down_trunk_angle
                # Debug: Check the touchdown trunk angle
                print(f"Touchdown trunk angle (left) at frame {start}: {touch_down_trunk_angle}")
            else:
                print(f"No matching frame found for start (left): {start}")

        # Calculate touchdown trunk angle for right leg
        for start, _ in plateau_widths_right:
            mask = df['frame'].astype(float) == float(start)
            if any(mask):
                touch_down_trunk_angle = df.loc[mask, 'trunk_angle'].iloc[0]
                df.loc[mask, 'touchdown_trunk_angle_right'] = touch_down_trunk_angle
                # Debug: Check the touchdown trunk angle
                print(f"Touchdown trunk angle (right) at frame {start}: {touch_down_trunk_angle}")
            else:
                print(f"No matching frame found for start (right): {start}")

        # Propagate touchdown trunk angle values across steps for left leg
        for step in unique_steps:
            step_mask = df['steps'] == step
            first_value = df.loc[step_mask, 'touchdown_trunk_angle_left'].first_valid_index()
            if first_value is not None:
                df.loc[step_mask, 'touchdown_trunk_angle_left'] = df.loc[first_value, 'touchdown_trunk_angle_left']
            else:
                print(f"No valid touchdown trunk angle (left) found for step: {step}")

        # Propagate touchdown trunk angle values across steps for right leg
        for step in unique_steps:
            step_mask = df['steps'] == step
            first_value = df.loc[step_mask, 'touchdown_trunk_angle_right'].first_valid_index()
            if first_value is not None:
                df.loc[step_mask, 'touchdown_trunk_angle_right'] = df.loc[first_value, 'touchdown_trunk_angle_right']
            else:
                print(f"No valid touchdown trunk angle (right) found for step: {step}")
        
        

        # Prep variables for Alpha trainer
        
        
        # Define optimal values and bounds based on gender
        if gender == 'male':
            optimal_values = {
                'stride_rate': {'optimal': 4.76, 'lower': 4.61, 'upper': 4.76},
                'gct': {'optimal': 0.087, 'lower': 0.087, 'upper': 0.094},
                'max_hip_angle': {'optimal': 260, 'lower': 250, 'upper': 270},
                'max_knee_angle': {'optimal': 47, 'lower': 44, 'upper': 47},
                'touchdown_trunk_angle': {'optimal': 170, 'lower': 165, 'upper': 170},
                'touchdown_knee_sep': {'optimal': 1, 'lower': 1, 'upper': 3},
                'touchdown_toe_velocity': {'optimal': 7.6, 'lower': 6.8, 'upper': 7.6}
            }
        else:  # female
            optimal_values = {
                'stride_rate': {'optimal': 4.85, 'lower': 4.70, 'upper': 4.85},
                'gct': {'optimal': 0.083, 'lower': 0.083, 'upper': 0.090},
                'max_hip_angle': {'optimal': 247, 'lower': 240, 'upper': 255},
                'max_knee_angle': {'optimal': 47, 'lower': 44, 'upper': 47},
                'touchdown_trunk_angle': {'optimal': 176, 'lower': 173, 'upper': 176},
                'touchdown_knee_sep': {'optimal': 1, 'lower': 1, 'upper': 3},
                'touchdown_toe_velocity': {'optimal': 7.6, 'lower': 6.8, 'upper': 7.6}
            }
        
        # Calculate means for each step
        try:
            grouped_means = df.groupby('steps').mean()
        except Exception as e:
            display_error(f"Error calculating means: {str(e)}")
            return
        
        # Get values for each metric (legacy and dual-sided)
        try:
            # Legacy values
            st_rt_values = grouped_means['st_rt'].tolist()
            gct_values = grouped_means['gct'].tolist()
            max_hip_angle_values = grouped_means['max_hip_angle'].tolist()
            max_knee_angle_values = grouped_means['max_knee_angle'].tolist()
            touchdown_trunk_angle_values = grouped_means['touchdown_trunk_angle'].tolist()
            touchdown_knee_sep_values = grouped_means['touchdown_knee_sep'].tolist()
            touchdown_toe_velocity_values = grouped_means['touchdown_toe_velocity'].tolist()
            
            # Left leg values
            st_rt_left_values = grouped_means['st_rt_left'].tolist() if 'st_rt_left' in grouped_means.columns else []
            gct_left_values = grouped_means['gct_left'].tolist() if 'gct_left' in grouped_means.columns else []
            max_hip_angle_left_values = grouped_means['max_hip_angle_left'].tolist() if 'max_hip_angle_left' in grouped_means.columns else []
            max_knee_angle_left_values = grouped_means['max_knee_angle_left'].tolist() if 'max_knee_angle_left' in grouped_means.columns else []
            touchdown_trunk_angle_left_values = grouped_means['touchdown_trunk_angle_left'].tolist() if 'touchdown_trunk_angle_left' in grouped_means.columns else []
            touchdown_knee_sep_left_values = grouped_means['touchdown_knee_sep_left'].tolist() if 'touchdown_knee_sep_left' in grouped_means.columns else []
            touchdown_toe_velocity_left_values = grouped_means['touchdown_toe_velocity_left'].tolist() if 'touchdown_toe_velocity_left' in grouped_means.columns else []
            
            # Right leg values
            st_rt_right_values = grouped_means['st_rt_right'].tolist() if 'st_rt_right' in grouped_means.columns else []
            gct_right_values = grouped_means['gct_right'].tolist() if 'gct_right' in grouped_means.columns else []
            max_hip_angle_right_values = grouped_means['max_hip_angle_right'].tolist() if 'max_hip_angle_right' in grouped_means.columns else []
            max_knee_angle_right_values = grouped_means['max_knee_angle_right'].tolist() if 'max_knee_angle_right' in grouped_means.columns else []
            touchdown_trunk_angle_right_values = grouped_means['touchdown_trunk_angle_right'].tolist() if 'touchdown_trunk_angle_right' in grouped_means.columns else []
            touchdown_knee_sep_right_values = grouped_means['touchdown_knee_sep_right'].tolist() if 'touchdown_knee_sep_right' in grouped_means.columns else []
            touchdown_toe_velocity_right_values = grouped_means['touchdown_toe_velocity_right'].tolist() if 'touchdown_toe_velocity_right' in grouped_means.columns else []
            
        except Exception as e:
            display_error(f"Error getting metric values: {str(e)}")
            return
        
        # Calculate averages with error handling
        # Helper function to filter NaN and calculate mean
        def safe_mean(values):
            if not values:
                return np.nan
            # Filter out NaN values
            valid_values = [v for v in values if pd.notna(v) and not np.isnan(v)]
            if len(valid_values) == 0:
                return np.nan
            return sum(valid_values) / len(valid_values)
        
        try:
            # Legacy averages
            avg_st_rt = safe_mean(st_rt_values)
            avg_gct = safe_mean(gct_values)
            avg_max_hip_angle = safe_mean(max_hip_angle_values)
            avg_max_knee_angle = safe_mean(max_knee_angle_values)
            avg_touchdown_trunk_angle = safe_mean(touchdown_trunk_angle_values)
            avg_touchdown_knee_sep = safe_mean(touchdown_knee_sep_values)
            avg_touchdown_toe_velocity = safe_mean(touchdown_toe_velocity_values)
            
            # Left leg averages
            avg_st_rt_left = safe_mean(st_rt_left_values)
            avg_gct_left = safe_mean(gct_left_values)
            avg_max_hip_angle_left = safe_mean(max_hip_angle_left_values)
            avg_max_knee_angle_left = safe_mean(max_knee_angle_left_values)
            avg_touchdown_trunk_angle_left = safe_mean(touchdown_trunk_angle_left_values)
            avg_touchdown_knee_sep_left = safe_mean(touchdown_knee_sep_left_values)
            avg_touchdown_toe_velocity_left = safe_mean(touchdown_toe_velocity_left_values)
            
            # Right leg averages
            avg_st_rt_right = safe_mean(st_rt_right_values)
            avg_gct_right = safe_mean(gct_right_values)
            avg_max_hip_angle_right = safe_mean(max_hip_angle_right_values)
            avg_max_knee_angle_right = safe_mean(max_knee_angle_right_values)
            avg_touchdown_trunk_angle_right = safe_mean(touchdown_trunk_angle_right_values)
            avg_touchdown_knee_sep_right = safe_mean(touchdown_knee_sep_right_values)
            avg_touchdown_toe_velocity_right = safe_mean(touchdown_toe_velocity_right_values)
            
        except ZeroDivisionError:
            pass
            # st.error("No valid steps detected in the analysis")
            return
        
        # Add optimal values and bounds to df
        df['opt_st_rt'] = optimal_values['stride_rate']['optimal']
        df['opt_gct'] = optimal_values['gct']['optimal']
        df['opt_max_hip_angle'] = optimal_values['max_hip_angle']['optimal']
        df['opt_max_knee_angle'] = optimal_values['max_knee_angle']['optimal']
        df['opt_touchdown_trunk_angle'] = optimal_values['touchdown_trunk_angle']['optimal']
        df['opt_touchdown_knee_sep'] = optimal_values['touchdown_knee_sep']['optimal']
        df['opt_touchdown_toe_velocity'] = optimal_values['touchdown_toe_velocity']['optimal']

        # Add upper bounds
        df['upper_st_rt'] = optimal_values['stride_rate']['upper']
        df['upper_gct'] = optimal_values['gct']['upper']
        df['upper_max_hip_angle'] = optimal_values['max_hip_angle']['upper']
        df['upper_max_knee_angle'] = optimal_values['max_knee_angle']['upper']
        df['upper_touchdown_trunk_angle'] = optimal_values['touchdown_trunk_angle']['upper']
        df['upper_touchdown_knee_sep'] = optimal_values['touchdown_knee_sep']['upper']
        df['upper_touchdown_toe_velocity'] = optimal_values['touchdown_toe_velocity']['upper']

        # Add lower bounds
        df['lower_st_rt'] = optimal_values['stride_rate']['lower']
        df['lower_gct'] = optimal_values['gct']['lower']
        df['lower_max_hip_angle'] = optimal_values['max_hip_angle']['lower']
        df['lower_max_knee_angle'] = optimal_values['max_knee_angle']['lower']
        df['lower_touchdown_trunk_angle'] = optimal_values['touchdown_trunk_angle']['lower']
        df['lower_touchdown_knee_sep'] = optimal_values['touchdown_knee_sep']['lower']
        df['lower_touchdown_toe_velocity'] = optimal_values['touchdown_toe_velocity']['lower']

        # Add averages (legacy and dual-sided)
        df['avg_st_rt'] = avg_st_rt
        df['avg_gct'] = avg_gct
        df['avg_max_hip_angle'] = avg_max_hip_angle
        df['avg_max_knee_angle'] = avg_max_knee_angle
        df['avg_touchdown_trunk_angle'] = avg_touchdown_trunk_angle
        df['avg_touchdown_knee_sep'] = avg_touchdown_knee_sep
        df['avg_touchdown_toe_velocity'] = avg_touchdown_toe_velocity
        
        # Add left leg averages
        df['avg_st_rt_left'] = avg_st_rt_left
        df['avg_gct_left'] = avg_gct_left
        df['avg_max_hip_angle_left'] = avg_max_hip_angle_left
        df['avg_max_knee_angle_left'] = avg_max_knee_angle_left
        df['avg_touchdown_trunk_angle_left'] = avg_touchdown_trunk_angle_left
        df['avg_touchdown_knee_sep_left'] = avg_touchdown_knee_sep_left
        df['avg_touchdown_toe_velocity_left'] = avg_touchdown_toe_velocity_left
        
        # Add right leg averages
        df['avg_st_rt_right'] = avg_st_rt_right
        df['avg_gct_right'] = avg_gct_right
        df['avg_max_hip_angle_right'] = avg_max_hip_angle_right
        df['avg_max_knee_angle_right'] = avg_max_knee_angle_right
        df['avg_touchdown_trunk_angle_right'] = avg_touchdown_trunk_angle_right
        df['avg_touchdown_knee_sep_right'] = avg_touchdown_knee_sep_right
        df['avg_touchdown_toe_velocity_right'] = avg_touchdown_toe_velocity_right

        # Connect to the database and save analysis data
        try:
            db_path = join_paths(destination_folder, f'{fn}_video.db')
            print(f"Attempting to save to database at: {db_path}")
            
            conn = connect_to_db(db_path)
            
            # Print DataFrame info before saving
            print(f"DataFrame shape before saving: {df.shape}")
            
            # Save the data
            df.to_sql('analysis_data', conn, if_exists='replace', index=False)
            
            # Verify the save by reading back
            verify_df = pd.read_sql("SELECT * FROM analysis_data", conn)
            print(f"Verified data shape: {verify_df.shape}")
            
            # Commit the changes
            conn.commit()
            print("Data successfully saved to database")
            
        except Exception as e:
            print(f"Detailed error while saving analysis data: {str(e)}")  # More detailed error
            display_error(f"Error saving analysis data: {str(e)}")
            return
        finally:
            conn.close()
            
        # Generate sprint feature videos
        try:
            print("Starting sprint feature video generation...")
            from src.utils.video.generate_sprint_feature_videos import generate_all_sprint_feature_videos
            from src.db.sqlite.analysis_data.position_data_database import select_all_position_data
            from src.db.sqlite.analysis_data.analysis_data_database import get_db_connection
            
            # CRITICAL: MediaPipe processes *_zoom.mp4 and stores normalized coordinates (0-1) 
            # relative to the dimensions of *_zoom.mp4 (720x960 for Max Velocity Analysis)
            # MediaPipe then creates *_zoom_processed.mp4 with the SAME dimensions as the input
            # For cv2 animations to align correctly, we MUST use the video that MediaPipe processed
            # OR ensure the video we use has the exact same dimensions
            # 
            # We want to use *_zoom_processed.mp4 so MediaPipe overlays are visible
            # But we need to verify dimensions match what MediaPipe processed
            import glob
            import os
            import cv2
            
            # Strip _original from fn for video lookup (videos don't have _original suffix)
            fn_for_video = fn.replace('_original', '') if '_original' in fn else fn
            video_base, video_ext = os.path.splitext(fn_for_video)
            
            # If fn doesn't have extension, try common extensions
            if not video_ext:
                video_ext = '.mp4'
            
            print(f"DEBUG apsprint feature videos: Using fn_for_video={fn_for_video} (original fn={fn})")
            print(f"DEBUG apsprint feature videos: video_base={video_base}, video_ext={video_ext}")
            
            # First, check what video MediaPipe actually processed to get its dimensions
            zoom_video_path = join_paths(destination_folder, f"{video_base}_zoom{video_ext}")
            mediapipe_input_dimensions = None
            if path_exists(zoom_video_path):
                # Get dimensions of the video MediaPipe processed
                test_cap = cv2.VideoCapture(zoom_video_path)
                if test_cap.isOpened():
                    mediapipe_input_dimensions = (
                        int(test_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(test_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    )
                    test_cap.release()
                    print(f"MediaPipe processed video dimensions: {mediapipe_input_dimensions[0]}x{mediapipe_input_dimensions[1]}")
            
            # Try to find the zoom_processed video (MediaPipe output with overlays)
            # Check multiple extensions
            zoom_processed_paths = [
                join_paths(destination_folder, f"{video_base}_zoom_processed.mp4"),
                join_paths(destination_folder, f"{video_base}_zoom_processed.mov"),
                join_paths(destination_folder, f"{video_base}_zoom_processed.MP4"),
                join_paths(destination_folder, f"{video_base}_zoom_processed.MOV"),
            ]
            
            regular_processed_paths = [
                join_paths(destination_folder, f"{fn_for_video}_processed.mp4"),
                join_paths(destination_folder, f"{fn_for_video}_processed.mov"),
                join_paths(destination_folder, f"{fn_for_video}_processed.MP4"),
                join_paths(destination_folder, f"{fn_for_video}_processed.MOV"),
            ]
            
            processed_video = None
            processed_video_dimensions = None
            
            # Try zoom_processed video first (preferred - has MediaPipe overlays)
            for zoom_proc_path in zoom_processed_paths:
                if path_exists(zoom_proc_path):
                    # Verify dimensions match what MediaPipe processed
                    test_cap = cv2.VideoCapture(zoom_proc_path)
                    if test_cap.isOpened():
                        processed_video_dimensions = (
                            int(test_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            int(test_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        )
                        test_cap.release()
                        
                        if mediapipe_input_dimensions:
                            if processed_video_dimensions == mediapipe_input_dimensions:
                                processed_video = zoom_proc_path
                                print(f"Using zoom_processed video (MediaPipe output with overlays): {os.path.basename(processed_video)}")
                                print(f"  Dimensions match MediaPipe input: {processed_video_dimensions[0]}x{processed_video_dimensions[1]}")
                                break
                            else:
                                print(f"WARNING: zoom_processed dimensions {processed_video_dimensions} don't match MediaPipe input {mediapipe_input_dimensions}")
                        else:
                            # Use it anyway if we can't verify
                            processed_video = zoom_proc_path
                            print(f"Using zoom_processed video (MediaPipe output with overlays): {os.path.basename(processed_video)}")
                            print(f"  Dimensions: {processed_video_dimensions[0]}x{processed_video_dimensions[1]}")
                            break
            
            # If no zoom_processed, try regular processed
            if not processed_video:
                for reg_proc_path in regular_processed_paths:
                    if path_exists(reg_proc_path):
                        processed_video = reg_proc_path
                        print(f"Using regular processed video (MediaPipe output): {os.path.basename(processed_video)}")
                        break
            
            # Fallback: try to find any processed video file
            if not processed_video:
                video_patterns = [
                    join_paths(destination_folder, f"{video_base}*_processed.mp4"),
                    join_paths(destination_folder, f"{video_base}*_processed.mov"),
                ]
                matches = []
                for pattern in video_patterns:
                    matches.extend(glob.glob(pattern))
                
                # Exclude flying_10 and other generated videos
                matches = [m for m in matches if '_flying_10' not in m 
                          and '_temp_flying_10' not in m and '_trimmed_flying_10' not in m
                          and '_distance_speed' not in m]
                
                if matches:
                    processed_video = matches[0]
                    print(f"Using found processed video: {os.path.basename(processed_video)}")
                else:
                    processed_video = None
                    print(f"ERROR: Could not find processed video file for {fn}")
                    print(f"  Searched for: {video_base}*_processed.*")
                    print(f"  In directory: {destination_folder}")
            
            if processed_video and path_exists(processed_video):
                print(f"Video file exists: {processed_video}")
                
                # Get position data from database
                # Try multiple database file patterns to find the correct one
                # Use fn_for_db (which has _original stripped) for database lookup
                possible_db_paths = [
                    join_paths(destination_folder, f'{fn_for_db}_video.db'),
                    join_paths(destination_folder, f'{video_base}_video.db'),
                    join_paths(destination_folder, f'{fn_for_db}.db'),
                    join_paths(destination_folder, f'{video_base}.db'),
                ]
                
                # Also search for any _video.db file in the folder
                import glob
                all_db_files = glob.glob(join_paths(destination_folder, '*_video.db'))
                possible_db_paths.extend(all_db_files)
                
                db_path = None
                for possible_path in possible_db_paths:
                    if path_exists(possible_path):
                        db_path = possible_path
                        print(f"Found database file: {basename(db_path)}")
                        break
                
                if not db_path:
                    print(f"ERROR: Database file not found. Searched for:")
                    for path in possible_db_paths[:4]:  # Show first 4 patterns
                        print(f"  - {path}")
                    print("Cannot generate feature videos without position data.")
                else:
                    conn = get_db_connection(db_path)
                    position_data = select_all_position_data(conn)
                    
                    # Also get analysis_data to merge with position_data for feature videos
                    # This contains angle values, distances, velocities, etc.
                    try:
                        analysis_data = pd.read_sql("SELECT * FROM analysis_data", conn)
                        print(f"Found {len(analysis_data)} rows of analysis_data")
                    except Exception as e:
                        print(f"WARNING: Could not load analysis_data: {e}")
                        analysis_data = pd.DataFrame()
                    
                    conn.close()
                    
                    if position_data is None or len(position_data) == 0:
                        print(f"ERROR: No position data found in database: {db_path}")
                        print("Cannot generate feature videos without position data.")
                    else:
                        print(f"Found {len(position_data)} rows of position data")
                        
                        # Merge position_data with analysis_data on frame and time
                        if not analysis_data.empty and 'frame' in analysis_data.columns:
                            # Merge on frame and time if both exist
                            merge_keys = ['frame']
                            if 'time' in position_data.columns and 'time' in analysis_data.columns:
                                merge_keys.append('time')
                            
                            # Merge, keeping all position_data rows
                            merged_data = position_data.merge(
                                analysis_data,
                                on=merge_keys,
                                how='left',
                                suffixes=('', '_analysis')
                            )
                            print(f"Merged position_data with analysis_data: {len(merged_data)} rows")
                            # Use merged data for feature videos
                            data_for_videos = merged_data
                        else:
                            print("WARNING: No analysis_data available, using position_data only")
                            data_for_videos = position_data
                        
                        # Calculate height in inches
                        height = height_feet * 12 + height_inches
                        
                        # Generate feature videos
                        print(f"Generating feature videos using video: {os.path.basename(processed_video)}")
                        # Pass meters_per_pixel to feature video generation for distance calculations
                        feature_videos_dir = generate_all_sprint_feature_videos(
                            processed_video,
                            data_for_videos,  # Use merged data (position_data + analysis_data)
                            destination_folder,
                            fps,
                            height,
                            "Yellow",  # Default band color
                            meters_per_pixel=meters_per_pixel  # Pass for distance calculations
                        )
                        print(f"Sprint feature videos generated in: {feature_videos_dir}")
            else:
                print(f"ERROR: Video file not found or invalid: {processed_video}")
                print("Cannot generate feature videos without the input video.")
                
        except Exception as e:
            print(f"Error generating sprint feature videos: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")