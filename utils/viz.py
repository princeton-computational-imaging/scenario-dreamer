import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as transforms
import os
from utils.geometry import *
import math
from cfgs.config import LANE_CONNECTION_TYPES_WAYMO, LANE_CONNECTION_TYPES_NUPLAN
from moviepy.editor import ImageSequenceClip
import wandb

def plot_scene(
        agent_states, 
        road_points, 
        agent_types, 
        lane_types, 
        name, 
        save_dir, 
        return_fig=False,
        tile_occupancy=None,
        adaptive_limits=False,
        route=None):
    """Plots a scene with lanes and agents."""

    # Create a figure and axes
    fig, ax = plt.subplots()

    if adaptive_limits:
        x_min, x_max, y_min, y_max = np.inf, -np.inf, np.inf, -np.inf
        for tile_corners in tile_occupancy:
            x_min = min(x_min, tile_corners[:, 0].min())
            x_max = max(x_max, tile_corners[:, 0].max())
            y_min = min(y_min, tile_corners[:, 1].min())
            y_max = max(y_max, tile_corners[:, 1].max())
    else:
        x_max = 32 
        x_min = -32
        y_max = 32 
        y_min = -32

    x_range = x_max - x_min
    y_range = y_max - y_min
    scale_factor = max(x_range, y_range) / 64  # Scale based on 64m x 64m reference
    base_linewidth = 1.5 / scale_factor
    road_width = 20 / scale_factor
    scatter_size = 8 / (scale_factor ** 2)
    bbox_linewidth = 0.35 / scale_factor
    heading_linewidth = 0.3 / scale_factor
    route_linewidth = 1.5 / scale_factor

    ct = 0
    for i in range(len(road_points)):
        if lane_types is None: # Centerlines
            color = 'grey'
            linestyle='dashed'
            zorder = 2 
        elif lane_types[i] == 0: # Centerlines
            color = 'grey'
            linestyle='dashed'
            zorder = 2 
        elif lane_types[i] == 1: # Green traffic light lanes
            color = 'green'
            linestyle='dashed'
            zorder = 3 
        else:
            color = 'red'
            linestyle='dashed' # Red traffic light lanes
            zorder = 3
        
        lane = road_points[i, :, :2]
        ax.plot(lane[:, 0], lane[:, 1], color=color, linewidth=base_linewidth, linestyle=linestyle, zorder=zorder)
        ct += 1
        
        # Road width
        draw_road_width = False
        if lane_types is None: # only centerlines
            draw_road_width = True
        elif lane_types[i] == 0:
            draw_road_width = True
        
        if draw_road_width:
            ax.plot(lane[:, 0], lane[:, 1], color="lightgrey", linewidth=road_width, linestyle="solid", zorder=zorder-1)

        # Lane end points
        ax.scatter(lane[0, 0], lane[0, 1], color=color, s=scatter_size, zorder=zorder+1)
        ax.scatter(lane[-1, 0], lane[-1, 1], color=color, s=scatter_size, zorder=zorder+1)

        # Lane annotations (for debugging)
        # label_idx = len(lane) // 2
        # ax.annotate(i, (lane[label_idx, 0], lane[label_idx, 1]), zorder=20, fontsize=1)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')

    # Plot route
    if route is not None:
        ax.plot(route[:, 0], route[:, 1], color='red', linestyle='solid', zorder=5, linewidth=route_linewidth)

    alpha = 1.0
    edgecolor = 'black'
    for a in range(len(agent_states)):
        if agent_types[a] == 0: # Vehicles
            color = '#de5959' if (a == 0) else '#87b3e6' # Light red for ego agent, Light blue for other vehicles
        elif agent_types[a] == 1: # Pedestrians
            color = '#bea9f5' # Light purple
        elif agent_types[a] == 2: # Immobile objects
            color = 'green' if lane_types is None else 'grey' # green for waymo dataset (cyclist), grey for nuplan (static objects)
        else:
            color = 'grey'  # Default color if agent type is unrecognized
        
        # Draw bounding boxes
        length = agent_states[a, 5]
        width = agent_states[a, 6]
        bbox_x_min = agent_states[a, 0] - width / 2
        bbox_y_min = agent_states[a, 1] - length / 2
        rectangle = mpatches.FancyBboxPatch(
            (bbox_x_min, bbox_y_min),
            width, length,
            ec=edgecolor, fc=color,
            linewidth=bbox_linewidth, alpha=alpha,
            boxstyle=mpatches.BoxStyle("Round", pad=0.3),
            zorder=4
        )

        # Calculate rotation angle
        cos_theta = agent_states[a, 3]
        sin_theta = agent_states[a, 4]
        theta = np.arctan2(sin_theta, cos_theta)
        rotation = transforms.Affine2D().rotate_deg_around(
            agent_states[a, 0], agent_states[a, 1], np.degrees(theta) - 90
        ) + ax.transData

        # Apply rotation to the rectangle
        rectangle.set_transform(rotation)
        ax.add_patch(rectangle)

        if lane_types is None:
            plot_heading_line = True # plot heading ling for vehicles, pedestrians, and cyclists
        else:
            plot_heading_line = agent_types[a] in [0, 1]  # Only plot heading line for vehicles and pedestrians, but not static objects
        
        
        if plot_heading_line:
            # Draw heading line
            heading_length = length / 2 + 1.5
            vehicle_center = agent_states[a, :2]
            line_end_x = vehicle_center[0] + heading_length * math.cos(theta)
            line_end_y = vehicle_center[1] + heading_length * math.sin(theta)
            ax.plot(
                [vehicle_center[0], line_end_x],
                [vehicle_center[1], line_end_y],
                color='black',
                alpha=0.5,
                linewidth=heading_linewidth,
                zorder=5
            )

    # Create the save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    if return_fig:
        # Return the figure object for logging
        return fig
    else:
        plt.margins(0)  # Remove margins
        ax.margins(0)  # Ensure no margins in axes
        # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Fill the entire figure canvas
        
        # Save the figure without margins
        fig.savefig(
            os.path.join(save_dir, name),
            dpi=1000,
            bbox_inches='tight',
            pad_inches=0
        )
        plt.close(fig)
        return None


def plot_lane_graph(
        road_points, 
        lane_conn, 
        edge_index_lane_to_lane, 
        lane_conn_type, 
        name, 
        save_dir, 
        return_fig=False):
    """
    Plots a lane graph with road points and semantic connections between lanes."""
    
    # Create a figure and axes
    fig, ax = plt.subplots()

    ct = 0
    for i in range(len(road_points)):
        lane = road_points[i, :, :2]
        
        ax.plot(lane[:, 0], lane[:, 1], color='black', linewidth=1.5)
        ct += 1

        label_idx = len(lane) // 2
        ax.annotate(
            str(i),
            (lane[label_idx, 0], lane[label_idx, 1]),
            zorder=5,
            fontsize=5
        )

    for j in range(lane_conn.shape[0]):
        if lane_conn[j, lane_conn_type] == 1:
            src_idx = edge_index_lane_to_lane[0, j]
            dest_idx = edge_index_lane_to_lane[1, j]
            
            lane_src = road_points[src_idx, :, :2]
            lane_dest = road_points[dest_idx, :, :2]
            src_pos = lane_src[10, :2]
            dest_pos = lane_dest[10, :2]

            if lane_conn.shape[1] == 6:
                edge_color = 'purple'
                if lane_conn[j, 2] == 1:
                    edge_color = 'red'
                elif lane_conn[j, 3] == 1:
                    edge_color = 'green'
                elif lane_conn[j, 4] == 1:
                    edge_color = 'blue'
            else:
                if lane_conn[j, 1] == 1:
                    edge_color = 'red'
                elif lane_conn[j, 2] == 1:
                    edge_color = 'green'

            ax.arrow(
                src_pos[0], src_pos[1],
                dest_pos[0] - src_pos[0], dest_pos[1] - src_pos[1],
                length_includes_head=True,
                head_width=1, head_length=1,
                zorder=10, color=edge_color
            )

    # Adjust plot settings if needed
    ax.set_aspect('equal', adjustable='box')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    if return_fig:
        # Return the figure object for logging
        return fig
    else:
        # Save the figure to disk and close it
        fig.savefig(os.path.join(save_dir, name), dpi=1000)
        plt.close(fig)
        return None
    

def visualize_batch(num_samples, 
                    agent_samples, 
                    lane_samples, 
                    agent_types, 
                    lane_types,
                    lane_conn_samples, 
                    data, 
                    save_dir,
                    epoch,
                    batch_idx,
                    save_wandb=False,
                    visualize_lane_graph=False):
    """ Visualize samples from the batch."""

    if lane_conn_samples.shape[-1] == 4:
        LANE_CONNECTION_TYPES = LANE_CONNECTION_TYPES_NUPLAN
    else:
        LANE_CONNECTION_TYPES = LANE_CONNECTION_TYPES_WAYMO

    agent_samples = agent_samples.detach().cpu().numpy()
    lane_samples = lane_samples.detach().cpu().numpy()
    agent_types = agent_types.detach().cpu().numpy()
    if lane_types is not None:
        lane_types = lane_types.detach().cpu().numpy()
    lane_conn_samples = lane_conn_samples.detach().cpu().numpy()
    
    # pyg data structures for indexing
    lane_batch = data['lane'].batch
    lane_row = data['lane', 'to', 'lane'].edge_index[0]
    lane_conn_batch = lane_batch[lane_row]
    edge_index_l2l = data['lane', 'to', 'lane'].edge_index
    lane_conn_batch = lane_conn_batch.cpu().numpy()
    agent_batch = data['agent'].batch.cpu().numpy()
    lane_batch = data['lane'].batch.cpu().numpy()

    images_to_log = {}
    for i in range(num_samples):
        # plot the scene
        scene_i_agents = agent_samples[agent_batch == i]
        scene_i_lanes = lane_samples[lane_batch == i]
        scene_i_agent_types = agent_types[agent_batch == i]
        if lane_types is not None:
            scene_i_lane_types = lane_types[lane_batch == i]
        else:
            scene_i_lane_types = None
        fig = plot_scene(
            scene_i_agents, 
            scene_i_lanes, 
            scene_i_agent_types, 
            scene_i_lane_types,
            name=f'epoch_{epoch}_batch_{batch_idx}_sample_{i}.png', 
            save_dir=save_dir,
            return_fig=save_wandb)
        if save_wandb:
            images_to_log[f'scene_plot/epoch_{epoch}_batch_{batch_idx}_sample_{i}'] = wandb.Image(fig)
            plt.close(fig)

        # plot the lane graph for each edge type
        if visualize_lane_graph:
            scene_i_lane_conns = lane_conn_samples[lane_conn_batch == i]
            shift = np.where(lane_batch == i)[0].min()
            edge_index_i_l2l = edge_index_l2l[:, lane_conn_batch == i].cpu().numpy() - shift
            # {"none": 0, "pred": 1, "succ": 2, "self": 3} (no left/right connections)
            if lane_conn_samples.shape[-1] == 4:
                edge_type_list = [LANE_CONNECTION_TYPES['pred'], LANE_CONNECTION_TYPES['succ']]
            else:
                edge_type_list = [LANE_CONNECTION_TYPES['pred'], LANE_CONNECTION_TYPES['succ'],
                                LANE_CONNECTION_TYPES['left'], LANE_CONNECTION_TYPES['right']]

            for typ in edge_type_list:
                fig = plot_lane_graph(
                    scene_i_lanes, 
                    scene_i_lane_conns, 
                    edge_index_i_l2l, 
                    typ, 
                    name=f'epoch_{epoch}_batch_{batch_idx}_sample_{i}_lanegraph_{typ}.png', 
                    save_dir=save_dir,
                    return_fig=save_wandb)
                if save_wandb:
                    images_to_log[f'lane_graph/epoch_{epoch}_batch_{batch_idx}_sample_{i}_type_{typ}'] = wandb.Image(fig)
                    plt.close(fig)
    
    # Log all images at once
    if save_wandb:
        return images_to_log
    else:
        return None


def plot_k_disks_vocabulary(V, png_path, dpi=1000):
    plt.figure(figsize=(18, 3))
    plt.ylim(-0.25, 0.25)
    for state in V:
        plt.scatter(state[0], state[1], s=1, color='blue')
        arrow_length = 0.08  # Define length of arrows
        dx = arrow_length * np.cos(state[2])  # Change in x
        dy = arrow_length * np.sin(state[2])  # Change in y
        plt.plot([state[0], state[0] + dx], [state[1], state[1] + dy], linewidth=0.5, color="black")
    plt.savefig(png_path, dpi=dpi)
    plt.clf()


def render_state(
        agent_states, 
        agent_types, 
        route, 
        lanes, 
        lanes_mask, 
        t, 
        name, 
        movie_path='video_frames', 
        lightweight=False
    ):
    """ Renders the current state of the simulation and saves it as a PNG image."""
    png_dir = f'{movie_path}/{name}'
    if not os.path.exists(png_dir):
        os.makedirs(png_dir, exist_ok=True)

    agent_alpha = 1.0
    agent_zord = 4
    ego_color = '#de5959'
    ego_alpha = 1.0
    ego_zord = 5

    x_min, y_min, x_max, y_max = -75, -75, 75, 75

    fig, ax = plt.subplots()
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')

    lanes = np.concatenate([lanes, lanes_mask[:, :, None]], axis=-1)

    # Plot lanes only
    color = 'grey'
    linestyle = 'dashed'
    zorder = 2
    for lane in lanes:
        exists = lane[:, 2] == 1
        plt.plot(
            lane[:, 0][exists], 
            lane[:, 1][exists], 
            color=color, 
            linewidth=1.5, 
            linestyle=linestyle, 
            zorder=zorder
        )
        plt.plot(
            lane[:, 0][exists], 
            lane[:, 1][exists], 
            color='lightgrey', 
            linewidth=20, 
            linestyle='solid', 
            zorder=zorder-1
        )
        if exists[0]:
            plt.scatter(
                lane[0, 0], 
                lane[0, 1], 
                color=color, 
                s=8, 
                zorder=zorder+1
            )
        if exists[-1]:
            plt.scatter(
                lane[-1, 0], 
                lane[-1, 1], 
                color=color, 
                s=8, 
                zorder=zorder+1
            )

    agent_types = np.argmax(agent_types, axis=1)
    
    # Plot agent bounding boxes and headings
    for a in range(len(agent_states)):
        if agent_states[a, -1] == 0:
            continue

        edgecolor = 'black'
        if a == len(agent_states) - 1:
            color = ego_color 
            alpha = ego_alpha 
            zord = ego_zord
        else:
            alpha = agent_alpha 
            zord = agent_zord
            
            if agent_types[a] == 1:
                color = '#87b3e6' # Light blue
            elif agent_types[a] == 2:
                color = '#bea9f5' # Light purple
            elif agent_types[a] == 3:
                color = 'grey'
            else:
                color = "grey"

        # Draw bounding boxes
        length = agent_states[a, 5] * 0.8
        width = agent_states[a, 6] * 0.8
        bbox_x_min = agent_states[a, 0] - width / 2
        bbox_y_min = agent_states[a, 1] - length / 2
        lw = 0.35 / ((x_max - x_min) / 140)
        rectangle = mpatches.FancyBboxPatch(
            (bbox_x_min, bbox_y_min), 
            width, 
            length, 
            ec=edgecolor, 
            fc=color,
            linewidth=lw, 
            alpha=alpha, 
            boxstyle=mpatches.BoxStyle("Round", pad=0.3), 
            zorder=zord
        )
        
        tr = transforms.Affine2D().rotate_deg_around(
            agent_states[a, 0], 
            agent_states[a, 1], 
            np.degrees(agent_states[a, 4]) - 90
        ) + ax.transData
        rectangle.set_transform(tr)
        ax.add_patch(rectangle)
        
        # Draw heading line
        if agent_types[a] in [1, 2]:
            heading_length = length / 2 + 1.5
            heading_angle_rad = agent_states[a, 4]
            vehicle_center = agent_states[a, :2]
            line_end_x = (vehicle_center[0] + 
                          heading_length * math.cos(heading_angle_rad))
            line_end_y = (vehicle_center[1] + 
                          heading_length * math.sin(heading_angle_rad))
            ax.plot(
                [vehicle_center[0], line_end_x], 
                [vehicle_center[1], line_end_y], 
                color='black', 
                zorder=zord+1, 
                alpha=0.25, 
                linewidth=0.3 / ((x_max - x_min) / 140))
    
    # for debugging
    # ax.annotate(a, (vehicle_center[0], vehicle_center[1]), zorder=8, fontsize=5) 
    
    if route is not None:
        plt.scatter(
            route[:, 0], 
            route[:, 1], 
            color=ego_color, 
            zorder=ego_zord, 
            s=8
        )
    plt.tight_layout()
    dpi = 100 if lightweight else 500
    plt.savefig(f'{png_dir}/frame_{t:03}.png', dpi=dpi)
    plt.close(fig)


def generate_video(name, output_dir, delete_images=False):
    """ Generates a video from a sequence of images saved in a directory."""
    image_folder = f'{output_dir}/{name}'
    
    # Get list of all image files in the directory
    images = [os.path.join(image_folder, img) for img in sorted(os.listdir(image_folder)) if img.endswith(".png")]
    images = [str1.replace('\n', '') for str1 in images]
    images.sort()  # Sort by filename

    # Create a video clip from the image sequence
    clip = ImageSequenceClip(images, fps=20)
    
    # Write the video file
    clip.write_videofile(f"{image_folder}.mp4", codec='libx264')

    if delete_images:
        for image in images:
            os.remove(image)