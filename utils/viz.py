import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as transforms
import os
from utils.geometry import *
import math
from cfgs.config import LANE_CONNECTION_TYPES_WAYMO, LANE_CONNECTION_TYPES_NUPLAN
import wandb

def plot_scene(
        agent_states, 
        road_points, 
        agent_types, 
        lane_types, 
        name, 
        save_dir, 
        return_fig=False):
    """Plots a scene with lanes and agents."""

    # Create a figure and axes
    fig, ax = plt.subplots()
    
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
        ax.plot(lane[:, 0], lane[:, 1], color=color, linewidth=1.5, linestyle=linestyle, zorder=zorder)
        ct += 1
        
        # Road width
        draw_road_width = False
        if lane_types is None: # only centerlines
            draw_road_width = True
        elif lane_types[i] == 0:
            draw_road_width = True
        
        if draw_road_width:
            ax.plot(lane[:, 0], lane[:, 1], color="lightgrey", linewidth=20, linestyle="solid", zorder=zorder-1)

        # Lane end points
        ax.scatter(lane[0, 0], lane[0, 1], color=color, s=8, zorder=zorder+1)
        ax.scatter(lane[-1, 0], lane[-1, 1], color=color, s=8, zorder=zorder+1)

        # Lane annotations (for debugging)
        # label_idx = len(lane) // 2
        # ax.annotate(i, (lane[label_idx, 0], lane[label_idx, 1]), zorder=5, fontsize=5)

    x_max = 32 
    x_min = -32
    y_max = 32 
    y_min = -32

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')

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
        lw = (0.35) / ((x_max - x_min) / 140)
        rectangle = mpatches.FancyBboxPatch(
            (bbox_x_min, bbox_y_min),
            width, length,
            ec=edgecolor, fc=color,
            linewidth=lw, alpha=alpha,
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
                linewidth=0.3 / ((x_max - x_min) / 140),
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