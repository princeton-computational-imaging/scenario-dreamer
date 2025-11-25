import torch
import numpy as np
from collections import deque
import json
import matplotlib.pyplot as plt
import random

from shapely import affinity
from shapely.geometry import box, LineString
from shapely.geometry.base import CAP_STYLE
from utils.collision_helpers import compute_collision_states_one_scene
from utils.data_helpers import modify_agent_states
from scipy.spatial import distance

PI = torch.pi

class IDMPolicy:
    """IDM actor.

    Args:
        env: Environment.
        device (str): Device to put the actions on.
    """

    MIN_GAP_TO_LEAD = 1.0 # [m]
    MAX_LOOKAHEAD_DIST = 200.0 # [m] (Max distance to look for leading agent)
    

    def __init__(self, cfg, env):
        self.cfg = cfg
        self.env = env

        self.strict_lane_follow = True # If False, will return action as acceleration and steering
        self.dt = self.cfg.sim.dt 

        random.seed(cfg.sim.seed)

        self.scenario_dict = None

        # for aggregating metrics
        self.agent_active_all = []
        self.sim_lin_speeds = []
        self.gt_lin_speeds = []
        self.sim_ang_speeds = []
        self.gt_ang_speeds = []
        self.sim_accels = []
        self.gt_accels = []
        self.sim_dist_near_veh = [] 
        self.gt_dist_near_veh = []
        self.collision_rate_scenario = []

        self.has_collided = None
        self.has_activated = None
        self.t = 0


    def reset(self, obs):
        self.scenario_dict = self.env.scenario_dict
        self.num_all_agents = self.scenario_dict['agents'].shape[0]
        self.lane_graph_dict, self.lane_geometries = self._get_lane_data()
        self.ego_paths = {}
        self.actors_current_lane = {}
        self.t = 0


    def update_running_statistics(self, data_dict, scenario_dict, scene_complete=False):
        # scenario_dict: agents: [A, 91, 8] (last idx is ego)
        # data_dict: agent: [self.t, A, 8]: [pos_x, pos_y, vel_x, vel_y, heading, length, width, existence]
        if self.t == 0:
            self.has_collided = np.zeros(data_dict['agent_active'].shape[0]).astype(bool)
            self.has_activated = data_dict['agent_active']
        else:
            self.has_activated = np.logical_or(
                self.has_activated,
                data_dict['agent_active']
            )
        
        agent_active = data_dict['agent_active']
        self.agent_active_all.append(agent_active)
        sim_agents = np.array(data_dict['agent'])[self.t, agent_active]
        gt_agents = np.array(scenario_dict['agents'][agent_active, self.t])

        sim_vels = sim_agents[:, 2:4]
        gt_vels = gt_agents[:, 2:4]
        sim_lin_speeds = np.linalg.norm(sim_vels, axis=-1)
        gt_lin_speeds = np.linalg.norm(gt_vels, axis=-1)
        self.sim_lin_speeds.append(sim_lin_speeds)
        self.gt_lin_speeds.append(gt_lin_speeds)

        sim_ang_speeds = np.rad2deg(sim_agents[:, 4]) / 0.1
        gt_ang_speeds = np.rad2deg(gt_agents[:, 4]) / 0.1
        self.sim_ang_speeds.append(sim_ang_speeds)
        self.gt_ang_speeds.append(gt_ang_speeds)

        if self.t > 0:
            accel_mask = np.logical_and(
                self.agent_active_all[self.t],
                self.agent_active_all[self.t - 1]
            )
            
            sim_vels_all_t = np.array(data_dict['agent'])[self.t, :, 2:4]
            gt_vels_all_t = np.array(scenario_dict['agents'][:, self.t, 2:4])
            sim_vels_all_tminus1 = np.array(data_dict['agent'])[self.t-1, :, 2:4]
            gt_vels_all_tminus1 = np.array(scenario_dict['agents'][:, self.t-1, 2:4])

            sim_vels_t = sim_vels_all_t[accel_mask]
            gt_vels_t = gt_vels_all_t[accel_mask]
            sim_vels_tminus1 = sim_vels_all_tminus1[accel_mask]
            gt_vels_tminus1 = gt_vels_all_tminus1[accel_mask]

            sim_accels = np.linalg.norm((sim_vels_t - sim_vels_tminus1) / 0.1, axis=-1)
            gt_accels = np.linalg.norm((gt_vels_t - gt_vels_tminus1) / 0.1, axis=-1)

            self.gt_accels.append(gt_accels)
            self.sim_accels.append(sim_accels)
        
        if sim_agents.shape[0] > 1:
            sim_pos = sim_agents[:, :2]
            sim_pairwise_distances = np.linalg.norm(sim_pos[:, np.newaxis, :] - sim_pos[np.newaxis, :, :], axis=-1)
            np.fill_diagonal(sim_pairwise_distances, np.inf)
            sim_dist_near_veh = np.min(sim_pairwise_distances, axis=1)

            gt_pos = gt_agents[:, :2]
            gt_pairwise_distances = np.linalg.norm(gt_pos[:, np.newaxis, :] - gt_pos[np.newaxis, :, :], axis=-1)
            np.fill_diagonal(gt_pairwise_distances, np.inf)
            gt_dist_near_veh = np.min(gt_pairwise_distances, axis=1)

            self.sim_dist_near_veh.append(sim_dist_near_veh)
            self.gt_dist_near_veh.append(gt_dist_near_veh)

            sim_agents_modified = modify_agent_states(sim_agents)
            agents_colliding = compute_collision_states_one_scene(sim_agents_modified)
            active_agent_idxs = np.where(agent_active == 1)[0]
            colliding_all = np.zeros(len(agent_active)).astype(bool)
            for active_agent_idx, agent_colliding in zip(active_agent_idxs, agents_colliding):
                colliding_all[active_agent_idx] = agent_colliding
            
            self.has_collided = np.logical_or(
                self.has_collided,
                colliding_all
            )

        if scene_complete:
            if np.sum(self.has_activated) > 0:
                collision_rate = np.sum(self.has_collided) / np.sum(self.has_activated)
            else:
                collision_rate = 0.
            self.collision_rate_scenario.append(collision_rate)  
            self.agent_active_all = []   


    def compute_metrics(self):
        metrics_dict = {}

        metrics_dict['collision_rate'] = np.array(self.collision_rate_scenario).mean()

        # lin speed jsd 
        lin_speeds_gt = np.concatenate(self.gt_lin_speeds, axis=0)
        lin_speeds_sim = np.concatenate(self.sim_lin_speeds, axis=0)
        lin_speeds_gt = np.clip(lin_speeds_gt, 0, 30)
        lin_speeds_sim = np.clip(lin_speeds_sim, 0, 30)
        bin_edges = np.arange(201) * 0.5 * (100 / 30)
        P_lin_speeds_sim = np.histogram(lin_speeds_sim, bins=bin_edges)[0] / len(lin_speeds_sim)
        Q_lin_speeds_sim = np.histogram(lin_speeds_gt, bins=bin_edges)[0] / len(lin_speeds_gt)
        metrics_dict['lin_speed_jsd'] = distance.jensenshannon(P_lin_speeds_sim, Q_lin_speeds_sim) ** 2 # compute the divergence
        
        # ang speed jsd
        ang_speeds_gt = np.concatenate(self.gt_ang_speeds, axis=0)
        ang_speeds_sim = np.concatenate(self.sim_ang_speeds, axis=0)
        ang_speeds_gt = np.clip(ang_speeds_gt, -50, 50)
        ang_speeds_sim = np.clip(ang_speeds_sim, -50, 50)
        bin_edges = np.arange(201) * 0.5 - 50 
        P_ang_speeds_sim = np.histogram(ang_speeds_sim, bins=bin_edges)[0] / len(ang_speeds_sim)
        Q_ang_speeds_sim = np.histogram(ang_speeds_gt, bins=bin_edges)[0] / len(ang_speeds_gt)
        metrics_dict['ang_speed_jsd'] = distance.jensenshannon(P_ang_speeds_sim, Q_ang_speeds_sim) ** 2

        # accel jsd
        accels_gt = np.concatenate(self.gt_accels, axis=0)
        accels_sim = np.concatenate(self.sim_accels, axis=0)
        accels_gt = np.clip(accels_gt, -10, 10)
        accels_sim = np.clip(accels_sim, -10, 10)
        bin_edges = np.arange(201) * 0.1 - 10
        P_accels_sim = np.histogram(accels_sim, bins=bin_edges)[0] / len(accels_sim)
        Q_accels_sim = np.histogram(accels_gt, bins=bin_edges)[0] / len(accels_gt)
        metrics_dict['accel_jsd'] = distance.jensenshannon(P_accels_sim, Q_accels_sim) ** 2

        # nearest dist jsd
        nearest_dists_gt = np.concatenate(self.gt_dist_near_veh, axis=0)
        nearest_dists_sim = np.concatenate(self.sim_dist_near_veh, axis=0)
        nearest_dists_gt = np.clip(nearest_dists_gt, 0, 40)
        nearest_dists_sim = np.clip(nearest_dists_sim, 0, 40)
        bin_edges = np.arange(201) * 0.5 * (100 / 40)
        P_nearest_dists_sim = np.histogram(nearest_dists_sim, bins=bin_edges)[0] / len(nearest_dists_sim)
        Q_nearest_dists_sim = np.histogram(nearest_dists_gt, bins=bin_edges)[0] / len(nearest_dists_gt)
        metrics_dict['nearest_dist_jsd'] = distance.jensenshannon(P_nearest_dists_sim, Q_nearest_dists_sim) ** 2
        
        return metrics_dict, ["{}: {:.6f}".format(k,v) for (k,v) in metrics_dict.items()]


    def act(self, obs, is_planner=True):
        return self.select_action(obs, is_planner=is_planner)


    def select_action(self, obs, is_planner=True):
        self.t += 1

        agent_states = obs # Use absolute agent states (world coord. system, to match with lanes)
        self.controlled_agent_ids = [len(agent_states) - 1] # control the ego
        self.is_planner = is_planner

        # Update agent lane locations
        self._compute_all_agent_lanes(agent_states)
        
        if not self.strict_lane_follow:
            # Steering: follow lane path to goal
            steerings = self._get_steerings(agent_states)
            
            # Acceleration is determined by IDM
            accelerations = self._get_accelerations(agent_states)

            # Reshape actions
            if not self.is_planner:
                actions = np.zeros([agent_states.shape[0] - 1, 2])
                for idx in self.controlled_agent_ids:
                    actions[idx][0] = accelerations[idx]
                    actions[idx][1] = steerings[idx]
            else:
                actions = np.array([accelerations[self.controlled_agent_ids[0]], steerings[self.controlled_agent_ids[0]]])
        else:
            # Strictly following lanes means we do not need to compute steering, only the next position along a lane path
            next_x, next_y, next_theta, next_vel = self._get_next_states(agent_states)

            if not self.is_planner:
                actions = np.zeros([agent_states.shape[0] - 1, 4])
                for idx in self.controlled_agent_ids:
                    actions[idx][0] = next_x[idx]
                    actions[idx][1] = next_y[idx]
                    actions[idx][2] = next_theta[idx]
                    actions[idx][3] = next_vel[idx]
            else:
                ego_id = self.controlled_agent_ids[0]
                actions = np.array([next_x[ego_id], next_y[ego_id], next_theta[ego_id], next_vel[ego_id]])

        return actions
    
    
    def _plot_lanes(self, lane_dict, agent_states, filename="lane_graph", lane_to_label=None):
        """
        For debugging the lane representations
        """

        plt.figure(figsize=(10, 8))
        
        # Loop over lanes
        for lane_id, points in enumerate(lane_dict):
            points = points
            x = points[:, 0]
            y = points[:, 1]
            
            # Plot the lane points with different colors
            # NOTE: Not sure why I need to inverse x & y so the plots match the gpudrive renders???
            plt.plot(x, y, marker=',', label=f"Lane {lane_id}: {len(points)}")

            if lane_to_label is not None and lane_id in lane_to_label:
                plt.text(points[0][0], points[0][1], f"S{lane_id}", ha="center", va="center")
                # plt.text(points[-1][1], points[-1][0], f"E{lane_id}")
        
        for idx, state in enumerate(agent_states):
            plt.plot(state[0], state[1], marker='x', color='r')
            plt.text(state[0], state[1], str(idx))

            heading_length = state[5] / 2 + 1.5
            heading_angle_rad = state[4]
            vehicle_center = state[:2]
            line_end_x = vehicle_center[0] + heading_length * np.cos(heading_angle_rad)
            line_end_y = vehicle_center[1] + heading_length * np.sin(heading_angle_rad)
            plt.plot([vehicle_center[0], line_end_x], [vehicle_center[1], line_end_y], color='black', zorder=6, alpha=0.75, linewidth=0.25)

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f"Scene ?")
        plt.legend()
        plt.grid(True)
        
        fig_path = f"./scenario-dreamer-dev/movie_frames/{filename}"
        plt.savefig(fig_path, dpi=500)
        print(f"Saved fig: {fig_path}")
        plt.close()

    def _plot_ego_path(self, ego_path_polygon, agent_occupancies, ego_id):
        # Plot for debugging purposes #
        fig, ax = plt.subplots()
        for agent_id, agent_occupancy in enumerate(agent_occupancies):
            x_vals, y_vals = agent_occupancy.exterior.xy
            if x_vals[0] < -10000:
                continue
            ax.plot(x_vals, y_vals, color="red", linewidth=1)
            plt.text(x_vals[0], y_vals[0], f"{agent_id}", ha="center", va="center")

        x_vals, y_vals = ego_path_polygon.exterior.xy
        ax.plot(x_vals, y_vals, color="blue", linewidth=1)
        ax.fill(x_vals, y_vals, color="lightblue", alpha=0.5)
        fig_path = f"./output/ego_path_{ego_id}"
        plt.savefig(fig_path)
        plt.close()
        print(f"Saved fig: {fig_path}")


    def _get_lane_data(self):

        lane_geometries = self.scenario_dict['lanes'] # [L,50,2] tensor lanes
        lane_graph = self.scenario_dict['lane_graph'] # nx.DiGraph

        suc_pairs = {}
        for l in range(len(lane_geometries)):
            suc_pairs[l] = []
        
        for edge in lane_graph.edges():
            src_lane = edge[0]
            dst_lane = edge[1]

            suc_pairs[src_lane].append(dst_lane)

        lane_graph_dict = {}
        lane_graph_dict['suc_pairs'] = suc_pairs

        return lane_graph_dict, lane_geometries


    def _get_closest_lane_point_from_position(self, lane_points, position):
        min_dist = np.linalg.norm(lane_points - position, axis=1).min()
        lane_point_idx = np.linalg.norm(lane_points - position, axis=1).argmin().item()
        return lane_point_idx, min_dist


    def _get_closest_lane_from_position(self, position, lane_geometries, max_dist=None):
        lane_closest_point_dist = np.inf
        closest_lane_idx = None
        closest_lane_point_idx = None
        for lane_idx, lane_points in enumerate(lane_geometries):
            lane_point_idx, dist = self._get_closest_lane_point_from_position(lane_points, position)
            if dist < lane_closest_point_dist:
                lane_closest_point_dist = dist
                closest_lane_idx = lane_idx
                closest_lane_point_idx = lane_point_idx
        
        return closest_lane_idx, closest_lane_point_idx, lane_closest_point_dist
    

    def _compute_all_agent_lanes(self, agent_states):
        # Computing start lanes for all agents

        self.agents_by_lane_points = {}
        self.lane_points_by_agent = {}
        self.agents_by_lane = {}
        for agent_id in range(self.num_all_agents):
            start_lane, lane_point, _ = self._get_closest_lane_from_position(agent_states[agent_id, :2], self.lane_geometries)
            # In case many agents can be near the same point
            prev_value = self.agents_by_lane_points.get((start_lane, lane_point))
            if prev_value is None:
                self.agents_by_lane_points[(start_lane, lane_point)] = [agent_id]
            else:
                self.agents_by_lane_points[(start_lane, lane_point)].append(agent_id)

            prev_value = self.agents_by_lane.get(start_lane)
            if prev_value is None:
                self.agents_by_lane[start_lane] = [agent_id]
            else:
                self.agents_by_lane[start_lane].append(agent_id)

            self.lane_points_by_agent[agent_id] = (start_lane, lane_point)


    def _get_ego_path(self, agent_id):
        cached_path = self.ego_paths.get(agent_id)
        if cached_path is None:
            self._initialize_ego_path(agent_id)
        
        return self.ego_paths.get(agent_id)


    def _initialize_ego_path(self, actor_id):
        """NOTE here that ego paths are just the paths for each agent in the scene"""

        if self.is_planner:
            if 'route_lane_indices' in self.scenario_dict:
                ego_path = self.scenario_dict['route_lane_indices']
            else:
                # Follow prescribed path
                # First we need to convert the route points into route lanes for compatibility 
                #   (NOTE: For future version, we could follow these points directly)
                ego_route_points = self.scenario_dict['route']

                # Search all lanes
                start_lane, _, _ = self._get_closest_lane_from_position(ego_route_points[0], self.lane_geometries)
                end_lane, _, _ =  self._get_closest_lane_from_position(ego_route_points[-1], self.lane_geometries)
                ego_path = [start_lane]
                # print("Ego path:", ego_path)

                # Now only search in successor lanes to find sequence
                excluded_lanes = []
                for point in ego_route_points[1:]:
                    if len(ego_path) == 0:
                        # print("ego_path is empty: possible broken lane graph")
                        ego_path = [start_lane] # Revert to just using the start lane as a minimal path
                        break

                    curr_lane = ego_path[-1]
                    next_lanes = self.lane_graph_dict['suc_pairs'][curr_lane]
                    lane_candidates = [l for l in next_lanes if l not in excluded_lanes]
                    lane_candidates.append(curr_lane)
                    if end_lane in lane_candidates:
                        # Favor picking the known end lane
                        next_lane = end_lane
                        dist = None
                    else:
                        lane_id, _, dist = self._get_closest_lane_from_position(point, self.lane_geometries[lane_candidates])
                        next_lane = lane_candidates[lane_id]

                    # Correct path if too far off route (unless due to initial position variance)
                    if dist is not None and dist > 5.0 and next_lane != start_lane: 
                        # Wrong path chosen, go back and exclude this lane
                        ego_path.pop()
                        excluded_lanes.append(next_lane)
                        continue
                    
                    if next_lane != curr_lane:
                        ego_path.append(next_lane)
            start_lane = ego_path[0]
        else:
            # Get a random path from the start lane
            start_lane, _ = self.lane_points_by_agent[actor_id]
            ego_path = [start_lane]
            max_depth = 20
            while True:
                next_lanes = self.lane_graph_dict['suc_pairs'][ego_path[-1]]
                if len(next_lanes) == 0 or len(ego_path) >= max_depth:
                    break

                next_lane = random.choice(next_lanes)
                ego_path.append(next_lane)

        self.ego_paths[actor_id] = ego_path
        self.actors_current_lane[actor_id] = start_lane
    
    
    def _compute_agent_occupancies(self, agent_states):
        agent_occupancies = {}
        # Always iterate over the actual agent_states we have right now
        for agent_id in range(agent_states.shape[0]):
            x_pos, y_pos = agent_states[agent_id, 0], agent_states[agent_id, 1]
            half_length, half_width = agent_states[agent_id, 5] / 2, agent_states[agent_id, 6] / 2
            if half_length == 0.0:
                continue
            orientation = agent_states[agent_id, 4]
            agent_box = box(x_pos - half_length, y_pos - half_width,
                            x_pos + half_length, y_pos + half_width)
            agent_polygon = affinity.rotate(agent_box, orientation, "center", use_radians=True)
            agent_occupancies[agent_id] = agent_polygon

        return agent_occupancies


    def _compute_leading_agents_occ(self, agent_states):
        # Construct occupancy map from agent geometries and paths
        agent_occupancies = self._compute_agent_occupancies(agent_states)

        leading_agents = {}
        for ego_id in self.controlled_agent_ids:
            ego_path = self._get_ego_path(ego_id)
            current_lane = self.actors_current_lane[ego_id]
            ego_position = agent_states[ego_id, :2]
            ego_width = agent_states[ego_id, 6]
            ego_lane_point, _ = self._get_closest_lane_point_from_position(
                self.lane_geometries[current_lane], ego_position
            )

            # Build path points ahead
            path_to_go = []
            lane_idx = current_lane
            point_idx = ego_lane_point
            path_done = False
            cumulative_distance_searched = 0.0
            prev_point_position = self.lane_geometries[current_lane][point_idx]

            while (not path_done) and (cumulative_distance_searched < IDMPolicy.MAX_LOOKAHEAD_DIST):
                next_point_position, lane_idx, point_idx, path_done = self._get_next_path_position(
                    ego_path, lane_idx, point_idx
                )
                if not path_done:
                    path_to_go.append(next_point_position)

                cumulative_distance_searched += np.linalg.norm(next_point_position - prev_point_position)
                prev_point_position = next_point_position

            leading_agent = None

            # Need at least two points to define a sensible direction and corridor
            if len(path_to_go) > 1:
                # Forward direction along path
                ego_forward = path_to_go[0] - ego_position
                if np.linalg.norm(ego_forward) < 1e-6:
                    ego_forward = path_to_go[1] - path_to_go[0]
                ego_forward_norm = np.linalg.norm(ego_forward)
                if ego_forward_norm < 1e-6:
                    leading_agents[ego_id] = None
                    continue
                ego_forward = ego_forward / ego_forward_norm

                corridor_half_width = ego_width * 0.6
                ego_path_polygon = LineString(
                    [(p[0], p[1]) for p in path_to_go]
                ).buffer(corridor_half_width, cap_style=CAP_STYLE.square)

                min_longitudinal = IDMPolicy.MAX_LOOKAHEAD_DIST

                for agent_id, agent_occupancy in agent_occupancies.items():
                    # skip ego and inactive agents
                    if agent_id == ego_id or agent_states[agent_id, -1] == 0:
                        continue

                    if ego_path_polygon.intersects(agent_occupancy):
                        delta = agent_states[agent_id, :2] - ego_position

                        longitudinal = np.dot(delta, ego_forward)
                        if longitudinal <= 0.0:
                            # Behind or exactly side-by-side in projection
                            continue

                        total_dist_sq = np.dot(delta, delta)
                        lateral_sq = total_dist_sq - longitudinal**2
                        lateral = np.sqrt(max(lateral_sq, 0.0))

                        # Require agent to be roughly in our lane / close to centerline
                        if lateral > ego_width:
                            continue

                        if longitudinal < min_longitudinal:
                            min_longitudinal = longitudinal
                            leading_agent = agent_id

            leading_agents[ego_id] = leading_agent

        self.leading_agents = leading_agents


    def _get_accelerations(self, agent_states):

        # Update leading agents
        self._compute_leading_agents_occ(agent_states)

        accelerations = {}
        for ego_id in self.controlled_agent_ids:

            min_gap_to_lead = IDMPolicy.MIN_GAP_TO_LEAD  # [m]
            headway_time_to_lead = 1.5  # [s]
            max_accel = 2.00  # [m/s^2]
            max_deccel = 6.0  # [m/s^2]
            accel_exponent = 4.0  # Usual value
            target_velocity = 15.0  # [m/s]

            # Per-timestep variables
            ego_speed = np.linalg.norm(agent_states[ego_id, 2:4])
            ego_half_length = agent_states[ego_id, 5] / 2

            # Clip speed to speed limit (this can be caused by initial speeds being set higher)
            if ego_speed > (target_velocity + 2):
                new_vel_x = target_velocity * np.cos(agent_states[ego_id, 4])
                new_vel_y = target_velocity * np.sin(agent_states[ego_id, 4])
                agent_states[ego_id, 2] = new_vel_x
                agent_states[ego_id, 3] = new_vel_y
                ego_speed = np.linalg.norm(agent_states[ego_id, 2:4])

            leading_agent = self.leading_agents[ego_id]
            # Default values when there is no leading agent
            lead_distance = np.inf
            lead_speed = 0.0
            lead_half_length = 4.0 / 2 
            if leading_agent is not None:
                lead_distance = np.linalg.norm(agent_states[leading_agent, :2] - agent_states[ego_id, :2]) - ego_half_length
                lead_speed = np.linalg.norm(agent_states[leading_agent, 2:4])
                lead_half_length = agent_states[leading_agent, 5] / 2

            # IDM equations
            s_star = (
                min_gap_to_lead 
                + (ego_speed * headway_time_to_lead) 
                + ( (ego_speed * (ego_speed - lead_speed)) / (2 * (max_accel * max_deccel)**0.5) )
            )
            s_alpha = max(lead_distance - lead_half_length, min_gap_to_lead)  # clamp to avoid zero division

            acceleration = max_accel * ( 1 - (ego_speed / target_velocity)**accel_exponent - (s_star / s_alpha)**2 )
            accelerations[ego_id] = acceleration
        
        return accelerations


    def _get_steerings(self, agent_states):
        TARGET_REACHED_DISTANCE = 5.0

        steerings = {}
        for ego_id in self.controlled_agent_ids:
            ego_orientation = agent_states[ego_id, 4]
            ego_position = agent_states[ego_id, :2]
            ego_path = self._get_ego_path(ego_id)

            # Find next target position: go to closest point if not near yet, else go to next point
            # current_lane, current_lane_point = self.lane_points_by_agent[world_idx][ego_id]
            current_lane = self.actors_current_lane[ego_id]
            current_lane_point, _ = self._get_closest_lane_point_from_position(self.lane_geometries[current_lane], ego_position)
            current_point_position = self.lane_geometries[current_lane][current_lane_point]
            
            # Determine if the current (nearest) point has been passed
            while True:
                next_point_position, next_lane, next_point_idx, path_done = self._get_next_path_position(ego_path, self.actors_current_lane[ego_id], current_lane_point)
                self.actors_current_lane[ego_id] = next_lane # Update that the agent will be changing lanes
                distance_to_next_point = np.linalg.norm(ego_position - next_point_position)
                distance_to_current_point = np.linalg.norm(current_point_position - ego_position)
                distance_between_points = np.linalg.norm(current_point_position - next_point_position)
                current_point_passed = distance_to_next_point < distance_between_points

                if distance_to_current_point > TARGET_REACHED_DISTANCE and not current_point_passed:
                    # Continue aiming for nearest point as it has not been reached yet
                    target_position = current_point_position
                    break
                else:
                    # Consider target reached, go to next point along path
                    current_point_position = next_point_position

                    if path_done:
                        target_position = next_point_position
                        break
                    
                    current_lane_point = next_point_idx
            
            target_delta = (target_position - ego_position)
            angle_to_target = np.arctan2(target_delta[1], target_delta[0])
            heading_angle_to_target = angle_to_target - ego_orientation
            heading_angle_to_target = (heading_angle_to_target + PI) % (2 * PI) - PI

            steering = np.clip(heading_angle_to_target, -PI/2.1, PI/2.1) # ~85 deg max turning
            # steering = heading_angle_to_target

            steerings[ego_id] = steering
        
        return steerings

    def _get_next_states(self, agent_states):
        accelerations = self._get_accelerations(agent_states)

        next_x = {}
        next_y = {}
        next_theta = {}
        next_vel = {}
        for ego_id in self.controlled_agent_ids:
            ego_position = agent_states[ego_id, :2]
            ego_path = self._get_ego_path(ego_id)

            # Find next target position: go to closest point if not near yet, else go to next point
            # current_lane, current_lane_point = self.lane_points_by_agent[world_idx][ego_id]
            current_lane = self.actors_current_lane[ego_id]
            current_lane_point, _ = self._get_closest_lane_point_from_position(self.lane_geometries[current_lane], ego_position)
            
            if self.t == 1:
                # Approximate the ego position to the nearest point at the first timestep
                closest_point_position = self.lane_geometries[current_lane][current_lane_point]
                ego_position = closest_point_position

            # Compute what distance the agent has covered in the past step
            ego_speed = np.linalg.norm(agent_states[ego_id, 2:4])
            dist_covered = ego_speed * self.dt

            # Compute next velocity
            ego_accel = accelerations[ego_id]
            ego_next_vel = ego_speed + (ego_accel * self.dt)

            # Find which point on the path best corresponds to this distance
            cumul_dist = 0.0
            prev_point_position = ego_position
            while True:
                next_point_position, next_lane, next_point_idx, path_done = self._get_next_path_position(ego_path, self.actors_current_lane[ego_id], current_lane_point)
                self.actors_current_lane[ego_id] = next_lane # Update that the agent will be changing lanes
                distance_between_points = np.linalg.norm(prev_point_position - next_point_position)

                cumul_dist += distance_between_points
                if (cumul_dist - dist_covered) > 0.0:

                    # Interpolate to find position that is the most accurate
                    prev_cumul_dist = cumul_dist - distance_between_points
                    inter_point_ratio = (dist_covered - prev_cumul_dist) / distance_between_points
                    delta_pos = (next_point_position - prev_point_position) * inter_point_ratio

                    target_position = prev_point_position + delta_pos
                    break

                if path_done:
                    target_position = next_point_position
                    break

                current_lane_point = next_point_idx
                prev_point_position = next_point_position

            # Compute the heading as the angle between the target point and the previous point
            pos_delta = (target_position - prev_point_position)
            lane_angle = np.arctan2(pos_delta[1], pos_delta[0])
            ego_next_theta = (lane_angle + PI) % (2 * PI) - PI  # Corrected angle

            next_x[ego_id] = target_position[0]
            next_y[ego_id] = target_position[1]
            next_theta[ego_id] = ego_next_theta
            next_vel[ego_id] = ego_next_vel

        return next_x, next_y, next_theta, next_vel


    def _get_next_path_position(self, ego_path, current_lane, current_lane_point):
        ego_path_lane_idx = ego_path.index(current_lane)
        next_lane_point = current_lane_point + 1
        if len(self.lane_geometries[current_lane]) > next_lane_point:
            return self.lane_geometries[current_lane][next_lane_point], current_lane, next_lane_point, False
        else:
            # Lane ended, look at next lane
            next_lane_idx = ego_path_lane_idx + 1
            if len(ego_path) > next_lane_idx:
                next_lane = ego_path[next_lane_idx]
                return self.lane_geometries[next_lane][1], next_lane, 1, False # Returning second point, because first should be the same as the last point of previous lane
            else:
                # Entire path is done (no more lanes), just return last point position
                lane_point = len(self.lane_geometries[current_lane]) - 1
                return self.lane_geometries[current_lane][lane_point], current_lane, lane_point, True  # Flag path termination


    