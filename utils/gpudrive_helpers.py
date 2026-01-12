""" Helper functions and classes for GPUDrive integration. 
Many of these functions are Python versions of C++ code in GPUDrive. 

This file syncs with the following GPUDrive source files:
- Constants: gpudrive/src/init.hpp, gpudrive/src/consts.hpp, gpudrive/gpudrive/env/constants.py
- Types and Enums: gpudrive/src/types.hpp
- JSON Serialization: gpudrive/src/json_serialization.hpp
- Map Reading: gpudrive/src/MapReader.cpp, gpudrive/src/MapReader.hpp
- Road Creation: gpudrive/src/level_gen.cpp
- Network Architecture: gpudrive/gpudrive/networks/late_fusion.py
- Observation Processing: gpudrive/gpudrive/env/env_torch.py, gpudrive/gpudrive/datatypes/observation.py
"""

import numpy as np
import torch
from torch import nn
from torch.distributions.utils import logits_to_probs
import math
from enum import Enum
from typing import List, Tuple, Union
import heapq

from itertools import product
from utils.geometry import normalize_agents, normalize_lanes

# Constants from gpudrive/src/init.hpp
MAX_OBJECTS = 515
MAX_ROADS = 956
MAX_POSITIONS = 91
MAX_GEOMETRY = 1746

# Constants from gpudrive/src/consts.hpp
TOP_K_ROAD_POINTS = 200  # kMaxAgentMapObservationsCount

# Constants from gpudrive/gpudrive/env/constants.py
EGO_FEAT_DIM = 6
PARTNER_FEAT_DIM = 6
ROAD_GRAPH_FEAT_DIM = 13
ROUTE_FEAT_DIM = 61  # 30 route points * 2 coords + 1 numPoints
MAX_SPEED = 100
MAX_VEH_LEN = 30
MAX_VEH_WIDTH = 15
MAX_VEH_HEIGHT = 10
MAX_ORIENTATION_RAD = 2 * np.pi
MIN_REL_GOAL_COORD = -1000
MAX_REL_GOAL_COORD = 1000
# Road graph constants from gpudrive/gpudrive/env/constants.py
MIN_RG_COORD = -1000
MAX_RG_COORD = 1000
MAX_ROAD_LINE_SEGMENT_LEN = 100
MAX_ROAD_SCALE = 100
AGENT_SCALE = 0.7


# EntityType enum from gpudrive/src/types.hpp
# Note: Python doesn't allow 'None' as an enum member, so we use 'NoneType'
# which corresponds to EntityType::None in the C++ code
class EntityType(Enum):
    """ Entity types. Synced with gpudrive/src/types.hpp """
    NoneType   = 0  # Corresponds to EntityType::None in C++
    RoadEdge   = 1
    RoadLine   = 2
    RoadLane   = 3
    CrossWalk  = 4
    SpeedBump  = 5
    StopSign   = 6
    Vehicle    = 7
    Pedestrian = 8
    Cyclist    = 9
    Padding    = 10
    NumTypes   = 11


# MapType enum from gpudrive/src/types.hpp
class MapType(Enum):
    """ Map element types. Synced with gpudrive/src/types.hpp """
    LANE_UNDEFINED = 0
    LANE_FREEWAY = 1
    LANE_SURFACE_STREET = 2
    LANE_BIKE_LANE = 3
    # Original definition skips 4
    ROAD_LINE_UNKNOWN = 5
    ROAD_LINE_BROKEN_SINGLE_WHITE = 6
    ROAD_LINE_SOLID_SINGLE_WHITE = 7
    ROAD_LINE_SOLID_DOUBLE_WHITE = 8
    ROAD_LINE_BROKEN_SINGLE_YELLOW = 9
    ROAD_LINE_BROKEN_DOUBLE_YELLOW = 10
    ROAD_LINE_SOLID_SINGLE_YELLOW = 11
    ROAD_LINE_SOLID_DOUBLE_YELLOW = 12
    ROAD_LINE_PASSING_DOUBLE_YELLOW = 13
    ROAD_EDGE_UNKNOWN = 14
    ROAD_EDGE_BOUNDARY = 15
    ROAD_EDGE_MEDIAN = 16
    STOP_SIGN = 17
    CROSSWALK = 18
    SPEED_BUMP = 19
    DRIVEWAY = 20  # New datatype in v1.2.0: Driveway entrances
    UNKNOWN = -1
    NUM_TYPES = 21


# MapVector2 struct from gpudrive/src/init.hpp
class MapVector2:
    """ 2D vector class. Synced with gpudrive/src/init.hpp """
    def __init__(self, x: float = 0.0, y: float = 0.0):
        self.x = x
        self.y = y


class MapVector3:
    """ 3D vector class. """
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z


# VehicleSize struct from gpudrive/src/types.hpp
class VehicleSize:
    """ Vehicle size struct. Synced with gpudrive/src/types.hpp """
    def __init__(self, length: float = 0.0, width: float = 0.0, height: float = 0.0):
        self.length = length
        self.width = width
        self.height = height


# MetaData struct from gpudrive/src/types.hpp
class MetaData:
    """ Metadata struct for objects. Synced with gpudrive/src/types.hpp """
    def __init__(self):
        self.isSdc = 0
        self.isObjectOfInterest = 0
        self.isTrackToPredict = 0
        self.difficulty = 0


# MapObject struct from gpudrive/src/init.hpp
class MapObject:
    """ GPUDrive object class. Synced with gpudrive/src/init.hpp """
    def __init__(self):
        self.position: List[MapVector2] = [MapVector2() for _ in range(MAX_POSITIONS)]
        self.vehicle_size = VehicleSize()
        self.heading: List[float] = [0.0 for _ in range(MAX_POSITIONS)]
        self.velocity: List[MapVector2] = [MapVector2() for _ in range(MAX_POSITIONS)]
        self.valid: List[bool] = [False for _ in range(MAX_POSITIONS)]
        self.goalPosition = MapVector2()
        self.type = EntityType.NoneType  # Corresponds to EntityType::None in C++
        self.metadata = MetaData()
        self.numPositions = 0
        self.numHeadings = 0
        self.numVelocities = 0
        self.numValid = 0
        self.id = 0
        self.mean = MapVector2()
        self.markAsExpert = False


# MapRoad struct from gpudrive/src/init.hpp
class MapRoad:
    """ GPUDrive road class. Synced with gpudrive/src/init.hpp """
    def __init__(self):
        self.geometry: List[MapVector2] = [MapVector2() for _ in range(MAX_GEOMETRY)]
        self.id = 0
        self.mapType = MapType.UNKNOWN
        self.type = EntityType.NoneType  # Corresponds to EntityType::None in C++
        self.numPoints = 0
        self.mean = MapVector2()


# Map struct from gpudrive/src/init.hpp
class Map:
    """ GPUDrive map class (roads and agents). Synced with gpudrive/src/init.hpp """
    def __init__(self):
        self.objects: List[MapObject] = [MapObject() for _ in range(MAX_OBJECTS)]
        self.roads: List[MapRoad] = [MapRoad() for _ in range(MAX_ROADS)]
        self.numObjects = 0
        self.numRoads = 0
        self.numRoadSegments = 0
        self.mean = MapVector2()
        self.mapName = ""  # Added mapName field (char[32] in C++)
        self.scenarioId = ""  # Added scenarioId field (char[32] in C++)
        self.route: List[MapVector2] = [MapVector2() for _ in range(1000)]  # Added route storage
        self.numRoutePoints = 0  # Added numRoutePoints field


def distance_2d(p1: MapVector2, p2: MapVector2) -> float:
    """Compute 2D distance between two points."""
    return math.hypot(p2.x - p1.x, p2.y - p1.y)


# Observation processing functions synced with gpudrive/gpudrive/env/env_torch.py
def get_ego_state(ego_state):
    """ Get ego state into format compatible with RL planners used in GPUDrive.
    Synced with gpudrive/gpudrive/env/env_torch.py _get_ego_state method.
    
    Args:
        ego_state: Array with structure [pos_x, pos_y, vel_x, vel_y, heading, length, width, existence]
                  - ego_state[2:4] contains vel_x and vel_y
                  - ego_state[5:7] contains length and width
    
    Returns:
        np.array: Shape (1, 6) containing:
            [0] speed (normalized, computed from vel_x and vel_y)
            [1] vehicle_length (normalized)
            [2] vehicle_width (normalized)
            [3] rel_goal_x (set to 0, matching gpudrive implementation)
            [4] rel_goal_y (set to 0, matching gpudrive implementation)
            [5] is_collided (set to 0, matching gpudrive implementation)
    """
    # Convert ego_state to float32 (creates a copy if needed, doesn't modify original)
    ego_state_f32 = np.asarray(ego_state, dtype=np.float32)
    
    # Convert constants to float32 for all calculations
    max_speed_f32 = np.float32(MAX_SPEED)
    agent_scale_f32 = np.float32(AGENT_SCALE)
    max_veh_len_f32 = np.float32(MAX_VEH_LEN)
    max_veh_width_f32 = np.float32(MAX_VEH_WIDTH)
    
    gpudrive_ego_state = np.zeros((1, 6), dtype=np.float32)
    
    # speed - computed from vel_x (ego_state[2]) and vel_y (ego_state[3])
    # All operations in float32
    vel_xy = ego_state_f32[2:4]
    speed = np.float32(np.linalg.norm(vel_xy))
    gpudrive_ego_state[0, 0] = speed / max_speed_f32
    
    # length - from ego_state[5]
    # All operations in float32
    length = ego_state_f32[5]
    gpudrive_ego_state[0, 1] = (length * agent_scale_f32) / max_veh_len_f32
    
    # width - from ego_state[6]
    # All operations in float32
    width = ego_state_f32[6]
    gpudrive_ego_state[0, 2] = (width * agent_scale_f32) / max_veh_width_f32
    
    # Elements [3], [4], [5] remain 0 (rel_goal_x, rel_goal_y, is_collided)
    # This matches gpudrive/gpudrive/env/env_torch.py _get_ego_state implementation
    return gpudrive_ego_state


def get_partner_obs(
        agents,
        ego_state,
        agent_active,
        num_partners=63,
        observation_radius=32,
    ):
    """ Get partner observations into format compatible with RL planners used in GPUDrive.
    Synced with gpudrive/gpudrive/env/env_torch.py _get_partner_obs method and
    gpudrive/src/sim.cpp collectPartnerObsSystem.
    
    This implementation matches gpudrive's behavior:
    - Partners are processed in the order they appear in the agents array (no sorting by distance)
    - Only active agents within observation_radius are included
    - Remaining slots are filled with zeros
    
    Args:
        agents: Array of agent states with structure [pos_x, pos_y, vel_x, vel_y, heading, length, width, existence]
                Shape: (num_agents, 8). Should NOT include ego vehicle.
        ego_state: Ego vehicle state [pos_x, pos_y, vel_x, vel_y, heading, length, width, existence]
        agent_active: Boolean array indicating which agents are active
        num_partners: Maximum number of partner observations to return (default 31)
        observation_radius: Maximum distance from ego to include partner (default inf, matching gpudrive
                          when observationRadius is not set). Set to a finite value to filter by distance.
    
    Returns:
        np.array: Shape (1, num_partners * 6) containing flattened partner observations.
                 Each partner observation has 6 features:
                 [0] speed (normalized, computed from vel_x and vel_y)
                 [1] rel_pos_x (normalized relative x position)
                 [2] rel_pos_y (normalized relative y position)
                 [3] orientation (normalized relative heading)
                 [4] vehicle_length (normalized)
                 [5] vehicle_width (normalized)
                 
                 Partners are processed in order (not sorted by distance), matching gpudrive behavior.
                 Matches gpudrive/gpudrive/env/env_torch.py _get_partner_obs output format.
    """
    local_frame = {
        'center': ego_state[:2],
        'yaw': ego_state[4]
    }
    
    # Normalize agents to ego-centric coordinate frame
    # After normalization: [rel_x, rel_y, rel_vel_x, rel_vel_y, rel_heading, length, width, ...]
    normalized_agents = normalize_agents(
        agents[:, None, :], 
        normalize_dict=local_frame,
        offset=0.
    )[:, 0]

    partner_obs = np.zeros((num_partners, 6), dtype=np.float32)
    arr_index = 0
    
    # Process agents in order (matching gpudrive/src/sim.cpp collectPartnerObsSystem)
    # No sorting by distance - just iterate through in order
    for agent_id in range(len(agents)):
        # Skip inactive agents
        if not agent_active[agent_id]:
            continue
        
        # Check if within observation radius (matching gpudrive line 222)
        normalized_agent = normalized_agents[agent_id]
        rel_pos = normalized_agent[:2]  # rel_x, rel_y
        dist = np.linalg.norm(rel_pos)
        
        if dist > observation_radius:
            continue
        
        # Stop if we've filled all partner slots
        if arr_index >= num_partners:
            break
        
        # Extract relative heading (normalized_agent[4] after coordinate transformation)
        head = normalized_agent[4]
        
        # Normalize relative positions to [-1, 1] range
        rel_x = _normalize_min_max(
            normalized_agent[0],  # rel_x after normalization
            MIN_REL_GOAL_COORD, 
            MAX_REL_GOAL_COORD
        )
        rel_y = _normalize_min_max(
            normalized_agent[1],  # rel_y after normalization
            MIN_REL_GOAL_COORD, 
            MAX_REL_GOAL_COORD
        )
        
        # Build partner observation matching gpudrive format:
        # [speed, rel_pos_x, rel_pos_y, orientation, vehicle_length, vehicle_width]
        partner = np.array([
            np.linalg.norm(normalized_agent[2:4]) / MAX_SPEED,  # speed from vel_x, vel_y
            rel_x,  # normalized relative x position
            rel_y,  # normalized relative y position
            head / MAX_ORIENTATION_RAD,  # normalized relative heading
            normalized_agent[5] * AGENT_SCALE / MAX_VEH_LEN,  # normalized length
            normalized_agent[6] * AGENT_SCALE / MAX_VEH_WIDTH  # normalized width
        ])

        partner_obs[arr_index] = partner
        arr_index += 1
    
    # Remaining slots are already zeros (matching gpudrive line 235-237)
    return partner_obs.flatten()[None,:].astype(np.float32)


def get_map_obs(
        lanes,
        ego_state,
        observation_radius=32.0,
        max_num_lanes=TOP_K_ROAD_POINTS,
        num_lane_features=ROAD_GRAPH_FEAT_DIM,
    ):
    """ Get map observations into format compatible with RL planners used in GPUDrive.
    Synced with gpudrive/gpudrive/env/env_torch.py _get_road_map_obs method and
    gpudrive/src/sim.cpp collectMapObservationsSystem.
    
    This implementation matches gpudrive's KNearestEntitiesWithRadiusFiltering algorithm:
    - Filters out road edges (EntityType.RoadEdge)
    - Selects K nearest road entities using heap-based KNN (K=200, not sorted by distance)
    - Filters by observation_radius
    - Normalizes using gpudrive constants
    
    Note: The final array is NOT sorted by distance (matches heap order from simulator).
    
    Args:
        lanes: Array of lane/road states with structure [pos_x, pos_y, segment_length, segment_width, 
               segment_height, orientation, type, ...]. Shape: (num_lanes, num_features)
               The type field (index 6) should contain EntityType enum values (road edges will be filtered).
        ego_state: Ego vehicle state [pos_x, pos_y, vel_x, vel_y, heading, length, width, existence]
        observation_radius: Maximum distance from ego to include road points (default inf, matching gpudrive
                          when observationRadius is not set). Set to a finite value to filter by distance.
        max_num_lanes: Maximum number of road observations to return (default TOP_K_ROAD_POINTS = 200,
                      matching kMaxAgentMapObservationsCount)
        num_lane_features: Number of features per lane after one-hot encoding (default ROAD_GRAPH_FEAT_DIM = 13)
    
    Returns:
        np.array: Shape (1, max_num_lanes * num_lane_features) containing flattened road map observations.
                 Each road observation has 13 features:
                 [0] x (normalized relative x position)
                 [1] y (normalized relative y position)
                 [2] segment_length (normalized)
                 [3] segment_width (normalized)
                 [4] segment_height (normalized)
                 [5] orientation (normalized relative heading)
                 [6-12] type_one_hot (7 classes: None, RoadLine, RoadEdge, RoadLane, CrossWalk, SpeedBump, StopSign)
                 
                 Matches gpudrive/gpudrive/env/env_torch.py _get_road_map_obs output format.
    """
    local_frame = {
        'center': ego_state[:2],
        'yaw': ego_state[4]
    }
    # Filter out road edges (matching gpudrive/src/knn.hpp line 116)
    # Type is one-hot encoded at indices 6-12
    # RoadEdge (EntityType.RoadEdge = 1) is at index 7 in the one-hot encoding
    road_edge_one_hot_idx = 6 + EntityType.RoadEdge.value  # 6 + 1 = 7
    # Keep only non-road-edge entities (where RoadEdge one-hot is 0)
    non_edge_mask = (lanes[:, road_edge_one_hot_idx] == 0.0)
    lanes_filtered = lanes[non_edge_mask]
    
    if len(lanes_filtered) == 0:
        # Return zero-padded observation
        map_tensor = np.zeros((max_num_lanes, num_lane_features))
        return map_tensor.flatten()[None, :]
    
    # Calculate distances to ego (in global coordinates before normalization)
    dist_to_ego = np.linalg.norm(
        lanes_filtered[:, :2] - ego_state[None, :2], axis=-1)
    
    # Use heap-based KNN selection (matching gpudrive/src/knn.hpp exactly)
    # The C++ code: 1) fills first K, 2) makes heap, 3) processes remaining with pop/push, 4) radiusFilter swaps
    # This creates a specific order that's not pure heap order due to radiusFilter swaps
    
    if len(lanes_filtered) <= max_num_lanes:
        # If we have fewer than K roads, just take all of them (after radius filtering)
        radius_mask = dist_to_ego <= observation_radius
        valid_indices = np.where(radius_mask)[0]
    else:
        # Step 1: Fill first K non-edge roads (matching C++ lines 114-129)
        heap_items = []  # List of (neg_dist, idx) tuples
        last_processed_idx = 0
        
        for idx, dist in enumerate(dist_to_ego):
            if len(heap_items) >= max_num_lanes:
                last_processed_idx = idx
                break
            # Note: radius filtering happens later in C++, so we add all here
            heap_items.append((-dist, idx))  # Negative for max-heap behavior
        
        # Step 2: Make heap (matching C++ line 137)
        # Convert to list of tuples and heapify
        heapq.heapify(heap_items)
        
        # Step 3: Process remaining roads (matching C++ lines 139-165)
        for idx in range(last_processed_idx, len(dist_to_ego)):
            dist = dist_to_ego[idx]
            
            # Check if current is closer than farthest in heap
            farthest_dist = -heap_items[0][0]  # Negate back to get actual distance
            if dist < farthest_dist:
                # Pop farthest, push current (matching C++ pop_heap/push_heap)
                heapq.heapreplace(heap_items, (-dist, idx))
        
        # Step 4: Apply radiusFilter (matching C++ lines 170, 83-96 exactly)
        # radiusFilter swaps elements beyond radius with elements from the end
        # This is the key to matching the exact order!
        heap_array = np.array([idx for _, idx in heap_items])
        heap_dists = dist_to_ego[heap_array]  # Get distances for indices in heap order
        
        # Implement radiusFilter exactly as in C++:
        # It iterates through the array and swaps invalid elements to the end
        new_beyond = len(heap_array)
        idx = 0
        while idx < new_beyond:
            if heap_dists[idx] <= observation_radius:
                idx += 1
                continue
            # Swap with element from the end (matching C++: heap[idx] = heap[--newBeyond])
            new_beyond -= 1
            heap_array[idx], heap_array[new_beyond] = heap_array[new_beyond], heap_array[idx]
            heap_dists[idx], heap_dists[new_beyond] = heap_dists[new_beyond], heap_dists[idx]
        
        # Valid indices are the first new_beyond elements (after swaps)
        valid_indices = heap_array[:new_beyond]
    
    if len(valid_indices) == 0:
        # Return zero-padded observation
        map_tensor = np.zeros((max_num_lanes, num_lane_features))
        return map_tensor.flatten()[None, :]
    
    # Get selected lanes (indices refer to lanes_filtered)
    lanes_selected = lanes_filtered[valid_indices]
    
    # Transform to ego-centric coordinates
    # Synced with gpudrive/src/utils.hpp ReferenceFrame::relativePosition
    # The transformation: translate to ego position, then rotate by negative ego heading
    # This puts the ego at origin (0,0) facing forward (positive y-axis in ego frame)
    # Normalize lanes to ego-centric coordinate frame
    # After normalization: [rel_x, rel_y, segment_length, segment_width, segment_height, rel_heading, type, ...]
    lanes_normalized = normalize_lanes(
        lanes_selected[:, None], 
        normalize_dict=local_frame,
        offset=0.
    )[:, 0]
    
    # Normalize features matching gpudrive/gpudrive/datatypes/roadgraph.py LocalRoadGraphPoints.normalize
    # x, y: normalize to [-1, 1] using MIN_RG_COORD, MAX_RG_COORD
    lanes_normalized[:, 0] = _normalize_min_max(
            lanes_normalized[:, 0], 
            MIN_RG_COORD, 
            MAX_RG_COORD)
    lanes_normalized[:, 1] = _normalize_min_max(
            lanes_normalized[:, 1], 
            MIN_RG_COORD, 
            MAX_RG_COORD)
    
    # segment_length: normalize by MAX_ROAD_LINE_SEGMENT_LEN
    lanes_normalized[:, 2] = lanes_normalized[:, 2] / MAX_ROAD_LINE_SEGMENT_LEN
    
    # segment_width: normalize by MAX_ROAD_SCALE
    lanes_normalized[:, 3] = lanes_normalized[:, 3] / MAX_ROAD_SCALE
    
    # segment_height: normalize by MAX_ROAD_SCALE
    if lanes_normalized.shape[1] > 4:
        lanes_normalized[:, 4] = lanes_normalized[:, 4] / MAX_ROAD_SCALE
    
    # orientation: normalize by MAX_ORIENTATION_RAD
    if lanes_normalized.shape[1] > 5:
        lanes_normalized[:, 5] = lanes_normalized[:, 5] / MAX_ORIENTATION_RAD
    
    # Create output tensor with zero-padding
    map_tensor = np.zeros((max_num_lanes, num_lane_features))
    num_valid = len(valid_indices)
    map_tensor[:num_valid] = lanes_normalized[:, :num_lane_features]
    # set the type of the invalid lanes to None
    map_tensor[num_valid:, int(EntityType.NoneType.value + 6)] = 1.0

    return map_tensor.flatten()[None, :].astype(np.float32)


def get_route_obs(
        route_points,
        ego_state,
        max_route_points=30,
        normalize_min=MIN_RG_COORD,
        normalize_max=MAX_RG_COORD,
        normalize=True
    ):
    """ Get route observations into format compatible with RL planners used in GPUDrive.
    Synced with gpudrive/src/sim.cpp routeProcessingSystem and gpudrive/gpudrive/env/env_torch.py _get_route_obs method.
    
    This implementation matches gpudrive's behavior:
    - Finds closest route point to ego's current position
    - Extracts up to 30 points starting from closest point
    - Transforms to ego-centric coordinates (relative position and rotation)
    - Normalizes using MIN_RG_COORD/MAX_RG_COORD (matching RouteObservation.normalize)
    
    Args:
        route_points: Array of route points in global coordinates, shape (N, 2) where N is number of route points.
                      Points should be centered on world mean (like in GPUDrive FullRoute component).
        ego_state: Ego vehicle state [x, y, vx, vy, heading, length, width, ...]
        max_route_points: Maximum number of route points to extract (default 30, matching ROUTE_FEAT_DIM // 2)
        normalize_min: Minimum value for normalization (default MIN_RG_COORD = -1000, matching gpudrive)
        normalize_max: Maximum value for normalization (default MAX_RG_COORD = 1000, matching gpudrive)
        normalize: Whether to normalize coordinates (default True, matching gpudrive when norm_obs=True)
    
    Returns:
        np.array: Flattened route observation, shape (1, 61) = (30 points * 2 coords + 1 numPoints)
                  Matches ROUTE_FEAT_DIM = 61
                  Format: [x0, y0, x1, y1, ..., x29, y29, numPoints]
    """
    if route_points is None or len(route_points) == 0:
        # Return zero-padded route observation
        route_obs = np.zeros((1, max_route_points * 2 + 1))
        route_obs[0, -1] = 0.0  # numPoints = 0
        return route_obs
    
    ego_pos = ego_state[:2]  # [x, y]
    ego_heading = ego_state[4]  # heading in radians
    
    # Find closest route point to ego's current position
    # Synced with gpudrive/src/sim.cpp routeProcessingSystem
    dists_sq = np.sum((route_points - ego_pos[None, :]) ** 2, axis=1)
    closest_idx = np.argmin(dists_sq)
    
    # Extract up to max_route_points starting from closest point
    num_extracted = min(max_route_points, len(route_points) - closest_idx)
    
    if num_extracted == 0:
        # Return zero-padded route observation
        route_obs = np.zeros((1, max_route_points * 2 + 1))
        route_obs[0, -1] = 0.0  # numPoints = 0
        return route_obs
    
    # Get the route segment to process
    route_segment = route_points[closest_idx:closest_idx + num_extracted]
    
    # Transform to ego-centric coordinates
    # Synced with gpudrive/src/utils.hpp ReferenceFrame::relativePosition
    # The transformation: translate to ego position, then rotate by negative ego heading
    # This puts the ego at origin (0,0) facing forward (positive y-axis in ego frame)
    
    # First translate: subtract ego position
    relative_positions = route_segment - ego_pos[None, :]
    
    # Then rotate by negative ego heading to transform to ego's local frame
    # This rotates the coordinate system so ego is facing forward
    cos_h = np.cos(-ego_heading)
    sin_h = np.sin(-ego_heading)
    rotation_matrix = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
    
    # Apply rotation to each point: (x', y') = R @ (x, y)
    ego_centric_points = (rotation_matrix @ relative_positions.T).T
    
    # Create route observation array
    route_obs = np.zeros((1, max_route_points * 2 + 1))
    
    # Fill in the extracted points
    route_obs[0, :num_extracted * 2] = ego_centric_points.flatten()
    
    # Normalize coordinates if requested
    if normalize:
        # Normalize x coordinates (even indices: 0, 2, 4, ...)
        x_coords = route_obs[0, 0:num_extracted * 2:2]
        route_obs[0, 0:num_extracted * 2:2] = _normalize_min_max(
            x_coords, normalize_min, normalize_max
        )
        # Normalize y coordinates (odd indices: 1, 3, 5, ...)
        y_coords = route_obs[0, 1:num_extracted * 2:2]
        route_obs[0, 1:num_extracted * 2:2] = _normalize_min_max(
            y_coords, normalize_min, normalize_max
        )
    else:
        route_obs[0, :num_extracted * 2] = route_obs[0, :num_extracted * 2]
    
    # Set numPoints (last element)
    route_obs[0, -1] = float(num_extracted)
    
    return route_obs.astype(np.float32)


class ForwardKinematics:
    """ Simple bicycle model forward kinematics for ego vehicle.
    Synced with gpudrive/src/dynamics.hpp forwardKinematics function (Classic dynamics model).
    """
    def __init__(
            self, 
            start_position, 
            start_velocity, 
            start_yaw, 
            length, 
            width, 
            dt=0.1, 
            max_speed=float('inf')
        ):
        self.length = length
        self.width = width
        self.yaw = start_yaw
        self.velocity = start_velocity
        self.position = start_position
        self.dt = dt
        self.max_speed = max_speed
        
    def forward_kinematics(self, action):
        """ Compute next state given current state and action using bicycle model.
        Synced with gpudrive/src/dynamics.hpp forwardKinematics (DynamicsModel::Classic).
        
        Args:
            action: Array with [acceleration, steering, ...] where:
                   - action[0] is acceleration
                   - action[1] is steering angle
        
        Returns:
            np.array: State vector [pos_x, pos_y, vel_x, vel_y, yaw, length, width, existence]
        """
        def clip_speed(speed: float) -> float:
            return max(min(speed, self.max_speed), -self.max_speed)
        
        def polar_to_vector2d(r: float, theta: float):
            return [r * math.cos(theta), r * math.sin(theta)]
        
        # Extract current speed and yaw (matching gpudrive line 26-27)
        speed = np.linalg.norm(self.velocity)
        yaw = self.yaw
        
        # Average speed for calculating direction vector (matching gpudrive line 29)
        v = clip_speed(speed + 0.5 * action[0] * self.dt)
        
        # Steering calculations (matching gpudrive lines 30-32)
        tan_delta = math.tan(action[1])
        # Assume center of mass lies at the middle of length, then l / L == 0.5
        beta = math.atan(0.5 * tan_delta)
        
        # Direction vector and angular velocity (matching gpudrive lines 33-34)
        d = polar_to_vector2d(v, yaw + beta)
        w = v * math.cos(beta) * tan_delta / self.length
        
        # Update yaw, speed, position, and velocity (matching gpudrive lines 39-46)
        new_yaw = _angle_add(yaw, w * self.dt)
        new_speed = clip_speed(speed + action[0] * self.dt)
        
        self.position[0] += d[0] * self.dt
        self.position[1] += d[1] * self.dt
        # Note: gpudrive sets position.z = 1, but we don't track z in 2D
        
        self.yaw = new_yaw
        self.velocity = np.array([new_speed * math.cos(new_yaw), new_speed * math.sin(new_yaw)])
        # Note: gpudrive sets velocity.angular.z = w, but we don't track angular velocity
        
        return np.array([self.position[0], self.position[1], self.velocity[0], self.velocity[1], 
                         self.yaw, self.length, self.width, 1.0])
    

def get_action_value_tensor() -> torch.Tensor:
    """ Generates a tensor mapping action indices to action values.
    Used for GPUDrive discrete action space.
    Synced with gpudrive/gpudrive/env/env_puffer.py and gpudrive/baselines/ppo/config/ppo_base_puffer.yaml.
    
    Default action space (matching gpudrive defaults):
    - steer_actions: 13 values from -π to π
    - accel_actions: 7 values from -4.0 to 4.0
    - head_tilt_actions: 1 value [0]
    - Total: 13 × 7 × 1 = 91 actions
    
    Returns:
        torch.Tensor: Shape (91, 3) containing [accel, steer, head_tilt] for each action index.
    """
    # Default action space matching gpudrive/baselines/ppo/config/ppo_base_puffer.yaml
    # and gpudrive/gpudrive/env/config.py defaults
    steer_actions: torch.Tensor = torch.linspace(-torch.pi, torch.pi, 13)
    accel_actions: torch.Tensor = torch.linspace(-4.0, 4.0, 7)
    head_tilt_actions: torch.Tensor = torch.Tensor([0]) 

    products = product(
        accel_actions, steer_actions, head_tilt_actions
    )
    
    # Create a mapping from action indices to action values
    # Matching gpudrive/gpudrive/env/env_torch.py _set_discrete_action_space
    action_key_to_values = {}
    for action_idx, (action_1, action_2, action_3) in enumerate(
        products
    ):
        action_key_to_values[action_idx] = [
            action_1.item(),
            action_2.item(),
            action_3.item(),
        ]

    action_keys_tensor = torch.tensor(
        [
            action_key_to_values[key]
            for key in sorted(action_key_to_values.keys())
        ]
    )
    return action_keys_tensor


def _normalize_min_max(tensor, min_val, max_val):
    """Normalizes an array of values to the range [-1, 1].

    Args:
        x (np.array): Array of values to normalize.
        min_val (float): Minimum value for normalization.
        max_val (float): Maximum value for normalization.

    Returns:
        np.array: Normalized array of values.
    """
    return 2 * ((tensor - min_val) / (max_val - min_val)) - 1


def _angle_add(angle1: float, angle2: float) -> float:
    """ Add two angles in radians, result wrapped to [-π, π]. """
    result = angle1 + angle2
    return (result + math.pi) % (2 * math.pi) - math.pi


# JSON serialization functions from gpudrive/src/json_serialization.hpp
def from_json_MapVector2(j: dict) -> MapVector2:
    """
    Equivalent to the C++ from_json(const nlohmann::json &j, MapVector2 &p).
    Synced with gpudrive/src/json_serialization.hpp
    """
    p = MapVector2()
    p.x = float(j["x"])
    p.y = float(j["y"])
    return p


def from_json_MapObject(j: dict) -> MapObject:
    """
    Equivalent to the C++ from_json(const nlohmann::json &j, MapObject &obj).
    Synced with gpudrive/src/json_serialization.hpp
    """
    obj = MapObject()
    obj.mean = MapVector2(0.0, 0.0)

    positions = j["position"]
    i = 0
    for pos in positions:
        if i >= MAX_POSITIONS:
            break
        p = from_json_MapVector2(pos)
        obj.position[i] = p
        
        # Incremental mean update
        obj.mean.x += (p.x - obj.mean.x) / (i + 1)
        obj.mean.y += (p.y - obj.mean.y) / (i + 1)
        
        i += 1
    obj.numPositions = i
    
    # Updated to use VehicleSize struct
    obj.vehicle_size.width = float(j["width"])
    obj.vehicle_size.length = float(j["length"])
    obj.vehicle_size.height = float(j["height"])  # Added height field
    
    # Added id field
    obj.id = int(j["id"])

    headings = j["heading"]
    i = 0
    for h in headings:
        if i >= MAX_POSITIONS:
            break
        obj.heading[i] = float(h)
        i += 1
    obj.numHeadings = i

    velocities = j["velocity"]
    i = 0
    for v in velocities:
        if i >= MAX_POSITIONS:
            break
        vel = from_json_MapVector2(v)
        obj.velocity[i] = vel
        i += 1
    obj.numVelocities = i

    valids = j["valid"]
    i = 0
    for v in valids:
        if i >= MAX_POSITIONS:
            break
        obj.valid[i] = bool(v)
        i += 1
    obj.numValid = i

    obj.goalPosition = from_json_MapVector2(j["goalPosition"])

    type_str = j["type"]
    if   type_str == "vehicle":
        obj.type = EntityType.Vehicle
    elif type_str == "pedestrian":
        obj.type = EntityType.Pedestrian
    elif type_str == "cyclist":
        obj.type = EntityType.Cyclist
    else:
        obj.type = EntityType.NoneType  # Corresponds to EntityType::None in C++

    if "mark_as_expert" in j:
        obj.markAsExpert = bool(j["mark_as_expert"])

    # Initialize metadata fields to 0 (matching C++ implementation)
    obj.metadata.isSdc = 0
    obj.metadata.isObjectOfInterest = 0
    obj.metadata.isTrackToPredict = 0
    obj.metadata.difficulty = 0

    return obj


def from_json_MapRoad(j: dict, polylineReductionThreshold: float = 0.0) -> MapRoad:
    """
    Equivalent to the C++ from_json(const nlohmann::json &j, MapRoad &road, float polylineReductionThreshold).
    Synced with gpudrive/src/json_serialization.hpp
    """
    road = MapRoad()
    road.mean = MapVector2(0.0, 0.0)
    
    type_str = j["type"]
    if   type_str == "road_edge":
        road.type = EntityType.RoadEdge
    elif type_str == "road_line":
        road.type = EntityType.RoadLine
    elif type_str == "lane":
        road.type = EntityType.RoadLane
    elif type_str == "crosswalk":
        road.type = EntityType.CrossWalk
    elif type_str == "speed_bump":
        road.type = EntityType.SpeedBump
    elif type_str == "stop_sign":
        road.type = EntityType.StopSign
    else:
        road.type = EntityType.NoneType  # Corresponds to EntityType::None in C++
    
    # Gather geometry points
    geometry_points = []
    for point in j["geometry"]:
        geometry_points.append(from_json_MapVector2(point))

    num_segments = len(geometry_points) - 1
    sample_every_n = 1
    num_sampled_points = (num_segments + sample_every_n - 1) // sample_every_n + 1
    
    # If big enough and is road-like entity, do polyline reduction
    # Updated condition to match C++ implementation exactly
    if num_segments >= 10 and (road.type == EntityType.RoadLane or road.type == EntityType.RoadEdge or road.type == EntityType.RoadLine):
        skip = [False] * num_sampled_points
        k = 0
        skip_changed = True
        
        while skip_changed:
            skip_changed = False
            k = 0
            while k < num_sampled_points - 1:
                # k_1 is the next point that is not skipped
                k1 = k + 1
                # Keep incrementing k_1 until we find a point that is not skipped
                while k1 < num_sampled_points - 1 and skip[k1]:
                    k1 += 1
                if k1 >= num_sampled_points - 1:
                    break
                
                k2 = k1 + 1
                # Keep incrementing k_2 until we find a point that is not skipped
                while k2 < num_sampled_points and skip[k2]:
                    k2 += 1
                if k2 >= num_sampled_points:
                    break

                point1 = geometry_points[k * sample_every_n]
                point2 = geometry_points[k1 * sample_every_n]
                point3 = geometry_points[k2 * sample_every_n]

                # Calculate triangle area (matching C++ implementation exactly)
                # C++ uses float (single precision) throughout, so we must convert
                # coordinates to float32 BEFORE arithmetic to match rounding behavior
                p1x = np.float32(point1.x)
                p1y = np.float32(point1.y)
                p2x = np.float32(point2.x)
                p2y = np.float32(point2.y)
                p3x = np.float32(point3.x)
                p3y = np.float32(point3.y)
                
                # Now compute in single precision, matching C++ exactly
                area = np.float32(0.5) * abs((p1x - p3x) * (p2y - p1y) - (p1x - p2x) * (p3y - p1y))

                # Convert threshold to float32 for comparison to match C++ precision
                threshold = np.float32(polylineReductionThreshold)
                if area < threshold:
                    # If the area is less than the threshold, then we skip the middle point
                    skip[k1] = True
                    # Skip the middle point and start from the next point
                    k = k2
                    skip_changed = True
                else:
                    # If the area is greater than the threshold, then we don't skip the middle point and start from the next point
                    k = k1

        # Create the road lines - force first and last point to not be skipped
        k = 0
        skip[0] = False
        skip[num_sampled_points - 1] = False
        new_geometry_points = []
        while k < num_sampled_points:
            if not skip[k]:
                # Add the point to the list if it is not skipped
                new_geometry_points.append(geometry_points[k * sample_every_n])
            k += 1

        for i in range(len(new_geometry_points)):
            if i == MAX_GEOMETRY:
                break
            road.geometry[i] = new_geometry_points[i]  # Create the road lines
        road.numPoints = len(new_geometry_points)
    
    else:
        # No polyline reduction
        for i in range(num_sampled_points):
            if i >= MAX_GEOMETRY:
                break
            road.geometry[i] = geometry_points[i * sample_every_n]
        road.numPoints = num_sampled_points

    # optional fields
    if "id" in j:
        road.id = int(j["id"])

    if "map_element_id" in j:
        map_element_id = int(j["map_element_id"])
        # Match C++ logic regarding valid range
        if map_element_id == 4 or map_element_id >= int(MapType.NUM_TYPES.value) or map_element_id < -1:
            road.mapType = MapType.UNKNOWN
        else:
            road.mapType = MapType(map_element_id)  # Direct cast, no try-except needed
    else:
        road.mapType = MapType.UNKNOWN

    # Compute incremental mean (matching C++ implementation)
    for i in range(road.numPoints):
        pt = road.geometry[i]
        road.mean.x += (pt.x - road.mean.x) / (i + 1)
        road.mean.y += (pt.y - road.mean.y) / (i + 1)

    return road


def calc_mean(j: dict) -> Tuple[float, float]:
    """
    Equivalent to the C++ calc_mean(const nlohmann::json &j).
    Computes a global (x, y) mean over object positions and road geometry.
    Synced with gpudrive/src/json_serialization.hpp
    """
    # Use float32 to match C++ float (single precision) throughout
    mean_x = np.float32(0.0)
    mean_y = np.float32(0.0)
    num_entities = 0

    for obj in j["objects"]:
        positions = obj["position"]
        valids = obj["valid"]
        i = 0
        for pos in positions:
            if not valids[i]:
                i += 1
                continue
            # Convert to float32 to match C++ float precision
            new_x = np.float32(pos["x"])
            new_y = np.float32(pos["y"])
            num_entities += 1
            # incremental mean (all arithmetic in float32)
            mean_x += (new_x - mean_x) / np.float32(num_entities)
            mean_y += (new_y - mean_y) / np.float32(num_entities)
            i += 1

    for rd in j["roads"]:
        for pt in rd["geometry"]:
            # Convert to float32 to match C++ float precision
            new_x = np.float32(pt["x"])
            new_y = np.float32(pt["y"])
            num_entities += 1
            # incremental mean (all arithmetic in float32)
            mean_x += (new_x - mean_x) / np.float32(num_entities)
            mean_y += (new_y - mean_y) / np.float32(num_entities)

    # Convert back to Python float for return type compatibility
    return float(mean_x), float(mean_y)


# Road edge creation from gpudrive/src/level_gen.cpp
def make_road_edge(road_init, j, world_mean) -> dict:
    """
    Create a 'road edge' object from two consecutive points in road_init.geometry.
    Synced with gpudrive/src/level_gen.cpp makeRoadEdge function.
    
    Args:
        road_init: A MapRoad instance containing geometry, ID, type, etc.
        j: An index for the starting point in the geometry list.
        world_mean: A Vector2 offset to subtract from positions (matching ctx.singleton<WorldMeans>() in C++).

    Returns:
        A dictionary describing a 'road edge' in Python with:
        - road_id: Road ID
        - road_type: EntityType of the road
        - start: Start position (MapVector3, with world_mean subtracted)
        - end: End position (MapVector3, with world_mean subtracted)
        - position: Center position (MapVector3)
        - rotation: Rotation angle in radians
        - scale: Scale tuple (half_length, 0.1, 0.1)
    """
    # Constants from gpudrive/src/consts.hpp
    LIDAR_ROAD_EDGE_OFFSET = np.float32(0.1)
    LIDAR_ROAD_LINE_OFFSET = np.float32(-0.1)
    
    p1 = road_init.geometry[j]
    p2 = road_init.geometry[j + 1]
    
    # Convert to float32 to match C++ float precision
    p1x = np.float32(p1.x)
    p1y = np.float32(p1.y)
    p2x = np.float32(p2.x)
    p2y = np.float32(p2.y)
    world_mean_x = np.float32(world_mean.x)
    world_mean_y = np.float32(world_mean.y)
    
    # Calculate z offset based on road type (matching gpudrive line 204)
    if road_init.type == EntityType.RoadEdge:
        z_offset = np.float32(1.0) + LIDAR_ROAD_EDGE_OFFSET
    else:
        z_offset = np.float32(1.0) + LIDAR_ROAD_LINE_OFFSET

    # Subtract world_mean from positions (matching gpudrive lines 206-207)
    # All arithmetic in float32 to match C++
    start_x = p1x # - world_mean_x
    start_y = p1y # - world_mean_y
    end_x = p2x # - world_mean_x
    end_y = p2y # - world_mean_y
    
    start = MapVector3(
        x = float(start_x),
        y = float(start_y),
        z = float(z_offset)
    )
    end = MapVector3(
        x = float(end_x),
        y = float(end_y),
        z = float(z_offset)
    )

    # Calculate center position (matching gpudrive line 212)
    # All arithmetic in float32
    pos_x = (start_x + end_x) / np.float32(2.0)
    pos_y = (start_y + end_y) / np.float32(2.0)
    pos = MapVector3(
        x = float(pos_x),
        y = float(pos_y),
        z = float(z_offset)
    )

    # Calculate rotation angle (matching gpudrive line 213)
    # All arithmetic in float32
    dx = end_x - start_x
    dy = end_y - start_y
    angle = math.atan2(float(dy), float(dx))
    rot = angle

    # Calculate scale (matching gpudrive line 214)
    # d0 = half_length, d1 = 0.1, d2 = 0.1
    # Use float32 for distance calculation to match C++ Vector3::distance()
    dx_sq = dx * dx
    dy_sq = dy * dy
    distance = np.sqrt(dx_sq + dy_sq)
    half_length = distance / np.float32(2.0)
    scale = (float(half_length), 0.1, 0.1)

    return {
        'road_id': road_init.id,
        'road_type': road_init.type,
        'start': start,
        'end': end,
        'position': pos,
        'rotation': rot,
        'scale': scale
    }


# Road edge creation from gpudrive/src/level_gen.cpp
def create_road_edges(data, world_mean, max_num_edges=10000) -> List[dict]:
    """
    Create road edges from map data. Synced with gpudrive/src/level_gen.cpp createRoadEntities.
    
    This function only processes RoadEdge, RoadLine, and RoadLane types (matching gpudrive's switch statement).
    CrossWalk, SpeedBump, and StopSign are handled differently in gpudrive and are not included here.
    
    Args:
        data: Map object containing roads
        world_mean: MapVector2 representing world mean (used for coordinate centering)
        max_num_edges: Maximum number of edges to create (default 10000, matching kMaxRoadEntityCount)
    
    Returns:
        List of edge tensors, each containing [pos_x, pos_y, scale[0], scale[1], scale[2], rotation, type]
    """
    # Constant from gpudrive/src/consts.hpp
    kMaxRoadEntityCount = 10000
    
    edges = []
    
    num_edges = data.numRoadSegments
    num_lanes = data.numRoads

    for idx in range(num_lanes):
        # Check bounds (matching gpudrive line 293)
        if len(edges) >= kMaxRoadEntityCount:
            break
            
        road = data.roads[idx]
        
        # Only process RoadEdge, RoadLine, and RoadLane types (matching gpudrive lines 297-299)
        if road.type not in [EntityType.RoadEdge, EntityType.RoadLine, EntityType.RoadLane]:
            continue
        
        num_points = road.numPoints
        
        if num_points < 2:
            continue
        
        # Create edges for each segment (matching gpudrive lines 302-304)
        # C++ loops j from 1 to numPoints-1 and calls makeRoadEdge with j-1
        # Python equivalent: loop j from 0 to num_points-2
        for j in range(num_points - 1):
            # Check bounds during iteration (matching gpudrive line 306)
            if len(edges) >= kMaxRoadEntityCount:
                break
                
            edge_data = make_road_edge(road, j, world_mean)
            edge_tensor = np.array(
                [
                    edge_data['position'].x,
                    edge_data['position'].y,
                    edge_data['scale'][0],
                    edge_data['scale'][1],
                    edge_data['scale'][2],
                    edge_data['rotation'],
                    edge_data['road_type'].value
                ],
                dtype=np.float32
            )
            edges.append(edge_tensor)

    # Verify we created the expected number of edges
    assert len(edges) == num_edges

    edges = np.array(edges, dtype=np.float32)

    edges = np.concatenate(
        [edges[:, :-1],
         np.eye(7)[edges[:, 6].astype(int)].astype(np.float32)], axis=-1
    )

    if len(edges) > max_num_edges:
        edges = edges[:max_num_edges]

    return edges

def from_json_Map(j: dict, polylineReductionThreshold: float = 0.0) -> dict:
    """
    Equivalent to the C++ from_json(const nlohmann::json &j, Map &map, float polylineReductionThreshold).
    Synced with gpudrive/src/json_serialization.hpp
    
    Note: This is a simplified version. The full C++ implementation includes:
    - Metadata processing (SDC, tracks_to_predict, objects_of_interest)
    - Route parsing
    - Map name and scenario ID
    - Prioritized object initialization (SDC first, then tracks_to_predict, etc.)
    
    Returns:
        dict: A dictionary with 'world_mean' and 'lanes_compressed' keys for backward compatibility.
              The C++ version returns a Map object directly, but this wrapper creates road edges
              and returns them in a convenient format.
    """
    the_map = Map()
    
    # Parse map name and scenario ID
    if "name" in j:
        the_map.mapName = str(j["name"])
    if "scenario_id" in j:
        the_map.scenarioId = str(j["scenario_id"])
    
    # Parse route if present
    the_map.numRoutePoints = 0
    if "route" in j and isinstance(j["route"], list):
        i = 0
        for point in j["route"]:
            if i < 1000:  # MAX_ROUTE_POINTS
                the_map.route[i] = from_json_MapVector2(point)
                i += 1
            else:
                break
        the_map.numRoutePoints = i
    
    # calculate global mean
    mx, my = calc_mean(j)
    the_map.mean = MapVector2(mx, my)

    # Process objects with metadata handling (matching gpudrive priority order)
    objects_data = j["objects"]
    the_map.numObjects = min(len(objects_data), MAX_OBJECTS)
    
    # Get metadata (matching gpudrive line 308)
    metadata = j.get("metadata", {})
    sdc_index = metadata.get("sdc_track_index", -1)
    tracks_to_predict = metadata.get("tracks_to_predict", [])
    objects_of_interest = metadata.get("objects_of_interest", [])
    
    # Create id to object index mapping (matching gpudrive line 312)
    id_to_obj_idx = {}
    idx = 0
    
    # Create sets for quick lookup (matching gpudrive lines 316-317)
    tracks_to_predict_indices = set()
    for track in tracks_to_predict:
        track_index = track.get("track_index", -1)
        if 0 <= track_index < len(objects_data):
            tracks_to_predict_indices.add(track_index)
        else:
            # Warning for invalid track_index (matching gpudrive line 325)
            print(f"Warning: Invalid track_index {track_index} in scene {j.get('name', 'unknown')}")
    
    objects_of_interest_ids = set(objects_of_interest)
    
    # Initialize SDC first if valid (matching gpudrive lines 335-361)
    if 0 <= sdc_index < len(objects_data):
        obj = from_json_MapObject(objects_data[sdc_index])
        obj.metadata.isSdc = 1
        
        # Set additional metadata if needed
        sdc_id = obj.id
        if sdc_index in tracks_to_predict_indices:
            obj.metadata.isTrackToPredict = 1
            # Find and set difficulty
            for track in tracks_to_predict:
                if track.get("track_index") == sdc_index:
                    obj.metadata.difficulty = track.get("difficulty", 0)
                    break
        if sdc_id in objects_of_interest_ids:
            obj.metadata.isObjectOfInterest = 1
        
        the_map.objects[0] = obj
        id_to_obj_idx[sdc_id] = 0
        idx = 1
        
        # Remove SDC from sets to avoid double processing (matching gpudrive lines 359-360)
        tracks_to_predict_indices.discard(sdc_index)
        objects_of_interest_ids.discard(sdc_id)
    
    # Initialize tracks_to_predict objects (excluding SDC) (matching gpudrive lines 364-388)
    for i in range(len(objects_data)):
        if idx >= the_map.numObjects:
            break
        if i == sdc_index:
            continue  # Skip SDC as it's already initialized
        
        if i in tracks_to_predict_indices:
            obj = from_json_MapObject(objects_data[i])
            obj.metadata.isTrackToPredict = 1
            
            # Find and set difficulty
            for track in tracks_to_predict:
                if track.get("track_index") == i:
                    obj.metadata.difficulty = track.get("difficulty", 0)
                    break
            
            # Check if also object of interest
            if obj.id in objects_of_interest_ids:
                obj.metadata.isObjectOfInterest = 1
                objects_of_interest_ids.discard(obj.id)
            
            the_map.objects[idx] = obj
            id_to_obj_idx[obj.id] = idx
            idx += 1
    
    # Initialize objects_of_interest (excluding those already processed) (matching gpudrive lines 391-402)
    for i in range(len(objects_data)):
        if idx >= the_map.numObjects:
            break
        if i == sdc_index:
            continue
        
        obj_id = objects_data[i].get("id")
        if obj_id in objects_of_interest_ids:
            obj = from_json_MapObject(objects_data[i])
            obj.metadata.isObjectOfInterest = 1
            
            the_map.objects[idx] = obj
            id_to_obj_idx[obj.id] = idx
            idx += 1
    
    # Initialize all remaining objects (matching gpudrive lines 405-414)
    for i in range(len(objects_data)):
        if idx >= the_map.numObjects:
            break
        if i == sdc_index:
            continue
        
        obj_id = objects_data[i].get("id")
        if obj_id not in id_to_obj_idx:  # Check if not already processed
            obj = from_json_MapObject(objects_data[i])
            the_map.objects[idx] = obj
            id_to_obj_idx[obj.id] = idx
            idx += 1
    
    the_map.numObjects = idx

    # roads
    roads_data = j["roads"]
    the_map.numRoads = min(len(roads_data), MAX_ROADS)
    
    count_road_points = 0
    for idx, rd_json in enumerate(roads_data[:the_map.numRoads]):
        road_obj = from_json_MapRoad(rd_json, polylineReductionThreshold)
        the_map.roads[idx] = road_obj
        
        # Count road segments (matching C++ logic)
        if road_obj.type.value <= EntityType.RoadLane.value:
            count_road_points += (road_obj.numPoints - 1)
        else:
            count_road_points += 1

    the_map.numRoadSegments = count_road_points
    
    # # BEGIN DEBUGGING
    # # Create map_roads tensor with structure [num_roads, 3498]
    # # Each road: [geometry (3492 floats), id, map_type, entity_type, num_points, mean_x, mean_y]
    # num_roads = the_map.numRoads
    # map_roads_list = []
    
    # for road_idx in range(num_roads):
    #     road = the_map.roads[road_idx]
        
    #     # Extract geometry: flatten (x, y) pairs from geometry list
    #     # Pad to MAX_GEOMETRY (1746) points if needed
    #     geometry_flat = []
    #     for i in range(MAX_GEOMETRY):
    #         if i < road.numPoints:
    #             geometry_flat.append(road.geometry[i].x)
    #             geometry_flat.append(road.geometry[i].y)
    #         else:
    #             # Pad with zeros if road has fewer than MAX_GEOMETRY points
    #             geometry_flat.append(0.0)
    #             geometry_flat.append(0.0)
        
    #     # Extract metadata
    #     id_val = float(road.id)
    #     map_type_val = float(road.mapType.value)
    #     entity_type_val = float(road.type.value)
    #     num_points_val = float(road.numPoints)
    #     mean_x = road.mean.x
    #     mean_y = road.mean.y
        
    #     # Combine geometry and metadata into single tensor
    #     road_tensor = geometry_flat + [id_val, map_type_val, entity_type_val, 
    #                                    num_points_val, mean_x, mean_y]
    #     map_roads_list.append(road_tensor)
    
    # # Convert to torch tensor: shape [num_roads, 3498]
    # map_roads = torch.tensor(map_roads_list, dtype=torch.float32)
    
    # with open('map_road.pkl', 'wb') as f:
    #     import pickle
    #     pickle.dump({'map_roads': map_roads}, f)
    
    # # END DEBUGGING
    
    # For backward compatibility with existing code, return a dict format
    # Note: The C++ version returns a Map object directly, but this wrapper
    # creates the road edges and returns them in a convenient format
    
    d = {}
    d['world_mean'] = np.array([the_map.mean.x, the_map.mean.y], dtype=np.float32)
    edges = create_road_edges(the_map, the_map.mean)
    d['lanes_compressed'] = edges

    # BEGIN DEBUGGING
    # with open('lanes_compressed.pkl', 'wb') as f:
    #     import pickle
    #     pickle.dump({'lanes_compressed': edges}, f)
    # END DEBUGGING
    
    return d


def load_policy(path_to_cpt, model_name, device):
    """Load a policy from a given path."""
    saved_cpt = torch.load(
        f=f"{path_to_cpt}/{model_name}.pt",
        map_location=device,
        weights_only=False,
    )

    print(f"Load model from {path_to_cpt}/{model_name}.pt")

    # Create policy architecture from saved checkpoint
    policy = NeuralNet(
        input_dim=saved_cpt["model_arch"]["input_dim"],
        action_dim=saved_cpt["action_dim"],
        hidden_dim=saved_cpt["model_arch"]["hidden_dim"],
    ).to(device)

    # Load the model parameters
    policy.load_state_dict(saved_cpt["parameters"])

    print("Loaded model parameters")

    return policy.eval()


# Network utility functions from gpudrive/gpudrive/networks/late_fusion.py
def log_prob(logits, value):
    """ Log probability function. Synced with gpudrive/gpudrive/networks/late_fusion.py """
    value = value.long().unsqueeze(-1)
    value, log_pmf = torch.broadcast_tensors(value, logits)
    value = value[..., :1]
    return log_pmf.gather(-1, value).squeeze(-1)


def entropy(logits):
    """ Entropy function. Synced with gpudrive/gpudrive/networks/late_fusion.py """
    min_real = torch.finfo(logits.dtype).min
    logits = torch.clamp(logits, min=min_real)
    p_log_p = logits * logits_to_probs(logits)
    return -p_log_p.sum(-1)


def sample_logits(
    logits: Union[torch.Tensor, List[torch.Tensor]],
    action=None,
    deterministic=False,
):
    """Sample logits: Supports deterministic sampling.
    Synced with gpudrive/gpudrive/networks/late_fusion.py
    """

    normalized_logits = [logits - logits.logsumexp(dim=-1, keepdim=True)]
    logits = [logits]

    if action is None:
        if deterministic:
            # Select the action with the maximum probability
            action = torch.stack([l.argmax(dim=-1) for l in logits])
        else:
            # Sample actions stochastically from the logits
            action = torch.stack(
                [
                    torch.multinomial(logits_to_probs(l), 1).squeeze()
                    for l in logits
                ]
            )
    else:
        batch = logits[0].shape[0]
        action = action.view(batch, -1).T

    assert len(logits) == len(action)

    logprob = torch.stack(
        [log_prob(l, a) for l, a in zip(normalized_logits, action)]
    ).T.sum(1)

    logits_entropy = torch.stack(
        [entropy(l) for l in normalized_logits]
    ).T.sum(1)

    return action.squeeze(0), logprob.squeeze(0), logits_entropy.squeeze(0)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """CleanRL's default layer initialization.
    Note: Current GPUDrive uses pufferlib.pytorch.layer_init instead.
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class NeuralNet(
    nn.Module,
):
    """ Neural network architecture. Synced with gpudrive/gpudrive/networks/late_fusion.py
    Note: Current GPUDrive uses pufferlib.pytorch.layer_init for initialization.
    """
    def __init__(
        self,
        action_dim=91,  # Default: 7 * 13
        input_dim=64,
        hidden_dim=128,
        dropout=0.00,
        act_func="tanh",
        max_controlled_agents=64,
        obs_dim=2984,  # Size of the flattened observation vector (hardcoded)
        config=None,  # Optional config
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.max_controlled_agents = max_controlled_agents
        self.max_observable_agents = max_controlled_agents - 1
        self.obs_dim = obs_dim
        self.num_modes = 4  # Ego, partner, road graph, route
        self.dropout = dropout
        self.act_func = nn.Tanh() if act_func == "tanh" else nn.GELU()

        # Indices for unpacking the observation
        self.ego_state_idx = EGO_FEAT_DIM
        self.partner_obs_idx = (
            PARTNER_FEAT_DIM * self.max_controlled_agents
        )
        self.vbd_in_obs = False
        # Calculate the VBD predictions size: 91 timesteps * 5 features = 455
        self.vbd_size = 91 * 5
        
        # Route observation size: 61 (30 points * 2 coords + 1 numPoints)
        self.route_obs_size = ROUTE_FEAT_DIM

        self.ego_embed = nn.Sequential(
            layer_init(
                nn.Linear(self.ego_state_idx, input_dim)
            ),
            nn.LayerNorm(input_dim),
            self.act_func,
            nn.Dropout(self.dropout),
            layer_init(nn.Linear(input_dim, input_dim)),
        )

        self.partner_embed = nn.Sequential(
            layer_init(
                nn.Linear(PARTNER_FEAT_DIM, input_dim)
            ),
            nn.LayerNorm(input_dim),
            self.act_func,
            nn.Dropout(self.dropout),
            layer_init(nn.Linear(input_dim, input_dim)),
        )

        self.road_map_embed = nn.Sequential(
            layer_init(
                nn.Linear(ROAD_GRAPH_FEAT_DIM, input_dim)
            ),
            nn.LayerNorm(input_dim),
            self.act_func,
            nn.Dropout(self.dropout),
            layer_init(nn.Linear(input_dim, input_dim)),
        )

        self.route_embed = nn.Sequential(
            layer_init(
                nn.Linear(self.route_obs_size, input_dim)
            ),
            nn.LayerNorm(input_dim),
            self.act_func,
            nn.Dropout(self.dropout),
            layer_init(nn.Linear(input_dim, input_dim)),
        )

        if self.vbd_in_obs:
            self.vbd_embed = nn.Sequential(
                layer_init(
                    nn.Linear(self.vbd_size, input_dim)
                ),
                nn.LayerNorm(input_dim),
                self.act_func,
                nn.Dropout(self.dropout),
                layer_init(nn.Linear(input_dim, input_dim)),
            )

        self.shared_embed = nn.Sequential(
            nn.Linear(self.input_dim * self.num_modes, self.hidden_dim),
            nn.Dropout(self.dropout),
        )

        self.actor = layer_init(
            nn.Linear(hidden_dim, action_dim), std=0.01
        )
        self.critic = layer_init(
            nn.Linear(hidden_dim, 1), std=1
        )

    def encode_observations(self, observation):

        if self.vbd_in_obs:
            (
                ego_state,
                road_objects,
                road_graph,
                route_obs,
                vbd_predictions,
            ) = self.unpack_obs(observation)
        else:
            ego_state, road_objects, road_graph, route_obs = self.unpack_obs(observation)

        # Embed the ego state
        ego_embed = self.ego_embed(ego_state)

        if self.vbd_in_obs:
            vbd_embed = self.vbd_embed(vbd_predictions)
            # Concatenate the VBD predictions with the ego state embedding
            ego_embed = torch.cat([ego_embed, vbd_embed], dim=1)

        # Max pool
        partner_embed, _ = self.partner_embed(road_objects).max(dim=1)
        road_map_embed, _ = self.road_map_embed(road_graph).max(dim=1)
        
        # Embed route observations
        route_embed = self.route_embed(route_obs)

        # Concatenate the embeddings
        embed = torch.cat([ego_embed, partner_embed, road_map_embed, route_embed], dim=1)

        return self.shared_embed(embed)

    def forward(self, obs, action=None, deterministic=False):

        # Encode the observations
        hidden = self.encode_observations(obs)

        # Decode the actions
        value = self.critic(hidden)
        logits = self.actor(hidden)

        action, logprob, entropy = sample_logits(logits, action, deterministic)

        return action, logprob, entropy, value

    def unpack_obs(self, obs_flat):
        """
        Unpack the flattened observation into the ego state, visible simulator state.

        Args:
            obs_flat (torch.Tensor): Flattened observation tensor of shape (batch_size, obs_dim).

        Returns:
            tuple: If vbd_in_obs is True, returns (ego_state, road_objects, road_graph, route_obs, vbd_predictions).
                Otherwise, returns (ego_state, road_objects, road_graph, route_obs).
        """

        # Unpack modalities
        ego_state = obs_flat[:, : self.ego_state_idx]
        partner_obs = obs_flat[:, self.ego_state_idx : self.partner_obs_idx]

        if self.vbd_in_obs:
            # Extract the VBD predictions (last 455 elements)
            vbd_predictions = obs_flat[:, -self.vbd_size :]
            # Road graph is everything between partner_obs and route+vb
            roadgraph_obs = obs_flat[:, self.partner_obs_idx : -self.route_obs_size - self.vbd_size]
            # Route is between road graph and VBD
            route_obs = obs_flat[:, -self.route_obs_size - self.vbd_size : -self.vbd_size]
        else:
            # Without VBD, road graph is everything between partner_obs and route
            roadgraph_obs = obs_flat[:, self.partner_obs_idx : -self.route_obs_size]
            # Route is the last route_obs_size elements
            route_obs = obs_flat[:, -self.route_obs_size :]
            vbd_predictions = None

        road_objects = partner_obs.view(
            -1, self.max_observable_agents, PARTNER_FEAT_DIM
        )
        road_graph = roadgraph_obs.view(
            -1, TOP_K_ROAD_POINTS, ROAD_GRAPH_FEAT_DIM
        )

        if self.vbd_in_obs:
            return ego_state, road_objects, road_graph, route_obs, vbd_predictions
        else:
            return ego_state, road_objects, road_graph, route_obs