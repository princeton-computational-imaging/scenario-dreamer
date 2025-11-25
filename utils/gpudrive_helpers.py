""" Helper functions and classes for GPUDrive integration. 
Many of these functions are Python versions of C++ code in GPUDrive. """

import numpy as np
import torch
import math
from enum import Enum
from typing import List, Tuple

from itertools import product
from utils.geometry import normalize_agents, normalize_lanes

MAX_OBJECTS = 515
MAX_ROADS = 956
MAX_POSITIONS = 1
MAX_GEOMETRY = 1746


class EntityType(Enum):
    """ Entity types. """
    NoneType   = 0  # used instead of 'None'
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


class MapType(Enum):
    """ Map element types. """
    LANE_UNDEFINED = 0
    LANE_FREEWAY = 1
    LANE_SURFACE_STREET = 2
    LANE_BIKE_LANE = 3
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
    DRIVEWAY = 20
    UNKNOWN = -1
    NUM_TYPES = 21


class MapVector2:
    """ 2D vector class. """
    def __init__(self, x: float = 0.0, y: float = 0.0):
        self.x = x
        self.y = y


class MapVector3:
    """ 3D vector class. """
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z


class MapObject:
    """ GPUDrive object class. """
    def __init__(self):
        self.position: List[MapVector2] = [MapVector2() for _ in range(MAX_POSITIONS)]
        self.numPositions = 0
        
        self.heading: List[float] = [0.0 for _ in range(MAX_POSITIONS)]
        self.numHeadings = 0
        
        self.velocity: List[MapVector2] = [MapVector2() for _ in range(MAX_POSITIONS)]
        self.numVelocities = 0
        
        self.valid: List[bool] = [False for _ in range(MAX_POSITIONS)]
        self.numValid = 0
        
        self.mean = MapVector2()
        self.goalPosition = MapVector2()
        self.width = 0.0
        self.length = 0.0
        self.type = EntityType.NoneType
        
        # Extra field
        self.markAsExpert = False


class MapRoad:
    """ GPUDrive road class. """
    def __init__(self):
        self.geometry: List[MapVector2] = [MapVector2() for _ in range(MAX_GEOMETRY)]
        self.numPoints = 0
        self.mean = MapVector2()
        self.type = EntityType.NoneType
        self.id = 0
        self.mapType = MapType.UNKNOWN


class Map:
    """ GPUDrive map class (roads and agents)."""
    def __init__(self):
        # mean of object positions and road positions
        self.mean = MapVector2()
        # list of "objects" or agents
        self.objects: List[MapObject] = [MapObject() for _ in range(MAX_OBJECTS)]
        self.numObjects = 0
        
        # list of map roads or lanes
        self.roads: List[MapRoad] = [MapRoad() for _ in range(MAX_ROADS)]
        self.numRoads = 0
        
        self.numRoadSegments = 0


def distance_2d(p1: MapVector2, p2: MapVector2) -> float:
    """Compute 2D distance between two points."""
    return math.hypot(p2.x - p1.x, p2.y - p1.y)


def get_ego_state(ego_state):
    """ Get ego state into format compatible with RL planners used in GPUDrive"""
    gpudrive_ego_state = np.zeros((1, 6))
    # speed
    gpudrive_ego_state[0, 0] = np.round(
        np.linalg.norm(ego_state[2:4]) 
        / 100, 4)
    # length
    gpudrive_ego_state[0, 1] = np.round(ego_state[5] / 30, 4)
    # width
    gpudrive_ego_state[0, 2] = np.round(ego_state[6] / 10, 4)
    return gpudrive_ego_state


def get_partner_obs(
        agents,
        ego_state,
        agent_active,
        local_frame,
        num_partners=31,
        normalize_min=-100,
        normalize_max=100,
        normalize_length=30,
        normalize_width=10
    ):
    """ Get partner observations into format compatible with RL planners used in GPUDrive"""
    dist_to_ego = np.linalg.norm(agents[:, :2] - ego_state[None, :2], axis=-1)
    closest_to_ego = np.argsort(dist_to_ego)
    active_agent_ids = np.where(agent_active)[0]
    closest_to_ego = [idx for idx in closest_to_ego if idx in active_agent_ids][:num_partners]

    normalized_agents = normalize_agents(
        agents[:, None, :], 
        normalize_dict=local_frame,
        offset=0.
    )[:, 0]

    partner_obs = np.zeros((31, 6))
    for i, agent_id in enumerate(closest_to_ego):
        normalized_agent = normalized_agents[agent_id]
        
        head = normalized_agent[4]
        rel_x = _normalize_min_max(
            normalized_agent[0],
            normalize_min, 
            normalize_max
        )
        rel_y = _normalize_min_max(
            normalized_agent[1], 
            normalize_min, 
            normalize_max
        )
        
        partner = np.array([
            np.round(np.linalg.norm(normalized_agent[2:4]) / normalize_max, 4), # speed
            np.round(rel_x, 4), # x
            np.round(rel_y, 4), # y
            np.round(head / (2 * np.pi), 4), # heading
            np.round(normalized_agent[5] / normalize_length, 4), # length
            np.round(normalized_agent[6] / normalize_width, 4) # width
        ])

        partner_obs[i] = partner
    
    return partner_obs.flatten()[None,:]


def get_map_obs(
        lanes,
        ego_state,
        local_frame,
        max_lane_dist=100,
        max_num_lanes=1000,
        num_lane_features=13,
        normalize_min=-100,
        normalize_max=100
    ):
    """ Get map observations into format compatible with RL planners used in GPUDrive"""
    dist_to_ego = np.linalg.norm(
        lanes[:, :2] - ego_state[None, :2], axis=-1)
    sorted_by_distance = np.argsort(dist_to_ego)

    greater_than_max_dist_to_ego = np.where(
        dist_to_ego[sorted_by_distance] 
        > max_lane_dist)[0]
    if len(greater_than_max_dist_to_ego):
        last_valid_idx = min(
            greater_than_max_dist_to_ego[0], 
            max_num_lanes
        )
    else:
        last_valid_idx = min(
            len(greater_than_max_dist_to_ego), 
            max_num_lanes
        )
    lanes_sorted = lanes[sorted_by_distance][:last_valid_idx]
    lanes_sorted = normalize_lanes(
        lanes_sorted[:, None], 
        normalize_dict=local_frame,
        offset=0.
    )[:, 0]

    lanes_sorted[:, 0] = np.round(
        _normalize_min_max(
            lanes_sorted[:, 0], 
            normalize_min, 
            normalize_max)
        , 4)
    lanes_sorted[:, 1] = np.round(
        _normalize_min_max(
            lanes_sorted[:, 1], 
            normalize_min, 
            normalize_max)
        , 4)
    lanes_sorted[:, 2] = np.round(
        lanes_sorted[:, 2] / normalize_max, 4)
    lanes_sorted[:, 3] = np.round(
        lanes_sorted[:, 3] / normalize_max, 4)
    lanes_sorted[:, 5] = np.round(
        lanes_sorted[:, 5] / (np.pi*2), 4)

    map_tensor = np.zeros((max_num_lanes, num_lane_features))
    map_tensor[:last_valid_idx] = lanes_sorted

    return map_tensor.flatten()[None, :]


class ForwardKinematics:
    """ Simple bicycle model forward kinematics for ego vehicle."""
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
        """ Compute next state given current state and action using bicycle model."""
        def clip_speed(speed: float) -> float:
            return max(min(speed, self.max_speed), -self.max_speed)
        
        def polar_to_vector2d(r: float, theta: float):
            return [r * math.cos(theta), r * math.sin(theta)]
        
        speed = np.linalg.norm(self.velocity)
        v = clip_speed(speed + 0.5 * action[0] * self.dt)
        tan_delta = math.tan(action[1])
        beta = math.atan(0.5 * tan_delta)
        d = polar_to_vector2d(v, self.yaw + beta)
        w = v * math.cos(beta) * tan_delta / self.length
        
        self.yaw = _angle_add(self.yaw, w * self.dt)
        new_speed = clip_speed(speed + action[0] * self.dt)
        self.position[0] += d[0] * self.dt
        self.position[1] += d[1] * self.dt
        self.velocity = np.array([ new_speed*np.cos(self.yaw), new_speed*np.sin(self.yaw)])
        return np.array([self.position[0], self.position[1], self.velocity[0], self.velocity[1], 
                         self.yaw, self.length, self.width, 1.0])
    

def get_action_value_tensor() -> None:
    """ Generates a tensor mapping action indices to action values.
    Used for GPUDrive discrete action space."""
    products = None

    steer_actions: torch.Tensor = torch.round(
        torch.linspace(-torch.pi, torch.pi, 72), decimals=3
    )
    accel_actions: torch.Tensor = torch.round(
        torch.linspace(-4.0, 4.0, 32), decimals=3
    )
    head_tilt_actions: torch.Tensor = torch.Tensor([0]) 

    products = product(
        accel_actions, steer_actions, head_tilt_actions
    )
    
    # Create a mapping from action indices to action values
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


def from_json_MapVector2(j: dict) -> MapVector2:
    """
    Equivalent to the C++ from_json(const nlohmann::json &j, MapVector2 &p).
    """
    p = MapVector2()
    p.x = float(j["x"])
    p.y = float(j["y"])
    return p


def from_json_MapObject(j: dict) -> MapObject:
    """
    Equivalent to the C++ from_json(const nlohmann::json &j, MapObject &obj).
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
    obj.width = float(j["width"])
    obj.length = float(j["length"])

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
        obj.type = EntityType.NoneType

    if "mark_as_expert" in j:
        obj.markAsExpert = bool(j["mark_as_expert"])

    return obj


def from_json_MapRoad(j: dict, polylineReductionThreshold: float = 0.0) -> MapRoad:
    """
    Equivalent to the C++ from_json(const nlohmann::json &j, MapRoad &road, float polylineReductionThreshold).
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
        road.type = EntityType.NoneType
    
    # Gather geometry points
    geometry_points = []
    for point in j["geometry"]:
        geometry_points.append(from_json_MapVector2(point))

    num_segments = len(geometry_points) - 1
    sample_every_n = 1
    num_sampled_points = (num_segments + sample_every_n - 1) // sample_every_n + 1
    
    # If big enough and is road-like entity, do polyline reduction
    if num_segments >= 10 and road.type in [EntityType.RoadLane, EntityType.RoadEdge, EntityType.RoadLine]:
        skip = [False] * num_sampled_points
        k = 0
        skip_changed = True
        
        while skip_changed:
            skip_changed = False
            k = 0
            while k < num_sampled_points - 1:
                k1 = k + 1
                # find next unskipped point
                while k1 < num_sampled_points - 1 and skip[k1]:
                    k1 += 1
                if k1 >= num_sampled_points - 1:
                    break
                
                k2 = k1 + 1
                # find next unskipped point
                while k2 < num_sampled_points and skip[k2]:
                    k2 += 1
                if k2 >= num_sampled_points:
                    break

                point1 = geometry_points[k * sample_every_n]
                point2 = geometry_points[k1 * sample_every_n]
                point3 = geometry_points[k2 * sample_every_n]

                # Calculate triangle area
                area = 0.5 * abs((point1.x - point3.x) * (point2.y - point1.y)
                                 - (point1.x - point2.x) * (point3.y - point1.y))

                if area < polylineReductionThreshold:
                    skip[k1] = True
                    k = k2
                    skip_changed = True
                else:
                    k = k1

        # Force first and last point to not be skipped
        skip[0] = False
        skip[num_sampled_points - 1] = False

        # Build new geometry
        new_geometry_points = []
        for idx, s in enumerate(skip):
            if not s:
                new_geometry_points.append(geometry_points[idx * sample_every_n])

        for i, pt in enumerate(new_geometry_points):
            if i >= MAX_GEOMETRY:
                break
            road.geometry[i] = pt
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
        # match your logic regarding valid range
        if map_element_id == 4 or map_element_id >= int(MapType.NUM_TYPES.value) or map_element_id < -1:
            road.mapType = MapType.UNKNOWN
        else:
            try:
                road.mapType = MapType(map_element_id)
            except ValueError:
                road.mapType = MapType.UNKNOWN
    else:
        road.mapType = MapType.UNKNOWN

    # Compute incremental mean
    for i in range(road.numPoints):
        pt = road.geometry[i]
        road.mean.x += (pt.x - road.mean.x) / (i + 1)
        road.mean.y += (pt.y - road.mean.y) / (i + 1)

    return road


def calc_mean(j: dict) -> Tuple[float, float]:
    """
    Equivalent to the C++ calc_mean(const nlohmann::json &j).
    Computes a global (x, y) mean over object positions and road geometry.
    """
    mean_x = 0.0
    mean_y = 0.0
    num_entities = 0

    for obj in j["objects"]:
        positions = obj["position"]
        valids = obj["valid"]
        for i, pos in enumerate(positions):
            if not valids[i]:
                continue
            new_x = float(pos["x"])
            new_y = float(pos["y"])
            num_entities += 1
            # incremental mean
            mean_x += (new_x - mean_x) / num_entities
            mean_y += (new_y - mean_y) / num_entities

    for rd in j["roads"]:
        for pt in rd["geometry"]:
            new_x = float(pt["x"])
            new_y = float(pt["y"])
            num_entities += 1
            # incremental mean
            mean_x += (new_x - mean_x) / num_entities
            mean_y += (new_y - mean_y) / num_entities

    return mean_x, mean_y


def make_road_edge(road_init, j, world_mean) -> dict:
    """
    Create a 'road edge' object from two consecutive points in road_init.geometry.

    Args:
        road_init: A MapRoad instance containing geometry, ID, type, etc.
        j: An index for the starting point in the geometry list.
        world_mean: A Vector2 offset we might need to subtract (similar to ctx.singleton<WorldMeans>() in C++).

    Returns:
        A dictionary describing a 'road edge' in Python.
    """

    p1 = road_init.geometry[j]
    p2 = road_init.geometry[j + 1]
    z_offset = 1.0

    start = MapVector3(
        x = p1.x,
        y = p1.y,
        z = z_offset
    )
    end = MapVector3(
        x = p2.x,
        y = p2.y,
        z = z_offset
    )

    # Calculate a "center" position
    pos = MapVector3(
        x = (start.x + end.x) / 2.0,
        y = (start.y + end.y) / 2.0,
        z = z_offset
    )

    dx = end.x - start.x
    dy = end.y - start.y
    angle = math.atan2(dy, dx)
    rot = angle

    half_length = distance_2d(MapVector2(start.x, start.y), MapVector2(end.x, end.y)) / 2.0
    scale = (half_length, 0.1, 0.1)

    return {
        'road_id': road_init.id,
        'road_type': road_init.type,
        'start': start,
        'end': end,
        'position': pos,
        'rotation': rot,
        'scale': scale
    }


def create_road_edges(data, world_mean, max_num_edges=10000) -> List[dict]:
    edges = []
    
    num_edges = data.numRoadSegments
    num_lanes = data.numRoads

    for idx in range(num_lanes):
        road = data.roads[idx]
        num_points = road.numPoints
        
        if num_points < 2:
            continue
        
        for j in range(num_points - 1):
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
                ]
            )
            edges.append(edge_tensor)


    assert len(edges) == num_edges

    edges = np.array(edges)

    edges = np.concatenate(
        [edges[:, :-1],
         np.eye(7)[edges[:, 6].astype(int)].astype(float)], axis=-1
    )

    if len(edges) > max_num_edges:
        edges = edges[:max_num_edges]

    return edges

def from_json_Map(j: dict, polylineReductionThreshold: float = 0.0) -> Map:
    """
    Equivalent to the C++ from_json(const nlohmann::json &j, Map &map, float polylineReductionThreshold).
    """
    the_map = Map()
    
    # calculate global mean
    mx, my = calc_mean(j)
    the_map.mean = MapVector2(mx, my)

    # objects
    objects_data = j["objects"]
    the_map.numObjects = min(len(objects_data), MAX_OBJECTS)
    for idx, obj_json in enumerate(objects_data[:the_map.numObjects]):
        the_map.objects[idx] = from_json_MapObject(obj_json)

    # roads
    roads_data = j["roads"]
    the_map.numRoads = min(len(roads_data), MAX_ROADS)
    
    count_road_points = 0
    for idx, rd_json in enumerate(roads_data[:the_map.numRoads]):
        road_obj = from_json_MapRoad(rd_json, polylineReductionThreshold)
        the_map.roads[idx] = road_obj
        
        if road_obj.type.value <= EntityType.RoadLane.value:
            count_road_points += (road_obj.numPoints - 1)
        else:
            count_road_points += 1

    the_map.numRoadSegments = count_road_points
    print(the_map.numRoadSegments)
    data = the_map
    
    d = {}
    d['world_mean'] = np.array([data.mean.x, data.mean.y])
    edges = create_road_edges(data, data.mean)
    d['lanes_compressed'] = edges
    
    return d