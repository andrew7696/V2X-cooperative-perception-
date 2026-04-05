import time
import numpy as np

try:
    import carla
    CARLA_AVAILABLE = True
except ImportError:
    CARLA_AVAILABLE = False


class CARLAEnv:
    """
    CARLA-based occlusion scenario for V2X cooperative perception.

    Scene layout (Vehicle B's approximate coordinate frame):
      Vehicle B  at (0, 0)      heading East (yaw=0°)
      Vehicle A  at (10, -20)   heading ~45° NE — has clear LOS to pedestrian
      Truck      at (30, 0)     stationary — blocks B's LOS to pedestrian
      Pedestrian at (45, 0)     stationary behind truck

    CARLA coordinate convention:
      X = East, Y = South, Z = Up   (left-hand system, yaw clockwise from North)
    We convert yaw to standard math radians (CCW from East) via:
      heading_rad = -math.radians(yaw_degrees)
    """

    def __init__(self, host: str = 'localhost', port: int = 2000, town: str = 'Town04'):
        if not CARLA_AVAILABLE:
            raise RuntimeError(
                "CARLA Python API not found. "
                "Install it from your CARLA installation: PythonAPI/carla/dist/"
            )
        import math
        self._math = math

        self.client = carla.Client(host, port)
        self.client.set_timeout(15.0)
        self.world = self.client.load_world(town)
        self.blueprint_library = self.world.get_blueprint_library()

        self._configure_sync()

        self.vehicle_a = None
        self.vehicle_b = None
        self.obstacle = None
        self.pedestrian = None
        self.lidar_a_actor = None
        self.lidar_b_actor = None

        self._lidar_a: np.ndarray = np.zeros((0, 4), dtype=np.float32)
        self._lidar_b: np.ndarray = np.zeros((0, 4), dtype=np.float32)

        self._spawn_scene()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _configure_sync(self):
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)

    def _spawn_scene(self):
        import carla

        bp_lib = self.blueprint_library
        vehicle_bp = bp_lib.filter('vehicle.tesla.model3')[0]
        vehicle_bp.set_attribute('role_name', 'ego')

        # Vehicle B — the cooperative recipient, blocked view
        self.vehicle_b = self.world.spawn_actor(
            vehicle_bp,
            carla.Transform(carla.Location(x=0, y=0, z=0.5), carla.Rotation(yaw=0)),
        )

        # Vehicle A — has line-of-sight to pedestrian
        self.vehicle_a = self.world.spawn_actor(
            vehicle_bp,
            carla.Transform(carla.Location(x=10, y=-20, z=0.5), carla.Rotation(yaw=45)),
        )

        # Blocking truck
        truck_bp = bp_lib.filter('vehicle.carlamotors.carlacola')[0]
        self.obstacle = self.world.spawn_actor(
            truck_bp,
            carla.Transform(carla.Location(x=30, y=0, z=0.5), carla.Rotation(yaw=0)),
        )

        # Pedestrian (target)
        ped_bps = bp_lib.filter('walker.pedestrian.*')
        ped_bp = ped_bps[0]
        self.pedestrian = self.world.spawn_actor(
            ped_bp,
            carla.Transform(carla.Location(x=45, y=0, z=0.5)),
        )

        # LiDAR sensors
        lidar_bp = bp_lib.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('range', '50')
        lidar_bp.set_attribute('rotation_frequency', '20')
        lidar_bp.set_attribute('channels', '64')
        lidar_bp.set_attribute('points_per_second', '500000')
        lidar_transform = carla.Transform(carla.Location(z=2.0))

        self.lidar_a_actor = self.world.spawn_actor(
            lidar_bp, lidar_transform, attach_to=self.vehicle_a
        )
        self.lidar_b_actor = self.world.spawn_actor(
            lidar_bp, lidar_transform, attach_to=self.vehicle_b
        )

        self.lidar_a_actor.listen(self._cb_lidar_a)
        self.lidar_b_actor.listen(self._cb_lidar_b)

        # Warm up
        for _ in range(5):
            self.world.tick()
        time.sleep(0.1)

    # ------------------------------------------------------------------
    # Sensor callbacks
    # ------------------------------------------------------------------

    def _cb_lidar_a(self, data):
        raw = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4).copy()
        self._lidar_a = raw

    def _cb_lidar_b(self, data):
        raw = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4).copy()
        self._lidar_b = raw

    # ------------------------------------------------------------------
    # Pose & GT helpers
    # ------------------------------------------------------------------

    def _get_pose(self, vehicle) -> dict:
        """Return world pose with heading in standard radians (CCW from East)."""
        t = vehicle.get_transform()
        return {
            'x': t.location.x,
            'y': t.location.y,
            # CARLA yaw: degrees, clockwise from North (+Y South axis)
            # Standard math heading: radians, CCW from East (+X axis)
            'heading': -self._math.radians(t.rotation.yaw),
        }

    def _get_gt_boxes(self) -> list:
        """
        Return ground-truth 3D boxes in Vehicle B's local BEV frame.
        Boxes are dicts: {x, y, z, w, l, h, class, score=1.0}
        """
        pose_b = self._get_pose(self.vehicle_b)
        cos_b = self._math.cos(-pose_b['heading'])
        sin_b = self._math.sin(-pose_b['heading'])

        gt = []
        for actor in [self.pedestrian, self.obstacle, self.vehicle_a]:
            loc = actor.get_location()
            bb = actor.bounding_box
            dx = loc.x - pose_b['x']
            dy = loc.y - pose_b['y']
            local_x = cos_b * dx - sin_b * dy
            local_y = sin_b * dx + cos_b * dy
            cls = 0 if 'walker' in actor.type_id else 1
            gt.append({
                'x': local_x, 'y': local_y, 'z': loc.z,
                'w': bb.extent.x * 2, 'l': bb.extent.y * 2, 'h': bb.extent.z * 2,
                'class': cls, 'score': 1.0,
            })
        return gt

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self) -> dict:
        """
        Advance simulation by one tick.

        Returns:
            {
              'lidar_a': np.ndarray (N, 4),
              'lidar_b': np.ndarray (N, 4),
              'pose_a':  {'x', 'y', 'heading'},
              'pose_b':  {'x', 'y', 'heading'},
              'gt_boxes': list of box dicts in Vehicle B's frame,
            }
        """
        self.world.tick()
        time.sleep(0.05)

        return {
            'lidar_a': self._lidar_a,
            'lidar_b': self._lidar_b,
            'pose_a': self._get_pose(self.vehicle_a),
            'pose_b': self._get_pose(self.vehicle_b),
            'gt_boxes': self._get_gt_boxes(),
        }

    def close(self):
        """Destroy all spawned actors and restore async mode."""
        for sensor in [self.lidar_a_actor, self.lidar_b_actor]:
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
        for actor in [self.vehicle_a, self.vehicle_b, self.obstacle, self.pedestrian]:
            if actor is not None:
                actor.destroy()
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        self.world.apply_settings(settings)
