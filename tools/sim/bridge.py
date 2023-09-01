#!/usr/bin/env python3
import argparse
import math
import os
import signal
import threading
import time
from multiprocessing import Process, Queue
from typing import Any
import socket
import json
import carla  # pylint: disable=import-error
import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array

import cereal.messaging as messaging
from cereal import log
from cereal.visionipc import VisionIpcServer, VisionStreamType
from common.basedir import BASEDIR
from common.numpy_fast import clip
from common.params import Params
from common.realtime import DT_DMON, Ratekeeper
from selfdrive.car.honda.values import CruiseButtons
from selfdrive.test.helpers import set_params_enabled
from tools.sim.lib.can import can_function

W, H = 1928, 1208
REPEAT_COUNTER = 5
PRINT_DECIMATION = 100
STEER_RATIO = 15.

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind(('127.0.0.1', 9999))
print('Bind UDP on 9999...')
print('Waiting for scenrio runner')

pm = messaging.PubMaster(['roadCameraState', 'wideRoadCameraState', 'accelerometer', 'gyroscope', 'can', "gpsLocationExternal"])
sm = messaging.SubMaster(['carControl', 'controlsState'])

def parse_args(add_args=None):
  parser = argparse.ArgumentParser(description='Bridge between CARLA and openpilot.')
  parser.add_argument('--joystick', action='store_true')
  parser.add_argument('--high_quality', action='store_true')
  parser.add_argument('--dual_camera', action='store_true')
  parser.add_argument('--town', type=str, default='Town04_Opt')
  parser.add_argument('--spawn_point')
  parser.add_argument('--v_x',default=0)
  parser.add_argument('--v_y',default=0)
  parser.add_argument('--host', dest='host', type=str, default='127.0.0.1')
  parser.add_argument('--port', dest='port', type=int, default=2000)

  return parser.parse_args(add_args)


class VehicleState:
  def __init__(self):
    self.speed = 0.0
    self.angle = 0.0
    self.bearing_deg = 0.0
    self.vel = carla.Vector3D()
    self.cruise_button = 0
    self.is_engaged = False
    self.ignition = True


def steer_rate_limit(old, new):
  # Rate limiting to 0.5 degrees per step
  limit = 0.5
  if new > old + limit:
    return old + limit
  elif new < old - limit:
    return old - limit
  else:
    return new


class Camerad:
  def __init__(self, dual_camera):
    self.frame_road_id = 0
    self.frame_wide_id = 0
    self.vipc_server = VisionIpcServer("camerad")

    self.vipc_server.create_buffers(VisionStreamType.VISION_STREAM_ROAD, 5, False, W, H)
    if dual_camera:
      self.vipc_server.create_buffers(VisionStreamType.VISION_STREAM_WIDE_ROAD, 5, False, W, H)
    self.vipc_server.start_listener()

    # set up for pyopencl rgb to yuv conversion
    self.ctx = cl.create_some_context()
    self.queue = cl.CommandQueue(self.ctx)
    cl_arg = f" -DHEIGHT={H} -DWIDTH={W} -DRGB_STRIDE={W * 3} -DUV_WIDTH={W // 2} -DUV_HEIGHT={H // 2} -DRGB_SIZE={W * H} -DCL_DEBUG "

    kernel_fn = os.path.join(BASEDIR, "tools/sim/rgb_to_nv12.cl")
    with open(kernel_fn) as f:
      prg = cl.Program(self.ctx, f.read()).build(cl_arg)
      self.krnl = prg.rgb_to_nv12
    self.Wdiv4 = W // 4 if (W % 4 == 0) else (W + (4 - W % 4)) // 4
    self.Hdiv4 = H // 4 if (H % 4 == 0) else (H + (4 - H % 4)) // 4

  def cam_callback_road(self, image):
    self._cam_callback(image, self.frame_road_id, 'roadCameraState', VisionStreamType.VISION_STREAM_ROAD)
    self.frame_road_id += 1

  def cam_callback_wide_road(self, image):
    self._cam_callback(image, self.frame_wide_id, 'wideRoadCameraState', VisionStreamType.VISION_STREAM_WIDE_ROAD)
    self.frame_wide_id += 1

  def _cam_callback(self, image, frame_id, pub_type, yuv_type):
    img = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    img = np.reshape(img, (H, W, 4))
    img = img[:, :, [0, 1, 2]].copy()

    # convert RGB frame to YUV
    rgb = np.reshape(img, (H, W * 3))
    rgb_cl = cl_array.to_device(self.queue, rgb)
    yuv_cl = cl_array.empty_like(rgb_cl)
    self.krnl(self.queue, (np.int32(self.Wdiv4), np.int32(self.Hdiv4)), None, rgb_cl.data, yuv_cl.data).wait()
    yuv = np.resize(yuv_cl.get(), rgb.size // 2)
    eof = int(frame_id * 0.05 * 1e9)

    self.vipc_server.send(yuv_type, yuv.data.tobytes(), frame_id, eof, eof)

    dat = messaging.new_message(pub_type)
    msg = {
      "frameId": frame_id,
      "transform": [1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0]
    }
    setattr(dat, pub_type, msg)
    pm.send(pub_type, dat)

def imu_callback(imu, vehicle_state):
  # send 5x since 'sensor_tick' doesn't seem to work. limited by the world tick?
  for _ in range(5):
    vehicle_state.bearing_deg = math.degrees(imu.compass)
    dat = messaging.new_message('accelerometer')
    dat.accelerometer.sensor = 4
    dat.accelerometer.type = 0x10
    dat.accelerometer.timestamp = dat.logMonoTime  # TODO: use the IMU timestamp
    dat.accelerometer.init('acceleration')
    dat.accelerometer.acceleration.v = [imu.accelerometer.x, imu.accelerometer.y, imu.accelerometer.z]
    pm.send('accelerometer', dat)

    # copied these numbers from locationd
    dat = messaging.new_message('gyroscope')
    dat.gyroscope.sensor = 5
    dat.gyroscope.type = 0x10
    dat.gyroscope.timestamp = dat.logMonoTime  # TODO: use the IMU timestamp
    dat.gyroscope.init('gyroUncalibrated')
    dat.gyroscope.gyroUncalibrated.v = [imu.gyroscope.x, imu.gyroscope.y, imu.gyroscope.z]
    pm.send('gyroscope', dat)
    time.sleep(0.01)


def panda_state_function(vs: VehicleState, exit_event: threading.Event):
  pm = messaging.PubMaster(['pandaStates'])
  while not exit_event.is_set():
    dat = messaging.new_message('pandaStates', 1)
    dat.valid = True
    dat.pandaStates[0] = {
      'ignitionLine': vs.ignition,
      'pandaType': "blackPanda",
      'controlsAllowed': True,
      'safetyModel': 'hondaNidec'
    }
    pm.send('pandaStates', dat)
    time.sleep(0.5)


def peripheral_state_function(exit_event: threading.Event):
  pm = messaging.PubMaster(['peripheralState'])
  while not exit_event.is_set():
    dat = messaging.new_message('peripheralState')
    dat.valid = True
    # fake peripheral state data
    dat.peripheralState = {
      'pandaType': log.PandaState.PandaType.blackPanda,
      'voltage': 12000,
      'current': 5678,
      'fanSpeedRpm': 1000
    }
    pm.send('peripheralState', dat)
    time.sleep(0.5)


def gps_callback(gps, vehicle_state):
  dat = messaging.new_message('gpsLocationExternal')

  # transform vel from carla to NED
  # north is -Y in CARLA
  velNED = [
    -vehicle_state.vel.y,  # north/south component of NED is negative when moving south
    vehicle_state.vel.x,  # positive when moving east, which is x in carla
    vehicle_state.vel.z,
  ]

  dat.gpsLocationExternal = {
    "unixTimestampMillis": int(time.time() * 1000),
    "flags": 1,  # valid fix
    "accuracy": 1.0,
    "verticalAccuracy": 1.0,
    "speedAccuracy": 0.1,
    "bearingAccuracyDeg": 0.1,
    "vNED": velNED,
    "bearingDeg": vehicle_state.bearing_deg,
    "latitude": gps.latitude,
    "longitude": gps.longitude,
    "altitude": gps.altitude,
    "speed": vehicle_state.speed,
    "source": log.GpsLocationData.SensorSource.ublox,
  }

  pm.send('gpsLocationExternal', dat)


def fake_driver_monitoring(exit_event: threading.Event):
  pm = messaging.PubMaster(['driverStateV2', 'driverMonitoringState'])
  while not exit_event.is_set():
    # dmonitoringmodeld output
    dat = messaging.new_message('driverStateV2')
    dat.driverStateV2.leftDriverData.faceOrientation = [0., 0., 0.]
    dat.driverStateV2.leftDriverData.faceProb = 1.0
    dat.driverStateV2.rightDriverData.faceOrientation = [0., 0., 0.]
    dat.driverStateV2.rightDriverData.faceProb = 1.0
    pm.send('driverStateV2', dat)

    # dmonitoringd output
    dat = messaging.new_message('driverMonitoringState')
    dat.driverMonitoringState = {
      "faceDetected": True,
      "isDistracted": False,
      "awarenessStatus": 1.,
    }
    pm.send('driverMonitoringState', dat)

    time.sleep(DT_DMON)


def can_function_runner(vs: VehicleState, exit_event: threading.Event):
  i = 0
  while not exit_event.is_set():
    can_function(pm, vs.speed, vs.angle, i, vs.cruise_button, vs.is_engaged)
    time.sleep(0.01)
    i += 1


def connect_carla_client(host: str, port: int):
  client = carla.Client(host, port)
  client.set_timeout(5)
  return client


class CarlaBridge:

  def __init__(self, arguments):
    set_params_enabled()

    self.params = Params()

    msg = messaging.new_message('liveCalibration')
    msg.liveCalibration.validBlocks = 20
    msg.liveCalibration.rpyCalib = [0.0, 0.0, 0.0]
    self.params.put("CalibrationParams", msg.to_bytes())
    self.params.put_bool("DisengageOnAccelerator", True)

    self._args = arguments
    self._carla_objects = []
    self._camerad = None
    self._exit_event = threading.Event()
    self._threads = []
    self._keep_alive = True
    self.started = False
    signal.signal(signal.SIGTERM, self._on_shutdown)
    self._exit = threading.Event()

  def _on_shutdown(self, signal, frame):
    self._keep_alive = False

  def bridge_keep_alive(self, q: Queue, retries: int):
    try:
      while self._keep_alive:
        try:
          self._run(q)
          break
        except RuntimeError as e:
          self.close()
          if retries == 0:
            raise

          # Reset for another try
          self._carla_objects = []
          self._threads = []
          self._exit_event = threading.Event()

          retries -= 1
          if retries <= -1:
            print(f"Restarting bridge. Error: {e} ")
          else:
            print(f"Restarting bridge. Retries left {retries}. Error: {e} ")
    finally:
      # Clean up resources in the opposite order they were created.
      self.close()
      time.sleep(0.5)
  
  def _run(self, q: Queue):
    client = connect_carla_client(self._args.host, self._args.port)
    self.world = client.load_world(self._args.town)
    settings = self.world.get_settings()
    settings.synchronous_mode = True  # Enables synchronous mode
    settings.fixed_delta_seconds = 0.05
    self.world.apply_settings(settings)
    self.world.set_weather(carla.WeatherParameters.ClearSunset)
    self.spectator = self.world.get_spectator()
    self.spectator.set_transform(carla.Transform(carla.Location(
        x=10, y=0, z=100), carla.Rotation(pitch=-90, roll=90)))
    # if not self._args.high_quality:
    #   self.world.unload_map_layer(carla.MapLayer.Foliage)
    #   self.world.unload_map_layer(carla.MapLayer.Buildings)
    #   self.world.unload_map_layer(carla.MapLayer.ParkedVehicles)
    #   self.world.unload_map_layer(carla.MapLayer.Props)
    #   self.world.unload_map_layer(carla.MapLayer.StreetLights)
    #   self.world.unload_map_layer(carla.MapLayer.Particles)

    blueprint_library = self.world.get_blueprint_library()

    world_map = self.world.get_map()

    vehicle_bp = blueprint_library.filter('vehicle.tesla.*')[1]
    vehicle_bp.set_attribute('role_name', 'Ego')
    # spawn_points = world_map.get_spawn_points()
    # vehicle = self.world.spawn_actor(vehicle_bp, spawn_points[20])
    spawn_point = self._args.spawn_point
    vehicle = self.world.spawn_actor(vehicle_bp,spawn_point)
    self._carla_objects.append(vehicle)
    max_steer_angle = vehicle.get_physics_control().wheels[0].max_steer_angle
    #*spawn some challenge vehicle
    # sp = spawn_points[self._args.num_selected_spawn_point + 99]
    # sp2 = carla.Transform(carla.Location(sp.location.x+4,sp.location.y+20,sp.location.z),sp.rotation)
    # adv = self.world.try_spawn_actor(vehicle_bp, sp2)
    # adv.set_target_velocity(carla.Vector3D(0,3,0))
    # self._carla_objects.append(adv)
    #*spawn some static cone
    # cone_bp = blueprint_library.find('static.prop.trafficcone01')
    # cone = world.spawn_actor(cone_bp,sp2)
    # self._carla_objects.append(cone)
    #*spawn_pedestrian
    # ped_bp = blueprint_library.find('walker.pedestrian.0001')
    # ped = world.spawn_actor(ped_bp,sp2)
    # self._carla_objects.append(ped)
    #*spawn_two_wheel_bicycle
    # bic_bp = blueprint_library.find('vehicle.bh.crossbike')
    # bic = world.spawn_actor(bic_bp,sp2)
    # self._carla_objects.append(bic)
    #*move spectator
    # spectator = self.world.get_spectator()
    # spectator.set_transform(carla.Transform(carla.Location(sp.location.x,sp.location.y,80),carla.Rotation(pitch=-90,roll = 90)))
    # make tires less slippery
    # wheel_control = carla.WheelPhysicsControl(tire_friction=5)
    physics_control = vehicle.get_physics_control()
    physics_control.mass = 2326
    # physics_control.wheels = [wheel_control]*4
    physics_control.torque_curve = [[20.0, 500.0], [5000.0, 500.0]]
    physics_control.gear_switch_time = 0.0
    vehicle.apply_physics_control(physics_control)
    # vehicle.set_autopilot()
    transform = carla.Transform(carla.Location(x=0.8, z=1.13))

    def create_camera(fov, callback):
      blueprint = blueprint_library.find('sensor.camera.rgb')
      blueprint.set_attribute('image_size_x', str(W))
      blueprint.set_attribute('image_size_y', str(H))
      blueprint.set_attribute('fov', str(fov))
      if not self._args.high_quality:
        blueprint.set_attribute('enable_postprocess_effects', 'False')
      camera = self.world.spawn_actor(blueprint, transform, attach_to=vehicle)
      camera.listen(callback)
      return camera

    self._camerad = Camerad(self._args.dual_camera)
    if self._args.dual_camera:
      road_wide_camera = create_camera(fov=120, callback=self._camerad.cam_callback_wide_road)  # fov bigger than 120 shows unwanted artifacts
      self._carla_objects.append(road_wide_camera)
    road_camera = create_camera(fov=40, callback=self._camerad.cam_callback_road)
    self._carla_objects.append(road_camera)

    vehicle_state = VehicleState()
    vehicle_state.speed = math.sqrt(math.pow(self._args.v_x,2)+math.pow(self._args.v_y,2))
    vehicle_state.cruise_button = 3
    vehicle_state.is_engaged = True

    # re-enable IMU
    imu_bp = blueprint_library.find('sensor.other.imu')
    imu_bp.set_attribute('sensor_tick', '0.01')
    imu = self.world.spawn_actor(imu_bp, transform, attach_to=vehicle)
    imu.listen(lambda imu: imu_callback(imu, vehicle_state))

    gps_bp = blueprint_library.find('sensor.other.gnss')
    gps = self.world.spawn_actor(gps_bp, transform, attach_to=vehicle)
    gps.listen(lambda gps: gps_callback(gps, vehicle_state))
    self.params.put_bool("UbloxAvailable", True)

    self._carla_objects.extend([imu, gps])
    # launch fake car threads
    self._threads.append(threading.Thread(target=panda_state_function, args=(vehicle_state, self._exit_event,)))
    self._threads.append(threading.Thread(target=peripheral_state_function, args=(self._exit_event,)))
    self._threads.append(threading.Thread(target=fake_driver_monitoring, args=(self._exit_event,)))
    self._threads.append(threading.Thread(target=can_function_runner, args=(vehicle_state, self._exit_event,)))
    for t in self._threads:
      t.start()

    # init
    # throttle_ease_out_counter = REPEAT_COUNTER
    # brake_ease_out_counter = REPEAT_COUNTER
    # steer_ease_out_counter = REPEAT_COUNTER
    vc = carla.VehicleControl(throttle=0, steer=0, brake=0, reverse=False)

    is_openpilot_engaged = True
    # throttle_out = steer_out = brake_out = 0.
    # throttle_op = steer_op = brake_op = 0.
    # throttle_manual = steer_manual = brake_manual = 0.
    old_steer = 0
    # old_steer = old_brake = old_throttle = 0.
    # throttle_manual_multiplier = 0.7  # keyboard signal is always 1
    # brake_manual_multiplier = 0.7  # keyboard signal is always 1
    # steer_manual_multiplier = 45 * STEER_RATIO  # keyboard signal is always 1
    # Simulation tends to be slow in the initial steps. This prevents lagging later
    #! 确认此处的等待时间
    for _ in range(300):
      self.world.tick()

    # loop
    rk = Ratekeeper(100, print_delay_threshold=0.05)
    # #*添加主车的初始速度
    vehicle.set_target_velocity(carla.Vector3D(self._args.v_x,self._args.v_y,0))
    self.world.tick()
    while self._keep_alive:
      # 1. Read the throttle, steer and brake from op or manual controls
      # 2. Set instructions in Carla
      # 3. Send current carstate to op via can
      # cruise_button = 0
      # throttle_out = steer_out = brake_out = 0.0
      # throttle_op = steer_op = brake_op = 0.0
      # throttle_manual = steer_manual = brake_manual = 0.0
      # --------------Step 1-------------------------------
      if not q.empty():
        message = q.get()
        m = message.split('_')
        if m[0] == "steer":
          steer_manual = float(m[1])
          is_openpilot_engaged = False
        elif m[0] == "throttle":
          throttle_manual = float(m[1])
          is_openpilot_engaged = False
        elif m[0] == "brake":
          brake_manual = float(m[1])
          is_openpilot_engaged = False
        elif m[0] == "reverse":
          cruise_button = CruiseButtons.CANCEL
          is_openpilot_engaged = False
        elif m[0] == "cruise":
          if m[1] == "down":
            cruise_button = CruiseButtons.DECEL_SET
            is_openpilot_engaged = True
          elif m[1] == "up":
            cruise_button = CruiseButtons.RES_ACCEL
            is_openpilot_engaged = True
          elif m[1] == "cancel":
            cruise_button = CruiseButtons.CANCEL
            is_openpilot_engaged = False
        elif m[0] == "ignition":
          vehicle_state.ignition = not vehicle_state.ignition
        elif m[0] == "quit":
          break
        # throttle_out = throttle_manual * throttle_manual_multiplier
        # steer_out = steer_manual * steer_manual_multiplier
        # brake_out = brake_manual * brake_manual_multiplier

        # old_steer = steer_out
        # old_throttle = throttle_out
        # old_brake = brake_out
      if is_openpilot_engaged:
        sm.update(0)
        # TODO gas and brake is deprecated
        throttle_op = clip(sm['carControl'].actuators.accel / 1.6, 0.0, 1.0)
        brake_op = clip(-sm['carControl'].actuators.accel / 4.0, 0.0, 1.0)
        steer_op = sm['carControl'].actuators.steeringAngleDeg

        throttle_out = throttle_op
        steer_out = steer_op
        brake_out = brake_op

        steer_out = steer_rate_limit(old_steer, steer_out)
        old_steer = steer_out

      # else:
      #   if throttle_out == 0 and old_throttle > 0:
      #     if throttle_ease_out_counter > 0:
      #       throttle_out = old_throttle
      #       throttle_ease_out_counter += -1
      #     else:
      #       throttle_ease_out_counter = REPEAT_COUNTER
      #       old_throttle = 0

      #   if brake_out == 0 and old_brake > 0:
      #     if brake_ease_out_counter > 0:
      #       brake_out = old_brake
      #       brake_ease_out_counter += -1
      #     else:
      #       brake_ease_out_counter = REPEAT_COUNTER
      #       old_brake = 0

      #   if steer_out == 0 and old_steer != 0:
      #     if steer_ease_out_counter > 0:
      #       steer_out = old_steer
      #       steer_ease_out_counter += -1
      #     else:
      #       steer_ease_out_counter = REPEAT_COUNTER
      #       old_steer = 0

      # --------------Step 2-------------------------------
      steer_carla = steer_out / (max_steer_angle * STEER_RATIO * -1)

      steer_carla = np.clip(steer_carla, -1, 1)
      steer_out = steer_carla * (max_steer_angle * STEER_RATIO * -1)
      old_steer = steer_carla * (max_steer_angle * STEER_RATIO * -1)

      vc.throttle = throttle_out / 0.6
      vc.steer = steer_carla
      vc.brake = brake_out
      vehicle.apply_control(vc)

      # --------------Step 3-------------------------------
      vel = vehicle.get_velocity()
      speed = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)  # in m/s
      vehicle_state.speed = speed
      vehicle_state.vel = vel
      vehicle_state.angle = steer_out
      vehicle_state.cruise_button = 0
      vehicle_state.is_engaged = is_openpilot_engaged
      if rk.frame % PRINT_DECIMATION == 0:
        print("frame: ", "engaged:", is_openpilot_engaged, "; throttle: ", round(vc.throttle, 3), "; steer(c/deg): ",
              round(vc.steer, 3), round(steer_out, 3), "; brake: ", round(vc.brake, 3))

      if rk.frame % 5 == 0:
        self.world.tick()
      rk.keep_time()
      self.started = True


  def close(self):
    print('close bridge')
    self.started = False
    self._exit_event.set()
    try:
      if hasattr(carla_bridge,'world'):
        for s in self._carla_objects:
          s.destroy()
        new_settings = self.world.get_settings()
        new_settings.fixed_delta_seconds = None
        new_settings.synchronous_mode = False
        self.world.apply_settings(new_settings)
        time.sleep(0.5)
    except Exception as e:
      print("Failed to destroy carla object", e)
    for t in reversed(self._threads):
      t.join()


  def run(self, queue, retries=-1):
    bridge_p = Process(target=self.bridge_keep_alive, args=(queue, retries), daemon=True)
    bridge_p.start()
    return bridge_p


if __name__ == "__main__":
  data, addr = s.recvfrom(1024)
  init_con = json.loads(bytes.decode(data))
  reply = 'Hello, %s!' % data.decode('utf-8')
  s.sendto(reply.encode('utf-8'), addr)
  print('received inital condition from test manager')
  q: Any = Queue()
  args = parse_args()
  args.town = init_con['map']
  args.spawn_point = carla.Transform(carla.Location(x=init_con['x'],y=init_con['y'],z=init_con['z']),carla.Rotation(pitch=init_con['pitch'],roll=init_con['roll'],yaw=init_con['yaw']))
  args.v_x = init_con['v_x']
  args.v_y = init_con['v_y']
  carla_bridge = CarlaBridge(args)
  p = carla_bridge.run(q)

  if args.joystick:
    # start input poll for joystick
    from tools.sim.lib.manual_ctrl import wheel_poll_thread

    wheel_poll_thread(q)
  else:
    # start input poll for keyboard
    from tools.sim.lib.keyboard_ctrl import keyboard_poll_thread

    keyboard_poll_thread(q)
  p.join()
  s.close()
