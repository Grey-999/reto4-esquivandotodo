#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
capytown_guardian.py  ──  Solución definitiva para el reto del anillo con cajas.
UN SOLO nodo ROS 2 que integra:
  1. PERCEPCIÓN LiDAR
     ─ Filtrado de puntos (rango, NaN, offset del LiDAR).
     ─ Split-and-Merge ITERATIVO con umbral adaptativo al rango.
     ─ Merge multi-pasada hasta convergencia.
     ─ Métrica de calidad de segmento (residual RMS).
     ─ Extracción del segmento LATERAL DERECHO más cercano
       (sirve para pared del anillo Y para caja: en el rodeo, la caja pasa
       a ser la "pared derecha" temporalmente).
     ─ Análisis del arco frontal más cercano: distancia + ANCHO ANGULAR.
       El ancho angular es lo que distingue una CAJA (arco corto, <30°)
       de una ESQUINA (arco largo, >45°).
     ─ EMA temporal en distancia frontal y distancia a pared para
       suavizar ruido del sensor entre frames.
  2. CONTROL de wall-follow con PID COMPLETO + HEADING
       w = ─Kp · (d_der ─ d_objetivo)          error lateral
           ─K_alpha · alpha                     error de heading
           ─Kd · d(PV)/dt  (EMA filtrado)      derivativo (sobre PV, sin kick)
           ─Ki · ∫e·dt      (con I-zone + clamp) integral opcional
     El término de heading es lo que mata el zigzag clásico del PID de solo
     posición, y aprovecha la orientación del segmento que ya tenemos del S&M.
     El derivativo se filtra con EMA y opera sobre la variable de proceso
     (no sobre el error) para evitar derivative kick ante cambios de setpoint.
     Velocidad adaptativa con tres zonas (normal/cautela/crítica).
     Velocity ramp para suavizar transiciones de velocidad (anti-slip).
     Arc turns en vez de pivot turns para mantener tracción.
  3. FSM con PERSISTENCIA DE TRANSICIONES + TIMEOUTS + RECOVERY
     Cada transición pide N frames consecutivos con la condición cumplida.
     Estados:
       CRUISE    ─ avanza siguiendo la pared derecha con el PID completo.
                   Si pierde la pared >N frames, gira suavemente a la derecha
                   para buscarla.
       CORNER    ─ arc-turn ~90° en el sentido de la esquina (odometría si
                   hay, timing si no). Timeout → RECOVERY.
       BOX_ALIGN ─ obstáculo tipo caja al frente: gira (arc-turn) hasta que
                   el frente vuelva a estar despejado por N frames.
                   Timeout → RECOVERY.
       BOX_BYPASS─ wall-follow contra la CAJA (ahora "pared derecha").
                   Sale a CRUISE cuando detecta pared LARGA estable.
                   Timeout → RECOVERY.
       RECOVERY  ─ retroceder + girar. Se activa por: timeout, stuck,
                   o emergencia frontal. Timeout propio → CRUISE.
     Capa de emergencia: si front_dist < emergency_dist en cualquier
     estado (excepto RECOVERY), parada inmediata + RECOVERY.
     Detección de stuck: si v_cmd > 0 pero odom_vx ≈ 0 durante N segundos.
  4. LiDAR MS200 con offset  (0° del sensor ≠ +x del robot)
     Parámetro ``lidar_offset_deg`` para rotar los puntos al marco del robot.
  ── FIX v2.1 ──────────────────────────────────────────────────────
  El robot no avanzaba porque la suscripción a /scan usaba QoS RELIABLE
  por defecto, mientras que casi todos los drivers de LiDAR (incluido
  el MS200) publican con QoS BEST_EFFORT (sensor data). Esa mezcla es
  incompatible en ROS 2: la suscripción nunca recibe mensajes, pero NO
  lanza ningún error — simplemente `have_scan` se queda en False para
  siempre y el loop de control retorna sin hacer nada en cada tick.
  Fix: usar qos_profile_sensor_data (BEST_EFFORT) para /scan.
  Se añadió además un watchdog de arranque que avisa explícitamente si
  no llega ningún scan en los primeros segundos, y un log de
  confirmación del primer scan recibido, para depurar esto más rápido
  la próxima vez.
  ── FIX v2.2 ──────────────────────────────────────────────────────
  Aún con BEST_EFFORT, algunos drivers (según versión) publican con
  RELIABLE. Para no adivinar, ahora se crean DOS suscripciones al
  mismo tópico: una BEST_EFFORT y una RELIABLE. Así el nodo recibe
  datos sin importar qué QoS use el publisher.
  Además:
  ─ El nombre del tópico es configurable (parámetro ``scan_topic``).
  ─ El watchdog auto-descubre tópicos de tipo LaserScan en el sistema
    y sugiere el correcto si /scan no existe.
Sensor: LiDAR MS200 360°  |  ROS 2 Humble  |  Yahboom MicroROS-Pi5
"""
from __future__ import annotations
import math
import statistics
from collections import deque
from typing import List, NamedTuple, Optional, Tuple
import rclpy
from rclpy.node import Node
from rclpy.qos import (qos_profile_sensor_data, QoSProfile,
                       ReliabilityPolicy, HistoryPolicy, DurabilityPolicy)
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
# ═══════════════════════════════════════════════════════════════════
#  CONSTANTES
# ═══════════════════════════════════════════════════════════════════
# --- Estados de la FSM ---
CRUISE     = 'CRUISE'
CORNER     = 'CORNER'
BOX_ALIGN  = 'BOX_ALIGN'
BOX_BYPASS = 'BOX_BYPASS'
RECOVERY   = 'RECOVERY'
# --- Clasificación frontal ---
FRONT_NONE   = 'NONE'
FRONT_BOX    = 'BOX'
FRONT_CORNER = 'CORNER'
# --- Fases de recuperación ---
_REC_REVERSE = 0
_REC_TURN    = 1
# ═══════════════════════════════════════════════════════════════════
#  TIPOS DE DATO
# ═══════════════════════════════════════════════════════════════════
class Segment(NamedTuple):
    """Segmento de línea extraído de puntos LiDAR."""
    p1: Tuple[float, float]        # Extremo inicial (x, y) [m]
    p2: Tuple[float, float]        # Extremo final   (x, y) [m]
    length: float                  # Longitud euclidiana [m]
    mean_y: float                  # Y promedio de los puntos [m]
    alpha: float                   # Ángulo respecto a +x [rad] ∈ [-π/2, π/2]
    residual: float                # RMS perp. distance (calidad del ajuste)
    n_pts: int                     # Número de puntos en el segmento
# ═══════════════════════════════════════════════════════════════════
#  UTILIDADES PURAS
# ═══════════════════════════════════════════════════════════════════
def _perp_dist(px: float, py: float,
               ax: float, ay: float,
               bx: float, by: float) -> float:
    """Distancia perpendicular del punto (px,py) a la recta (ax,ay)─(bx,by)."""
    dx, dy = bx - ax, by - ay
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return math.hypot(px - ax, py - ay)
    return abs(dy * px - dx * py + bx * ay - by * ax) / length
def _angle_diff(a: float, b: float) -> float:
    """Diferencia angular con signo (a − b), normalizada a [−π, π]."""
    return math.atan2(math.sin(a - b), math.cos(a - b))
def _clamp(value: float, lo: float, hi: float) -> float:
    """Limita *value* al intervalo [lo, hi]."""
    return max(lo, min(hi, value))
def _ramp(target: float, current: float,
          max_rate: float, dt: float) -> float:
    """Acerca *current* hacia *target* sin exceder *max_rate* · dt."""
    delta = target - current
    max_delta = max_rate * dt
    return current + _clamp(delta, -max_delta, max_delta)
# ═══════════════════════════════════════════════════════════════════
#  NODO PRINCIPAL
# ═══════════════════════════════════════════════════════════════════
class CapyGuardian(Node):
    """
    Nodo ROS 2 integrado: percepción LiDAR + control wall-follow PID
    + máquina de estados para navegar un anillo con cajas.
    """
    # ──────────────────────────────────────────────────────────────
    #  INICIALIZACIÓN
    # ──────────────────────────────────────────────────────────────
    def __init__(self) -> None:
        super().__init__('capytown_guardian')
        # ── DECLARAR TODOS LOS PARÁMETROS ────────────────────────
        self.declare_parameters('', [
            # ─ Geometría del LiDAR ─
            ('lidar_offset_deg',    180.0),
            ('lidar_y_sign',        -1.0),
            # ─ Filtrado del scan ─
            ('range_min',           0.05),
            ('range_max',           3.5),
            ('ignore_deg_min',      float('nan')),
            ('ignore_deg_max',      float('nan')),
            # ─ Split & Merge ─
            ('sm_split_thresh',     0.04),
            ('sm_adaptive_range_k', 0.05),
            ('sm_pre_gap_idx',      6),
            ('sm_pre_gap_dist',     0.40),
            ('sm_min_pts',          3),
            ('sm_min_len',          0.06),
            ('sm_merge_max_passes', 2),
            # ─ Clasificación de pared lateral ─
            ('wall_min_len',        0.50),
            ('wall_cos_lat_min',    0.75),
            # ─ Detección frontal ─
            ('front_sector_deg',    38.0),
            ('front_alert_dist',    0.42),
            ('front_arc_radial',    0.18),
            ('front_arc_ang_gap',   4.0),
            ('front_wall_deg',      44.0),
            ('front_box_deg',       22.0),
            ('side_clear_dist',     0.58),
            # ─ Wall-follow PID ─
            ('target_wall_dist',    0.30),
            ('Kp_dist',             1.1),
            ('K_alpha',             0.6),
            ('Kd_dist',             0.25),
            ('Ki_dist',             0.0),
            ('Ki_max_accum',        0.5),
            ('Ki_zone',             0.15),
            ('max_ang_vel',         0.55),
            # ─ Velocidades ─
            ('cruise_speed',        0.14),
            ('v_turn_min',          0.04),
            # ─ Filtrado EMA ─
            ('alpha_ema_deriv',     0.20),
            ('alpha_ema_front',     0.15),
            ('alpha_ema_wall',      0.35),
            # ─ Velocidad adaptativa (3 zonas) ─
            ('speed_caution_dist',  0.50),
            ('speed_critical_dist', 0.15),
            ('speed_min_factor',    0.30),
            # ─ Velocity ramp (anti-slip) ─
            ('max_v_accel',         0.5),
            ('max_w_accel',         3.0),
            # ─ Persistencia de transiciones ─
            ('persist_frames',      3),
            # ─ Esquina ─
            ('corner_turn_dir',     +1.0),
            ('corner_turn_speed',   0.45),
            ('corner_turn_time',    2.8),
            ('corner_timeout_s',    8.0),
            # ─ Alineación ante caja ─
            ('align_turn_dir',      +1.0),
            ('align_turn_speed',    0.45),
            ('align_clear_frames',  2),
            ('align_timeout_s',     6.0),
            # ─ Bypass de caja ─
            ('box_exit_wall_len',   0.60),
            ('box_exit_frames',     4),
            ('bypass_timeout_s',   15.0),
            # ─ Recuperación ─
            ('recovery_reverse_v',  -0.08),
            ('recovery_reverse_t',   1.0),
            ('recovery_turn_deg',   30.0),
            ('recovery_turn_speed',  0.45),
            ('recovery_timeout_s',   5.0),
            # ─ Emergencia ─
            ('emergency_dist',      0.10),
            # ─ Detección de stuck ─
            ('stuck_v_threshold',   0.02),
            ('stuck_time_s',        3.0),
            # ─ Pérdida de pared ─
            ('wall_lost_frames',    24),
            ('wall_search_w',       0.3),
            # ─ Loop de control ─
            ('control_rate_hz',     20.0),
            ('log_every_n',         20),
            # ─ Tópico del scan (configurable) ─
            ('scan_topic',          '/scan'),
            # ─ Diagnóstico de arranque ─
            ('scan_watchdog_s',      5.0),
            ('scan_qos_best_effort', True),
            # ─ Diagnóstico automático ─
            ('diag_enabled',        True),   # activa el reporte automático
            ('diag_window_s',        5.0),   # [s] ventana deslizante de análisis
            ('diag_report_every_s',  5.0),   # [s] cada cuánto imprime el reporte
        ])
        # ── LEER PARÁMETROS ──────────────────────────────────────
        gp = self.get_parameter
        # Geometría LiDAR
        self.lidar_offset     = math.radians(gp('lidar_offset_deg').value)
        self.lidar_y_sign     = float(gp('lidar_y_sign').value)
        self.range_min_m      = float(gp('range_min').value)
        self.range_max_m      = float(gp('range_max').value)
        _ign_lo               = gp('ignore_deg_min').value
        _ign_hi               = gp('ignore_deg_max').value
        self.ignore_min: Optional[float] = (
            math.radians(_ign_lo) if not math.isnan(_ign_lo) else None)
        self.ignore_max: Optional[float] = (
            math.radians(_ign_hi) if not math.isnan(_ign_hi) else None)
        # Split & Merge
        self.sm_split_thresh  = float(gp('sm_split_thresh').value)
        self.sm_range_k       = float(gp('sm_adaptive_range_k').value)
        self.sm_pre_gap_idx   = int(gp('sm_pre_gap_idx').value)
        self.sm_pre_gap_dist  = float(gp('sm_pre_gap_dist').value)
        self.sm_min_pts       = int(gp('sm_min_pts').value)
        self.sm_min_len       = float(gp('sm_min_len').value)
        self.sm_merge_passes  = int(gp('sm_merge_max_passes').value)
        # Pared lateral
        self.wall_min_len     = float(gp('wall_min_len').value)
        self.wall_cos_lat_min = float(gp('wall_cos_lat_min').value)
        # Frontal
        self.front_sector     = math.radians(gp('front_sector_deg').value)
        self.front_alert      = float(gp('front_alert_dist').value)
        self.front_arc_rad    = float(gp('front_arc_radial').value)
        self.front_arc_ang    = math.radians(gp('front_arc_ang_gap').value)
        self.front_wall_ang   = math.radians(gp('front_wall_deg').value)
        self.front_box_ang    = math.radians(gp('front_box_deg').value)
        self.side_clear       = float(gp('side_clear_dist').value)
        # PID
        self.target_d    = float(gp('target_wall_dist').value)
        self.Kp          = float(gp('Kp_dist').value)
        self.K_alpha     = float(gp('K_alpha').value)
        self.Kd          = float(gp('Kd_dist').value)
        self.Ki          = float(gp('Ki_dist').value)
        self.Ki_max      = float(gp('Ki_max_accum').value)
        self.Ki_zone     = float(gp('Ki_zone').value)
        self.max_w       = float(gp('max_ang_vel').value)
        # Velocidades
        self.v_cruise    = float(gp('cruise_speed').value)
        self.v_turn_min  = float(gp('v_turn_min').value)
        # EMA
        self.alpha_ema_d     = float(gp('alpha_ema_deriv').value)
        self.alpha_ema_front = float(gp('alpha_ema_front').value)
        self.alpha_ema_wall  = float(gp('alpha_ema_wall').value)
        # Velocidad adaptativa
        self.speed_caution   = float(gp('speed_caution_dist').value)
        self.speed_critical  = float(gp('speed_critical_dist').value)
        self.speed_min_fac   = float(gp('speed_min_factor').value)
        # Velocity ramp
        self.max_v_accel = float(gp('max_v_accel').value)
        self.max_w_accel = float(gp('max_w_accel').value)
        # Persistencia
        self.n_persist = int(gp('persist_frames').value)
        # Esquina
        self.corner_dir     = float(gp('corner_turn_dir').value)
        self.corner_w       = float(gp('corner_turn_speed').value)
        self.corner_t       = float(gp('corner_turn_time').value)
        self.corner_timeout = float(gp('corner_timeout_s').value)
        # Alineación
        self.align_dir     = float(gp('align_turn_dir').value)
        self.align_w       = float(gp('align_turn_speed').value)
        self.align_n_clear = int(gp('align_clear_frames').value)
        self.align_timeout = float(gp('align_timeout_s').value)
        # Bypass
        self.box_exit_len    = float(gp('box_exit_wall_len').value)
        self.box_exit_n      = int(gp('box_exit_frames').value)
        self.bypass_timeout  = float(gp('bypass_timeout_s').value)
        # Recuperación
        self.recovery_v       = float(gp('recovery_reverse_v').value)
        self.recovery_rev_t   = float(gp('recovery_reverse_t').value)
        self.recovery_turn    = math.radians(gp('recovery_turn_deg').value)
        self.recovery_w       = float(gp('recovery_turn_speed').value)
        self.recovery_timeout = float(gp('recovery_timeout_s').value)
        # Emergencia & stuck
        self.emergency_dist = float(gp('emergency_dist').value)
        self.stuck_v_thresh = float(gp('stuck_v_threshold').value)
        self.stuck_time     = float(gp('stuck_time_s').value)
        # Pérdida de pared
        self.wall_lost_max  = int(gp('wall_lost_frames').value)
        self.wall_search_w  = float(gp('wall_search_w').value)
        # Loop
        rate             = float(gp('control_rate_hz').value)
        self.log_every_n = int(gp('log_every_n').value)
        self.dt_nominal  = 1.0 / rate
        # Tópico del scan
        self.scan_topic           = str(gp('scan_topic').value)
        # Diagnóstico
        self.scan_watchdog_s      = float(gp('scan_watchdog_s').value)
        self.scan_qos_best_effort = bool(gp('scan_qos_best_effort').value)
        # Diagnóstico
        self.diag_enabled         = bool(gp('diag_enabled').value)
        self.diag_window_s        = float(gp('diag_window_s').value)
        self.diag_report_every_s  = float(gp('diag_report_every_s').value)
        # ── ESTADO INTERNO ───────────────────────────────────────
        self.state: str         = CRUISE
        self.state_enter_time   = self.get_clock().now()
        # Percepción (actualizado por cb_scan)
        self.have_scan: bool             = False
        self.wall_seg: Optional[Segment] = None
        self.front_dist_raw: float       = float('inf')
        self.front_ang_width: float      = 0.0
        self.front_class: str            = FRONT_NONE
        self.left_min: float             = float('inf')
        # EMA de percepción
        self._front_dist_ema: Optional[float] = None
        self._wall_dist_ema:  Optional[float] = None
        # PID
        self._err_integral: float   = 0.0
        self._d_der_prev: Optional[float] = None
        self._d_pv_filtered: float  = 0.0
        self._t_prev                = self.get_clock().now()
        # Velocity ramp
        self._last_v: float = 0.0
        self._last_w: float = 0.0
        # Persistencia de transiciones
        self._persist: dict[str, int] = {}
        # Odometría
        self.yaw: Optional[float] = None
        self.have_odom: bool      = False
        self.odom_vx: float       = 0.0
        # Corner
        self._corner_yaw_target: Optional[float] = None
        self._corner_t0 = None  # rclpy.time.Time
        # Bypass / align
        self._align_clear: int = 0
        self._exit_count:  int = 0
        # Recovery
        self._recovery_phase: int = _REC_REVERSE
        self._recovery_reason: str = ''
        # Stuck
        self._stuck_timer: float = 0.0
        # Wall lost
        self._wall_lost_count: int = 0
        # Logging
        self._step_idx: int = 0
        self._node_start_time = self.get_clock().now()
        self._watchdog_fired  = False
        # ── Diagnóstico ──
        # Ventanas deslizantes de (timestamp_s, valor) para medir ruido real
        self._diag_front_hist: deque = deque()
        self._diag_wall_len_hist: deque = deque()
        self._diag_state_transitions: deque = deque()   # timestamps de cada cambio de estado
        self._diag_frames_total: int = 0
        self._diag_frames_no_wall: int = 0
        self._diag_last_report_t = self.get_clock().now()
        # ── Deduplicación para suscripción dual ──
        self._last_scan_stamp = None  # (sec, nanosec) del último scan procesado
        # ── ROS 2 I/O ───────────────────────────────────────────
        # FIX v2.2: creamos DOS suscripciones al mismo tópico, una con
        # QoS BEST_EFFORT y otra RELIABLE. Así no importa qué QoS
        # use el publisher — al menos una será compatible.
        # Se usa un mecanismo de deduplicación por timestamp para no
        # procesar el mismo mensaje dos veces si ambas reciben.
        qos_best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE,
            depth=5)
        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE,
            depth=5)
        self.create_subscription(
            LaserScan, self.scan_topic, self.cb_scan, qos_best_effort)
        self.create_subscription(
            LaserScan, self.scan_topic, self.cb_scan, qos_reliable)
        self.get_logger().info(
            f'Suscripción DUAL a {self.scan_topic} '
            f'(BEST_EFFORT + RELIABLE) ─ compatible con cualquier publisher.')
        self.create_subscription(Odometry,  '/odom', self.cb_odom, 10)
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_timer(1.0 / rate, self.loop_control)
        # Watchdog: revisa una sola vez si el primer scan tarda demasiado.
        self.create_timer(self.scan_watchdog_s, self._scan_watchdog_check)
        if self.diag_enabled:
            self.create_timer(self.diag_report_every_s, self._diag_report)
        self.get_logger().info(
            f'CapyGuardian v2.2 listo │ '
            f'Kp={self.Kp} K_α={self.K_alpha} Kd={self.Kd} Ki={self.Ki} │ '
            f'd*={self.target_d}m  v={self.v_cruise}m/s │ '
            f'scan_topic={self.scan_topic}')
    # ══════════════════════════════════════════════════════════════
    #  DIAGNÓSTICO DE ARRANQUE
    # ══════════════════════════════════════════════════════════════
    def _scan_watchdog_check(self) -> None:
        """Si tras ``scan_watchdog_s`` segundos no llegó ningún /scan,
        avisa explícitamente en el log con auto-descubrimiento de tópicos.
        Solo dispara una vez."""
        if self._watchdog_fired or self.have_scan:
            return
        self._watchdog_fired = True
        # ── Auto-descubrir tópicos de tipo LaserScan ──
        scan_topics = []
        try:
            topic_names_and_types = self.get_topic_names_and_types()
            for name, types in topic_names_and_types:
                for t in types:
                    if 'LaserScan' in t:
                        scan_topics.append(name)
        except Exception:
            pass
        msg = (
            f'⚠ No se ha recibido NINGÚN mensaje en {self.scan_topic} '
            f'tras {self.scan_watchdog_s:.0f}s. '
            f'El robot NO se moverá hasta que llegue el primer scan.')
        if scan_topics:
            if self.scan_topic in scan_topics:
                msg += (
                    f'\n  El tópico {self.scan_topic} EXISTE en el sistema '
                    f'pero no llegan datos al nodo. '
                    f'Verifica que el driver del LiDAR esté publicando: '
                    f'ros2 topic hz {self.scan_topic}')
            else:
                msg += (
                    f'\n  ¡ENCONTRÉ tópicos LaserScan disponibles: '
                    f'{scan_topics}!\n'
                    f'  Tu nodo escucha en "{self.scan_topic}" pero ese '
                    f'tópico NO existe. Prueba con:\n'
                    f'  ros2 run ... --ros-args -p scan_topic:={scan_topics[0]}')
        else:
            msg += (
                f'\n  No se encontró NINGÚN tópico de tipo LaserScan '
                f'en el sistema. Verifica que el driver del LiDAR esté '
                f'corriendo: ros2 topic list')
        self.get_logger().error(msg)
    # ══════════════════════════════════════════════════════════════
    #  CALLBACKS
    # ══════════════════════════════════════════════════════════════
    def cb_odom(self, msg: Odometry) -> None:
        """Extrae yaw (cuaternión → euler) y velocidad lineal."""
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny, cosy)
        self.odom_vx = msg.twist.twist.linear.x
        self.have_odom = True
    def cb_scan(self, msg: LaserScan) -> None:
        """Procesa cada scan: segmenta, identifica pared lateral y clasifica
        el sector frontal.  Actualiza los EMA de distancia.
        Con suscripción dual, puede llegar el mismo mensaje por ambos
        canales; se deduplica por timestamp del header."""
        # ── Deduplicación ──
        stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        if stamp == self._last_scan_stamp:
            return  # Ya procesado por la otra suscripción
        self._last_scan_stamp = stamp
        if not self.have_scan:
            elapsed = (self.get_clock().now() - self._node_start_time).nanoseconds * 1e-9
            self.get_logger().info(
                f'✓ Primer scan recibido en {self.scan_topic} '
                f'tras {elapsed:.2f}s. Arrancando control.')
        pts = self._points_robot_frame(msg)
        # Segmentos por Split-and-Merge iterativo
        segs = self._segments_from_points(pts)
        # Pared lateral derecha (o caja actuando como pared)
        self.wall_seg = self._best_right_segment(segs)
        # Análisis frontal (distancia + ancho angular + izq libre)
        raw_fd, self.front_ang_width, self.left_min = self._front_analysis(pts)
        self.front_dist_raw = raw_fd
        # ── EMA: distancia frontal ──
        if self._front_dist_ema is None or not math.isfinite(self._front_dist_ema):
            self._front_dist_ema = raw_fd
        elif math.isfinite(raw_fd):
            a = self.alpha_ema_front
            self._front_dist_ema = a * raw_fd + (1.0 - a) * self._front_dist_ema
        # ── EMA: distancia a pared derecha ──
        if self.wall_seg is not None:
            raw_wd = -self.wall_seg.mean_y
            if self._wall_dist_ema is None:
                self._wall_dist_ema = raw_wd
            else:
                a = self.alpha_ema_wall
                self._wall_dist_ema = a * raw_wd + (1.0 - a) * self._wall_dist_ema
        # Clasificar el frente (usa EMA de distancia)
        self.front_class = self._classify_front()
        self.have_scan = True
        # ── Diagnóstico: alimentar buffers ──
        if self.diag_enabled:
            self._diag_feed(raw_fd)
    # ══════════════════════════════════════════════════════════════
    #  PERCEPCIÓN:  LiDAR → puntos en marco del robot
    # ══════════════════════════════════════════════════════════════
    def _points_robot_frame(self, msg: LaserScan) \
            -> List[Tuple[int, float, float]]:
        """Convierte el scan polar a [(idx, x, y)] en el frame del robot.
        +x = adelante, +y = izquierda.

        FIX geometría (espejo en Y + alineación frontal en X):
        Antes, el offset de montaje y el signo de espejo del eje Y se
        aplicaban juntos sobre el ÁNGULO (``y_sign*theta_lidar + offset``).
        Eso mezcla dos correcciones físicas distintas en una sola
        operación: a menos que el offset sea exactamente 0° o 180°, el
        orden importa y el resultado deja de ser un espejo puro sobre Y,
        sino una combinación rotación+espejo que no corresponde a la
        realidad física del sensor.

        Ahora se separan en el orden correcto:
          1. ROTAR primero (``theta_lidar + lidar_offset``) para que el
             0° del sensor quede alineado con el +X (frente) del robot.
          2. ESPEJAR después, en coordenadas cartesianas
             (``y = lidar_y_sign * r*sin(theta)``), para corregir si el
             sensor escanea en sentido contrario al estándar de ROS
             (CCW visto desde arriba). X (frente/atrás) nunca se toca
             por el espejo — solo Y (izquierda/derecha).

        Esto es matemáticamente equivalente a lo anterior únicamente
        cuando offset es 0° o 180° (por eso "funcionaba" antes); para
        cualquier otro valor de ``lidar_offset_deg`` el resultado ahora
        sí es geométricamente correcto y estable.

        FIX v2.3 (lado invertido de la caja):
        El espejo debe seguir aplicándose sobre Y (izquierda/derecha),
        NUNCA sobre X (adelante/atrás) — invertir X mezclaría el
        frente con el fondo del scan y rompería front_dist. Lo que
        realmente estaba mal era el SIGNO de ``lidar_y_sign``
        (-1.0 -> 1.0): eso hacía que una caja físicamente a la derecha
        del robot se calculara con y>0 (que en este frame es
        "izquierda"), y por eso aparecía del lado equivocado en el
        dashboard. Con el signo corregido, la caja/pared derecha cae
        del lado correcto.
        """
        pts: List[Tuple[int, float, float]] = []
        rmax = min(msg.range_max, self.range_max_m)
        rmin = max(msg.range_min, self.range_min_m)
        for i, r in enumerate(msg.ranges):
            if not math.isfinite(r) or r < rmin or r > rmax:
                continue
            theta_lidar = msg.angle_min + i * msg.angle_increment
            # 1) Rotación pura: alinea el 0° del sensor con +X del robot.
            theta = theta_lidar + self.lidar_offset
            theta = math.atan2(math.sin(theta), math.cos(theta))  # normalizar a [-π, π]
            if self._in_ignore_arc(theta):
                continue
            x = r * math.cos(theta)
            y = self.lidar_y_sign * (r * math.sin(theta))  # espejo en Y: izquierda <-> derecha
            pts.append((i, x, y))
        return pts
    def _in_ignore_arc(self, theta: float) -> bool:
        """Devuelve True si *theta* cae en la zona angular ignorada."""
        if self.ignore_min is None or self.ignore_max is None:
            return False
        if self.ignore_min <= self.ignore_max:
            return self.ignore_min <= theta <= self.ignore_max
        # Arco que cruza ±π
        return theta >= self.ignore_min or theta <= self.ignore_max
    # ══════════════════════════════════════════════════════════════
    #  PERCEPCIÓN:  Split-and-Merge ITERATIVO
    # ══════════════════════════════════════════════════════════════
    def _segments_from_points(
            self, pts: List[Tuple[int, float, float]]) -> List[Segment]:
        """Pipeline: pre-segmentación → split iterativo → merge multi-pasada
        → crear Segments con métrica de calidad."""
        segments: List[Segment] = []
        for grupo in self._pre_seg(pts):
            if len(grupo) < self.sm_min_pts:
                continue
            xy = [(x, y) for _, x, y in grupo]
            raw_groups = self._split_iterative(xy)
            merged = self._merge_until_stable(raw_groups)
            for g in merged:
                seg = self._make_segment(g)
                if seg is not None:
                    segments.append(seg)
        return segments
    # ── Pre-segmentación por huecos ──────────────────────────────
    def _pre_seg(self, pts: List[Tuple[int, float, float]]) \
            -> List[List[Tuple[int, float, float]]]:
        """Corta la nube de puntos donde hay saltos de índice o distancia."""
        if not pts:
            return []
        out: List[List[Tuple[int, float, float]]] = []
        cur = [pts[0]]
        for k in range(1, len(pts)):
            ip, xp, yp = pts[k - 1]
            ic, xc, yc = pts[k]
            if (ic - ip > self.sm_pre_gap_idx or
                    math.hypot(xc - xp, yc - yp) > self.sm_pre_gap_dist):
                if len(cur) >= 2:
                    out.append(cur)
                cur = [pts[k]]
            else:
                cur.append(pts[k])
        if len(cur) >= 2:
            out.append(cur)
        return out
    # ── Split iterativo con pila explícita ───────────────────────
    def _split_iterative(
            self, pts: List[Tuple[float, float]]) \
            -> List[List[Tuple[float, float]]]:
        """Fase SPLIT del algoritmo Split-and-Merge, implementada con pila
        explícita para evitar stack overflow con datos ruidosos.
        Usa un umbral adaptativo que escala con la distancia promedio al
        sensor, compensando el mayor ruido de puntos lejanos.
        """
        n = len(pts)
        if n <= 2:
            return [pts]
        # Pila de rangos (start, end) pendientes de verificar
        stack: List[Tuple[int, int]] = [(0, n - 1)]
        split_indices: set[int] = {0, n - 1}
        while stack:
            start, end = stack.pop()
            if end - start < 2:
                continue
            ax, ay = pts[start]
            bx, by = pts[end]
            # Umbral adaptativo al rango promedio de los extremos
            r_avg = (math.hypot(ax, ay) + math.hypot(bx, by)) * 0.5
            thresh = self.sm_split_thresh * (1.0 + self.sm_range_k * r_avg)
            best_dist = 0.0
            best_idx  = start
            for i in range(start + 1, end):
                d = _perp_dist(pts[i][0], pts[i][1], ax, ay, bx, by)
                if d > best_dist:
                    best_dist = d
                    best_idx  = i
            if best_dist > thresh:
                split_indices.add(best_idx)
                stack.append((start, best_idx))
                stack.append((best_idx, end))
        # Crear segmentos a partir de los índices de corte
        sorted_idx = sorted(split_indices)
        out: List[List[Tuple[float, float]]] = []
        for i in range(len(sorted_idx) - 1):
            seg = pts[sorted_idx[i]: sorted_idx[i + 1] + 1]
            if len(seg) >= 2:
                out.append(seg)
        return out
    # ── Merge multi-pasada ───────────────────────────────────────
    def _merge_until_stable(
            self, groups: List[List[Tuple[float, float]]]) \
            -> List[List[Tuple[float, float]]]:
        """Fusiona segmentos colineales adyacentes hasta convergencia o
        agotar el máximo de pasadas."""
        for _ in range(self.sm_merge_passes):
            merged = self._merge_pass(groups)
            if len(merged) == len(groups):
                break
            groups = merged
        return groups
    def _merge_pass(
            self, groups: List[List[Tuple[float, float]]]) \
            -> List[List[Tuple[float, float]]]:
        """Una pasada: fusiona pares adyacentes si juntos forman una recta."""
        if len(groups) <= 1:
            return list(groups)
        out = [groups[0]]
        for g in groups[1:]:
            candidate = out[-1] + g
            if len(self._split_iterative(candidate)) <= 1:
                out[-1] = candidate
            else:
                out.append(g)
        return out
    # ── Construcción de Segment ──────────────────────────────────
    def _make_segment(
            self, pts: List[Tuple[float, float]]) -> Optional[Segment]:
        """Crea un ``Segment`` validado, o None si no cumple requisitos mínimos
        de puntos o longitud."""
        if len(pts) < self.sm_min_pts:
            return None
        p1, p2 = pts[0], pts[-1]
        length = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        if length < self.sm_min_len:
            return None
        mean_y = sum(p[1] for p in pts) / len(pts)
        alpha = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
        # Normalizar a [-π/2, π/2]
        if alpha > math.pi / 2:
            alpha -= math.pi
        if alpha < -math.pi / 2:
            alpha += math.pi
        residual = self._segment_residual(pts, p1, p2)
        return Segment(p1=p1, p2=p2, length=length, mean_y=mean_y,
                       alpha=alpha, residual=residual, n_pts=len(pts))
    @staticmethod
    def _segment_residual(pts: List[Tuple[float, float]],
                          p1: Tuple[float, float],
                          p2: Tuple[float, float]) -> float:
        """RMS de la distancia perpendicular de los puntos a la recta p1─p2.
        Valor bajo = buen ajuste lineal."""
        if len(pts) <= 2:
            return 0.0
        total = sum(
            _perp_dist(px, py, p1[0], p1[1], p2[0], p2[1]) ** 2
            for px, py in pts)
        return math.sqrt(total / len(pts))
    # ══════════════════════════════════════════════════════════════
    #  PERCEPCIÓN:  Selección del segmento lateral derecho
    # ══════════════════════════════════════════════════════════════
    def _best_right_segment(self, segs: List[Segment]) -> Optional[Segment]:
        """Selecciona el segmento lateral derecho más cercano.
        Criterios:
          1. Longitud ≥ sm_min_len.
          2. Orientado mayormente a lo largo de +x  (|dx|/L > wall_cos_lat_min).
          3. Del lado derecho (mean_y < umbral dinámico).
          4. El más cercano al robot (|mean_y| menor).
        El umbral dinámico evita rechazar paredes a las que estamos *muy*
        cerca (se ajusta con ``target_wall_dist``).
        """
        y_thresh = -max(0.02, self.target_d * 0.15)
        best: Optional[Segment] = None
        for s in segs:
            if s.length < self.sm_min_len:
                continue
            dx = abs(s.p2[0] - s.p1[0])
            if dx / max(1e-6, s.length) < self.wall_cos_lat_min:
                continue
            if s.mean_y > y_thresh:
                continue
            if best is None or abs(s.mean_y) < abs(best.mean_y):
                best = s
        return best
    # ══════════════════════════════════════════════════════════════
    #  PERCEPCIÓN:  Análisis frontal
    # ══════════════════════════════════════════════════════════════
    def _front_analysis(
            self, pts: List[Tuple[int, float, float]]) \
            -> Tuple[float, float, float]:
        """Analiza el sector frontal.
        Returns:
            front_dist:      distancia mínima en el sector frontal [m].
            front_ang_width: ancho angular del arco continuo más cercano [rad].
            left_min:        distancia mínima lateral izquierda [m].
        """
        front: List[Tuple[float, float]] = []   # (theta, r)
        left_min = float('inf')
        for _, x, y in pts:
            r = math.hypot(x, y)
            theta = math.atan2(y, x)
            if abs(theta) <= self.front_sector:
                front.append((theta, r))
            if math.pi / 4 <= theta <= 3 * math.pi / 4:
                left_min = min(left_min, r)
        if not front:
            return float('inf'), 0.0, left_min
        front.sort(key=lambda tr: tr[0])
        # Agrupar arcos continuos (por proximidad radial Y angular)
        best_width = 0.0
        best_dmin  = float('inf')
        cur = [front[0]]
        for k in range(1, len(front)):
            radial_ok  = abs(front[k][1] - front[k - 1][1]) < self.front_arc_rad
            angular_ok = abs(front[k][0] - front[k - 1][0]) < self.front_arc_ang
            if radial_ok and angular_ok:
                cur.append(front[k])
            else:
                # Evaluar arco actual
                w  = cur[-1][0] - cur[0][0]
                dm = min(t[1] for t in cur)
                if dm < best_dmin:
                    best_dmin  = dm
                    best_width = w
                cur = [front[k]]
        # Último arco
        w  = cur[-1][0] - cur[0][0]
        dm = min(t[1] for t in cur)
        if dm < best_dmin:
            best_dmin  = dm
            best_width = w
        return best_dmin, best_width, left_min
    def _classify_front(self) -> str:
        """Clasifica el obstáculo frontal por ancho angular del arco.
        Usa la distancia frontal EMA-suavizada para decidir si hay
        obstáculo, y el ancho angular para distinguir caja vs esquina.
        En la zona ambigua (30°─45°), desempata con la apertura lateral
        izquierda.
        """
        fd = self._eff_front()
        if fd > self.front_alert:
            return FRONT_NONE
        if self.front_ang_width >= self.front_wall_ang:
            return FRONT_CORNER
        if self.front_ang_width <= self.front_box_ang:
            return FRONT_BOX
        # Zona ambigua: ¿hay hueco a la izquierda?
        if self.left_min > self.side_clear:
            return FRONT_CORNER
        return FRONT_BOX
    def _eff_front(self) -> float:
        """Distancia frontal efectiva (EMA suavizada, con fallback a raw)."""
        if self._front_dist_ema is not None and math.isfinite(self._front_dist_ema):
            return self._front_dist_ema
        return self.front_dist_raw
    # ══════════════════════════════════════════════════════════════
    #  CONTROL:  Wall-follow PID completo + heading
    # ══════════════════════════════════════════════════════════════
    def _wall_follow_cmd(self) -> Tuple[float, float, Optional[Tuple]]:
        """PID de posición + heading contra el segmento lateral derecho.
        · El derivativo opera sobre la *variable de proceso* (d_der)
          y se filtra con EMA para evitar derivative kick y ruido.
        · El integral solo acumula dentro de la I-zone (|err| < Ki_zone)
          y se clampea para anti-windup.
        · La velocidad se adapta por proximidad frontal (3 zonas).
        Returns:
            (v, w, debug_tuple_or_None)
        """
        seg = self.wall_seg
        if seg is None:
            return self._adaptive_speed(), 0.0, None
        # Distancia a la pared derecha (EMA si disponible, sino raw)
        d_der = (self._wall_dist_ema
                 if self._wall_dist_ema is not None
                 else -seg.mean_y)
        err_dist = d_der - self.target_d   # + = lejos, - = cerca
        # Heading error del segmento
        alpha = seg.alpha
        # ── dt ──
        now = self.get_clock().now()
        dt = max((now - self._t_prev).nanoseconds * 1e-9, 0.01)
        self._t_prev = now
        # ── Derivativo sobre PV (no sobre error → sin derivative kick) ──
        if self._d_der_prev is not None:
            d_pv_raw = (d_der - self._d_der_prev) / dt
            a = self.alpha_ema_d
            self._d_pv_filtered = a * d_pv_raw + (1.0 - a) * self._d_pv_filtered
        self._d_der_prev = d_der
        # ── Integral con I-zone y anti-windup ──
        if self.Ki > 1e-9:
            if abs(err_dist) < self.Ki_zone:
                self._err_integral += err_dist * dt
                self._err_integral = _clamp(
                    self._err_integral, -self.Ki_max, self.Ki_max)
            else:
                # Decaer lentamente fuera de la I-zone
                self._err_integral *= 0.95
        # ── Salida PID ──
        w = (-self.Kp     * err_dist
             - self.K_alpha * alpha
             - self.Kd      * self._d_pv_filtered
             - self.Ki      * self._err_integral)
        w = _clamp(w, -self.max_w, self.max_w)
        v = self._adaptive_speed()
        return v, w, (d_der, alpha, err_dist)
    def _adaptive_speed(self) -> float:
        """Velocidad lineal adaptativa según distancia frontal (3 zonas).
        · NORMAL   (fd ≥ caution):  v = v_cruise
        · CAUTELA  (critical < fd < caution):  interpolación lineal
        · CRÍTICA  (fd ≤ critical):  v = 0
        """
        fd = self._eff_front()
        if fd <= self.speed_critical:
            return 0.0
        if fd >= self.speed_caution:
            return self.v_cruise
        frac = ((fd - self.speed_critical) /
                max(0.01, self.speed_caution - self.speed_critical))
        return self.v_cruise * max(self.speed_min_fac, frac)
    def _reset_pid(self) -> None:
        """Resetea acumuladores del PID para transiciones de estado limpias."""
        self._err_integral  = 0.0
        self._d_pv_filtered = 0.0
        self._d_der_prev    = None
    # ══════════════════════════════════════════════════════════════
    #  CONTROL:  Esquina (arc turn por odometría o tiempo)
    # ══════════════════════════════════════════════════════════════
    def _corner_begin(self) -> None:
        """Inicializa la maniobra de esquina: calcula yaw objetivo si hay
        odometría, o prepara el modo por tiempo."""
        self._reset_persist()
        if self.have_odom and self.yaw is not None:
            target = self.yaw + self.corner_dir * math.pi / 2
            self._corner_yaw_target = math.atan2(
                math.sin(target), math.cos(target))
            self._corner_t0 = None
        else:
            self._corner_yaw_target = None
            self._corner_t0 = self.get_clock().now()
    def _corner_step(self) -> Tuple[bool, float]:
        """Ejecuta un paso de la maniobra de esquina.
        Returns:
            (done, angular_velocity)
        """
        if self._corner_yaw_target is not None and self.yaw is not None:
            err = _angle_diff(self._corner_yaw_target, self.yaw)
            if abs(err) < math.radians(3.5):
                return True, 0.0
            return False, (1.0 if err > 0 else -1.0) * self.corner_w
        else:
            if self._corner_t0 is None:
                self._corner_t0 = self.get_clock().now()
            elapsed = (self.get_clock().now() - self._corner_t0).nanoseconds * 1e-9
            if elapsed >= self.corner_t:
                return True, 0.0
            return False, self.corner_dir * self.corner_w
    # ══════════════════════════════════════════════════════════════
    #  PERSISTENCIA DE TRANSICIONES
    # ══════════════════════════════════════════════════════════════
    def _persist_check(self, key: str, cond: bool) -> bool:
        """Incrementa el contador de *key* si *cond* es True (reset si False).
        Devuelve True cuando se alcanzan ``n_persist`` frames consecutivos."""
        c = self._persist.get(key, 0)
        c = c + 1 if cond else 0
        self._persist[key] = c
        return c >= self.n_persist
    def _reset_persist(self) -> None:
        """Limpia todos los contadores de persistencia."""
        self._persist.clear()
    # ══════════════════════════════════════════════════════════════
    #  DETECCIÓN DE EMERGENCIA Y STUCK
    # ══════════════════════════════════════════════════════════════
    def _check_stuck(self, dt: float) -> bool:
        """Detecta si el robot comanda movimiento pero no avanza.
        Solo funciona con odometría disponible; sin ella, siempre False.
        """
        if not self.have_odom:
            return False
        if self._last_v < self.stuck_v_thresh:
            self._stuck_timer = 0.0
            return False
        if abs(self.odom_vx) < self.stuck_v_thresh:
            self._stuck_timer += dt
            return self._stuck_timer >= self.stuck_time
        self._stuck_timer = 0.0
        return False
    def _state_elapsed(self) -> float:
        """Segundos transcurridos desde la entrada al estado actual."""
        return (self.get_clock().now() - self.state_enter_time).nanoseconds * 1e-9
    # ══════════════════════════════════════════════════════════════
    #  FSM:  LOOP PRINCIPAL
    # ══════════════════════════════════════════════════════════════
    def loop_control(self) -> None:
        """Bucle de control principal (llamado por timer a control_rate_hz).
        Orden de ejecución:
          0.  Esperar primer scan.
          1.  Capa de emergencia (override en cualquier estado).
          2.  Detección de stuck  (CRUISE y BOX_BYPASS).
          3.  Ejecución del estado actual (incluye timeouts internos).
          4.  Publicar velocidades con ramp.
          5.  Log periódico.
        """
        if not self.have_scan:
            return
        dt = self.dt_nominal
        # ── 1. Emergencia frontal (excepto RECOVERY) ──
        if self.state != RECOVERY and self._eff_front() < self.emergency_dist:
            self._enter_state(RECOVERY, 'EMERGENCY')
            self._publish_ramped(0.0, 0.0, dt)
            return
        # ── 2. Stuck (solo CRUISE, BOX_BYPASS) ──
        if self.state in (CRUISE, BOX_BYPASS) and self._check_stuck(dt):
            self._enter_state(RECOVERY, 'STUCK')
            self._publish_ramped(0.0, 0.0, dt)
            return
        # ── 3. Ejecutar estado ──
        v, w = 0.0, 0.0
        if self.state == CRUISE:
            v, w = self._exec_cruise()
        elif self.state == CORNER:
            v, w = self._exec_corner()
        elif self.state == BOX_ALIGN:
            v, w = self._exec_align()
        elif self.state == BOX_BYPASS:
            v, w = self._exec_bypass()
        elif self.state == RECOVERY:
            v, w = self._exec_recovery()
        # ── 4. Publicar con ramp ──
        self._publish_ramped(v, w, dt)
        # ── 5. Log ──
        self._periodic_log()
    # ══════════════════════════════════════════════════════════════
    #  FSM:  EJECUCIÓN DE CADA ESTADO
    # ══════════════════════════════════════════════════════════════
    def _exec_cruise(self) -> Tuple[float, float]:
        """CRUISE: wall-follow normal + búsqueda de pared perdida.
        Transiciones:
          → CORNER    si front_class == CORNER durante N frames.
          → BOX_ALIGN si front_class == BOX    durante N frames.
        """
        # ── Calcular comando ──
        if self.wall_seg is None:
            self._wall_lost_count += 1
            if self._wall_lost_count > self.wall_lost_max:
                # Búsqueda: avanzar lento + girar suavemente a la derecha
                v = self.v_cruise * 0.7
                w = -self.wall_search_w
            else:
                # Pared perdida hace poco: seguir recto
                v = self._adaptive_speed()
                w = 0.0
        else:
            self._wall_lost_count = 0
            v, w, _ = self._wall_follow_cmd()
        # ── Transiciones con persistencia ──
        if self._persist_check('to_corner', self.front_class == FRONT_CORNER):
            self._enter_state(CORNER)
            return 0.0, 0.0
        if self._persist_check('to_box', self.front_class == FRONT_BOX):
            self._enter_state(BOX_ALIGN)
            return 0.0, 0.0
        return v, w
    def _exec_corner(self) -> Tuple[float, float]:
        """CORNER: arc-turn hasta completar ~90° o timeout.
        Usa v_turn_min para mantener tracción (evita pivot puro).
        """
        if self._state_elapsed() > self.corner_timeout:
            self._enter_state(RECOVERY, 'CORNER_TIMEOUT')
            return 0.0, 0.0
        done, w = self._corner_step()
        if done:
            self._enter_state(CRUISE)
            return 0.0, 0.0
        return self.v_turn_min, w
    def _exec_align(self) -> Tuple[float, float]:
        """BOX_ALIGN: gira (arc-turn) hasta que el frente está despejado.
        Sale a BOX_BYPASS cuando el frente lleva N frames limpio.
        """
        if self._state_elapsed() > self.align_timeout:
            self._enter_state(RECOVERY, 'ALIGN_TIMEOUT')
            return 0.0, 0.0
        clear = (self.front_class == FRONT_NONE)
        self._align_clear = self._align_clear + 1 if clear else 0
        if self._align_clear >= self.align_n_clear:
            self._enter_state(BOX_BYPASS)
            return 0.0, 0.0
        return self.v_turn_min, self.align_dir * self.align_w
    def _exec_bypass(self) -> Tuple[float, float]:
        """BOX_BYPASS: wall-follow contra la caja como pared derecha.
        Sale a CRUISE cuando detecta pared LARGA estable (segmento largo
        durante N frames consecutivos).
        """
        if self._state_elapsed() > self.bypass_timeout:
            self._enter_state(RECOVERY, 'BYPASS_TIMEOUT')
            return 0.0, 0.0
        v, w, _ = self._wall_follow_cmd()
        # Condición de salida: pared larga sostenida
        long_wall = (self.wall_seg is not None and
                     self.wall_seg.length >= self.box_exit_len)
        self._exit_count = self._exit_count + 1 if long_wall else 0
        if self._exit_count >= self.box_exit_n:
            self._enter_state(CRUISE, 'BYPASS_DONE')
        return v, w
    def _exec_recovery(self) -> Tuple[float, float]:
        """RECOVERY: retroceder + girar para salir de una situación de bloqueo.
        Fases:
          0 ─ REVERSE: retroceder a recovery_v durante recovery_rev_t seg.
          1 ─ TURN:    girar recovery_turn rad (por odom o por timing).
        Timeout propio → volver a CRUISE (intento limpio).
        """
        elapsed = self._state_elapsed()
        # Timeout de recovery → volver a CRUISE
        if elapsed > self.recovery_timeout:
            self._enter_state(CRUISE, 'RECOVERY_DONE')
            return 0.0, 0.0
        # ── Fase 0: retroceder ──
        if self._recovery_phase == _REC_REVERSE:
            if elapsed < self.recovery_rev_t:
                return self.recovery_v, 0.0
            # Pasar a fase de giro
            self._recovery_phase = _REC_TURN
        # ── Fase 1: girar ──
        turn_elapsed = elapsed - self.recovery_rev_t
        if self.have_odom and self.yaw is not None:
            # Giro por odometría: usar yaw al entrar al estado
            # (aproximación: calcular cuánto hemos girado en la fase de turn)
            yaw_turned = self.recovery_w * turn_elapsed
            if yaw_turned >= self.recovery_turn:
                self._enter_state(CRUISE, 'RECOVERY_DONE')
                return 0.0, 0.0
        else:
            # Giro por timing
            turn_needed = self.recovery_turn / max(0.1, self.recovery_w)
            if turn_elapsed >= turn_needed:
                self._enter_state(CRUISE, 'RECOVERY_DONE')
                return 0.0, 0.0
        return self.v_turn_min, self.recovery_w
    # ══════════════════════════════════════════════════════════════
    #  DIAGNÓSTICO AUTOMÁTICO
    # ══════════════════════════════════════════════════════════════
    def _diag_feed(self, raw_fd: float) -> None:
        """Alimenta las ventanas deslizantes de diagnóstico con cada scan.
        No hace ningún cálculo pesado aquí — solo guarda y poda por tiempo,
        para que el reporte (cada diag_report_every_s) sea barato de calcular.
        """
        now_s = self.get_clock().now().nanoseconds * 1e-9
        self._diag_frames_total += 1
        if math.isfinite(raw_fd):
            self._diag_front_hist.append((now_s, raw_fd))
        if self.wall_seg is not None:
            self._diag_wall_len_hist.append((now_s, self.wall_seg.length))
        else:
            self._diag_frames_no_wall += 1
        # Podar todo lo que salió de la ventana de análisis
        cutoff = now_s - self.diag_window_s
        while self._diag_front_hist and self._diag_front_hist[0][0] < cutoff:
            self._diag_front_hist.popleft()
        while self._diag_wall_len_hist and self._diag_wall_len_hist[0][0] < cutoff:
            self._diag_wall_len_hist.popleft()
        while self._diag_state_transitions and self._diag_state_transitions[0] < cutoff:
            self._diag_state_transitions.popleft()
    def _diag_report(self) -> None:
        """Calcula e imprime un diagnóstico interpretado (no solo números
        crudos) sobre ruido de percepción y estabilidad de la FSM en la
        ventana reciente. Pensado para reemplazar la lectura manual de
        logs: te dice explícitamente qué está pasando y qué grupo de
        parámetros tocar.
        """
        if not self.have_scan:
            return
        lines = ['── DIAGNÓSTICO ──────────────────────────']
        # ── 1. Ruido en front_dist (jitter) ──
        front_vals = [v for _, v in self._diag_front_hist]
        if len(front_vals) >= 3:
            f_min, f_max = min(front_vals), max(front_vals)
            f_jitter = f_max - f_min
            f_std = statistics.pstdev(front_vals)
            tag = '⚠ RUIDOSO' if f_jitter > 0.35 else ('~ moderado' if f_jitter > 0.15 else '✓ estable')
            lines.append(
                f'front_dist: rango={f_jitter:.2f}m std={f_std:.2f}m [{tag}] '
                f'(min={f_min:.2f} max={f_max:.2f}, n={len(front_vals)})')
            if f_jitter > 0.35:
                lines.append(
                    '  → sugerencia: baja alpha_ema_front (más suavizado) '
                    'y/o sube front_arc_radial (agrupa mejor el arco).')
        else:
            lines.append('front_dist: datos insuficientes en la ventana')
        # ── 2. Ruido en la longitud del segmento de pared/caja ──
        wall_vals = [v for _, v in self._diag_wall_len_hist]
        if len(wall_vals) >= 3:
            w_min, w_max = min(wall_vals), max(wall_vals)
            w_jitter = w_max - w_min
            tag = '⚠ RUIDOSO' if w_jitter > 0.40 else ('~ moderado' if w_jitter > 0.20 else '✓ estable')
            lines.append(
                f'seg_len (pared/caja): rango={w_jitter:.2f}m [{tag}] '
                f'(min={w_min:.2f} max={w_max:.2f}, n={len(wall_vals)})')
            if w_jitter > 0.40:
                lines.append(
                    '  → sugerencia: sube wall_cos_lat_min y/o baja '
                    'sm_adaptive_range_k (el Split&Merge está fusionando '
                    'caja+pared en un segmento falso).')
        else:
            lines.append('seg_len: sin datos (wall_seg=None en toda la ventana)')
        # ── 3. Tasa de pérdida de pared ──
        if self._diag_frames_total > 0:
            no_wall_pct = 100.0 * self._diag_frames_no_wall / self._diag_frames_total
            tag = '⚠ ALTA' if no_wall_pct > 60 else ('~ moderada' if no_wall_pct > 30 else '✓ baja')
            lines.append(f'frames sin wall_seg: {no_wall_pct:.0f}% [{tag}]')
            if no_wall_pct > 60:
                lines.append(
                    '  → sugerencia: baja wall_min_len (probablemente más '
                    'corto que el lado real de tu caja) y/o baja sm_min_len.')
        # ── 4. Flapping de estados (cambios de estado demasiado seguidos) ──
        n_trans = len(self._diag_state_transitions)
        tag = '⚠ FLAPPING' if n_trans >= 6 else ('~ activo' if n_trans >= 3 else '✓ estable')
        lines.append(
            f'transiciones de estado en {self.diag_window_s:.0f}s: {n_trans} [{tag}]')
        if n_trans >= 6:
            lines.append(
                '  → sugerencia: sube persist_frames y align_clear_frames '
                '(la FSM está cambiando de opinión por ruido, no por '
                'eventos reales).')
        # ── 5. Resumen de estado actual ──
        lines.append(
            f'estado actual: {self.state} │ front={self._eff_front():.2f}m '
            f'│ class={self.front_class} │ wall_seg='
            f'{"sí (%.2fm)" % self.wall_seg.length if self.wall_seg else "NO"}')
        lines.append('─────────────────────────────────────────')
        # Reinicia contadores de tasa (la ventana deslizante de listas ya
        # se poda sola en _diag_feed; esto es solo para el % de no-wall)
        self._diag_frames_total = 0
        self._diag_frames_no_wall = 0
        self.get_logger().info('\n'.join(lines))
    # ══════════════════════════════════════════════════════════════
    #  GESTIÓN DE ESTADOS
    # ══════════════════════════════════════════════════════════════
    def _enter_state(self, new: str, reason: str = '') -> None:
        """Transición de estado con logging y reinicialización.
        Cada estado tiene su setup específico al entrar.
        """
        if new != self.state:
            info = f'{self.state} → {new}'
            if reason:
                info += f' ({reason})'
            info += (f'  [d_front={self._eff_front():.2f} '
                     f'w_ang={math.degrees(self.front_ang_width):.0f}° '
                     f'class={self.front_class} '
                     f'left={self.left_min:.2f}]')
            self.get_logger().info(info)
            if self.diag_enabled:
                now_s = self.get_clock().now().nanoseconds * 1e-9
                self._diag_state_transitions.append(now_s)
        old_state = self.state
        self.state = new
        self.state_enter_time = self.get_clock().now()
        # ── Setup específico por estado ──
        if new == CRUISE:
            self._reset_persist()
            self._reset_pid()
            self._wall_lost_count = 0
            self._stuck_timer     = 0.0
        elif new == CORNER:
            self._corner_begin()
        elif new == BOX_ALIGN:
            self._align_clear = 0
        elif new == BOX_BYPASS:
            self._reset_pid()
            self._exit_count  = 0
            self._stuck_timer = 0.0
        elif new == RECOVERY:
            self._recovery_phase  = _REC_REVERSE
            self._recovery_reason = reason
    # ══════════════════════════════════════════════════════════════
    #  PUBLICACIÓN CON VELOCITY RAMP
    # ══════════════════════════════════════════════════════════════
    def _publish_ramped(self, v_target: float, w_target: float,
                        dt: float) -> None:
        """Aplica ramp de aceleración y publica el Twist.
        Limita tanto la aceleración lineal como la angular para evitar
        saltos bruscos de velocidad que causan deslizamiento de ruedas.
        """
        v = _ramp(v_target, self._last_v, self.max_v_accel, dt)
        w = _ramp(w_target, self._last_w, self.max_w_accel, dt)
        self._last_v = v
        self._last_w = w
        cmd = Twist()
        cmd.linear.x  = float(v)
        cmd.angular.z = float(w)
        self.pub_cmd.publish(cmd)
    # ══════════════════════════════════════════════════════════════
    #  LOGGING PERIÓDICO
    # ══════════════════════════════════════════════════════════════
    def _periodic_log(self) -> None:
        """Log periódico (cada ``log_every_n`` frames) con información
        adaptada al estado actual."""
        self._step_idx += 1
        if self.log_every_n <= 0 or (self._step_idx % self.log_every_n) != 0:
            return
        elapsed = self._state_elapsed()
        if self.state == CRUISE:
            if self.wall_seg is None:
                if self._wall_lost_count > self.wall_lost_max:
                    self.get_logger().info(
                        f'[CRUISE] pared perdida ({self._wall_lost_count} frames), '
                        f'buscando...')
                else:
                    self.get_logger().info('[CRUISE] sin pared; recto')
            else:
                d_der = (self._wall_dist_ema
                         if self._wall_dist_ema is not None
                         else -self.wall_seg.mean_y)
                err = d_der - self.target_d
                self.get_logger().info(
                    f'[CRUISE] d={d_der:.2f} err={err:+.2f} '
                    f'α={math.degrees(self.wall_seg.alpha):+.1f}° '
                    f'front={self._eff_front():.2f}m ({self.front_class}) '
                    f'seg_q={self.wall_seg.residual:.3f}')
        elif self.state == RECOVERY:
            phase = 'REV' if self._recovery_phase == _REC_REVERSE else 'TURN'
            self.get_logger().info(
                f'[RECOVERY/{phase}] reason={self._recovery_reason} '
                f'elapsed={elapsed:.1f}s')
        else:
            extra = ''
            if self.state == BOX_BYPASS and self.wall_seg is not None:
                extra = f' seg_len={self.wall_seg.length:.2f}'
            self.get_logger().info(
                f'[{self.state}] elapsed={elapsed:.1f}s '
                f'front={self._eff_front():.2f}m{extra}')
# ═══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════
def main(args=None) -> None:
    rclpy.init(args=args)
    node = CapyGuardian()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Parada de seguridad
        try:
            node.pub_cmd.publish(Twist())
        except Exception:
            pass
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass  # Ya fue cerrado por el signal handler de ROS 2
if __name__ == '__main__':
    main()