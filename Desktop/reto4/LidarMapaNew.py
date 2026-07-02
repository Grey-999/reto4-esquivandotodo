#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LidarMapa.py  ──  Visualización gráfica del LiDAR en tiempo real (matplotlib).

Replica la visualización de referencia con DOS paneles:

  ┌─────────────────────────────────────────┐
  │  PANEL SUPERIOR                         │
  │  "Marco robot — scan + clasificación"   │
  │  - Puntos LiDAR crudos (cyan)           │
  │  - Segmentos clasificados:              │
  │      PARED  → naranja/amarillo          │
  │      CAJA   → rojo                      │
  │      OTRO   → azul claro                │
  │  - Robot al centro (triángulo blanco)   │
  │  - Sector frontal sombreado             │
  │  - Anotaciones: frente, pared, etc.     │
  │  - Barra de estado FSM + distancias     │
  └─────────────────────────────────────────┘
  ┌─────────────────────────────────────────┐
  │  PANEL INFERIOR                         │
  │  "Recorrido + cajas vivas — N vueltas"  │
  │  - Trayectoria acumulada (odometría)    │
  │  - Posición actual del robot            │
  │  - Cajas detectadas fijas               │
  │  - Leyenda                              │
  └─────────────────────────────────────────┘

Uso:
    python3 LidarMapa.py

Se suscribe a:
    /scan   (LaserScan)  – datos del LiDAR
    /odom   (Odometry)   – posición del robot
    /cmd_vel (Twist)     – comandos de velocidad (para inferir acción)

ROS 2 Humble  |  Yahboom MicroROS-Pi5  |  LiDAR MS200
"""
from __future__ import annotations

import math
import threading
import time
from collections import deque
from typing import List, Optional, Tuple, NamedTuple

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Wedge, FancyBboxPatch
from matplotlib.collections import LineCollection
from matplotlib.widgets import Slider, Button
import matplotlib.patheffects as pe
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

# ═══════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN (mismos defaults que capytown_guardian)
# ═══════════════════════════════════════════════════════════════════
LIDAR_OFFSET_DEG  = 180.0
# FIX ESPEJO: con -1.0 el eje Y quedaba invertido (izquierda/derecha
# reflejadas). Se deja en +1.0 para que el ángulo crezca en sentido
# antihorario (convención estándar ROS REP-103: +x adelante, +y
# izquierda). Si tras esto ves la pared/caja en el lado contrario al
# real, tu LiDAR gira en sentido horario: vuelve a poner -1.0 aquí.
LIDAR_Y_SIGN      = 1.0
RANGE_MIN         = 0.05
RANGE_MAX         = 3.5

# Split & Merge
SM_SPLIT_THRESH   = 0.04
SM_RANGE_K        = 0.05
SM_PRE_GAP_IDX    = 6
SM_PRE_GAP_DIST   = 0.40
SM_MIN_PTS        = 3
SM_MIN_LEN        = 0.06
SM_MERGE_PASSES   = 2

# Clasificación
WALL_MIN_LEN      = 0.35
WALL_COS_LAT      = 0.75
BOX_MAX_LEN       = 0.35

# Frontal
FRONT_SECTOR_DEG  = 38.0
FRONT_ALERT_DIST  = 0.42
FRONT_WALL_DEG    = 44.0
FRONT_BOX_DEG     = 31.0

# Pared lateral
TARGET_WALL_DIST  = 0.30

# Visualización
VIEW_RANGE        = 1.2       # rango visible [m] (cada lado) panel superior
REFRESH_HZ        = 8.0       # FPS del mapa
SCAN_TOPIC        = '/scan'
TRAIL_MAX_PTS     = 8000      # máximo de puntos en la trayectoria

# Colores estilo "aurora nocturna" (rediseño, paleta violeta/verde-agua)
BG_COLOR          = '#0b0e14'    # casi negro con tinte azulado
PANEL_BG          = '#121722'    # panel un poco más claro que el fondo
GRID_COLOR        = '#232a3a'    # rejilla sutil
TEXT_COLOR        = '#eef1f7'    # blanco cálido para texto
ACCENT_COLOR      = '#7dd3fc'    # celeste para flechas/heading
WALL_COLOR        = '#a78bfa'    # violeta para paredes genéricas
WALL_RIGHT_COLOR  = '#34d399'    # verde esmeralda para pared derecha seguida
BOX_COLOR         = '#fb7185'    # rosa-coral para cajas
UNK_COLOR         = '#94a3b8'    # gris azulado para segmentos sin clasificar
POINT_COLOR       = '#38bdf8'    # azul cian para puntos crudos del LiDAR
ROBOT_COLOR       = '#fefce8'    # blanco marfil para el robot
TRAIL_COLOR       = '#facc15'    # ámbar para la trayectoria
BOX_MARKER_COLOR  = '#f472b6'    # rosa fuerte para marcadores de cajas
FRONT_SECTOR_COLOR = '#fbbf24'   # ámbar para el sector frontal

# ═══════════════════════════════════════════════════════════════════
#  TIPOS
# ═══════════════════════════════════════════════════════════════════
class Segment(NamedTuple):
    p1: Tuple[float, float]
    p2: Tuple[float, float]
    length: float
    mean_y: float
    alpha: float
    n_pts: int
    kind: str          # 'WALL', 'BOX', 'UNK'

# ═══════════════════════════════════════════════════════════════════
#  UTILIDADES
# ═══════════════════════════════════════════════════════════════════
def _perp_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return math.hypot(px - ax, py - ay)
    return abs(dy * px - dx * py + bx * ay - by * ax) / length

# ═══════════════════════════════════════════════════════════════════
#  NODO ROS 2 – Recolector de datos
# ═══════════════════════════════════════════════════════════════════
class LidarMapaNode(Node):
    """Nodo ROS 2 que recolecta datos de /scan, /odom y /cmd_vel
    para alimentar la visualización matplotlib."""

    def __init__(self):
        super().__init__('lidar_mapa')
        self.lidar_offset = math.radians(LIDAR_OFFSET_DEG)
        self.front_sector = math.radians(FRONT_SECTOR_DEG)
        self.front_wall_ang = math.radians(FRONT_WALL_DEG)
        self.front_box_ang  = math.radians(FRONT_BOX_DEG)

        self._lock = threading.Lock()
        self._last_stamp = None
        self._pts: List[Tuple[float, float]] = []
        self._segs: List[Segment] = []
        self._wall_seg: Optional[Segment] = None
        self._front_dist = float('inf')
        self._front_class = 'NONE'
        self._front_ang_width = 0.0
        self._have_scan = False

        # Odometría
        self._odom_x = 0.0
        self._odom_y = 0.0
        self._odom_yaw = 0.0
        self._have_odom = False
        self._trail: deque = deque(maxlen=TRAIL_MAX_PTS)

        # cmd_vel (para inferir acción)
        self._cmd_v = 0.0
        self._cmd_w = 0.0

        # Detección de cajas acumulada (posiciones globales)
        self._boxes_detected: List[Tuple[float, float]] = []
        self._box_detect_cooldown = 0.0

        # FSM state (inferido del guardian via /cmd_vel patterns)
        self._inferred_action = 'ESPERANDO'

        # Contador de vueltas
        self._lap_count = 0
        self._lap_start_x = None
        self._lap_start_y = None
        self._lap_dist_from_start = 0.0
        self._lap_armed = False

        # Suscripción dual a /scan
        qos_be = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE, depth=5)
        qos_rel = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE, depth=5)
        self.create_subscription(LaserScan, SCAN_TOPIC, self._cb_scan, qos_be)
        self.create_subscription(LaserScan, SCAN_TOPIC, self._cb_scan, qos_rel)
        self.create_subscription(Odometry, '/odom', self._cb_odom, 10)
        self.create_subscription(Twist, '/cmd_vel', self._cb_cmd, 10)

        self.get_logger().info(f'LidarMapa listo — escuchando {SCAN_TOPIC}')

    # ──────────────────────────────────────────────────────────────
    #  CALLBACKS
    # ──────────────────────────────────────────────────────────────
    def _cb_scan(self, msg: LaserScan):
        stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        if stamp == self._last_stamp:
            return
        self._last_stamp = stamp

        if not self._have_scan:
            self.get_logger().info('✓ Primer scan recibido.')
            self._have_scan = True

        # Puntos en frame robot
        pts = []
        rmax = min(msg.range_max, RANGE_MAX)
        rmin = max(msg.range_min, RANGE_MIN)
        for i, r in enumerate(msg.ranges):
            if not math.isfinite(r) or r < rmin or r > rmax:
                continue
            theta_lidar = msg.angle_min + i * msg.angle_increment
            # Rotación: alinea 0° del sensor con +X del robot
            theta = theta_lidar + self.lidar_offset
            theta = math.atan2(math.sin(theta), math.cos(theta))
            x = r * math.cos(theta)
            y = LIDAR_Y_SIGN * (r * math.sin(theta))
            pts.append((x, y))

        # Segmentar
        indexed = [(i, x, y) for i, (x, y) in enumerate(pts)]
        segs = self._segment_pipeline(indexed)

        # Pared derecha
        wall_seg = self._best_right_segment(segs)

        # Frontal
        fd, faw, _ = self._front_analysis_full(pts)
        fc = self._classify_front_full(fd, faw, pts)

        # Detección de cajas en coordenadas globales
        if self._have_odom:
            for seg in segs:
                if seg.kind == 'BOX':
                    cx_local = (seg.p1[0] + seg.p2[0]) / 2.0
                    cy_local = (seg.p1[1] + seg.p2[1]) / 2.0
                    cos_y = math.cos(self._odom_yaw)
                    sin_y = math.sin(self._odom_yaw)
                    gx = self._odom_x + cos_y * cx_local - sin_y * cy_local
                    gy = self._odom_y + sin_y * cx_local + cos_y * cy_local
                    self._register_box(gx, gy)

        with self._lock:
            self._pts = pts
            self._segs = segs
            self._wall_seg = wall_seg
            self._front_dist = fd
            self._front_ang_width = faw
            self._front_class = fc

    def _cb_odom(self, msg: Odometry):
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny, cosy)

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        with self._lock:
            self._odom_x = x
            self._odom_y = y
            self._odom_yaw = yaw
            self._have_odom = True
            self._trail.append((x, y))

            # Detección de vueltas
            if self._lap_start_x is None:
                self._lap_start_x = x
                self._lap_start_y = y
            else:
                d = math.hypot(x - self._lap_start_x, y - self._lap_start_y)
                if d > 1.5:
                    self._lap_armed = True
                if self._lap_armed and d < 0.5:
                    self._lap_count += 1
                    self._lap_armed = False

    def _cb_cmd(self, msg: Twist):
        v = msg.linear.x
        w = msg.angular.z
        with self._lock:
            self._cmd_v = v
            self._cmd_w = w
            # Inferir acción del robot
            if v < -0.01:
                self._inferred_action = 'RETROCEDIENDO'
            elif v < 0.01 and abs(w) > 0.1:
                self._inferred_action = 'GIRANDO'
            elif v > 0.01:
                if abs(w) > 0.2:
                    self._inferred_action = 'ESQUIVANDO'
                else:
                    self._inferred_action = 'AVANZANDO'
            else:
                self._inferred_action = 'DETENIDO'

    def _register_box(self, gx, gy):
        """Registra una caja en coordenadas globales (evita duplicados)."""
        min_dist = 0.25
        for bx, by in self._boxes_detected:
            if math.hypot(gx - bx, gy - by) < min_dist:
                return
        self._boxes_detected.append((gx, gy))

    def clear_trail(self):
        """Limpia la trayectoria acumulada y las cajas detectadas (thread-safe).
        Usado por el botón 'Limpiar' de la UI interactiva."""
        with self._lock:
            self._trail.clear()
            self._boxes_detected.clear()
            self._lap_count = 0
            self._lap_start_x = None
            self._lap_start_y = None
            self._lap_armed = False
        self.get_logger().info('Trayectoria y cajas reiniciadas desde la UI.')

    def get_data(self):
        """Retorna snapshot thread-safe de todos los datos."""
        with self._lock:
            return {
                'pts': list(self._pts),
                'segs': list(self._segs),
                'wall_seg': self._wall_seg,
                'front_dist': self._front_dist,
                'front_class': self._front_class,
                'front_ang_width': self._front_ang_width,
                'have_scan': self._have_scan,
                'odom_x': self._odom_x,
                'odom_y': self._odom_y,
                'odom_yaw': self._odom_yaw,
                'have_odom': self._have_odom,
                'trail': list(self._trail),
                'boxes': list(self._boxes_detected),
                'action': self._inferred_action,
                'cmd_v': self._cmd_v,
                'cmd_w': self._cmd_w,
                'laps': self._lap_count,
            }

    # ──────────────────────────────────────────────────────────────
    #  PERCEPCIÓN (replicada de capytown_guardian)
    # ──────────────────────────────────────────────────────────────
    def _segment_pipeline(self, pts) -> List[Segment]:
        segs = []
        for grupo in self._pre_seg(pts):
            if len(grupo) < SM_MIN_PTS:
                continue
            xy = [(x, y) for _, x, y in grupo]
            raw = self._split_iter(xy)
            merged = self._merge(raw)
            for g in merged:
                seg = self._make_seg(g)
                if seg:
                    segs.append(seg)
        return segs

    def _pre_seg(self, pts):
        if not pts:
            return []
        out, cur = [], [pts[0]]
        for k in range(1, len(pts)):
            ip, xp, yp = pts[k-1]
            ic, xc, yc = pts[k]
            if (ic - ip > SM_PRE_GAP_IDX or
                    math.hypot(xc-xp, yc-yp) > SM_PRE_GAP_DIST):
                if len(cur) >= 2:
                    out.append(cur)
                cur = [pts[k]]
            else:
                cur.append(pts[k])
        if len(cur) >= 2:
            out.append(cur)
        return out

    def _split_iter(self, pts):
        n = len(pts)
        if n <= 2:
            return [pts]
        stack = [(0, n-1)]
        splits = {0, n-1}
        while stack:
            s, e = stack.pop()
            if e - s < 2:
                continue
            ax, ay = pts[s]
            bx, by = pts[e]
            r_avg = (math.hypot(ax, ay) + math.hypot(bx, by)) * 0.5
            thresh = SM_SPLIT_THRESH * (1.0 + SM_RANGE_K * r_avg)
            best_d, best_i = 0.0, s
            for i in range(s+1, e):
                d = _perp_dist(pts[i][0], pts[i][1], ax, ay, bx, by)
                if d > best_d:
                    best_d, best_i = d, i
            if best_d > thresh:
                splits.add(best_i)
                stack.append((s, best_i))
                stack.append((best_i, e))
        si = sorted(splits)
        return [pts[si[i]:si[i+1]+1] for i in range(len(si)-1) if si[i+1]-si[i] >= 1]

    def _merge(self, groups):
        for _ in range(SM_MERGE_PASSES):
            if len(groups) <= 1:
                break
            out = [groups[0]]
            for g in groups[1:]:
                c = out[-1] + g
                if len(self._split_iter(c)) <= 1:
                    out[-1] = c
                else:
                    out.append(g)
            if len(out) == len(groups):
                break
            groups = out
        return groups

    def _make_seg(self, pts) -> Optional[Segment]:
        if len(pts) < SM_MIN_PTS:
            return None
        p1, p2 = pts[0], pts[-1]
        length = math.hypot(p2[0]-p1[0], p2[1]-p1[1])
        if length < SM_MIN_LEN:
            return None
        mean_y = sum(p[1] for p in pts) / len(pts)
        alpha = math.atan2(p2[1]-p1[1], p2[0]-p1[0])
        if alpha > math.pi/2:
            alpha -= math.pi
        if alpha < -math.pi/2:
            alpha += math.pi
        # Clasificar
        dx = abs(p2[0] - p1[0])
        cos_lat = dx / max(1e-6, length)
        if length >= WALL_MIN_LEN and cos_lat >= WALL_COS_LAT:
            kind = 'WALL'
        elif length < BOX_MAX_LEN:
            cx = (p1[0] + p2[0]) / 2
            cy = (p1[1] + p2[1]) / 2
            theta_c = abs(math.atan2(cy, cx))
            if theta_c < self.front_sector and cx > 0:
                kind = 'BOX'
            else:
                kind = 'UNK'
        else:
            kind = 'UNK'
        return Segment(p1=p1, p2=p2, length=length, mean_y=mean_y,
                       alpha=alpha, n_pts=len(pts), kind=kind)

    def _best_right_segment(self, segs: List[Segment]) -> Optional[Segment]:
        y_thresh = -max(0.02, TARGET_WALL_DIST * 0.15)
        best = None
        for s in segs:
            if s.length < SM_MIN_LEN:
                continue
            dx = abs(s.p2[0] - s.p1[0])
            if dx / max(1e-6, s.length) < WALL_COS_LAT:
                continue
            if s.mean_y > y_thresh:
                continue
            if best is None or abs(s.mean_y) < abs(best.mean_y):
                best = s
        return best

    def _front_analysis_full(self, pts):
        front = []
        left_min = float('inf')
        for x, y in pts:
            r = math.hypot(x, y)
            theta = math.atan2(y, x)
            if abs(theta) <= self.front_sector:
                front.append((theta, r))
            if math.pi / 4 <= theta <= 3 * math.pi / 4:
                left_min = min(left_min, r)
        if not front:
            return float('inf'), 0.0, left_min
        front.sort(key=lambda tr: tr[0])
        best_width = 0.0
        best_dmin = float('inf')
        cur = [front[0]]
        for k in range(1, len(front)):
            radial_ok = abs(front[k][1] - front[k-1][1]) < 0.18
            angular_ok = abs(front[k][0] - front[k-1][0]) < math.radians(4.0)
            if radial_ok and angular_ok:
                cur.append(front[k])
            else:
                w = cur[-1][0] - cur[0][0]
                dm = min(t[1] for t in cur)
                if dm < best_dmin:
                    best_dmin = dm
                    best_width = w
                cur = [front[k]]
        w = cur[-1][0] - cur[0][0]
        dm = min(t[1] for t in cur)
        if dm < best_dmin:
            best_dmin = dm
            best_width = w
        return best_dmin, best_width, left_min

    def _classify_front_full(self, fd, faw, pts):
        if fd > FRONT_ALERT_DIST:
            return 'NONE'
        if faw >= self.front_wall_ang:
            return 'PARED'
        if faw <= self.front_box_ang:
            return 'CAJA'
        return 'CAJA'

# ═══════════════════════════════════════════════════════════════════
#  VISUALIZACIÓN MATPLOTLIB
# ═══════════════════════════════════════════════════════════════════

def build_figure():
    """Construye la figura con dos subplots apilados + barra de controles
    interactivos (zoom, pausa, limpiar) en la parte inferior."""
    fig = plt.figure(figsize=(7, 10.6), facecolor=BG_COLOR)
    fig.canvas.manager.set_window_title('LidarMapa — Visualización en vivo')

    # Layout: panel superior (scan) más grande, inferior (mapa), y una
    # franja delgada al fondo para los controles interactivos.
    gs = fig.add_gridspec(3, 1, height_ratios=[1.3, 1.0, 0.09],
                          hspace=0.32, left=0.10, right=0.95,
                          top=0.93, bottom=0.035)

    ax_scan = fig.add_subplot(gs[0])
    ax_map  = fig.add_subplot(gs[1])

    for ax in [ax_scan, ax_map]:
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors='#666666', labelsize=7)
        for spine in ax.spines.values():
            spine.set_color('#333355')
            spine.set_linewidth(0.8)
        ax.grid(True, color=GRID_COLOR, linewidth=0.3, alpha=0.6)
        ax.set_aspect('equal')

    # Panel superior
    ax_scan.set_xlim(-VIEW_RANGE * 0.35, VIEW_RANGE)
    ax_scan.set_ylim(-VIEW_RANGE, VIEW_RANGE)
    ax_scan.set_xlabel('atrás          x [m] — adelante →', color='#666666', fontsize=7)
    ax_scan.set_ylabel('← der          y [m]          izq →', color='#666666', fontsize=7)

    # Panel inferior
    ax_map.set_xlabel('x [m]', color='#666666', fontsize=7)
    ax_map.set_ylabel('y [m]', color='#666666', fontsize=7)

    # ── Barra de controles interactivos ──
    ctrl_row = gs[2].subgridspec(1, 4, width_ratios=[2.2, 0.9, 0.9, 1.4],
                                 wspace=0.25)
    ax_slider = fig.add_subplot(ctrl_row[0])
    ax_btn_pause = fig.add_subplot(ctrl_row[1])
    ax_btn_clear = fig.add_subplot(ctrl_row[2])
    ax_hint = fig.add_subplot(ctrl_row[3])
    ax_hint.axis('off')
    ax_hint.text(0, 0.5, 'teclas: [espacio] pausa  [c] limpiar  [t] puntos  [+/-] zoom',
                color='#666666', fontsize=6, va='center', ha='left',
                transform=ax_hint.transAxes)

    slider_zoom = Slider(ax_slider, 'Zoom', 0.4, 3.0, valinit=VIEW_RANGE,
                        valstep=0.1, color=ACCENT_COLOR)
    slider_zoom.label.set_color('#888888')
    slider_zoom.label.set_fontsize(7)
    slider_zoom.valtext.set_color('#888888')
    slider_zoom.valtext.set_fontsize(7)

    btn_pause = Button(ax_btn_pause, 'Pausar', color='#1a1a2e', hovercolor='#2a2a4e')
    btn_pause.label.set_color('#dddddd')
    btn_pause.label.set_fontsize(7.5)

    btn_clear = Button(ax_btn_clear, 'Limpiar', color='#1a1a2e', hovercolor='#2a2a4e')
    btn_clear.label.set_color('#dddddd')
    btn_clear.label.set_fontsize(7.5)

    widgets = {
        'slider_zoom': slider_zoom,
        'btn_pause': btn_pause,
        'btn_clear': btn_clear,
    }

    return fig, ax_scan, ax_map, widgets


def render_scan_panel(ax, data):
    """Renderiza el panel superior: scan en vivo con clasificación.

    FIX de orientación: antes el frente del robot (+x_robot) se graficaba
    en el eje VERTICAL del plot (x_plot = -y_robot, y_plot = x_robot,
    "adelante = arriba"). Ahora el frente se grafica en el eje
    HORIZONTAL (x_plot = x_robot, y_plot = y_robot), que es la convención
    estándar de "robot mirando hacia +X, izquierda = +Y arriba,
    derecha = -Y abajo" — más intuitiva para leer junto con los logs
    de capytown_guardian.py, que ya usan +x=adelante, +y=izquierda.
    """
    ax.cla()
    ax.set_facecolor(PANEL_BG)
    ax.grid(True, color=GRID_COLOR, linewidth=0.3, alpha=0.6)
    ax.set_aspect('equal')
    view_range = data.get('view_range', VIEW_RANGE)
    ax.set_xlim(-view_range * 0.35, view_range)   # un poco de espacio detrás del robot
    ax.set_ylim(-view_range, view_range)
    ax.tick_params(colors='#666666', labelsize=7)
    for spine in ax.spines.values():
        spine.set_color('#333355')
        spine.set_linewidth(0.8)

    pts = data['pts']
    segs = data['segs']
    wall_seg = data['wall_seg']
    front_dist = data['front_dist']
    front_class = data['front_class']
    action = data['action']
    show_points = data.get('show_points', True)

    # ── Título con acción ──
    action_colors = {
        'AVANZANDO': '#34d399',
        'RETROCEDIENDO': '#fb7185',
        'GIRANDO': '#fbbf24',
        'ESQUIVANDO': '#fbbf24',
        'DETENIDO': '#94a3b8',
        'ESPERANDO': '#64748b',
    }
    ac = action_colors.get(action, '#94a3b8')
    ax.set_title(
        f'Marco robot — scan + clasificación',
        color=TEXT_COLOR, fontsize=9, fontweight='bold', pad=4,
        loc='left')
    ax.text(view_range * 0.98, view_range * 0.92,
            f'ACCION: {action}',
            color=ac, fontsize=8, fontweight='bold',
            ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e',
                      edgecolor=ac, alpha=0.9, linewidth=0.8))

    # ── Sector frontal (abanico semitransparente) ──
    sector_deg = math.degrees(math.radians(FRONT_SECTOR_DEG))
    # "adelante" del robot es +x_robot → +x_plot → 0° en el plot (eje horizontal)
    wedge = Wedge((0, 0), FRONT_ALERT_DIST,
                  -sector_deg, sector_deg,
                  color=FRONT_SECTOR_COLOR, alpha=0.06, zorder=1)
    ax.add_patch(wedge)

    # ── Puntos LiDAR crudos ──
    if pts and show_points:
        # Mapeo directo: x_plot = x_robot (adelante→derecha), y_plot = y_robot (izq→arriba)
        px = np.array([p[0] for p in pts])
        py = np.array([p[1] for p in pts])
        ax.scatter(px, py, s=1.5, color=POINT_COLOR, alpha=0.35, zorder=2,
                   linewidths=0)

    # ── Segmentos clasificados ──
    for seg in segs:
        x1_p, y1_p = seg.p1[0], seg.p1[1]
        x2_p, y2_p = seg.p2[0], seg.p2[1]

        is_wall_right = (wall_seg is not None and
                         seg.p1 == wall_seg.p1 and seg.p2 == wall_seg.p2)

        if is_wall_right:
            color = WALL_RIGHT_COLOR
            lw = 2.8
            label_text = 'pared'
        elif seg.kind == 'WALL':
            color = WALL_COLOR
            lw = 2.2
            label_text = None
        elif seg.kind == 'BOX':
            color = BOX_COLOR
            lw = 2.5
            label_text = 'caja'
        else:
            color = UNK_COLOR
            lw = 1.5
            label_text = None

        ax.plot([x1_p, x2_p], [y1_p, y2_p], color=color,
                linewidth=lw, solid_capstyle='round', zorder=4)
        # Puntos extremos del segmento
        ax.plot([x1_p, x2_p], [y1_p, y2_p], 'o', color=color,
                markersize=3, zorder=5)

        # Etiqueta
        if label_text:
            mx = (x1_p + x2_p) / 2
            my = (y1_p + y2_p) / 2
            offset_x = 0.06
            offset_y = 0.06 if my >= 0 else -0.06
            ax.annotate(
                label_text,
                (mx, my),
                xytext=(mx + offset_x, my + offset_y),
                color=color, fontsize=6.5, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=0.6, connectionstyle='arc3,rad=0.15'),
                zorder=6,
                bbox=dict(boxstyle='round,pad=0.15', facecolor=PANEL_BG,
                          edgecolor=color, alpha=0.85, linewidth=0.5))

    # ── Línea de distancia al frente (ahora horizontal, hacia +x) ──
    if math.isfinite(front_dist) and front_dist < 3.0:
        fd_x = min(front_dist, view_range * 0.95)
        ax.plot([0, fd_x], [0, 0],
                '--', color=FRONT_SECTOR_COLOR, linewidth=0.8, alpha=0.7, zorder=3)
        fc_color = '#34d399' if front_class == 'NONE' else (
            BOX_COLOR if front_class == 'CAJA' else WALL_COLOR)
        ax.text(min(front_dist, view_range * 0.85), 0.08,
                f'frente\n{front_dist:.2f}m',
                color=fc_color, fontsize=6, fontweight='bold',
                ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.15', facecolor=PANEL_BG,
                          edgecolor=fc_color, alpha=0.85, linewidth=0.5),
                zorder=7)

    # ── Línea de distancia a pared derecha (ahora vertical, hacia -y) ──
    if wall_seg is not None:
        wd = abs(wall_seg.mean_y)
        y_wall_plot = wall_seg.mean_y  # ya negativo si está a la derecha (-y)
        ax.plot([0, 0], [0, y_wall_plot], '--',
                color=WALL_RIGHT_COLOR, linewidth=0.8, alpha=0.7, zorder=3)
        ax.text(0.08, y_wall_plot * 0.5,
                f'{wd:.2f}m',
                color=WALL_RIGHT_COLOR, fontsize=6, fontweight='bold',
                ha='left', va='center', zorder=7)

    # ── Robot (triángulo, ahora apuntando hacia +x = derecha) ──
    robot_size = 0.05
    triangle = plt.Polygon(
        [[robot_size * 1.8, 0],
         [-robot_size, -robot_size],
         [-robot_size, robot_size]],
        closed=True, facecolor=ROBOT_COLOR, edgecolor='#333333',
        linewidth=0.8, zorder=10)
    ax.add_patch(triangle)

    # ── Eje de referencia del robot (flecha corta hacia +x = frente) ──
    ax.annotate('', xy=(0.12, 0), xytext=(0.04, 0),
                arrowprops=dict(arrowstyle='->', color='#aaaaaa',
                                lw=0.6), zorder=8)

    # ── Barra de estado inferior ──
    fd_str = f'{front_dist:.2f}' if math.isfinite(front_dist) and front_dist < 10 else '∞'
    wd_str = f'{abs(wall_seg.mean_y):.2f}' if wall_seg else '—'
    fc_str = front_class

    status = (f'FSM: {action}   frente: {fd_str} m ({fc_str})   '
              f'pared der: {wd_str} m   segs: {len(segs)}')

    ax.text(view_range * 0.32, -view_range * 1.12, status,
            color='#aaaaaa', fontsize=6.5, ha='center', va='top',
            transform=ax.transData,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#0d0d1e',
                      edgecolor='#333355', alpha=0.95, linewidth=0.5))

    ax.set_xlabel('atrás          x [m] — adelante →', color='#555555', fontsize=7)
    ax.set_ylabel('← der          y [m]          izq →', color='#555555', fontsize=7)


def render_map_panel(ax, data):
    """Renderiza el panel inferior: trayectoria acumulada + cajas."""
    ax.cla()
    ax.set_facecolor(PANEL_BG)
    ax.grid(True, color=GRID_COLOR, linewidth=0.3, alpha=0.6)
    ax.set_aspect('equal')
    ax.tick_params(colors='#666666', labelsize=7)
    for spine in ax.spines.values():
        spine.set_color('#333355')
        spine.set_linewidth(0.8)

    trail = data['trail']
    boxes = data['boxes']
    ox = data['odom_x']
    oy = data['odom_y']
    oyaw = data['odom_yaw']
    have_odom = data['have_odom']
    laps = data['laps']

    ax.set_title(
        f'Recorrido + cajas vivas — {laps} vueltas',
        color=TEXT_COLOR, fontsize=9, fontweight='bold', pad=4,
        loc='left')

    if not have_odom:
        ax.text(0.5, 0.5, 'Esperando odometría...',
                color='#555555', fontsize=10, ha='center', va='center',
                transform=ax.transAxes)
        return

    # ── Trayectoria ──
    if len(trail) >= 2:
        tx = [p[0] for p in trail]
        ty = [p[1] for p in trail]

        # Gradiente de color a lo largo del trail
        n_trail = len(tx)
        if n_trail > 2:
            points = np.array([tx, ty]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            # Color gradient: ámbar oscuro → ámbar brillante (combina con TRAIL_COLOR)
            colors = np.zeros((n_trail - 1, 4))
            for i in range(n_trail - 1):
                t = i / max(1, n_trail - 2)
                colors[i] = [0.55 + 0.43 * t, 0.35 + 0.45 * t, 0.03 + 0.05 * t, 0.3 + 0.5 * t]

            lc = LineCollection(segments, colors=colors, linewidths=1.2, zorder=2)
            ax.add_collection(lc)
        else:
            ax.plot(tx, ty, color=TRAIL_COLOR, linewidth=1.0, alpha=0.6, zorder=2)

    # ── Cajas detectadas ──
    if boxes:
        bx = [b[0] for b in boxes]
        by = [b[1] for b in boxes]
        ax.scatter(bx, by, s=60, color=BOX_MARKER_COLOR, marker='s',
                   edgecolors='#fecdd3', linewidths=1.0, alpha=0.85,
                   zorder=5, label=f'cajas detectadas ({len(boxes)})')
        # Etiquetas
        for i, (bxi, byi) in enumerate(boxes):
            ax.text(bxi + 0.05, byi + 0.05, f'C{i+1}',
                    color=BOX_MARKER_COLOR, fontsize=5.5, fontweight='bold',
                    zorder=6)

    # ── Robot actual ──
    robot_size = 0.08
    cos_y = math.cos(oyaw)
    sin_y = math.sin(oyaw)
    tri_pts = [
        (ox + robot_size * 1.5 * cos_y, oy + robot_size * 1.5 * sin_y),
        (ox - robot_size * cos_y - robot_size * sin_y,
         oy - robot_size * sin_y + robot_size * cos_y),
        (ox - robot_size * cos_y + robot_size * sin_y,
         oy - robot_size * sin_y - robot_size * cos_y),
    ]
    robot_tri = plt.Polygon(tri_pts, closed=True,
                            facecolor=ROBOT_COLOR, edgecolor='#333333',
                            linewidth=1.0, zorder=10)
    ax.add_patch(robot_tri)

    # Flecha de heading
    arr_len = 0.15
    ax.annotate('',
                xy=(ox + arr_len * cos_y, oy + arr_len * sin_y),
                xytext=(ox, oy),
                arrowprops=dict(arrowstyle='->', color=ACCENT_COLOR,
                                lw=1.2), zorder=9)

    # ── Punto de inicio ──
    if trail:
        ax.plot(trail[0][0], trail[0][1], 'o', color='#34d399',
                markersize=6, markeredgecolor='#065f46', markeredgewidth=1.0,
                zorder=7, label='inicio')

    # ── Auto-ajustar límites ──
    if trail:
        all_x = [p[0] for p in trail]
        all_y = [p[1] for p in trail]
        if boxes:
            all_x += [b[0] for b in boxes]
            all_y += [b[1] for b in boxes]
        margin = 0.5
        x_min, x_max = min(all_x) - margin, max(all_x) + margin
        y_min, y_max = min(all_y) - margin, max(all_y) + margin
        # Mantener aspecto cuadrado
        dx = x_max - x_min
        dy = y_max - y_min
        if dx > dy:
            cy = (y_min + y_max) / 2
            y_min = cy - dx / 2
            y_max = cy + dx / 2
        else:
            cx = (x_min + x_max) / 2
            x_min = cx - dy / 2
            x_max = cx + dy / 2
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
    else:
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)

    # ── Leyenda ──
    legend_elements = [
        plt.Line2D([0], [0], color=TRAIL_COLOR, linewidth=1.5,
                   label='ruta estimada (odom)'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=BOX_MARKER_COLOR,
                   markersize=6, label=f'cajas detectadas (fijas)',
                   linestyle='None'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#34d399',
                   markersize=5, label='punto de inicio',
                   linestyle='None'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor=ROBOT_COLOR,
                   markersize=6, label='robot (posición actual)',
                   linestyle='None'),
    ]
    leg = ax.legend(handles=legend_elements, loc='upper right',
                    fontsize=5.5, facecolor='#1a1a2e', edgecolor='#333355',
                    labelcolor='#bbbbbb', framealpha=0.9)
    leg.get_frame().set_linewidth(0.5)

    # ── Info de cajas en esquina inferior ──
    n_boxes = len(boxes)
    ax.text(0.02, 0.02,
            f'Cajas conocidas (fodo derecho): {n_boxes}',
            color='#888888', fontsize=6, ha='left', va='bottom',
            transform=ax.transAxes)


# ═══════════════════════════════════════════════════════════════════
#  ESTADO INTERACTIVO DE LA UI
# ═══════════════════════════════════════════════════════════════════
class AppState:
    """Estado mutable de la interfaz, controlado por widgets/teclado.
    Vive solo en el proceso de visualización — no afecta al robot."""
    def __init__(self):
        self.paused = False
        self.view_range = VIEW_RANGE
        self.show_points = True


def main():
    rclpy.init()
    node = LidarMapaNode()

    # ROS 2 en hilo de fondo
    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    fig, ax_scan, ax_map, widgets = build_figure()
    state = AppState()
    plt.ion()
    plt.show()

    # Texto de espera
    wait_text = fig.text(0.5, 0.5, 'Esperando datos del LiDAR...',
                         color='#555555', fontsize=14, ha='center', va='center')

    # ── Callbacks de los widgets ──
    def on_zoom(val):
        state.view_range = float(val)

    def on_pause(event):
        state.paused = not state.paused
        widgets['btn_pause'].label.set_text('Reanudar' if state.paused else 'Pausar')
        fig.canvas.draw_idle()

    def on_clear(event):
        node.clear_trail()

    widgets['slider_zoom'].on_changed(on_zoom)
    widgets['btn_pause'].on_clicked(on_pause)
    widgets['btn_clear'].on_clicked(on_clear)

    # ── Teclado: atajos rápidos sin tocar el mouse ──
    def on_key(event):
        if event.key == ' ':
            on_pause(event)
        elif event.key == 'c':
            node.clear_trail()
        elif event.key == 't':
            state.show_points = not state.show_points
        elif event.key in ('+', '='):
            state.view_range = max(0.4, state.view_range - 0.2)
            widgets['slider_zoom'].set_val(round(state.view_range, 1))
        elif event.key == '-':
            state.view_range = min(3.0, state.view_range + 0.2)
            widgets['slider_zoom'].set_val(round(state.view_range, 1))

    fig.canvas.mpl_connect('key_press_event', on_key)

    # ── Clic en el panel de scan: lee coordenadas (x,y) en el frame del
    #    robot bajo el cursor — útil para verificar a mano que el frente
    #    quedó alineado en +X tras el fix de geometría. ──
    click_annotation = {'artist': None}

    def on_click(event):
        if event.inaxes != ax_scan or event.xdata is None:
            return
        x_r, y_r = event.xdata, event.ydata
        if click_annotation['artist'] is not None:
            try:
                click_annotation['artist'].remove()
            except Exception:
                pass
        txt = ax_scan.annotate(
            f'({x_r:.2f}, {y_r:.2f})m',
            (x_r, y_r), xytext=(8, 8), textcoords='offset points',
            color='#ffeb3b', fontsize=7, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#1a1a2e',
                      edgecolor='#ffeb3b', alpha=0.9, linewidth=0.6),
            zorder=20)
        click_annotation['artist'] = txt
        node.get_logger().info(
            f'Clic en panel scan → frame robot: x={x_r:.3f}m (adelante) '
            f'y={y_r:.3f}m ({"izq" if y_r >= 0 else "der"})')
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', on_click)

    frame_count = 0

    try:
        while plt.fignum_exists(fig.number):
            if not state.paused:
                data = node.get_data()

                if data['have_scan']:
                    if wait_text.get_visible():
                        wait_text.set_visible(False)

                    data['view_range'] = state.view_range
                    data['show_points'] = state.show_points

                    render_scan_panel(ax_scan, data)
                    render_map_panel(ax_map, data)

                    frame_count += 1

                    fig.canvas.draw_idle()

            plt.pause(1.0 / REFRESH_HZ)

    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
