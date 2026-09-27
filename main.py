import pygame
import cv2
import numpy as np
import random
import math
import time

from core.heart import generate_all_shapes
from vision.hand_tracking import HandTracker


# =========================================================
# SETTINGS
# =========================================================

WIDTH = 1100
HEIGHT = 800

PARTICLE_COUNT = 9000
STAR_COUNT = 450
TRAIL_COUNT = 700

FPS = 60

GESTURE_STABILITY_FRAMES = 6

SHAPE_GESTURES = {
    "heart",
    "galaxy",
    "cloud",
    "star",
}


# =========================================================
# PYGAME
# =========================================================

# Ask pygame for a small, low-latency audio buffer before init.
try:
    pygame.mixer.pre_init(44100, -16, 1, 512)
except Exception:
    pass

pygame.init()

screen = pygame.display.set_mode(
    (WIDTH, HEIGHT),
    pygame.DOUBLEBUF
)

pygame.display.set_caption(
    "Nebula Hand Controlled Particle Universe"
)

clock = pygame.time.Clock()


# =========================================================
# PARTICLE SHAPES
# =========================================================

shapes = generate_all_shapes(PARTICLE_COUNT)

current_shape = shapes["heart"].copy()
target_shape = shapes["heart"].copy()

# Spring-like morph velocity for smoother shape transitions.
morph_velocity = np.zeros_like(current_shape)


# =========================================================
# PARTICLE COLORS
# =========================================================

particle_colors = np.zeros(
    (PARTICLE_COUNT, 3),
    dtype=np.uint8
)

for i in range(PARTICLE_COUNT):
    choice = random.random()

    if choice < 0.55:
        particle_colors[i] = [
            255,
            random.randint(40, 110),
            random.randint(130, 255)
        ]

    elif choice < 0.80:
        particle_colors[i] = [
            random.randint(120, 220),
            random.randint(30, 90),
            255
        ]

    else:
        particle_colors[i] = [
            255,
            255,
            random.randint(180, 255)
        ]


# =========================================================
# STARS
# =========================================================

stars = np.random.uniform(
    -2.5,
    2.5,
    (STAR_COUNT, 3)
)

star_brightness = np.random.randint(
    80,
    230,
    STAR_COUNT
)


# =========================================================
# CAMERA
# =========================================================

camera = cv2.VideoCapture(0)

camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    640
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    480
)

tracker = HandTracker()


# =========================================================
# VARIABLES
# =========================================================

rotation_x = 0.0
rotation_y = 0.0

target_rotation_x = 0.0
target_rotation_y = 0.0

heart_x = 0.0
heart_y = 0.0

target_x = 0.0
target_y = 0.0

particle_scale = 1.0
target_scale = 1.0

current_mode = "heart"

# Gesture stability.
candidate_gesture = None
candidate_frames = 0
stable_gesture = "heart"

# Pinch release detection.
was_pinching = False

# Particle explosion state.
burst_offset = np.zeros(
    (PARTICLE_COUNT, 3),
    dtype=np.float32
)

burst_velocity = np.zeros(
    (PARTICLE_COUNT, 3),
    dtype=np.float32
)

explosion_flash = 0.0

# Time for heartbeat and idle animation.
start_time = time.perf_counter()
heartbeat_high = False

# HUD / hand telemetry state.
hand_position_smooth = None
hud_scan_phase = 0.0
energy_level = 68.0


# =========================================================
# FONT
# =========================================================

font = pygame.font.SysFont(
    "consolas",
    18
)

small_font = pygame.font.SysFont(
    "consolas",
    15
)

big_font = pygame.font.SysFont(
    "consolas",
    26,
    bold=True
)


# =========================================================
# GLOW SURFACE
# =========================================================

glow_surface = pygame.Surface(
    (WIDTH, HEIGHT),
    pygame.SRCALPHA
)


# =========================================================
# TRAILS
# =========================================================

trail_surface = pygame.Surface(
    (WIDTH, HEIGHT),
    pygame.SRCALPHA
)

trail_indices = np.random.choice(
    PARTICLE_COUNT,
    min(TRAIL_COUNT, PARTICLE_COUNT),
    replace=False
)

previous_trail_positions = np.full(
    (len(trail_indices), 2),
    (WIDTH // 2, HEIGHT // 2),
    dtype=np.int32
)


# =========================================================
# PROCEDURAL AUDIO
# =========================================================

AUDIO_AVAILABLE = pygame.mixer.get_init() is not None
sound_muted = False


def make_tone(start_freq, duration, volume=0.25, end_freq=None):
    """Create a short procedural mono sound without external assets."""
    if not AUDIO_AVAILABLE:
        return None

    sample_rate = 44100
    samples = max(1, int(sample_rate * duration))
    t = np.arange(samples, dtype=np.float32) / sample_rate

    if end_freq is None:
        end_freq = start_freq

    sweep_rate = (end_freq - start_freq) / max(duration, 1e-6)
    phase = 2.0 * np.pi * (
        start_freq * t
        + 0.5 * sweep_rate * t * t
    )

    tone = np.sin(phase)

    # Fast attack + exponential decay gives a clean game-like response.
    attack = np.clip(t / 0.008, 0.0, 1.0)
    decay = np.exp(-5.5 * t / max(duration, 1e-6))
    envelope = attack * decay

    audio = (
        tone
        * envelope
        * float(volume)
        * 32767
    ).astype(np.int16)

    try:
        return pygame.sndarray.make_sound(audio)
    except Exception:
        return None


MODE_SOUNDS = {
    "heart": make_tone(520, 0.13, 0.18, 720),
    "galaxy": make_tone(330, 0.16, 0.15, 520),
    "cloud": make_tone(220, 0.18, 0.14, 360),
    "star": make_tone(760, 0.10, 0.16, 980),
}

EXPLOSION_SOUND = make_tone(140, 0.26, 0.23, 980)
PINCH_RELEASE_SOUND = make_tone(620, 0.09, 0.12, 420)
HEARTBEAT_SOUND = make_tone(92, 0.12, 0.20, 72)


def play_sound(sound):
    if sound is None or sound_muted:
        return

    try:
        sound.play()
    except Exception:
        pass


def toggle_sound():
    global sound_muted
    sound_muted = not sound_muted


# =========================================================
# EFFECT HELPERS
# =========================================================

def heartbeat_pulse(elapsed):
    """
    Soft double-beat pulse.
    The cycle is roughly 0.86 seconds (~70 BPM).
    """
    phase = elapsed % 0.86

    first = math.exp(
        -((phase - 0.18) / 0.055) ** 2
    )

    second = math.exp(
        -((phase - 0.38) / 0.075) ** 2
    )

    return 1.0 + (0.075 * first) + (0.045 * second)


def trigger_explosion(power=0.075):
    """
    Launch every particle slightly outward.
    The burst is rendered as an offset on top of the current
    particle shape, then naturally settles back.
    """
    global explosion_flash

    directions = np.random.normal(
        0.0,
        1.0,
        (PARTICLE_COUNT, 3)
    ).astype(np.float32)

    lengths = np.linalg.norm(
        directions,
        axis=1,
        keepdims=True
    )

    lengths[lengths == 0] = 1.0
    directions /= lengths

    strength = np.random.uniform(
        power * 0.35,
        power,
        (PARTICLE_COUNT, 1)
    ).astype(np.float32)

    burst_velocity[:] = directions * strength

    # Slight random starting offset makes the explosion feel less uniform.
    burst_offset[:] = directions * (
        strength * np.random.uniform(
            0.05,
            0.25,
            (PARTICLE_COUNT, 1)
        ).astype(np.float32)
    )

    explosion_flash = 1.0
    play_sound(EXPLOSION_SOUND)


def update_explosion():
    global explosion_flash

    burst_offset[:] += burst_velocity
    burst_velocity[:] *= 0.90
    burst_offset[:] *= 0.93

    explosion_flash *= 0.88

    if explosion_flash < 0.01:
        explosion_flash = 0.0


def set_mode(mode, trigger_fx=True):
    """
    Centralized mode switch so keyboard and gestures behave identically.
    """
    global current_mode
    global target_shape
    global stable_gesture
    global candidate_gesture
    global candidate_frames

    if mode not in SHAPE_GESTURES:
        return

    if current_mode == mode:
        stable_gesture = mode
        return

    current_mode = mode
    target_shape = shapes[mode]

    stable_gesture = mode
    candidate_gesture = None
    candidate_frames = 0

    if trigger_fx:
        play_sound(MODE_SOUNDS.get(mode))
        trigger_explosion(
            0.095 if mode == "heart" else 0.075
        )


def update_gesture_stability(raw_gesture):
    """
    Require the same shape gesture for several consecutive frames.
    Pinch is treated only as a scaling gesture.
    """
    global candidate_gesture
    global candidate_frames

    if raw_gesture in SHAPE_GESTURES:
        if raw_gesture == candidate_gesture:
            candidate_frames += 1
        else:
            candidate_gesture = raw_gesture
            candidate_frames = 1

        if (
            candidate_frames >= GESTURE_STABILITY_FRAMES
            and candidate_gesture != stable_gesture
        ):
            set_mode(candidate_gesture)

    elif raw_gesture in ("pinch", "none"):
        candidate_gesture = None
        candidate_frames = 0


# =========================================================
# 3D FUNCTIONS
# =========================================================

def rotate_points(points, rx, ry):
    cos_x = math.cos(rx)
    sin_x = math.sin(rx)

    cos_y = math.cos(ry)
    sin_y = math.sin(ry)

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # X rotation
    y2 = y * cos_x - z * sin_x
    z2 = y * sin_x + z * cos_x

    # Y rotation
    x3 = x * cos_y + z2 * sin_y
    z3 = -x * sin_y + z2 * cos_y

    return x3, y2, z3


def project_points(x, y, z):
    camera_distance = 3.2

    # Protect perspective division during a strong particle burst.
    denominator = np.maximum(
        camera_distance + z,
        0.25
    )

    depth = camera_distance / denominator

    screen_x = (
        x * depth * 360
        + WIDTH / 2
        + heart_x
    )

    screen_y = (
        -y * depth * 360
        + HEIGHT / 2
        + heart_y
    )

    return (
        screen_x.astype(np.int32),
        screen_y.astype(np.int32),
        depth
    )


# =========================================================
# DRAW TRAILS
# =========================================================

def draw_particle_trails(sx, sy):
    # Fade the previous frame's trails instead of clearing the surface.
    trail_surface.fill(
        (0, 0, 0, 32),
        special_flags=pygame.BLEND_RGBA_MULT
    )

    for j, particle_index in enumerate(trail_indices):
        current_x = int(sx[particle_index])
        current_y = int(sy[particle_index])

        previous_x = int(previous_trail_positions[j, 0])
        previous_y = int(previous_trail_positions[j, 1])

        dx = current_x - previous_x
        dy = current_y - previous_y
        distance_sq = dx * dx + dy * dy

        # Only draw visible movement, avoiding a wall of static lines.
        if distance_sq > 4 and distance_sq < 5000:
            speed = math.sqrt(distance_sq)
            alpha = int(
                np.clip(
                    25 + speed * 5.0 + explosion_flash * 170,
                    25,
                    230
                )
            )

            pygame.draw.line(
                trail_surface,
                (255, 90, 220, alpha),
                (previous_x, previous_y),
                (current_x, current_y),
                1 if explosion_flash < 0.35 else 2
            )

        previous_trail_positions[j, 0] = current_x
        previous_trail_positions[j, 1] = current_y

    screen.blit(
        trail_surface,
        (0, 0)
    )


# =========================================================
# DRAW PARTICLES
# =========================================================

def draw_particles(render_scale):
    # Add the temporary explosion offset to the normal shape.
    render_points = (
        current_shape
        + burst_offset
    )

    scaled = render_points * render_scale

    x, y, z = rotate_points(
        scaled,
        rotation_x,
        rotation_y
    )

    sx, sy, depth = project_points(
        x,
        y,
        z
    )

    valid = (
        (sx >= 0)
        & (sx < WIDTH)
        & (sy >= 0)
        & (sy < HEIGHT)
    )

    indices = np.where(valid)[0]

    draw_particle_trails(sx, sy)

    if len(indices) == 0:
        return

    # Draw far particles first.
    order = indices[
        np.argsort(depth[indices])
    ]

    for i in order:
        size = int(
            np.clip(
                depth[i] * 2.2,
                1,
                4
            )
        )

        # Make burst particles slightly brighter.
        if explosion_flash > 0.12:
            boost = int(
                min(
                    55,
                    explosion_flash * 55
                )
            )

            base = particle_colors[i]
            color = (
                min(255, int(base[0]) + boost),
                min(255, int(base[1]) + boost // 2),
                min(255, int(base[2]) + boost)
            )

        else:
            color = tuple(
                int(value)
                for value in particle_colors[i]
            )

        pygame.draw.circle(
            screen,
            color,
            (sx[i], sy[i]),
            size
        )


# =========================================================
# DRAW STARS
# =========================================================

def draw_stars():
    x, y, z = rotate_points(
        stars,
        rotation_x * 0.25,
        rotation_y * 0.25
    )

    sx, sy, depth = project_points(
        x,
        y,
        z
    )

    for i in range(STAR_COUNT):
        if (
            sx[i] < 0
            or sx[i] >= WIDTH
            or sy[i] < 0
            or sy[i] >= HEIGHT
        ):
            continue

        brightness = int(
            star_brightness[i]
        )

        # A little variation keeps the background alive.
        twinkle = int(
            18 * math.sin(
                time.perf_counter() * 1.8
                + i * 0.37
            )
        )

        brightness = int(
            np.clip(
                brightness + twinkle,
                40,
                255
            )
        )

        pygame.draw.circle(
            screen,
            (
                brightness,
                brightness,
                brightness
            ),
            (
                int(sx[i]),
                int(sy[i])
            ),
            1
        )


# =========================================================
# DRAW GLOW
# =========================================================

def draw_glow(heartbeat):
    glow_surface.fill(
        (0, 0, 0, 0)
    )

    flash = explosion_flash

    center = (
        int(WIDTH / 2 + heart_x),
        int(HEIGHT / 2 + heart_y)
    )

    if current_mode == "heart":
        base_scale = heartbeat
    else:
        base_scale = 1.0

    layers = [
        (
            int(230 * base_scale),
            int(10 + 18 * flash)
        ),
        (
            int(180 * base_scale),
            int(14 + 24 * flash)
        ),
        (
            int(130 * base_scale),
            int(20 + 35 * flash)
        ),
        (
            int(90 * base_scale),
            int(28 + 48 * flash)
        ),
    ]

    for radius, alpha in layers:
        radius = max(radius, 1)
        alpha = min(alpha, 150)

        pygame.draw.circle(
            glow_surface,
            (
                255,
                30,
                170,
                alpha
            ),
            center,
            radius
        )

    screen.blit(
        glow_surface,
        (0, 0)
    )


# =========================================================
# FUTURISTIC HUD
# =========================================================

MODE_COLORS = {
    "heart": (255, 70, 190),
    "galaxy": (90, 210, 255),
    "cloud": (200, 160, 255),
    "star": (255, 210, 90),
}

hud_surface = pygame.Surface(
    (WIDTH, HEIGHT),
    pygame.SRCALPHA
)

hud_panel = pygame.Surface(
    (300, 335),
    pygame.SRCALPHA
)


def hud_color():
    return MODE_COLORS.get(current_mode, (255, 100, 220))


def draw_corner_brackets(surface, rect, color, length=20, width=2):
    x, y, w, h = rect

    # Top-left
    pygame.draw.line(surface, color, (x, y), (x + length, y), width)
    pygame.draw.line(surface, color, (x, y), (x, y + length), width)

    # Top-right
    pygame.draw.line(surface, color, (x + w - length, y), (x + w, y), width)
    pygame.draw.line(surface, color, (x + w, y), (x + w, y + length), width)

    # Bottom-left
    pygame.draw.line(surface, color, (x, y + h), (x + length, y + h), width)
    pygame.draw.line(surface, color, (x, y + h - length), (x, y + h), width)

    # Bottom-right
    pygame.draw.line(surface, color, (x + w - length, y + h), (x + w, y + h), width)
    pygame.draw.line(surface, color, (x + w, y + h - length), (x + w, y + h), width)


def draw_hand_reticle(position, elapsed):
    if position is None:
        return

    px = int(np.clip(position[0], 0.0, 1.0) * WIDTH)
    py = int(np.clip(position[1], 0.0, 1.0) * HEIGHT)

    pulse = 4.0 + 3.0 * math.sin(elapsed * 5.0)
    radius = int(18 + max(0.0, target_scale - 1.0) * 18 + pulse)

    color = hud_color()
    reticle_color = (*color, 210)

    pygame.draw.circle(
        hud_surface,
        reticle_color,
        (px, py),
        radius,
        2
    )

    pygame.draw.circle(
        hud_surface,
        (255, 255, 255, 185),
        (px, py),
        3
    )

    pygame.draw.line(
        hud_surface,
        reticle_color,
        (px - radius - 8, py),
        (px - radius + 2, py),
        2
    )
    pygame.draw.line(
        hud_surface,
        reticle_color,
        (px + radius - 2, py),
        (px + radius + 8, py),
        2
    )
    pygame.draw.line(
        hud_surface,
        reticle_color,
        (px, py - radius - 8),
        (px, py - radius + 2),
        2
    )
    pygame.draw.line(
        hud_surface,
        reticle_color,
        (px, py + radius - 2),
        (px, py + radius + 8),
        2
    )

    tag = small_font.render(
        f"HAND // {raw_gesture.upper()}",
        True,
        color
    )

    hud_surface.blit(
        tag,
        (px + radius + 10, py - 9)
    )


def draw_center_target(elapsed, heartbeat):
    cx = int(WIDTH / 2 + heart_x)
    cy = int(HEIGHT / 2 + heart_y)

    color = hud_color()
    ring_alpha = int(70 + min(70, explosion_flash * 100))

    radius = int(145 + 8 * math.sin(elapsed * 1.9))
    if current_mode == "heart":
        radius += int((heartbeat - 1.0) * 180)

    pygame.draw.circle(
        hud_surface,
        (*color, ring_alpha),
        (cx, cy),
        max(60, radius),
        1
    )

    pygame.draw.arc(
        hud_surface,
        (*color, 190),
        (cx - radius - 8, cy - radius - 8, (radius + 8) * 2, (radius + 8) * 2),
        elapsed % (math.tau),
        (elapsed % math.tau) + 0.75,
        2
    )

    pygame.draw.arc(
        hud_surface,
        (255, 255, 255, 150),
        (cx - radius - 8, cy - radius - 8, (radius + 8) * 2, (radius + 8) * 2),
        (elapsed % math.tau) + 2.2,
        (elapsed % math.tau) + 2.65,
        2
    )

    # Small axis ticks create a scanner / targeting feel.
    tick = radius + 18
    for ang in np.linspace(0, math.tau, 8, endpoint=False):
        x1 = cx + int(math.cos(ang) * tick)
        y1 = cy + int(math.sin(ang) * tick)
        x2 = cx + int(math.cos(ang) * (tick + 10))
        y2 = cy + int(math.sin(ang) * (tick + 10))
        pygame.draw.line(
            hud_surface,
            (*color, 105),
            (x1, y1),
            (x2, y2),
            1
        )


def draw_hud_panel(raw_gesture, camera_ok, heartbeat, elapsed):
    global hud_scan_phase
    global energy_level

    hud_scan_phase = (hud_scan_phase + 0.035) % 1.0

    color = hud_color()
    hud_surface.fill((0, 0, 0, 0))

    # --- Top-left title frame ---
    title_rect = (20, 18, 430, 78)
    pygame.draw.rect(
        hud_surface,
        (6, 5, 18, 150),
        title_rect,
        border_radius=10
    )
    pygame.draw.rect(
        hud_surface,
        (*color, 95),
        title_rect,
        width=1,
        border_radius=10
    )
    draw_corner_brackets(
        hud_surface,
        title_rect,
        (*color, 210),
        14,
        2
    )

    title = big_font.render(
        "NEBULA / 3D HEART",
        True,
        color
    )
    hud_surface.blit(title, (36, 30))

    subtitle = small_font.render(
        "HAND INTERFACE // PARTICLE CORE ONLINE",
        True,
        (170, 180, 205)
    )
    hud_surface.blit(subtitle, (38, 63))

    # --- Top-right status ---
    online_text = "CAMERA ONLINE" if camera_ok else "CAMERA OFFLINE"
    online_color = (100, 255, 175) if camera_ok else (255, 110, 110)
    status = font.render(online_text, True, online_color)
    hud_surface.blit(status, (WIDTH - 235, 25))

    sound_status = "AUDIO ACTIVE" if not sound_muted else "AUDIO MUTED"
    sound_color = (255, 170, 220) if not sound_muted else (150, 150, 165)
    hud_surface.blit(
        small_font.render(sound_status, True, sound_color),
        (WIDTH - 235, 50)
    )

    # --- Right telemetry panel ---
    panel_x = WIDTH - 325
    panel_y = 105
    panel_w = 300
    panel_h = 335

    panel_rect = pygame.Rect(panel_x, panel_y, panel_w, panel_h)
    pygame.draw.rect(
        hud_panel,
        (4, 4, 14, 192),
        (0, 0, panel_w, panel_h),
        border_radius=12
    )
    pygame.draw.rect(
        hud_panel,
        (*color, 100),
        (0, 0, panel_w, panel_h),
        width=1,
        border_radius=12
    )
    draw_corner_brackets(
        hud_panel,
        (0, 0, panel_w, panel_h),
        (*color, 190),
        16,
        2
    )

    hud_surface.blit(hud_panel, (panel_x, panel_y))

    panel_title = font.render(
        "LIVE TELEMETRY",
        True,
        color
    )
    hud_surface.blit(panel_title, (panel_x + 18, panel_y + 16))

    hud_surface.blit(
        small_font.render("SYSTEM ID  NBU-03D-HEART", True, (130, 145, 175)),
        (panel_x + 18, panel_y + 42)
    )

    def telemetry_row(label, value, y, value_color=(235, 240, 255)):
        hud_surface.blit(
            small_font.render(label, True, (135, 150, 178)),
            (panel_x + 18, y)
        )
        hud_surface.blit(
            small_font.render(value, True, value_color),
            (panel_x + 118, y)
        )

    telemetry_row(
        "MODE",
        current_mode.upper(),
        panel_y + 76,
        color
    )
    telemetry_row(
        "RAW GESTURE",
        raw_gesture.upper(),
        panel_y + 101,
        (220, 225, 240)
    )
    telemetry_row(
        "STABLE",
        stable_gesture.upper(),
        panel_y + 126,
        (165, 255, 210)
    )
    telemetry_row(
        "PARTICLES",
        f"{PARTICLE_COUNT:,}",
        panel_y + 151
    )
    telemetry_row(
        "ROT X",
        f"{rotation_x:+.2f}",
        panel_y + 176
    )
    telemetry_row(
        "ROT Y",
        f"{rotation_y:+.2f}",
        panel_y + 201
    )
    telemetry_row(
        "SCALE",
        f"{particle_scale:.2f}x",
        panel_y + 226
    )

    # Energy / pinch bar.
    target_energy = np.interp(
        np.clip(target_scale, 0.65, 1.75),
        [0.65, 1.75],
        [22.0, 100.0]
    )
    energy_level += (target_energy - energy_level) * 0.08

    bar_x = panel_x + 18
    bar_y = panel_y + 277
    bar_w = 260
    bar_h = 10

    pygame.draw.rect(
        hud_surface,
        (26, 28, 42, 220),
        (bar_x, bar_y, bar_w, bar_h),
        border_radius=5
    )

    fill_w = int(bar_w * np.clip(energy_level / 100.0, 0.0, 1.0))
    if fill_w > 0:
        pygame.draw.rect(
            hud_surface,
            (*color, 215),
            (bar_x, bar_y, fill_w, bar_h),
            border_radius=5
        )

    hud_surface.blit(
        small_font.render(
            f"ENERGY  {int(energy_level):02d}%",
            True,
            (185, 200, 225)
        ),
        (bar_x, bar_y + 15)
    )

    # Stability lock progress.
    lock_progress = (
        min(candidate_frames, GESTURE_STABILITY_FRAMES)
        / GESTURE_STABILITY_FRAMES
        if candidate_gesture in SHAPE_GESTURES
        else 1.0
    )

    lock_y = panel_y + 309
    hud_surface.blit(
        small_font.render("GESTURE LOCK", True, (135, 150, 178)),
        (bar_x, lock_y)
    )

    lock_x = panel_x + 118
    lock_w = 160
    pygame.draw.rect(
        hud_surface,
        (25, 27, 40, 220),
        (lock_x, lock_y + 2, lock_w, 7),
        border_radius=4
    )

    if lock_progress > 0:
        pygame.draw.rect(
            hud_surface,
            (120, 255, 205, 205),
            (
                lock_x,
                lock_y + 2,
                int(lock_w * np.clip(lock_progress, 0.0, 1.0)),
                7
            ),
            border_radius=4
        )

    # --- Center object targeting ring ---
    draw_center_target(elapsed, heartbeat)

    # --- Hand tracking reticle ---
    draw_hand_reticle(hand_position_smooth, elapsed)

    # --- Scan line ---
    scan_y = int(120 + hud_scan_phase * (HEIGHT - 220))
    pygame.draw.line(
        hud_surface,
        (*color, 22),
        (18, scan_y),
        (WIDTH - 18, scan_y),
        1
    )

    # --- Bottom command strip ---
    command_rect = pygame.Rect(18, HEIGHT - 58, WIDTH - 36, 38)
    pygame.draw.rect(
        hud_surface,
        (4, 4, 13, 185),
        command_rect,
        border_radius=8
    )
    pygame.draw.rect(
        hud_surface,
        (*color, 70),
        command_rect,
        width=1,
        border_radius=8
    )

    controls = small_font.render(
        "1-4 SHAPES    SPACE BURST    M AUDIO    ESC EXIT    |    PINCH = SCALE",
        True,
        (170, 180, 205)
    )
    hud_surface.blit(
        controls,
        (command_rect.x + 16, command_rect.y + 10)
    )

    # Burst alert.
    if explosion_flash > 0.18:
        burst_alpha = int(np.clip(explosion_flash * 255, 0, 255))
        burst = big_font.render(
            "// PARTICLE BURST //",
            True,
            (255, 205, 245)
        )
        burst.set_alpha(burst_alpha)
        rect = burst.get_rect(
            center=(WIDTH // 2, 120)
        )
        hud_surface.blit(burst, rect)

    screen.blit(hud_surface, (0, 0))


def update_hand_hud_position(position):
    global hand_position_smooth

    if position is None:
        return

    px, py = position
    target = np.array(
        [
            float(np.clip(px, 0.0, 1.0)),
            float(np.clip(py, 0.0, 1.0)),
        ],
        dtype=np.float32
    )

    if hand_position_smooth is None:
        hand_position_smooth = target
        return

    hand_position_smooth += (
        target - hand_position_smooth
    ) * 0.18


def draw_camera_status(raw_gesture, camera_ok):
    # Keep a compact legacy status block so the Pygame HUD and
    # the hand-tracking window both expose camera state.
    status = "CAMERA: ONLINE" if camera_ok else "CAMERA: OFFLINE"
    status_color = (100, 255, 170) if camera_ok else (255, 120, 120)

    screen.blit(
        small_font.render(status, True, status_color),
        (WIDTH - 235, 78)
    )

# =========================================================
# HEARTBEAT AUDIO
# =========================================================

def update_heartbeat_sound(heartbeat):
    global heartbeat_high

    if current_mode != "heart":
        heartbeat_high = False
        return

    high_now = heartbeat > 1.045

    if high_now and not heartbeat_high:
        play_sound(HEARTBEAT_SOUND)

    heartbeat_high = high_now


# =========================================================
# MAIN LOOP
# =========================================================

running = True
raw_gesture = "none"
camera_ok = False

try:
    while running:
        clock.tick(FPS)

        elapsed = time.perf_counter() - start_time

        # -----------------------------------------------------
        # EVENTS
        # -----------------------------------------------------

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if (
                event.type == pygame.KEYDOWN
                and event.key == pygame.K_ESCAPE
            ):
                running = False

            # Keyboard testing.
            if (
                event.type == pygame.KEYDOWN
                and event.key == pygame.K_1
            ):
                set_mode("heart")

            if (
                event.type == pygame.KEYDOWN
                and event.key == pygame.K_2
            ):
                set_mode("galaxy")

            if (
                event.type == pygame.KEYDOWN
                and event.key == pygame.K_3
            ):
                set_mode("cloud")

            if (
                event.type == pygame.KEYDOWN
                and event.key == pygame.K_4
            ):
                set_mode("star")

            # Manual explosion test.
            if (
                event.type == pygame.KEYDOWN
                and event.key == pygame.K_SPACE
            ):
                trigger_explosion(0.11)

            if (
                event.type == pygame.KEYDOWN
                and event.key == pygame.K_m
            ):
                toggle_sound()

        # -----------------------------------------------------
        # CAMERA / HAND TRACKING
        # -----------------------------------------------------

        success, frame = camera.read()

        if success:
            camera_ok = True

            frame = cv2.flip(
                frame,
                1
            )

            result = tracker.process(
                frame
            )

            tracker.draw_landmarks(
                frame,
                result
            )

            raw_gesture = tracker.get_gesture(
                result
            )

            position = tracker.get_hand_position(
                result
            )

            pinch_scale = tracker.get_pinch_strength(
                result
            )

            # Gesture stability.
            update_gesture_stability(
                raw_gesture
            )

            # -------------------------------------------------
            # HAND POSITION
            # -------------------------------------------------

            if position:
                update_hand_hud_position(position)

                hand_x, hand_y = position

                target_x = (
                    (hand_x - 0.5)
                    * WIDTH
                    * 0.8
                )

                target_y = (
                    (hand_y - 0.5)
                    * HEIGHT
                    * 0.7
                )

                target_rotation_y = (
                    hand_x - 0.5
                ) * 2.5

                target_rotation_x = (
                    hand_y - 0.5
                ) * 2.0

            # -------------------------------------------------
            # PINCH
            # -------------------------------------------------

            pinching = (
                raw_gesture == "pinch"
            )

            if pinching:
                target_scale = float(
                    np.clip(
                        pinch_scale,
                        0.65,
                        1.75
                    )
                )
            else:
                target_scale = 1.0

            # Pinch release -> small particle burst.
            if was_pinching and not pinching:
                play_sound(PINCH_RELEASE_SOUND)
                trigger_explosion(0.055)

            was_pinching = pinching

            # -------------------------------------------------
            # CAMERA WINDOW
            # -------------------------------------------------

            cv2.putText(
                frame,
                f"RAW: {raw_gesture.upper()}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 100, 220),
                2
            )

            cv2.putText(
                frame,
                f"STABLE: {stable_gesture.upper()}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (100, 255, 200),
                2
            )

            cv2.imshow(
                "Hand Tracking",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                running = False

        else:
            camera_ok = False
            raw_gesture = "none"
            was_pinching = False
            target_scale = 1.0

        # =====================================================
        # SMOOTH PARTICLE MORPH
        # =====================================================

        # Spring-style interpolation:
        # smoother than direct linear lerping and gives a softer
        # "living" transition between shapes.
        morph_force = (
            target_shape - current_shape
        ) * 0.028

        morph_velocity += morph_force
        morph_velocity *= 0.82
        current_shape += morph_velocity

        # =====================================================
        # UPDATE EXPLOSION
        # =====================================================

        update_explosion()

        # =====================================================
        # SMOOTH MOVEMENT
        # =====================================================

        heart_x += (
            target_x - heart_x
        ) * 0.08

        heart_y += (
            target_y - heart_y
        ) * 0.08

        rotation_x += (
            target_rotation_x - rotation_x
        ) * 0.08

        rotation_y += (
            target_rotation_y - rotation_y
        ) * 0.08

        particle_scale += (
            target_scale - particle_scale
        ) * 0.08

        # Continuous idle rotation.
        rotation_y += 0.005

        # =====================================================
        # HEARTBEAT
        # =====================================================

        heartbeat = (
            heartbeat_pulse(elapsed)
            if current_mode == "heart"
            else 1.0
        )

        update_heartbeat_sound(heartbeat)

        render_scale = (
            particle_scale
            * heartbeat
        )

        # =====================================================
        # BACKGROUND
        # =====================================================

        screen.fill(
            (2, 1, 8)
        )

        # =====================================================
        # RENDER
        # =====================================================

        draw_stars()

        draw_glow(
            heartbeat
        )

        draw_particles(
            render_scale
        )

        # =====================================================
        # HUD
        # =====================================================

        draw_hud_panel(
            raw_gesture,
            camera_ok,
            heartbeat,
            elapsed
        )

        draw_camera_status(
            raw_gesture,
            camera_ok
        )

        pygame.display.flip()

finally:
    # =========================================================
    # CLEANUP
    # =========================================================

    try:
        tracker.release()
    except Exception:
        pass

    camera.release()
    cv2.destroyAllWindows()
    pygame.quit()
