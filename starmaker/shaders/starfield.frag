#version 330 core

// Starfield + warp-speed streaks pass.
// Outputs an RGBA layer: RGB = star colour, A = star opacity.
// The compositor blends this over the nebula background using alpha.
//
// Each star occupies a virtual 3D position that cycles in depth (Z-axis),
// creating the illusion of flying through an infinite field.
// Stars close to the camera (shallow Z) are drawn as stretched streaks;
// distant stars are rendered as soft round glows.

uniform float u_time;
uniform vec2  u_resolution;
uniform int   u_star_density;   // 50 – 2000
uniform float u_star_size;      // glow radius multiplier
uniform float u_warp_speed;     // travel speed
uniform float u_seed;           // float from integer seed

out vec4 fragColor;

// ---- Pseudo-random helpers -------------------------------------------
float hash1(float n) {
    return fract(sin(n) * 43758.5453123);
}

float hash1_2(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}

// ---- Star colour palette: white, blue-white, yellow-white, red-giant --
vec3 starColour(float idx) {
    float t = hash1(idx * 0.317 + 5.1);
    if (t < 0.55) return vec3(1.0, 1.0, 1.0);          // white
    if (t < 0.75) return vec3(0.75, 0.85, 1.0);         // blue-white
    if (t < 0.90) return vec3(1.0, 0.95, 0.75);         // yellow-white
    return vec3(1.0, 0.6, 0.4);                          // red giant
}

void main() {
    vec2 uv = gl_FragCoord.xy / u_resolution;
    float aspect = u_resolution.x / u_resolution.y;
    vec2 screen = (uv - 0.5) * vec2(aspect, 1.0);

    float depth_range = 256.0;
    float max_speed = u_warp_speed * 60.0;  // depth units per second

    vec3 acc_colour = vec3(0.0);
    float acc_alpha  = 0.0;

    int star_count = u_star_density;

    for (int i = 0; i < star_count; i++) {
        float fi = float(i) + u_seed;

        // Star 2D position in a ±250 virtual-unit field, using sin() as a
        // compact hash (no uniform array needed).
        vec2 star_pos = vec2(sin(fi * 1.7) * 250.0,
                             sin(fi * fi * 0.003 + fi) * 250.0);

        // Z-depth cycles from 0..depth_range, approaching viewer over time
        float raw_z = mod(fi * fi * 0.005 + max_speed * u_time, depth_range);
        float z = depth_range - raw_z;  // 256 = far, 0 = right here

        float fade = raw_z / depth_range; // 0=just appeared, 1=just passed

        // Project: stars map ±250 world units to the screen via perspective
        float inv_z = 1.0 / max(z, 0.5);
        vec2 proj = star_pos * inv_z;

        // Distance from this pixel to the star's projected centre
        float dist = length(screen - proj);

        // Streak length proportional to velocity (closer = longer streak)
        // Direction is radially outward (warp effect)
        float speed_factor = max_speed * inv_z * 0.016; // pixels-ish per frame
        float streak_len = speed_factor * u_star_size * 3.0;
        streak_len = clamp(streak_len, 0.0, 0.4);

        // Build a capsule shape: a point at proj, elongated toward the centre
        vec2 streak_dir = normalize(proj + vec2(0.001, 0.001)); // away from centre
        // Project pixel onto the streak axis
        vec2 to_pixel = screen - proj;
        float along  = dot(to_pixel, streak_dir);
        float perp_d = length(to_pixel - along * streak_dir);

        // Streak body: from 0 to streak_len along the outward axis
        float along_clamped = clamp(along, -streak_len * 0.1, streak_len);
        float capsule_dist  = length(vec2(along - along_clamped, perp_d));

        // Point radius scales with size param and fades with distance
        float point_r = 0.0025 * u_star_size * (fade * 0.5 + 0.3);
        float glow_r  = point_r * 4.0;

        // Soft glow around the streak/point
        float core  = smoothstep(point_r, 0.0, capsule_dist);
        float glow  = smoothstep(glow_r,  0.0, capsule_dist) * 0.3;
        float star_brightness = (core + glow) * fade * fade;

        // Twinkle: very subtle, slower for distant stars
        float twinkle = 1.0 + 0.08 * sin(u_time * (2.5 + hash1(fi) * 3.0) + fi);
        star_brightness *= twinkle;

        vec3 col = starColour(fi) * star_brightness;

        // Additive accumulation (stars are emissive)
        acc_colour += col;
        acc_alpha   = min(1.0, acc_alpha + star_brightness);
    }

    acc_colour = clamp(acc_colour, 0.0, 1.0);
    fragColor = vec4(acc_colour, clamp(acc_alpha, 0.0, 1.0));
}
