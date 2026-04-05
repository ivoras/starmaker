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
uniform float u_warp_speed;     // travel speed (scene uses ~0.1–9)
uniform float u_seed;           // float from integer seed

out vec4 fragColor;

float soft_disc(float dist, float radius) {
    return 1.0 - smoothstep(0.0, radius, dist);
}

// ---- Pseudo-random helpers -------------------------------------------
float hash1(float n) {
    return fract(sin(n) * 43758.5453123);
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

    vec3 acc_colour = vec3(0.0);
    float acc_alpha = 0.0;

    int star_count = u_star_density;

    for (int i = 0; i < star_count; i++) {
        float fi = float(i) + u_seed;

        // Stable star position in a small square around the origin. Perspective
        // projection and cycling depth create the impression of flying forward.
        vec2 base = vec2(
            hash1(fi * 11.31) * 2.0 - 1.0,
            hash1(fi * 17.17) * 2.0 - 1.0
        );
        base.x *= aspect;
        base *= 0.82;

        // Stars move toward the viewer as depth shrinks toward zero, then wrap.
        float cycle = fract(hash1(fi * 0.071) + u_time * (0.045 + 0.11 * u_warp_speed));
        float depth = mix(1.25, 0.05, cycle);
        vec2 proj = base / depth;

        // Skip stars far outside the viewport to reduce haze.
        if (abs(proj.x) > aspect * 1.3 || abs(proj.y) > 1.3) {
            continue;
        }

        vec2 dir = normalize(proj + vec2(1e-4));
        vec2 to_pixel = screen - proj;
        float along = dot(to_pixel, dir);
        float perp_d = length(to_pixel - along * dir);

        float proximity = 1.0 - depth / 1.25;
        float streak_len = (0.004 + proximity * proximity * 0.17) * u_star_size * (0.65 + u_warp_speed * 0.55);
        float point_r = (0.0009 + proximity * 0.0032) * u_star_size;
        float glow_r = point_r * 5.0;

        float along_clamped = clamp(along, -point_r, streak_len);
        float capsule_dist = length(vec2(along - along_clamped, perp_d));

        float core = soft_disc(capsule_dist, point_r);
        float glow = soft_disc(capsule_dist, glow_r) * (0.08 + proximity * 0.12);
        float star_brightness = (core + glow) * (0.10 + proximity * proximity * 2.2);

        float twinkle = 0.96 + 0.04 * sin(u_time * (1.0 + hash1(fi) * 2.0) + fi);
        star_brightness *= twinkle;

        vec3 col = starColour(fi) * star_brightness;
        acc_colour += col;
        acc_alpha = min(1.0, acc_alpha + star_brightness * 0.12);
    }

    // Allow >1 so composite + Reinhard can recover detail where many stars overlap.
    fragColor = vec4(max(acc_colour, vec3(0.0)), clamp(acc_alpha, 0.0, 1.0));
}
