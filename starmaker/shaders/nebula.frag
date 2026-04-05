#version 330 core

// Nebula background pass.
// Generates dark space with broad procedural nebula clouds. The previous
// version pushed too much high-frequency energy into the frame, which made the
// result look like static instead of scenery. This version keeps most of the
// frame near-black and lets the nebula appear only in large cloud islands.

uniform float u_time;          // seconds since start
uniform vec2  u_resolution;    // viewport size (pixels)
uniform float u_nebula_intensity; // 0.0 – 3.0
uniform float u_nebula_scale;    // 0.1 – 5.0
uniform float u_nebula_color_cycle_period; // seconds; full cycle through 3 palettes
uniform float u_seed;            // float derived from integer seed

out vec4 fragColor;

// Unit-ish gradient at lattice point (Perlin-style); avoids value-noise “squares”
vec2 hash22(vec2 p) {
    float n = sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123;
    float a = fract(n) * 6.28318530718;
    return vec2(cos(a), sin(a));
}

// Improved Perlin: quintic fade removes second-derivative kinks along grid lines
float grad_noise2(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);

    vec2 g00 = hash22(i + vec2(0.0, 0.0));
    vec2 g10 = hash22(i + vec2(1.0, 0.0));
    vec2 g01 = hash22(i + vec2(0.0, 1.0));
    vec2 g11 = hash22(i + vec2(1.0, 1.0));

    float n00 = dot(g00, f - vec2(0.0, 0.0));
    float n10 = dot(g10, f - vec2(1.0, 0.0));
    float n01 = dot(g01, f - vec2(0.0, 1.0));
    float n11 = dot(g11, f - vec2(1.0, 1.0));

    return mix(mix(n00, n10, u.x), mix(n01, n11, u.x), u.y);
}

float fbm(vec2 p) {
    float value = 0.0;
    float amp = 0.5;
    float wsum = 0.0;
    for (int i = 0; i < 5; i++) {
        // Perlin is ~[-0.55,0.55] per octave; map to [0,1] like old value noise
        value += amp * (grad_noise2(p) * 0.55 + 0.5);
        wsum += amp;
        p = p * 2.03 + vec2(17.1, 9.2);
        amp *= 0.5;
    }
    return value / wsum;
}

// 2D rotation helper
vec2 rot2(vec2 v, float a) {
    float c = cos(a), s = sin(a);
    return vec2(v.x * c - v.y * s, v.x * s + v.y * c);
}

void main() {
    vec2 uv = gl_FragCoord.xy / u_resolution;
    float aspect = u_resolution.x / u_resolution.y;
    vec2 p = (uv - 0.5) * vec2(aspect, 1.0);

    float seed_off = u_seed * 0.013;
    p = rot2(p, u_time * 0.01);

    float scale = mix(2.8, 0.7, clamp((u_nebula_scale - 0.1) / 4.9, 0.0, 1.0));
    vec2 q1 = p * scale + vec2(u_time * 0.010, -u_time * 0.006) + vec2(seed_off, seed_off * 0.7);
    vec2 q2 = rot2(p, 0.8) * (scale * 1.7) + vec2(-u_time * 0.005, u_time * 0.008) + vec2(seed_off * 1.7, -seed_off);

    float n1 = fbm(q1);
    float n2 = fbm(q2);
    float density = n1 * 0.72 + n2 * 0.43;
    // Wider soft band reads as fluffy cloud edges, not hard blobs
    float mask = smoothstep(0.46, 0.82, density);

    vec3 base_space = vec3(0.003, 0.005, 0.012);

    // Three palettes (cool / mid / warm accent), cross-faded on a slow clock
    float period = max(u_nebula_color_cycle_period, 1.0);
    float t3 = fract(u_time / period) * 3.0;

    vec3 p0_cool = vec3(0.12, 0.08, 0.42);
    vec3 p0_mid  = vec3(0.52, 0.14, 0.62);
    vec3 p0_warm = vec3(0.82, 0.22, 0.58);

    vec3 p1_cool = vec3(0.20, 0.07, 0.03);
    vec3 p1_mid  = vec3(0.68, 0.32, 0.06);
    vec3 p1_warm = vec3(0.92, 0.52, 0.10);

    vec3 p2_cool = vec3(0.03, 0.14, 0.06);
    vec3 p2_mid  = vec3(0.28, 0.58, 0.10);
    vec3 p2_warm = vec3(0.65, 0.06, 0.12);

    vec3 ca, cb, ma, mb, wa, wb;
    float blend;
    if (t3 < 1.0) {
        ca = p0_cool; cb = p1_cool; ma = p0_mid; mb = p1_mid;
        wa = p0_warm; wb = p1_warm; blend = t3;
    } else if (t3 < 2.0) {
        ca = p1_cool; cb = p2_cool; ma = p1_mid; mb = p2_mid;
        wa = p1_warm; wb = p2_warm; blend = t3 - 1.0;
    } else {
        ca = p2_cool; cb = p0_cool; ma = p2_mid; mb = p0_mid;
        wa = p2_warm; wb = p0_warm; blend = t3 - 2.0;
    }
    vec3 cool = mix(ca, cb, blend);
    vec3 midc = mix(ma, mb, blend);
    vec3 warm = mix(wa, wb, blend);

    float hue_mix = smoothstep(0.30, 0.75, n2);
    vec3 nebula_colour = mix(cool, midc, hue_mix);
    nebula_colour = mix(nebula_colour, warm, smoothstep(0.72, 0.92, density) * 0.35);

    float inner_glow = smoothstep(0.54, 0.90, density);
    vec3 nebula = nebula_colour * mask * (0.16 + inner_glow * 0.70) * u_nebula_intensity;

    // No upper clamp: nebula renders to float16 FBO. Clamping here caused white
    // plateaus at high u_nebula_intensity before tone mapping in post.frag.
    vec3 out_rgb = base_space + nebula;
    fragColor = vec4(max(out_rgb, vec3(0.0)), 1.0);
}
