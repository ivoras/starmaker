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
uniform float u_seed;            // float derived from integer seed

out vec4 fragColor;

float hash12(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}

float value_noise2(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    float a = hash12(i);
    float b = hash12(i + vec2(1.0, 0.0));
    float c = hash12(i + vec2(0.0, 1.0));
    float d = hash12(i + vec2(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

float fbm(vec2 p) {
    float value = 0.0;
    float amp = 0.5;
    for (int i = 0; i < 5; i++) {
        value += amp * value_noise2(p);
        p = p * 2.03 + vec2(17.1, 9.2);
        amp *= 0.5;
    }
    return value;
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
    float mask = smoothstep(0.50, 0.78, density);

    vec3 base_space = vec3(0.003, 0.005, 0.012);
    vec3 cool = vec3(0.16, 0.32, 0.72);
    vec3 magenta = vec3(0.56, 0.14, 0.64);
    vec3 warm = vec3(0.95, 0.46, 0.18);

    float hue_mix = smoothstep(0.30, 0.75, n2);
    vec3 nebula_colour = mix(cool, magenta, hue_mix);
    nebula_colour = mix(nebula_colour, warm, smoothstep(0.72, 0.92, density) * 0.35);

    float inner_glow = smoothstep(0.58, 0.92, density);
    vec3 nebula = nebula_colour * mask * (0.16 + inner_glow * 0.70) * u_nebula_intensity;

    fragColor = vec4(clamp(base_space + nebula, 0.0, 1.0), 1.0);
}
