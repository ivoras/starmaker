#version 330 core

// Nebula background pass.
// Generates a rich, slowly-evolving nebula using fast 3D computed noise with
// fractal Brownian motion (fBm).  Two independent turbulence surfaces are
// combined and colour-mapped through power functions to produce a palette of
// deep purples, cobalt blues, electric teals, and warm amber filaments.
// The whole field rotates very slowly and the noise coordinates are animated
// so the nebula subtly breathes over the full video duration.

uniform float u_time;          // seconds since start
uniform vec2  u_resolution;    // viewport size (pixels)
uniform float u_nebula_intensity; // 0.0 – 3.0
uniform float u_nebula_scale;    // 0.1 – 5.0
uniform float u_seed;            // float derived from integer seed

out vec4 fragColor;

// ---- Fast computed 3D noise (no texture, GPU-friendly) -----------------
// Reference: http://www.gamedev.net/topic/502913-fast-computed-noise/

const float NA = 1.0;
const float NB = 57.0;
const float NC = 113.0;
const vec3  NABC = vec3(NA, NB, NC);
const vec4  NA3  = vec4(0.0, NB, NC, NC + NB);
const vec4  NA4  = vec4(NA, NA + NB, NC + NA, NC + NA + NB);

vec4 _rand4(vec4 x) {
    vec4 z = mod(x, vec4(5612.0));
    z = mod(z, vec4(3.1415927 * 2.0));
    return fract(cos(z) * vec4(56812.5453));
}

float cnoise(vec3 xx) {
    vec3 x  = mod(xx + 32768.0, 65536.0);
    vec3 ix = floor(x);
    vec3 fx = fract(x);
    vec3 wx = fx * fx * (3.0 - 2.0 * fx);  // smoothstep
    float nn = dot(ix, NABC);
    vec4 R1 = _rand4(nn + NA3);
    vec4 R2 = _rand4(nn + NA4);
    vec4 R  = mix(R1, R2, wx.x);
    return 1.0 - 2.0 * mix(mix(R.x, R.y, wx.y), mix(R.z, R.w, wx.y), wx.z);
}

// fBm turbulence surface -- absolute values create filamentary structure
float turbulence(vec3 p, float freq) {
    float n = 0.0;
    float amp = 1.0;
    float f = freq;
    for (int i = 0; i < 6; i++) {
        n   += amp * abs(cnoise(p * f));
        amp *= 0.5;
        f   *= 2.02;
    }
    return n;
}

// 2D rotation helper
vec2 rot2(vec2 v, float a) {
    float c = cos(a), s = sin(a);
    return vec2(v.x * c - v.y * s, v.x * s + v.y * c);
}

// ---- Colour palette helpers -------------------------------------------
// Maps a 0-1 luminance to a colour using three independent power curves.
// Different exponent sets produce different hue families.

vec3 nebulaColour(float lum, vec3 exponents) {
    return pow(vec3(1.0 - lum), exponents);
}

void main() {
    vec2 uv = gl_FragCoord.xy / u_resolution;

    // Aspect-correct UV, centred at 0.5
    float aspect = u_resolution.x / u_resolution.y;
    vec2 p = (uv - 0.5) * vec2(aspect, 1.0);

    // Slow global rotation
    float rot_angle = u_time * 0.018;
    p = rot2(p, rot_angle);

    // Seed offset so different seeds look different
    float seed_off = u_seed * 0.001;

    // Scale by nebula_scale (inverted: larger scale = larger features = smaller freq)
    float base_freq = 0.9 / u_nebula_scale;

    // 3D coordinate with seed offset and time animation
    // Two independent surfaces for layered complexity
    vec3 q1 = vec3(p * sin(u_time * 0.05 + seed_off),
                   u_time * 0.04 + seed_off);
    // Slight Y/Z rotation for volume illusion
    q1 = q1 * mat3(1, 0, 0,
                   0, 0.8, 0.6,
                   0, -0.6, 0.8);

    vec3 q2 = vec3(p * cos(u_time * 0.04 + seed_off + 1.57),
                   u_time * 0.035 + seed_off + 7.3);
    q2 = q2 * mat3(1, 0, 0,
                   0, 0.8, 0.6,
                   0, -0.6, 0.8);

    float n1 = turbulence(q1, base_freq);
    float n2 = turbulence(q2, base_freq * 0.88);

    float lum1 = length(n1);
    float lum2 = length(n2);

    // Primary layer: cool blues / purples
    float ex_r = sin(p.x * 2.0) + cos(u_time * 0.1) + 4.0;
    float ex_g = 8.0 + sin(u_time * 0.07) + 4.0;
    float ex_b = 80.0;
    vec3 col1 = nebulaColour(lum1, vec3(ex_r, ex_g, ex_b)) * 6.8;

    // Secondary layer: warm amber / teal
    float ex2_r = 5.0;
    float ex2_g = p.y + cos(u_time * 0.08) + 7.0;
    float ex2_b = sin(p.x * 1.5) + sin(u_time * 0.11) + 2.0;
    vec3 col2 = nebulaColour(lum2, vec3(ex2_r, ex2_g, ex2_b)) * 2.5;

    // Blend and apply intensity
    vec3 nebula = (col1 + col2) * u_nebula_intensity;

    // Keep space dark where nebula is thin (avoid grey background)
    float brightness = dot(nebula, vec3(0.299, 0.587, 0.114));
    nebula *= smoothstep(0.0, 0.3, brightness);

    fragColor = vec4(clamp(nebula, 0.0, 1.0), 1.0);
}
