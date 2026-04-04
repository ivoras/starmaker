#version 330 core

// Post-processing pass.
// Applied last, to the composited scene texture.
// Implements:
//   1. Bloom  -- bright pixels bleed light into neighbours (2-pass approximation)
//   2. Vignette -- subtle edge darkening
//   3. Film grain -- mild organic noise to prevent banding in dark regions
//   4. Colour grading -- slight blue-shift and contrast lift for "cold space" feel
//   (Scene tonemap lives in composite.frag; this pass outputs display-referred RGB.)

uniform sampler2D u_scene;
uniform float     u_time;
uniform vec2      u_resolution;

out vec4 fragColor;

// ---- Helpers --------------------------------------------------------
float hash21(vec2 p) {
    p = fract(p * vec2(234.34, 435.345));
    p += dot(p, p + 34.23);
    return fract(p.x * p.y);
}

// Single-pass approximate bloom: samples a cross-shaped neighbourhood of
// the brightest pixels and smears them outward.
vec3 bloom(sampler2D tex, vec2 uv, vec2 texel_size) {
    float bloom_radius = 2.5;
    vec3 acc = vec3(0.0);
    float weight_sum = 0.0;

    // 13-tap cross + diagonals (cheap approximation of Kawase bloom)
    const int TAPS = 13;
    vec2 offsets[13] = vec2[13](
        vec2( 0.0,  0.0),
        vec2( 1.0,  0.0), vec2(-1.0,  0.0),
        vec2( 0.0,  1.0), vec2( 0.0, -1.0),
        vec2( 1.5,  1.5), vec2(-1.5,  1.5),
        vec2( 1.5, -1.5), vec2(-1.5, -1.5),
        vec2( 3.0,  0.0), vec2(-3.0,  0.0),
        vec2( 0.0,  3.0), vec2( 0.0, -3.0)
    );
    float weights[13] = float[13](
        1.0,
        0.6, 0.6,
        0.6, 0.6,
        0.3, 0.3,
        0.3, 0.3,
        0.15, 0.15,
        0.15, 0.15
    );

    for (int i = 0; i < TAPS; i++) {
        vec2 offset_uv = uv + offsets[i] * texel_size * bloom_radius;
        vec3 sample_col = texture(tex, offset_uv).rgb;
        float lum = dot(sample_col, vec3(0.299, 0.587, 0.114));
        float bloom_mask = smoothstep(0.25, 0.85, lum);
        acc += sample_col * bloom_mask * weights[i];
        weight_sum += weights[i];
    }

    return acc / weight_sum;
}

void main() {
    vec2 uv = gl_FragCoord.xy / u_resolution;
    vec2 texel = 1.0 / u_resolution;

    vec3 col = texture(u_scene, uv).rgb;

    // 1. Bloom
    vec3 bloom_col = bloom(u_scene, uv, texel);
    col = col + bloom_col * 0.14;

    // 2. Vignette
    vec2 vig_uv = uv * 2.0 - 1.0;
    float vig = 1.0 - dot(vig_uv * vec2(0.9, 0.85), vig_uv * vec2(0.9, 0.85));
    vig = clamp(pow(vig, 1.5), 0.0, 1.0);
    col *= (0.5 + 0.5 * vig);

    // 3. Film grain
    float grain_amount = 0.006;
    float grain = hash21(uv + fract(u_time * 0.1)) * 2.0 - 1.0;
    col += grain * grain_amount;

    // 4. Colour grading: slight blue-cyan tint; shadow lift only (no wash on highlights)
    float lin_lum = dot(col, vec3(0.299, 0.587, 0.114));
    float shadow_w = clamp(1.0 - lin_lum * 2.5, 0.0, 1.0);
    col = col * 0.98 + 0.004 * shadow_w * vec3(0.1, 0.15, 0.25);
    col.b = col.b * 1.06;
    col.r = col.r * 0.97;
    // Mild midtone contrast
    vec3 curved = smoothstep(0.02, 0.88, col);
    col = mix(col, curved, 0.08);

    // Tone mapping already applied in composite.frag; second Reinhard here
    // crushed bright HDR and made the frame a flat near-white sheet.

    // Gamma correction: convert from linear to gamma 2.2
    col = pow(clamp(col, 0.0, 1.0), vec3(1.0 / 2.2));

    fragColor = vec4(col, 1.0);
}
