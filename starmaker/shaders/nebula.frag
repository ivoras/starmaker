#version 330 core

// Nebula background pass.
// Procedural emission nebula using domain-warped Perlin fBm.
// Most of the frame is pitch black deep space; nebulas appear as isolated,
// vivid clouds with wispy tendrils inspired by Hubble / JWST imagery.
//
// Key techniques:
//   - Domain warping (Inigo Quilez, 2002): warp noise coordinates with another
//     noise field to produce organic, filamentary structure.
//   - Ridged noise: abs(noise) inverted to create sharp filament edges
//     resembling ionisation fronts in real emission nebulae.
//   - Emission-line colour palettes: Hα pink, [OIII] teal, [SII] crimson.

uniform float u_time;          // seconds since start
uniform vec2  u_resolution;    // viewport size (pixels)
uniform float u_nebula_intensity; // 0.0 – 3.0
uniform float u_nebula_scale;    // 0.1 – 5.0
uniform float u_nebula_color_cycle_period; // seconds; full cycle through 3 palettes
uniform float u_seed;            // float derived from integer seed

out vec4 fragColor;

// Gradient at lattice point — two independent sin-hashes.
vec2 hash22(vec2 p) {
    vec2 d = vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)));
    return fract(sin(d) * 43758.5453) * 2.0 - 1.0;
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

// Full-detail fBm (5 octaves) for primary density field.
float fbm(vec2 p) {
    float value = 0.0;
    float amp = 0.5;
    float wsum = 0.0;
    for (int i = 0; i < 5; i++) {
        value += amp * (grad_noise2(p) * 0.55 + 0.5);
        wsum += amp;
        p = p * 2.03 + vec2(17.1, 9.2);
        amp *= 0.5;
    }
    return value / wsum;
}

// Low-detail fBm (3 octaves) for domain-warp displacement.
// Broad-scale only: we need the general flow direction, not fine detail.
float fbm_warp(vec2 p) {
    float value = 0.0;
    float amp = 0.5;
    float wsum = 0.0;
    for (int i = 0; i < 3; i++) {
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

    // Higher base scale → smaller nebula features relative to the viewport.
    float scale = mix(4.0, 1.2, clamp((u_nebula_scale - 0.1) / 4.9, 0.0, 1.0));
    vec2 q = p * scale
           + vec2(u_time * 0.008, -u_time * 0.005)
           + vec2(seed_off, seed_off * 0.7);

    // ---- Domain warping ------------------------------------------------
    // Displaces noise coordinates by the output of another noise field,
    // producing the organic, filamentary tendrils seen in real nebulae.
    float w1 = fbm_warp(q);
    float w2 = fbm_warp(q + vec2(5.2, 1.3));
    vec2 warp = vec2(w1, w2);
    vec2 q_warped = q + warp * 0.45;

    float density = fbm(q_warped);

    // ---- Ridged noise for filament detail ------------------------------
    // |noise| creates ridges at zero-crossings — resembles the sharp
    // ionisation fronts visible at nebula boundaries in Hubble images.
    float ridged = 1.0 - abs(grad_noise2(q_warped * 2.5) * 2.0);
    ridged *= ridged;   // sharpen the ridges
    density = mix(density, density * (0.6 + ridged * 0.5), 0.4);

    // ---- Mask: keep most of the frame pitch black ----------------------
    float mask = smoothstep(0.58, 0.88, density);

    // Deep space is black — no ambient tint.
    vec3 base_space = vec3(0.0);

    // ---- Emission-nebula colour palettes -------------------------------
    // Inspired by natural-colour Hubble/JWST images:
    //   Cool  → [OIII] oxygen-III teal / cyan
    //   Mid   → Hα hydrogen-alpha pink / magenta
    //   Warm  → [SII] sulfur-II crimson / warm dust gold
    float period = max(u_nebula_color_cycle_period, 1.0);
    float t3 = fract(u_time / period) * 3.0;

    // Palette 0: Classic emission (Hα pink + OIII teal)
    vec3 p0_cool = vec3(0.05, 0.18, 0.30);
    vec3 p0_mid  = vec3(0.65, 0.12, 0.35);
    vec3 p0_warm = vec3(0.90, 0.25, 0.20);

    // Palette 1: Reflection / dust (deep blue + gold)
    vec3 p1_cool = vec3(0.08, 0.10, 0.35);
    vec3 p1_mid  = vec3(0.30, 0.18, 0.55);
    vec3 p1_warm = vec3(0.85, 0.55, 0.12);

    // Palette 2: Oxygen-dominated (teal-green + rose accent)
    vec3 p2_cool = vec3(0.03, 0.22, 0.18);
    vec3 p2_mid  = vec3(0.15, 0.50, 0.42);
    vec3 p2_warm = vec3(0.72, 0.18, 0.25);

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

    // Colour mapping driven by warp magnitude (how much the field was
    // displaced) and density.  This ties colour variation to structure,
    // mimicking how different emission lines dominate at different depths.
    float warp_mag = length(warp) * 2.0;
    float hue_mix = smoothstep(0.25, 0.70, warp_mag);
    vec3 nebula_colour = mix(cool, midc, hue_mix);
    nebula_colour = mix(nebula_colour, warm, smoothstep(0.72, 0.92, density) * 0.45);

    // Bright rim at nebula edges — ionisation-front glow effect.
    float edge_glow = smoothstep(0.55, 0.72, density) * (1.0 - smoothstep(0.72, 0.90, density));
    nebula_colour += edge_glow * cool * 0.3;

    float inner_glow = smoothstep(0.60, 0.92, density);
    vec3 nebula = nebula_colour * mask * (0.12 + inner_glow * 0.80) * u_nebula_intensity;

    // No upper clamp: nebula renders to float16 FBO.  Clamping here caused
    // white plateaus at high u_nebula_intensity before tone mapping in post.
    vec3 out_rgb = base_space + nebula;
    fragColor = vec4(max(out_rgb, vec3(0.0)), 1.0);
}
