#version 330 core

// Composite pass.
// Blends the nebula background (tex0) with the starfield/warp layer (tex1)
// and optionally a dust layer represented by a procedural noise pass.
// Also renders foreground micro-dust particles using animated noise.

uniform sampler2D u_nebula;     // background nebula texture
uniform sampler2D u_stars;      // RGBA starfield layer
uniform float     u_time;
uniform vec2      u_resolution;
uniform float     u_dust_amount;
uniform float     u_seed;

out vec4 fragColor;

float soft_disc(float dist, float radius) {
    return 1.0 - smoothstep(0.0, radius, dist);
}

// ---- Micro-dust / particle noise ------------------------------------
// Very small, faint luminous specks scattered across the field.

float hash1(float n) {
    return fract(sin(n) * 43758.5453);
}

float hash2(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

float dustNoise(vec2 uv) {
    // Tile into cells and place a dust mote in each cell
    float scale = 120.0;  // grid density
    vec2 cell = floor(uv * scale);
    vec2 frac = fract(uv * scale);

    float dust = 0.0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            vec2 neighbour = cell + vec2(dx, dy);
            float h = hash2(neighbour + u_seed * 0.001);
            float h2 = hash2(neighbour + vec2(7.3, 2.1) + u_seed * 0.001);
            // Mote position within cell (0.1..0.9 to avoid edge overlap)
            vec2 mote_pos = vec2(0.1 + 0.8 * h, 0.1 + 0.8 * h2);
            // Very slow drift
            float drift_phase = h * 6.283 + u_time * (0.005 + h * 0.01);
            mote_pos += 0.04 * vec2(cos(drift_phase), sin(drift_phase));
            float d = length(frac - vec2(dx, dy) - mote_pos);
            float radius = 0.006 + 0.014 * hash2(neighbour + vec2(3.1, 9.9));
            float brightness = hash2(neighbour + vec2(1.1, 5.5)) * 0.35;
            dust += brightness * soft_disc(d, radius);
        }
    }
    return dust;
}

void main() {
    vec2 uv = gl_FragCoord.xy / u_resolution;

    vec4 nebula  = texture(u_nebula, uv);
    vec4 stars   = texture(u_stars,  uv);

    // Stars are emissive, so additive composition reads more naturally than
    // alpha blending and avoids washing out the nebula.
    vec3 combined = nebula.rgb + stars.rgb;

    // Foreground dust
    if (u_dust_amount > 0.0) {
        float dust = dustNoise(uv) * u_dust_amount * 0.05;
        // Dust is slightly blue-tinted
        combined += dust * vec3(0.7, 0.8, 1.0);
    }

    fragColor = vec4(clamp(combined, 0.0, 1.0), 1.0);
}
