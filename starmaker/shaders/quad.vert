#version 330 core

// Full-screen quad vertex shader.
// Draws a clip-space triangle that covers the entire viewport.
// No vertex buffer needed -- just call ctx.screen.render()
// or draw a VAO-less triangle with 3 vertices.

in vec2 in_vert;
out vec2 v_uv;

void main() {
    v_uv = in_vert * 0.5 + 0.5;
    gl_Position = vec4(in_vert, 0.0, 1.0);
}
