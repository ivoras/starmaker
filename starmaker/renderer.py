"""ModernGL headless renderer.

Creates a standalone OpenGL context, compiles the shader pipeline, and
exposes a render() method that advances the simulation by one frame and
writes the raw RGB24 output into a caller-provided moderngl.Buffer.

Multi-pass pipeline
-------------------
Pass 1  nebula_fbo    Full-screen nebula background
Pass 2  stars_fbo     Starfield + warp streaks (RGBA)
Pass 3  composite_fbo Blend nebula + stars + dust
Pass 4  post_fbo      Bloom, vignette, grain, grade, tone-map → final
"""

from __future__ import annotations

import importlib.resources
from pathlib import Path
from typing import Iterator

import moderngl
import numpy as np

from starmaker.config import Config


def _load_shader(name: str) -> str:
    """Load a GLSL source file from the starmaker/shaders package directory."""
    shader_dir = Path(__file__).parent / "shaders"
    return (shader_dir / name).read_text(encoding="utf-8")


def _make_quad_vao(ctx: moderngl.Context, prog: moderngl.Program) -> moderngl.VertexArray:
    """Create a VAO for a full-screen quad (two triangles as a single strip)."""
    vertices = np.array([
        -1.0, -1.0,
         1.0, -1.0,
        -1.0,  1.0,
         1.0,  1.0,
    ], dtype="f4")
    vbo = ctx.buffer(vertices)
    return ctx.vertex_array(prog, [(vbo, "2f", "in_vert")])


class Renderer:
    """Headless moderngl renderer for the space warp scene."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.ctx = moderngl.create_context(standalone=True, require=330)

        w, h = cfg.width, cfg.height
        self._w = w
        self._h = h

        vert_src = _load_shader("quad.vert")

        # Compile shader programs
        self._prog_nebula   = self.ctx.program(
            vertex_shader=vert_src,
            fragment_shader=_load_shader("nebula.frag"),
        )
        self._prog_stars = self.ctx.program(
            vertex_shader=vert_src,
            fragment_shader=_load_shader("starfield.frag"),
        )
        self._prog_composite = self.ctx.program(
            vertex_shader=vert_src,
            fragment_shader=_load_shader("composite.frag"),
        )
        self._prog_post = self.ctx.program(
            vertex_shader=vert_src,
            fragment_shader=_load_shader("post.frag"),
        )

        # Full-screen quad VAOs
        self._vao_nebula    = _make_quad_vao(self.ctx, self._prog_nebula)
        self._vao_stars     = _make_quad_vao(self.ctx, self._prog_stars)
        self._vao_composite = _make_quad_vao(self.ctx, self._prog_composite)
        self._vao_post      = _make_quad_vao(self.ctx, self._prog_post)

        # FBOs ---------------------------------------------------------------
        # nebula: RGB, no alpha needed
        self._nebula_tex = self.ctx.texture((w, h), 3, dtype="f2")
        self._nebula_fbo = self.ctx.framebuffer(color_attachments=[self._nebula_tex])

        # stars: RGBA (A = opacity)
        self._stars_tex = self.ctx.texture((w, h), 4, dtype="f2")
        self._stars_fbo = self.ctx.framebuffer(color_attachments=[self._stars_tex])

        # composite: RGB
        self._comp_tex = self.ctx.texture((w, h), 3, dtype="f2")
        self._comp_fbo = self.ctx.framebuffer(color_attachments=[self._comp_tex])

        # post / final output: RGB8 for easy readback
        self._post_tex = self.ctx.texture((w, h), 3, dtype="u1")
        self._post_fbo = self.ctx.framebuffer(color_attachments=[self._post_tex])

        # Double-buffered output buffers (for TurboPipe)
        bytes_per_frame = w * h * 3
        self.buffers = [self.ctx.buffer(reserve=bytes_per_frame) for _ in range(2)]
        self._buf_idx = 0

        # Cache the seed as a float uniform value
        self._seed_f = float(cfg.seed % 100000)

        # Set static uniforms
        self._set_static_uniforms()

    def _set_static_uniforms(self) -> None:
        cfg = self.cfg
        res = (float(cfg.width), float(cfg.height))

        for prog in (self._prog_nebula, self._prog_stars,
                     self._prog_composite, self._prog_post):
            if "u_resolution" in prog:
                prog["u_resolution"].value = res
            if "u_seed" in prog:
                prog["u_seed"].value = self._seed_f

        if "u_nebula_intensity" in self._prog_nebula:
            self._prog_nebula["u_nebula_intensity"].value = cfg.nebula_intensity
        if "u_nebula_scale" in self._prog_nebula:
            self._prog_nebula["u_nebula_scale"].value = cfg.nebula_scale

        if "u_star_density" in self._prog_stars:
            self._prog_stars["u_star_density"].value = cfg.star_density
        if "u_star_size" in self._prog_stars:
            self._prog_stars["u_star_size"].value = cfg.star_size
        if "u_warp_speed" in self._prog_stars:
            self._prog_stars["u_warp_speed"].value = cfg.warp_speed

        if "u_dust_amount" in self._prog_composite:
            self._prog_composite["u_dust_amount"].value = cfg.dust_amount

    def render_frame(self, frame_index: int) -> moderngl.Buffer:
        """Render one frame and return the double-buffered output Buffer.

        The caller is responsible for calling turbopipe.sync(buf) before
        re-using the buffer, and turbopipe.pipe(buf, fd) to consume it.
        """
        t = frame_index / self.cfg.fps

        # -- Pass 1: Nebula -------------------------------------------------
        self._nebula_fbo.use()
        self._nebula_fbo.clear()
        if "u_time" in self._prog_nebula:
            self._prog_nebula["u_time"].value = t
        self._vao_nebula.render(moderngl.TRIANGLE_STRIP)

        # -- Pass 2: Starfield ----------------------------------------------
        self._stars_fbo.use()
        self._stars_fbo.clear(0.0, 0.0, 0.0, 0.0)
        if "u_time" in self._prog_stars:
            self._prog_stars["u_time"].value = t
        self._vao_stars.render(moderngl.TRIANGLE_STRIP)

        # -- Pass 3: Composite ----------------------------------------------
        self._comp_fbo.use()
        self._comp_fbo.clear()
        self._nebula_tex.use(location=0)
        self._stars_tex.use(location=1)
        if "u_nebula" in self._prog_composite:
            self._prog_composite["u_nebula"].value = 0
        if "u_stars" in self._prog_composite:
            self._prog_composite["u_stars"].value = 1
        if "u_time" in self._prog_composite:
            self._prog_composite["u_time"].value = t
        self._vao_composite.render(moderngl.TRIANGLE_STRIP)

        # -- Pass 4: Post-processing ----------------------------------------
        self._post_fbo.use()
        self._post_fbo.clear()
        self._comp_tex.use(location=0)
        if "u_scene" in self._prog_post:
            self._prog_post["u_scene"].value = 0
        if "u_time" in self._prog_post:
            self._prog_post["u_time"].value = t
        self._vao_post.render(moderngl.TRIANGLE_STRIP)

        # -- Readback into double-buffered output ---------------------------
        buf = self.buffers[self._buf_idx]
        self._buf_idx ^= 1
        self._post_fbo.read_into(buf, components=3, dtype="u1")
        return buf

    def release(self) -> None:
        """Free GPU resources."""
        for buf in self.buffers:
            buf.release()
        for obj in (
            self._nebula_tex, self._stars_tex, self._comp_tex, self._post_tex,
            self._nebula_fbo, self._stars_fbo, self._comp_fbo, self._post_fbo,
            self._vao_nebula, self._vao_stars, self._vao_composite, self._vao_post,
            self._prog_nebula, self._prog_stars, self._prog_composite, self._prog_post,
        ):
            obj.release()
        self.ctx.release()
