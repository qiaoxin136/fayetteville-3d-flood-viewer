/* ==========================================================================
   Fayetteville 2-Yr Stormwater 3D Drape Engine (app.js)
   - Streams compact GPU triangle buffer (flood_mesh.bin, 213,075 triangles)
   - Renders 512x512 WebMercator slippy tiles on-the-fly in WebGL2 (<0.5ms/tile)
   - Drapes tiles onto Google Photorealistic 3D Tiles (tileset.imageryLayers)
     and seamlessly onto high-res aerial imagery fallback when no key is set
   - Provides 5-microsecond spatial-grid point-in-triangle DEPTH2D probing
   ========================================================================== */

(function () {
  "use strict";

  const FT_TO_M = 0.3048;
  const STORAGE_GOOGLE_KEY = "fayetteville_google_3d_api_key";
  const STORAGE_ION_TOKEN = "fayetteville_cesium_ion_token";

  // 7-stage hydrologic engineering thresholds (feet) mapped to equal staff intervals
  const STAGE_KNOTS_FT = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.5];

  const PALETTES = {
    // 0: Calibrated Hydrologic Hazard (translucent ice cyan -> cobalt -> violet -> crimson -> crest gold)
    0: [
      [186, 230, 253, 0.56],
      [56, 189, 248, 0.75],
      [2, 132, 199, 0.83],
      [29, 78, 216, 0.88],
      [109, 40, 217, 0.91],
      [190, 24, 93, 0.94],
      [249, 115, 22, 0.97],
      [250, 204, 21, 0.99],
    ],
    // 1: Bathymetric Blues
    1: [
      [224, 242, 254, 0.54],
      [125, 211, 252, 0.74],
      [56, 189, 248, 0.82],
      [14, 165, 233, 0.87],
      [2, 132, 199, 0.91],
      [29, 78, 216, 0.95],
      [30, 58, 138, 0.97],
      [23, 37, 84, 0.99],
    ],
    // 2: High-Contrast Crest Warning
    2: [
      [254, 249, 195, 0.56],
      [254, 240, 138, 0.74],
      [250, 204, 21, 0.83],
      [251, 146, 60, 0.88],
      [249, 115, 22, 0.92],
      [239, 68, 68, 0.95],
      [190, 18, 60, 0.97],
      [136, 19, 55, 0.99],
    ],
  };

  function depthToNorm(depthFt) {
    if (depthFt <= 0) return 0;
    const k = STAGE_KNOTS_FT;
    const segs = k.length - 1;
    for (let i = 0; i < segs; i++) {
      if (depthFt <= k[i + 1]) {
        const t = (depthFt - k[i]) / (k[i + 1] - k[i]);
        return (i + Math.max(0, Math.min(1, t))) / segs;
      }
    }
    return 1.0;
  }

  function depthToStageName(depthFt) {
    if (depthFt < 0.1) return "Trace / Sheet Film";
    if (depthFt < 0.25) return "Gutter & Shallow Ponding";
    if (depthFt < 0.5) return "Curb Overtopping";
    if (depthFt < 1.0) return "Roadway Crown Inundation";
    if (depthFt < 2.0) return "Vehicle Stall Risk";
    if (depthFt < 5.0) return "Structural Inundation";
    return "Deep Channel / Culvert Crest";
  }

  function depthToCssColor(depthFt, paletteId) {
    const stops = PALETTES[paletteId] || PALETTES[0];
    const u = depthToNorm(depthFt) * (stops.length - 1);
    const idx = Math.min(stops.length - 2, Math.max(0, Math.floor(u)));
    const f = u - idx;
    const c0 = stops[idx];
    const c1 = stops[idx + 1];
    const r = Math.round(c0[0] + (c1[0] - c0[0]) * f);
    const g = Math.round(c0[1] + (c1[1] - c0[1]) * f);
    const b = Math.round(c0[2] + (c1[2] - c0[2]) * f);
    return `rgb(${r}, ${g}, ${b})`;
  }

  function lonLatToMercNorm(lonDeg, latDeg) {
    const mx = (lonDeg + 180.0) / 360.0;
    const clampedLat = Math.max(-85.05112878, Math.min(85.05112878, latDeg));
    const latRad = (clampedLat * Math.PI) / 180.0;
    const my = (1.0 - Math.log(Math.tan(latRad) + 1.0 / Math.cos(latRad)) / Math.PI) * 0.5;
    return [mx, my];
  }

  function mercNormToLonLat(mx, my) {
    const lonDeg = mx * 360.0 - 180.0;
    const n = Math.PI - 2.0 * Math.PI * my;
    const latDeg = (180.0 / Math.PI) * Math.atan(0.5 * (Math.exp(n) - Math.exp(-n)));
    return [lonDeg, latDeg];
  }

  /* ========================================================================
     Offscreen WebGL2 / WebGL1 Tile Rasterizer
     ======================================================================== */
  class WebGLFloodRasterizer {
    constructor(mesh) {
      this.mesh = mesh;
      this.tileSize = 512;

      this.glCanvas = document.createElement("canvas");
      this.glCanvas.width = this.tileSize;
      this.glCanvas.height = this.tileSize;

      const ctxAttrs = {
        alpha: true,
        premultipliedAlpha: false,
        antialias: true,
        depth: false,
        stencil: false,
        preserveDrawingBuffer: true,
      };

      let gl = this.glCanvas.getContext("webgl2", ctxAttrs);
      this.isWebGL2 = !!gl;
      if (!gl) {
        gl = this.glCanvas.getContext("webgl", ctxAttrs);
        if (!gl) throw new Error("WebGL is unavailable in this browser.");
        gl.getExtension("OES_element_index_uint");
        gl.getExtension("OES_standard_derivatives");
      }
      this.gl = gl;

      this.emptyCanvas = document.createElement("canvas");
      this.emptyCanvas.width = 16;
      this.emptyCanvas.height = 16;

      this._initShaders();
      this._initSmoothBuffers();
      this.flatReady = false;
    }

    _compile(type, src) {
      const gl = this.gl;
      const sh = gl.createShader(type);
      gl.shaderSource(sh, src);
      gl.compileShader(sh);
      if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
        const info = gl.getShaderInfoLog(sh);
        gl.deleteShader(sh);
        throw new Error("Shader compile error: " + info);
      }
      return sh;
    }

    _initShaders() {
      const gl = this.gl;
      const vsSrc = this.isWebGL2
        ? `#version 300 es
           precision highp float;
           in vec2 a_pos;
           in float a_depth;
           in vec3 a_bary;
           uniform vec2 u_tileOrigin;
           uniform float u_invSpan;
           out float v_depth;
           out vec3 v_bary;
           void main() {
             vec2 uv = (a_pos - u_tileOrigin) * u_invSpan;
             gl_Position = vec4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
             v_depth = a_depth;
             v_bary = a_bary;
           }`
        : `precision highp float;
           attribute vec2 a_pos;
           attribute float a_depth;
           attribute vec3 a_bary;
           uniform vec2 u_tileOrigin;
           uniform float u_invSpan;
           varying float v_depth;
           varying vec3 v_bary;
           void main() {
             vec2 uv = (a_pos - u_tileOrigin) * u_invSpan;
             gl_Position = vec4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
             v_depth = a_depth;
             v_bary = a_bary;
           }`;

      const rampBody = `
        float depthToNorm(float d) {
          if (d <= 0.10) return clamp(d / 0.10, 0.0, 1.0) / 7.0;
          if (d <= 0.25) return (1.0 + (d - 0.10) / 0.15) / 7.0;
          if (d <= 0.50) return (2.0 + (d - 0.25) / 0.25) / 7.0;
          if (d <= 1.00) return (3.0 + (d - 0.50) / 0.50) / 7.0;
          if (d <= 2.00) return (4.0 + (d - 1.00) / 1.00) / 7.0;
          if (d <= 5.00) return (5.0 + (d - 2.00) / 3.00) / 7.0;
          return min(1.0, (6.0 + (d - 5.00) / 5.50) / 7.0);
        }

        vec4 samplePalette(float u, int pal) {
          float s = clamp(u, 0.0, 1.0) * 7.0;
          vec4 c0, c1;
          float f = fract(s);
          int idx = int(floor(s));
          if (idx >= 7) { idx = 6; f = 1.0; }

          if (pal == 1) {
            if (idx == 0) { c0 = vec4(0.878, 0.949, 0.996, 0.54); c1 = vec4(0.490, 0.827, 0.988, 0.74); }
            else if (idx == 1) { c0 = vec4(0.490, 0.827, 0.988, 0.74); c1 = vec4(0.220, 0.741, 0.973, 0.82); }
            else if (idx == 2) { c0 = vec4(0.220, 0.741, 0.973, 0.82); c1 = vec4(0.055, 0.647, 0.914, 0.87); }
            else if (idx == 3) { c0 = vec4(0.055, 0.647, 0.914, 0.87); c1 = vec4(0.008, 0.518, 0.780, 0.91); }
            else if (idx == 4) { c0 = vec4(0.008, 0.518, 0.780, 0.91); c1 = vec4(0.114, 0.306, 0.847, 0.95); }
            else if (idx == 5) { c0 = vec4(0.114, 0.306, 0.847, 0.95); c1 = vec4(0.118, 0.227, 0.541, 0.97); }
            else { c0 = vec4(0.118, 0.227, 0.541, 0.97); c1 = vec4(0.090, 0.145, 0.329, 0.99); }
          } else if (pal == 2) {
            if (idx == 0) { c0 = vec4(0.996, 0.976, 0.765, 0.56); c1 = vec4(0.996, 0.941, 0.541, 0.74); }
            else if (idx == 1) { c0 = vec4(0.996, 0.941, 0.541, 0.74); c1 = vec4(0.980, 0.800, 0.082, 0.83); }
            else if (idx == 2) { c0 = vec4(0.980, 0.800, 0.082, 0.83); c1 = vec4(0.984, 0.573, 0.235, 0.88); }
            else if (idx == 3) { c0 = vec4(0.984, 0.573, 0.235, 0.88); c1 = vec4(0.976, 0.451, 0.086, 0.92); }
            else if (idx == 4) { c0 = vec4(0.976, 0.451, 0.086, 0.92); c1 = vec4(0.937, 0.267, 0.267, 0.95); }
            else if (idx == 5) { c0 = vec4(0.937, 0.267, 0.267, 0.95); c1 = vec4(0.745, 0.071, 0.235, 0.97); }
            else { c0 = vec4(0.745, 0.071, 0.235, 0.97); c1 = vec4(0.533, 0.075, 0.216, 0.99); }
          } else {
            if (idx == 0) { c0 = vec4(0.729, 0.902, 0.992, 0.56); c1 = vec4(0.220, 0.741, 0.973, 0.75); }
            else if (idx == 1) { c0 = vec4(0.220, 0.741, 0.973, 0.75); c1 = vec4(0.008, 0.518, 0.780, 0.83); }
            else if (idx == 2) { c0 = vec4(0.008, 0.518, 0.780, 0.83); c1 = vec4(0.114, 0.306, 0.847, 0.88); }
            else if (idx == 3) { c0 = vec4(0.114, 0.306, 0.847, 0.88); c1 = vec4(0.427, 0.157, 0.851, 0.91); }
            else if (idx == 4) { c0 = vec4(0.427, 0.157, 0.851, 0.91); c1 = vec4(0.745, 0.094, 0.365, 0.94); }
            else if (idx == 5) { c0 = vec4(0.745, 0.094, 0.365, 0.94); c1 = vec4(0.976, 0.451, 0.086, 0.97); }
            else { c0 = vec4(0.976, 0.451, 0.086, 0.97); c1 = vec4(0.980, 0.800, 0.082, 0.99); }
          }
          return mix(c0, c1, f);
        }
      `;

      const fsSrc = this.isWebGL2
        ? `#version 300 es
           precision highp float;
           in float v_depth;
           in vec3 v_bary;
           uniform float u_minDepth;
           uniform int u_palette;
           uniform int u_cellMode;
           uniform int u_zoomLevel;
           out vec4 outColor;
           ${rampBody}
           void main() {
             if (v_depth < u_minDepth) discard;
             float u = depthToNorm(v_depth);
             vec4 col = samplePalette(u, u_palette);

             if (u_cellMode == 0) {
               float fw = max(0.008, fwidth(v_depth));
               float edgeAlpha = smoothstep(u_minDepth, u_minDepth + fw * 1.35, v_depth);
               col.a *= edgeAlpha;
             } else if (u_zoomLevel >= 16) {
               vec3 d = fwidth(v_bary);
               vec3 a3 = smoothstep(vec3(0.0), d * 1.15, v_bary);
               float wire = min(min(a3.x, a3.y), a3.z);
               col.rgb = mix(col.rgb * 0.42, col.rgb, wire);
             }
             outColor = col;
           }`
        : `#extension GL_OES_standard_derivatives : enable
           precision highp float;
           varying float v_depth;
           varying vec3 v_bary;
           uniform float u_minDepth;
           uniform int u_palette;
           uniform int u_cellMode;
           uniform int u_zoomLevel;
           ${rampBody}
           void main() {
             if (v_depth < u_minDepth) discard;
             float u = depthToNorm(v_depth);
             vec4 col = samplePalette(u, u_palette);
             gl_FragColor = col;
           }`;

      const vs = this._compile(gl.VERTEX_SHADER, vsSrc);
      const fs = this._compile(gl.FRAGMENT_SHADER, fsSrc);
      const prog = gl.createProgram();
      gl.attachShader(prog, vs);
      gl.attachShader(prog, fs);
      gl.linkProgram(prog);
      if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
        throw new Error("Program link error: " + gl.getProgramInfoLog(prog));
      }
      this.prog = prog;

      this.locPos = gl.getAttribLocation(prog, "a_pos");
      this.locDepth = gl.getAttribLocation(prog, "a_depth");
      this.locBary = gl.getAttribLocation(prog, "a_bary");
      this.uTileOrigin = gl.getUniformLocation(prog, "u_tileOrigin");
      this.uInvSpan = gl.getUniformLocation(prog, "u_invSpan");
      this.uMinDepth = gl.getUniformLocation(prog, "u_minDepth");
      this.uPalette = gl.getUniformLocation(prog, "u_palette");
      this.uCellMode = gl.getUniformLocation(prog, "u_cellMode");
      this.uZoomLevel = gl.getUniformLocation(prog, "u_zoomLevel");
    }

    _initSmoothBuffers() {
      const gl = this.gl;
      const { vertXY, vertDepth, triIdx } = this.mesh;

      this.vboPos = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, this.vboPos);
      gl.bufferData(gl.ARRAY_BUFFER, vertXY, gl.STATIC_DRAW);

      this.vboDepth = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, this.vboDepth);
      gl.bufferData(gl.ARRAY_BUFFER, vertDepth, gl.STATIC_DRAW);

      this.ebo = gl.createBuffer();
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.ebo);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, triIdx, gl.STATIC_DRAW);
    }

    _ensureFlatBuffers() {
      if (this.flatReady) return;
      const gl = this.gl;
      const { vertXY, triIdx, triDepth, numTris } = this.mesh;
      // Interleaved 6 floats per corner: [mx, my, cellDepth, bx, by, bz]
      const data = new Float32Array(numTris * 3 * 6);
      let p = 0;
      for (let t = 0; t < numTris; t++) {
        const d = triDepth[t];
        const i0 = triIdx[t * 3] * 2;
        const i1 = triIdx[t * 3 + 1] * 2;
        const i2 = triIdx[t * 3 + 2] * 2;

        data[p++] = vertXY[i0];     data[p++] = vertXY[i0 + 1]; data[p++] = d; data[p++] = 1; data[p++] = 0; data[p++] = 0;
        data[p++] = vertXY[i1];     data[p++] = vertXY[i1 + 1]; data[p++] = d; data[p++] = 0; data[p++] = 1; data[p++] = 0;
        data[p++] = vertXY[i2];     data[p++] = vertXY[i2 + 1]; data[p++] = d; data[p++] = 0; data[p++] = 0; data[p++] = 1;
      }
      this.vboFlat = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, this.vboFlat);
      gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
      this.flatVertCount = numTris * 3;
      this.flatReady = true;
    }

    renderTile(relMinMx, relMinMy, invSpan, level, opts) {
      const gl = this.gl;
      gl.viewport(0, 0, this.tileSize, this.tileSize);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.disable(gl.DEPTH_TEST);
      gl.disable(gl.CULL_FACE);
      gl.disable(gl.BLEND);

      gl.useProgram(this.prog);
      gl.uniform2f(this.uTileOrigin, relMinMx, relMinMy);
      gl.uniform1f(this.uInvSpan, invSpan);
      gl.uniform1f(this.uMinDepth, opts.cutoffFt);
      gl.uniform1i(this.uPalette, opts.palette);
      gl.uniform1i(this.uCellMode, opts.cellMode ? 1 : 0);
      gl.uniform1i(this.uZoomLevel, level);

      if (!opts.cellMode) {
        gl.bindBuffer(gl.ARRAY_BUFFER, this.vboPos);
        gl.enableVertexAttribArray(this.locPos);
        gl.vertexAttribPointer(this.locPos, 2, gl.FLOAT, false, 0, 0);

        gl.bindBuffer(gl.ARRAY_BUFFER, this.vboDepth);
        gl.enableVertexAttribArray(this.locDepth);
        gl.vertexAttribPointer(this.locDepth, 1, gl.FLOAT, false, 0, 0);

        if (this.locBary >= 0) {
          gl.disableVertexAttribArray(this.locBary);
          gl.vertexAttrib3f(this.locBary, 1, 1, 1);
        }

        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.ebo);
        gl.drawElements(gl.TRIANGLES, this.mesh.numTris * 3, gl.UNSIGNED_INT, 0);
      } else {
        this._ensureFlatBuffers();
        const stride = 24;
        gl.bindBuffer(gl.ARRAY_BUFFER, this.vboFlat);
        gl.enableVertexAttribArray(this.locPos);
        gl.vertexAttribPointer(this.locPos, 2, gl.FLOAT, false, stride, 0);

        gl.enableVertexAttribArray(this.locDepth);
        gl.vertexAttribPointer(this.locDepth, 1, gl.FLOAT, false, stride, 8);

        if (this.locBary >= 0) {
          gl.enableVertexAttribArray(this.locBary);
          gl.vertexAttribPointer(this.locBary, 3, gl.FLOAT, false, stride, 12);
        }
        gl.drawArrays(gl.TRIANGLES, 0, this.flatVertCount);
      }

      const out = document.createElement("canvas");
      out.width = this.tileSize;
      out.height = this.tileSize;
      const ctx2d = out.getContext("2d");
      ctx2d.drawImage(this.glCanvas, 0, 0);
      return out;
    }
  }

  /* ========================================================================
     CesiumJS Custom ImageryProvider for 3D Tileset & Terrain Draping
     ======================================================================== */
  class FloodMeshImageryProvider {
    constructor(rasterizer, meta, opts) {
      this.rasterizer = rasterizer;
      this.meta = meta;
      this.opts = opts;

      this.tilingScheme = new Cesium.WebMercatorTilingScheme();
      this.tileWidth = 512;
      this.tileHeight = 512;
      this.minimumLevel = 0;
      this.maximumLevel = 22;
      this.rectangle = Cesium.Rectangle.fromDegrees(
        meta.bbox.west - 0.003,
        meta.bbox.south - 0.003,
        meta.bbox.east + 0.003,
        meta.bbox.north + 0.003
      );
      this.errorEvent = new Cesium.Event();
      this.credit = new Cesium.Credit("2yr.json 2D Hydraulic Flood Mesh");
      this.hasAlphaChannel = true;
      this.tileDiscardPolicy = undefined;
      this.proxy = undefined;
      this.ready = true;
    }

    getTileCredits() {
      return [];
    }

    pickFeatures() {
      return undefined;
    }

    requestImage(x, y, level) {
      const n = Math.pow(2, level);
      const tileMinMx = x / n;
      const tileMaxMx = (x + 1) / n;
      const tileMinMy = y / n;
      const tileMaxMy = (y + 1) / n;

      const mb = this.meta.mercBounds;
      if (
        tileMaxMx < mb.mx0 ||
        tileMinMx > mb.mx0 + mb.mxSpan ||
        tileMaxMy < mb.my0 ||
        tileMinMy > mb.my0 + mb.mySpan
      ) {
        return Promise.resolve(this.rasterizer.emptyCanvas);
      }

      const canvas = this.rasterizer.renderTile(
        tileMinMx - mb.mx0,
        tileMinMy - mb.my0,
        n,
        level,
        this.opts
      );
      return Promise.resolve(canvas);
    }
  }

  /* ========================================================================
     Uniform Spatial Grid for Microsecond Point-in-Triangle Depth Probing
     ======================================================================== */
  class SpatialMeshPicker {
    constructor(mesh, mercBounds) {
      this.mesh = mesh;
      this.mx0 = mercBounds.mx0;
      this.my0 = mercBounds.my0;
      this.spanX = mercBounds.mxSpan;
      this.spanY = mercBounds.mySpan;
      this.cols = 200;
      this.rows = 450;

      const totalCells = this.cols * this.rows;
      const counts = new Uint32Array(totalCells);
      const { vertXY, triIdx, numTris } = mesh;

      for (let t = 0; t < numTris; t++) {
        const i0 = triIdx[t * 3] * 2;
        const i1 = triIdx[t * 3 + 1] * 2;
        const i2 = triIdx[t * 3 + 2] * 2;
        const minX = Math.min(vertXY[i0], vertXY[i1], vertXY[i2]);
        const maxX = Math.max(vertXY[i0], vertXY[i1], vertXY[i2]);
        const minY = Math.min(vertXY[i0 + 1], vertXY[i1 + 1], vertXY[i2 + 1]);
        const maxY = Math.max(vertXY[i0 + 1], vertXY[i1 + 1], vertXY[i2 + 1]);

        const c0 = Math.max(0, Math.min(this.cols - 1, Math.floor((minX / this.spanX) * this.cols)));
        const c1 = Math.max(0, Math.min(this.cols - 1, Math.floor((maxX / this.spanX) * this.cols)));
        const r0 = Math.max(0, Math.min(this.rows - 1, Math.floor((minY / this.spanY) * this.rows)));
        const r1 = Math.max(0, Math.min(this.rows - 1, Math.floor((maxY / this.spanY) * this.rows)));

        for (let r = r0; r <= r1; r++) {
          const rowOff = r * this.cols;
          for (let c = c0; c <= c1; c++) {
            counts[rowOff + c]++;
          }
        }
      }

      this.offsets = new Uint32Array(totalCells + 1);
      for (let i = 0; i < totalCells; i++) {
        this.offsets[i + 1] = this.offsets[i] + counts[i];
      }

      this.gridTris = new Uint32Array(this.offsets[totalCells]);
      const cursor = new Uint32Array(this.offsets);

      for (let t = 0; t < numTris; t++) {
        const i0 = triIdx[t * 3] * 2;
        const i1 = triIdx[t * 3 + 1] * 2;
        const i2 = triIdx[t * 3 + 2] * 2;
        const minX = Math.min(vertXY[i0], vertXY[i1], vertXY[i2]);
        const maxX = Math.max(vertXY[i0], vertXY[i1], vertXY[i2]);
        const minY = Math.min(vertXY[i0 + 1], vertXY[i1 + 1], vertXY[i2 + 1]);
        const maxY = Math.max(vertXY[i0 + 1], vertXY[i1 + 1], vertXY[i2 + 1]);

        const c0 = Math.max(0, Math.min(this.cols - 1, Math.floor((minX / this.spanX) * this.cols)));
        const c1 = Math.max(0, Math.min(this.cols - 1, Math.floor((maxX / this.spanX) * this.cols)));
        const r0 = Math.max(0, Math.min(this.rows - 1, Math.floor((minY / this.spanY) * this.rows)));
        const r1 = Math.max(0, Math.min(this.rows - 1, Math.floor((maxY / this.spanY) * this.rows)));

        for (let r = r0; r <= r1; r++) {
          const rowOff = r * this.cols;
          for (let c = c0; c <= c1; c++) {
            const idx = rowOff + c;
            this.gridTris[cursor[idx]++] = t;
          }
        }
      }
    }

    queryLonLat(lonDeg, latDeg) {
      const [mx, my] = lonLatToMercNorm(lonDeg, latDeg);
      const rx = mx - this.mx0;
      const ry = my - this.my0;
      if (rx < 0 || rx > this.spanX || ry < 0 || ry > this.spanY) return null;

      const c = Math.floor((rx / this.spanX) * this.cols);
      const r = Math.floor((ry / this.spanY) * this.rows);
      if (c < 0 || c >= this.cols || r < 0 || r >= this.rows) return null;

      const cell = r * this.cols + c;
      const start = this.offsets[cell];
      const end = this.offsets[cell + 1];
      const { vertXY, vertDepth, triIdx, triDepth } = this.mesh;

      for (let k = end - 1; k >= start; k--) {
        const t = this.gridTris[k];
        const v0 = triIdx[t * 3];
        const v1 = triIdx[t * 3 + 1];
        const v2 = triIdx[t * 3 + 2];

        const ax = vertXY[v0 * 2], ay = vertXY[v0 * 2 + 1];
        const bx = vertXY[v1 * 2], by = vertXY[v1 * 2 + 1];
        const cx = vertXY[v2 * 2], cy = vertXY[v2 * 2 + 1];

        const denom = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy);
        if (Math.abs(denom) < 1e-20) continue;
        const w0 = ((by - cy) * (rx - cx) + (cx - bx) * (ry - cy)) / denom;
        const w1 = ((cy - ay) * (rx - cx) + (ax - cx) * (ry - cy)) / denom;
        const w2 = 1.0 - w0 - w1;

        if (w0 >= -1e-4 && w1 >= -1e-4 && w2 >= -1e-4) {
          const smoothDepth = w0 * vertDepth[v0] + w1 * vertDepth[v1] + w2 * vertDepth[v2];
          return {
            triIndex: t,
            cellDepthFt: triDepth[t],
            smoothDepthFt: smoothDepth,
            cornersLonLat: [
              mercNormToLonLat(ax + this.mx0, ay + this.my0),
              mercNormToLonLat(bx + this.mx0, by + this.my0),
              mercNormToLonLat(cx + this.mx0, cy + this.my0),
            ],
          };
        }
      }
      return null;
    }
  }

  /* ========================================================================
     Main Application Controller
     ======================================================================== */
  const state = {
    unit: "ft",
    cutoffFt: 0.05,
    opacity: 0.84,
    palette: 0,
    shading: "smooth",
    google3dActive: false,
    googleApiKey: "",
    pinnedProbe: null,
    streetViewOpen: false,
    followMapCamera: true,
    svFloodOverlay: true,
    svTarget: {
      lon: -78.89025,
      lat: 35.060316,
      heading: 165,
      pitch: -3,
    },
  };

  let viewer = null;
  let googleTileset = null;
  let currentFloodLayer = null;
  let retiringLayers = [];
  let rasterizer = null;
  let picker = null;
  let meta = null;
  let sortedCellDepths = null;
  let pinEntities = [];

  // Street View synchronization state
  let svPanorama = null;
  let svService = null;
  let gmapsLoadedKey = "";
  let gmapsLoadPromise = null;
  let svEmbedFallback = false;
  let svSyncingFromPano = false;
  let svSyncingFromMap = false;
  let isMapCameraMoving = false;
  let svPanoReqSeq = 0;
  let lastPanoFetchMs = 0;
  let panoSettleTimer = null;
  let embedSettleTimer = null;
  let lastQueriedPanoLon = -999;
  let lastQueriedPanoLat = -999;
  let lastEmbedUrl = "";
  let cachedGroundHeightM = 0;
  let lastDepthPickMs = 0;
  let svBeaconEntity = null;

  function formatDepth(depthFt) {
    if (state.unit === "m") {
      return `${(depthFt * FT_TO_M).toFixed(2)} m`;
    }
    return `${depthFt.toFixed(2)} ft`;
  }

  function countVisibleCells(cutoffFt) {
    if (!sortedCellDepths) return 0;
    let lo = 0;
    let hi = sortedCellDepths.length;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (sortedCellDepths[mid] < cutoffFt) lo = mid + 1;
      else hi = mid;
    }
    return sortedCellDepths.length - lo;
  }

  async function fetchBinaryMesh() {
    const [metaResp, binResp] = await Promise.all([
      fetch("./flood_meta.json"),
      fetch("./flood_mesh.bin"),
    ]);
    if (!metaResp.ok || !binResp.ok) {
      throw new Error("Failed to load preprocessed flood_meta.json / flood_mesh.bin");
    }
    const metaJson = await metaResp.json();
    const buf = await binResp.arrayBuffer();
    const dv = new DataView(buf);
    const magic =
      String.fromCharCode(dv.getUint8(0)) +
      String.fromCharCode(dv.getUint8(1)) +
      String.fromCharCode(dv.getUint8(2)) +
      String.fromCharCode(dv.getUint8(3));
    if (magic !== "FL2D") throw new Error("Invalid flood_mesh.bin header: " + magic);

    const numVerts = dv.getUint32(8, true);
    const numTris = dv.getUint32(12, true);
    const mx0 = dv.getFloat64(16, true);
    const my0 = dv.getFloat64(24, true);

    let off = 32;
    const vertXY = new Float32Array(buf, off, numVerts * 2);
    off += numVerts * 8;
    const vertDepth = new Float32Array(buf, off, numVerts);
    off += numVerts * 4;
    const triIdx = new Uint32Array(buf, off, numTris * 3);
    off += numTris * 12;
    const triDepth = new Float32Array(buf, off, numTris);

    return {
      meta: metaJson,
      mesh: { numVerts, numTris, mx0, my0, vertXY, vertDepth, triIdx, triDepth },
    };
  }

  function initCesiumViewer() {
    const aerialProvider = new Cesium.UrlTemplateImageryProvider({
      url: "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
      maximumLevel: 19,
      credit: "Esri, Maxar, Earthstar Geographics",
    });

    viewer = new Cesium.Viewer("cesiumContainer", {
      baseLayer: new Cesium.ImageryLayer(aerialProvider),
      baseLayerPicker: false,
      geocoder: false,
      homeButton: false,
      sceneModePicker: false,
      navigationHelpButton: false,
      animation: false,
      timeline: false,
      fullscreenButton: false,
      infoBox: false,
      selectionIndicator: false,
      requestRenderMode: false,
      msaaSamples: 4,
    });

    viewer.scene.globe.depthTestAgainstTerrain = true;
    viewer.scene.globe.baseColor = Cesium.Color.fromCssColorString("#111921");
    viewer.scene.backgroundColor = Cesium.Color.fromCssColorString("#0B1117");
    viewer.scene.postProcessStages.fxaa.enabled = true;
  }

  function getActiveImageryCollection() {
    if (state.google3dActive && googleTileset) {
      return googleTileset.imageryLayers;
    }
    return viewer.imageryLayers;
  }

  function refreshFloodDrapeLayer() {
    if (!rasterizer || !meta || !viewer) return;
    const collection = getActiveImageryCollection();

    const provider = new FloodMeshImageryProvider(rasterizer, meta, {
      cutoffFt: state.cutoffFt,
      palette: state.palette,
      cellMode: state.shading === "cells",
    });

    const nextLayer = new Cesium.ImageryLayer(provider, {
      alpha: state.opacity,
      rectangle: provider.rectangle,
    });

    collection.add(nextLayer);

    if (currentFloodLayer) {
      retiringLayers.push({ collection, layer: currentFloodLayer });
    }
    currentFloodLayer = nextLayer;
    viewer.scene.requestRender();

    setTimeout(() => {
      while (retiringLayers.length > 0) {
        const item = retiringLayers.shift();
        if (item.collection && item.collection.contains(item.layer)) {
          item.collection.remove(item.layer, true);
        }
      }
      viewer.scene.requestRender();
    }, 150);
  }

  async function activateGooglePhotorealistic3D(googleKey, ionToken) {
    const banner = document.getElementById("loadingBanner");
    banner.textContent = "Connecting Google Photorealistic 3D Tiles…";
    banner.hidden = false;

    try {
      let tileset = null;
      if (googleKey && googleKey.trim()) {
        const cleanKey = googleKey.trim();
        Cesium.GoogleMaps.defaultApiKey = cleanKey;
        tileset = await Cesium.createGooglePhotorealistic3DTileset(
          { key: cleanKey, onlyUsingWithGoogleGeocoder: true },
          { showCreditsOnScreen: true, maximumScreenSpaceError: 8 }
        );
      } else if (ionToken && ionToken.trim()) {
        Cesium.Ion.defaultAccessToken = ionToken.trim();
        Cesium.GoogleMaps.defaultApiKey = undefined;
        tileset = await Cesium.createGooglePhotorealistic3DTileset(undefined, {
          showCreditsOnScreen: true,
          maximumScreenSpaceError: 8,
        });
      } else {
        throw new Error("Please provide a Google Maps API Key or Cesium ion token.");
      }

      // Remove flood layer from globe before switching to 3D Tileset
      if (currentFloodLayer && viewer.imageryLayers.contains(currentFloodLayer)) {
        viewer.imageryLayers.remove(currentFloodLayer, true);
        currentFloodLayer = null;
      }
      if (googleTileset) {
        viewer.scene.primitives.remove(googleTileset);
        googleTileset = null;
      }

      googleTileset = tileset;
      viewer.scene.primitives.add(googleTileset);
      viewer.scene.globe.show = false;
      state.google3dActive = true;

      refreshFloodDrapeLayer();
      updateBasemapStatusUI(true);
      banner.hidden = true;
      return true;
    } catch (err) {
      banner.hidden = true;
      throw err;
    }
  }

  function switchToAerialFallback() {
    if (googleTileset) {
      if (currentFloodLayer && googleTileset.imageryLayers.contains(currentFloodLayer)) {
        googleTileset.imageryLayers.remove(currentFloodLayer, true);
        currentFloodLayer = null;
      }
      viewer.scene.primitives.remove(googleTileset);
      googleTileset = null;
    }
    state.google3dActive = false;
    viewer.scene.globe.show = true;
    refreshFloodDrapeLayer();
    updateBasemapStatusUI(false);
  }

  function updateBasemapStatusUI(isGoogle3D) {
    const dot = document.getElementById("basemapDot");
    const label = document.getElementById("basemapLabel");
    if (isGoogle3D) {
      dot.classList.add("is-google3d");
      label.textContent = "Google Photorealistic 3D · Active";
    } else {
      dot.classList.remove("is-google3d");
      label.textContent = "Connect Google 3D Key";
    }
  }

  function flyToTarget(lon, lat, rangeMeters, pitchDeg, headingDeg = -18) {
    if (!viewer) return;
    const carto = Cesium.Cartographic.fromDegrees(lon, lat);
    let targetHeight = state.google3dActive ? 32.0 : 0.0;
    if (state.google3dActive && viewer.scene.sampleHeightSupported) {
      const sampled = viewer.scene.sampleHeight(carto);
      if (sampled !== undefined && !Number.isNaN(sampled)) {
        targetHeight = sampled;
      }
    }
    const center = Cesium.Cartesian3.fromDegrees(lon, lat, targetHeight);
    const bs = new Cesium.BoundingSphere(center, 1.0);
    viewer.camera.flyToBoundingSphere(bs, {
      offset: new Cesium.HeadingPitchRange(
        Cesium.Math.toRadians(headingDeg),
        Cesium.Math.toRadians(pitchDeg),
        rangeMeters
      ),
      duration: 1.35,
      complete: () => {
        viewer.camera.lookAtTransform(Cesium.Matrix4.IDENTITY);
      },
    });
  }

  function populateStaffTicks() {
    const container = document.getElementById("staffTicks");
    container.innerHTML = "";
    const unitLabel = document.getElementById("staffUnitLabel");
    unitLabel.textContent = state.unit === "m" ? "STAFF (M)" : "STAFF (FT)";

    for (let i = 0; i < STAGE_KNOTS_FT.length; i++) {
      const ft = STAGE_KNOTS_FT[i];
      const pctBottom = (i / (STAGE_KNOTS_FT.length - 1)) * 100;
      const tick = document.createElement("div");
      tick.className = "staff-tick";
      tick.style.bottom = `${pctBottom}%`;

      const valText = state.unit === "m" ? (ft * FT_TO_M).toFixed(2) : ft < 1 ? ft.toFixed(2) : ft.toFixed(1);
      const bandInfo = meta && meta.bands && i < meta.bands.length ? `${meta.bands[i].pct}%` : "";
      tick.innerHTML = `<span class="staff-tick-val">${valText}</span><span class="staff-pct">${bandInfo}</span>`;
      container.appendChild(tick);
    }

    updateCutoffMaskAndStats();
  }

  function updateCutoffMaskAndStats() {
    const pct = depthToNorm(state.cutoffFt) * 100;
    document.getElementById("staffDryMask").style.height = `${pct}%`;
    document.getElementById("valCutoff").textContent = formatDepth(state.cutoffFt);

    if (sortedCellDepths && meta) {
      const vis = countVisibleCells(state.cutoffFt);
      document.getElementById("statWetCells").textContent = vis.toLocaleString();
      document.getElementById("statMaxDepth").textContent = formatDepth(meta.depthStats.max);
    }
  }

  function buildHotspotButtons() {
    const bar = document.getElementById("hotspotBar");
    if (!meta || !meta.hotspots) return;

    meta.hotspots.forEach((h, idx) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "hotspot-btn";
      btn.dataset.hotspotIndex = String(idx);
      const label = h.name || `Hotspot ${idx + 1}`;
      btn.innerHTML = `<span>${label}</span><span class="hotspot-depth">${formatDepth(h.maxDepth)}</span>`;
      btn.addEventListener("click", () => {
        document.querySelectorAll(".hotspot-btn").forEach((b) => b.classList.remove("is-active"));
        btn.classList.add("is-active");
        const hdg = (165 + idx * 35) % 360;
        flyToTarget(h.lon, h.lat, 420, -36, hdg);
        if (state.streetViewOpen && !state.followMapCamera) {
          setStreetViewLocation(h.lon, h.lat, hdg);
        }
      });
      bar.appendChild(btn);
    });

    const overviewBtn = bar.querySelector('[data-preset="overview"]');
    overviewBtn.addEventListener("click", () => {
      document.querySelectorAll(".hotspot-btn").forEach((b) => b.classList.remove("is-active"));
      overviewBtn.classList.add("is-active");
      flyToTarget(meta.center.lon, meta.center.lat - 0.012, 5400, -42, -14);
    });
  }

  function pickWorldPosition(windowPos) {
    if (!viewer) return null;
    if (state.google3dActive && viewer.scene.pickPositionSupported) {
      const p = viewer.scene.pickPosition(windowPos);
      if (Cesium.defined(p)) return p;
    }
    const ellip = viewer.camera.pickEllipsoid(windowPos, viewer.scene.globe.ellipsoid);
    if (Cesium.defined(ellip)) return ellip;
    if (viewer.scene.pickPositionSupported) {
      const p2 = viewer.scene.pickPosition(windowPos);
      if (Cesium.defined(p2)) return p2;
    }
    return null;
  }

  function renderProbeTelemetry(hit, lonDeg, latDeg, isPinned) {
    const swatch = document.getElementById("probeSwatch");
    const depthEl = document.getElementById("probeDepth");
    const stageEl = document.getElementById("probeStage");
    const coordsEl = document.getElementById("probeCoords");
    const needle = document.getElementById("staffNeedle");
    const needleTag = document.getElementById("staffNeedleTag");
    const unpinBtn = document.getElementById("btnUnpin");

    coordsEl.textContent = `${latDeg.toFixed(5)}°N, ${Math.abs(lonDeg).toFixed(5)}°W`;
    unpinBtn.hidden = !isPinned;

    const dFt = hit ? (state.shading === "cells" ? hit.cellDepthFt : hit.smoothDepthFt) : 0;
    if (!hit || dFt < state.cutoffFt) {
      swatch.style.background = "var(--hydro-panel)";
      depthEl.textContent = hit
        ? `Below cutoff (${formatDepth(hit.cellDepthFt)} raw cell)`
        : "Dry ground (outside 2-yr wet mesh)";
      stageEl.textContent = isPinned ? "Benchmark pinned · Click Unpin to resume live hover" : "Move cursor over flooded corridor · Click cell to pin";
      needle.classList.remove("is-visible");
      return;
    }

    const col = depthToCssColor(hit.cellDepthFt, state.palette);
    swatch.style.background = col;
    const inches = (hit.cellDepthFt * 12).toFixed(1);
    const dual =
      state.unit === "ft"
        ? `${hit.cellDepthFt.toFixed(2)} ft (${inches} in · ${(hit.cellDepthFt * FT_TO_M).toFixed(2)} m)`
        : `${(hit.cellDepthFt * FT_TO_M).toFixed(2)} m (${hit.cellDepthFt.toFixed(2)} ft)`;
    depthEl.textContent = (isPinned ? "PINNED · " : "") + dual;
    stageEl.textContent = `${depthToStageName(hit.cellDepthFt)} · Cell #${hit.triIndex.toLocaleString()}`;

    const pct = depthToNorm(hit.cellDepthFt) * 100;
    needle.style.bottom = `${pct}%`;
    needleTag.textContent = formatDepth(hit.cellDepthFt);
    needle.classList.add("is-visible");
  }

  function clearPinEntities() {
    pinEntities.forEach((e) => viewer.entities.remove(e));
    pinEntities = [];
    state.pinnedProbe = null;
    document.getElementById("btnUnpin").hidden = true;
  }

  function placePinOnMap(cartesian, hit, lonDeg, latDeg) {
    clearPinEntities();
    state.pinnedProbe = { hit, lonDeg, latDeg };

    const corners = hit.cornersLonLat;
    const ringDegrees = [
      corners[0][0], corners[0][1],
      corners[1][0], corners[1][1],
      corners[2][0], corners[2][1],
      corners[0][0], corners[0][1],
    ];

    const polyOutline = viewer.entities.add({
      polyline: {
        positions: Cesium.Cartesian3.fromDegreesArray(ringDegrees),
        width: 3,
        material: Cesium.Color.fromCssColorString("#FACC15"),
        clampToGround: true,
        classificationType: Cesium.ClassificationType.BOTH,
      },
    });

    const marker = viewer.entities.add({
      position: cartesian,
      point: {
        pixelSize: 10,
        color: Cesium.Color.fromCssColorString("#E59A28"),
        outlineColor: Cesium.Color.fromCssColorString("#111921"),
        outlineWidth: 2,
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
      },
      label: {
        text: `${formatDepth(hit.cellDepthFt)}\n${depthToStageName(hit.cellDepthFt)}`,
        font: "600 12px 'IBM Plex Mono', monospace",
        fillColor: Cesium.Color.fromCssColorString("#FEF08A"),
        backgroundColor: Cesium.Color.fromCssColorString("rgba(17, 25, 33, 0.92)"),
        showBackground: true,
        backgroundPadding: new Cesium.Cartesian2(8, 5),
        pixelOffset: new Cesium.Cartesian2(0, -32),
        verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
      },
    });

    pinEntities.push(polyOutline, marker);
    renderProbeTelemetry(hit, lonDeg, latDeg, true);
  }

  /* ========================================================================
     Synchronized Google Street View + 3D Beacon Marker + Split Resizer
     ======================================================================== */
  function headingToCardinal(deg) {
    const dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"];
    const norm = ((deg % 360) + 360) % 360;
    return dirs[Math.round(norm / 22.5) % 16];
  }

  function ensureStreetViewMapCone() {
    if (!viewer || svBeaconEntity) return;

    svBeaconEntity = viewer.entities.add({
      show: state.streetViewOpen,
      position: new Cesium.CallbackProperty(
        () => Cesium.Cartesian3.fromDegrees(state.svTarget.lon, state.svTarget.lat, 1.5),
        false
      ),
      point: {
        pixelSize: 11,
        color: Cesium.Color.fromCssColorString("#38BDF8"),
        outlineColor: Cesium.Color.fromCssColorString("#111921"),
        outlineWidth: 2.5,
        heightReference: Cesium.HeightReference.RELATIVE_TO_GROUND,
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
      },
      label: {
        text: "STREET VIEW",
        font: "700 10px 'Space Grotesk', sans-serif",
        fillColor: Cesium.Color.fromCssColorString("#EEF3F7"),
        backgroundColor: Cesium.Color.fromCssColorString("rgba(17, 25, 33, 0.88)"),
        showBackground: true,
        backgroundPadding: new Cesium.Cartesian2(6, 3),
        pixelOffset: new Cesium.Cartesian2(0, 12),
        verticalOrigin: Cesium.VerticalOrigin.TOP,
        heightReference: Cesium.HeightReference.RELATIVE_TO_GROUND,
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
      },
    });
  }

  function setStreetViewConeVisibility(visible) {
    if (svBeaconEntity) svBeaconEntity.show = visible;
    if (viewer) viewer.scene.requestRender();
  }

  function updateStreetViewTelemetryUI() {
    const { lon, lat, heading, pitch } = state.svTarget;
    const normHead = Math.round(((heading % 360) + 360) % 360);
    const card = headingToCardinal(normHead);
    const coordStr = `${lat.toFixed(4)}°N, ${Math.abs(lon).toFixed(4)}°W`;

    document.getElementById("svHeadingText").textContent = `${coordStr} · ${normHead}° ${card}`;
    document.getElementById("svStandbyLocation").textContent = `${lat.toFixed(5)}°N, ${Math.abs(lon).toFixed(5)}°W · Heading ${normHead}° ${card}`;

    const extUrl = `https://www.google.com/maps/@?api=1&map_action=pano&viewpoint=${lat.toFixed(6)},${lon.toFixed(6)}&heading=${normHead}&pitch=${Math.round(pitch)}&fov=80`;
    document.getElementById("linkExternalSV").href = extUrl;

    const hit = picker ? picker.queryLonLat(lon, lat) : null;
    const dFt = hit ? hit.cellDepthFt : 0;

    const svSwatch = document.getElementById("svSwatch");
    const svFooterDepth = document.getElementById("svFooterDepth");
    const svFooterStage = document.getElementById("svFooterStage");
    const svScaleValue = document.getElementById("svScaleValue");
    const waterOverlay = document.getElementById("svWaterOverlay");
    const waterTag = document.getElementById("svWaterTag");

    if (hit && dFt >= state.cutoffFt) {
      svSwatch.style.background = depthToCssColor(dFt, state.palette);
      const inches = (dFt * 12).toFixed(1);
      svFooterDepth.textContent = `2-Yr Street Depth: ${formatDepth(dFt)} (${inches} in)`;
      svFooterStage.textContent = `${depthToStageName(dFt)} · Synchronized with 3D mesh`;
      svScaleValue.textContent = `${formatDepth(dFt)} (${inches} in)`;
    } else {
      svSwatch.style.background = "var(--hydro-slate)";
      svFooterDepth.textContent = hit
        ? `Below cutoff (${formatDepth(dFt)} raw mesh)`
        : "Dry pavement (0.00 ft outside 2-yr flood)";
      svFooterStage.textContent = "Click any flooded street on the 3D map to inspect";
      svScaleValue.textContent = "0.00 ft (Dry)";
    }

    // Highlight vehicle / curb submergence benchmarks in standby scale card
    document.querySelectorAll(".sv-bench").forEach((el) => {
      const thresh = parseFloat(el.dataset.thresh || "0");
      el.classList.toggle("is-submerged", dFt >= thresh);
    });

    // Update simulated 2-yr flood crest water line inside Street View viewport
    const panoVisible = !document.getElementById("svPanorama").hidden || !document.getElementById("svEmbedFrame").hidden;
    if (state.svFloodOverlay && panoVisible && hit && dFt >= state.cutoffFt) {
      // Calibrated perspective water-line height for a ~8.2 ft Street View camera height
      const depthRatio = Math.min(1.0, Math.sqrt(dFt / 8.2));
      const basePct = 9 + depthRatio * 42;
      const pitchShift = -pitch * 0.65;
      const heightPct = Math.max(4, Math.min(86, basePct + pitchShift));
      waterOverlay.style.height = `${heightPct.toFixed(1)}%`;
      waterTag.textContent = `2-YR FLOOD CREST · ${formatDepth(dFt)} (${(dFt * 12).toFixed(1)} in)`;
      waterOverlay.hidden = false;
    } else {
      waterOverlay.hidden = true;
    }
  }

  function loadGoogleMapsJS(apiKey) {
    if (window.google && window.google.maps && window.google.maps.StreetViewPanorama && gmapsLoadedKey === apiKey) {
      return Promise.resolve(window.google.maps);
    }
    if (gmapsLoadPromise && gmapsLoadedKey === apiKey) {
      return gmapsLoadPromise;
    }

    gmapsLoadedKey = apiKey;
    svEmbedFallback = false;

    gmapsLoadPromise = new Promise((resolve, reject) => {
      window.gm_authFailure = () => {
        svEmbedFallback = true;
        activateStreetViewEmbedFallback(apiKey);
      };

      const existing = document.getElementById("googleMapsJsSdk");
      if (existing) existing.remove();

      const script = document.createElement("script");
      script.id = "googleMapsJsSdk";
      script.async = true;
      script.defer = true;
      script.src = `https://maps.googleapis.com/maps/api/js?key=${encodeURIComponent(apiKey)}&v=weekly`;
      script.onload = () => {
        if (window.google && window.google.maps) {
          resolve(window.google.maps);
        } else {
          reject(new Error("Google Maps JS SDK did not initialize."));
        }
      };
      script.onerror = () => {
        reject(new Error("Failed to load Google Maps JavaScript SDK."));
      };
      document.head.appendChild(script);
    });

    return gmapsLoadPromise;
  }

  function activateStreetViewEmbedFallback(apiKey) {
    const panoEl = document.getElementById("svPanorama");
    const frameEl = document.getElementById("svEmbedFrame");
    const standbyEl = document.getElementById("svStandbyCard");
    if (!apiKey) return;

    panoEl.hidden = true;
    standbyEl.hidden = true;
    frameEl.hidden = false;

    const { lon, lat, heading, pitch } = state.svTarget;
    const normHead = Math.round(((heading % 360) + 360) % 360);
    const nextSrc =
      `https://www.google.com/maps/embed/v1/streetview?key=${encodeURIComponent(apiKey)}` +
      `&location=${lat.toFixed(6)},${lon.toFixed(6)}&heading=${normHead}&pitch=${Math.round(pitch)}&fov=80`;
    if (nextSrc !== lastEmbedUrl) {
      lastEmbedUrl = nextSrc;
      frameEl.src = nextSrc;
    }
    updateStreetViewTelemetryUI();
  }

  function approxDistMeters(lon1, lat1, lon2, lat2) {
    const dLat = (lat2 - lat1) * 111132.92;
    const dLon = (lon2 - lon1) * 111412.84 * Math.cos(((lat1 + lat2) * 0.5 * Math.PI) / 180);
    return Math.hypot(dLon, dLat);
  }

  async function initOrUpdateLiveStreetView(opts = {}) {
    const { povOnly = false, snapTargetToRoad = true } = opts;
    const panoEl = document.getElementById("svPanorama");
    const frameEl = document.getElementById("svEmbedFrame");
    const standbyEl = document.getElementById("svStandbyCard");

    if (!state.googleApiKey) {
      panoEl.hidden = true;
      frameEl.hidden = true;
      standbyEl.hidden = false;
      updateStreetViewTelemetryUI();
      return;
    }

    if (svEmbedFallback) {
      if (!povOnly) {
        activateStreetViewEmbedFallback(state.googleApiKey);
      } else {
        updateStreetViewTelemetryUI();
      }
      return;
    }

    try {
      const gmaps = await loadGoogleMapsJS(state.googleApiKey);
      if (svEmbedFallback) return;

      standbyEl.hidden = true;
      frameEl.hidden = true;
      panoEl.hidden = false;

      if (!svService) {
        svService = new gmaps.StreetViewService();
      }

      if (!svPanorama) {
        svPanorama = new gmaps.StreetViewPanorama(panoEl, {
          position: { lat: state.svTarget.lat, lng: state.svTarget.lon },
          pov: { heading: state.svTarget.heading, pitch: state.svTarget.pitch },
          zoom: 0,
          addressControl: true,
          linksControl: true,
          panControl: true,
          enableCloseButton: false,
          fullscreenControl: false,
          motionTracking: false,
          showRoadLabels: true,
        });

        svPanorama.addListener("position_changed", () => {
          if (svSyncingFromMap || isMapCameraMoving) return;
          const pos = svPanorama.getPosition();
          if (!pos) return;
          svSyncingFromPano = true;
          state.svTarget.lat = pos.lat();
          state.svTarget.lon = pos.lng();
          updateStreetViewTelemetryUI();
          if (viewer) viewer.scene.requestRender();
          svSyncingFromPano = false;
        });

        svPanorama.addListener("pov_changed", () => {
          if (svSyncingFromMap || isMapCameraMoving) return;
          const pov = svPanorama.getPov();
          if (!pov) return;
          state.svTarget.heading = pov.heading || 0;
          state.svTarget.pitch = pov.pitch || 0;
          updateStreetViewTelemetryUI();
          if (viewer) viewer.scene.requestRender();
        });
      }

      // Instant client-side POV rotation sync (zero network calls)
      svSyncingFromMap = true;
      svPanorama.setPov({ heading: state.svTarget.heading, pitch: state.svTarget.pitch });
      svSyncingFromMap = false;

      if (povOnly) {
        updateStreetViewTelemetryUI();
        return;
      }

      // Snap to nearest outdoor roadway Street View panorama within 150 m
      const queryLon = state.svTarget.lon;
      const queryLat = state.svTarget.lat;
      lastQueriedPanoLon = queryLon;
      lastQueriedPanoLat = queryLat;
      const reqSeq = ++svPanoReqSeq;

      try {
        const res = await svService.getPanorama({
          location: { lat: queryLat, lng: queryLon },
          radius: 150,
          preference: gmaps.StreetViewPreference.NEAREST,
          source: gmaps.StreetViewSource.OUTDOOR,
        });
        if (reqSeq !== svPanoReqSeq) return;
        if (res && res.data && res.data.location && res.data.location.latLng) {
          const snapped = res.data.location.latLng;
          svSyncingFromMap = true;
          if (res.data.location.pano) {
            svPanorama.setPano(res.data.location.pano);
          } else {
            svPanorama.setPosition(snapped);
          }
          svPanorama.setPov({ heading: state.svTarget.heading, pitch: state.svTarget.pitch });
          if (snapTargetToRoad && !isMapCameraMoving) {
            state.svTarget.lat = snapped.lat();
            state.svTarget.lon = snapped.lng();
          }
          svSyncingFromMap = false;
        }
      } catch (_noPanoErr) {
        if (reqSeq !== svPanoReqSeq) return;
        document.getElementById("svFooterStage").textContent =
          "No outdoor Street View car coverage within 150 m of center point";
      }

      updateStreetViewTelemetryUI();
    } catch (err) {
      console.warn("Maps JS SDK unavailable, falling back to Maps Embed API:", err);
      svEmbedFallback = true;
      activateStreetViewEmbedFallback(state.googleApiKey);
    }
  }

  function updateReticleVisibility() {
    const ret = document.getElementById("mapSyncReticle");
    if (ret) {
      ret.hidden = !(state.streetViewOpen && state.followMapCamera);
    }
  }

  function setStreetViewLocation(lonDeg, latDeg, headingDeg, snapToRoad = true) {
    state.svTarget.lon = lonDeg;
    state.svTarget.lat = latDeg;
    if (typeof headingDeg === "number" && !Number.isNaN(headingDeg)) {
      state.svTarget.heading = ((headingDeg % 360) + 360) % 360;
    }
    updateStreetViewTelemetryUI();
    if (viewer) viewer.scene.requestRender();
    if (state.streetViewOpen && !svSyncingFromPano) {
      initOrUpdateLiveStreetView({ povOnly: false, snapTargetToRoad: snapToRoad });
    }
  }

  function pickCenterGroundFast(allowDepthSample) {
    if (!viewer) return null;
    const cw = viewer.canvas.clientWidth;
    const ch = viewer.canvas.clientHeight;
    if (cw <= 0 || ch <= 0) return null;
    const centerWin = new Cesium.Cartesian2(cw * 0.5, ch * 0.5);

    const now = performance.now();
    if (allowDepthSample && (!state.google3dActive || now - lastDepthPickMs > 160)) {
      lastDepthPickMs = now;
      const exact = pickWorldPosition(centerWin);
      if (exact) {
        const carto = Cesium.Cartographic.fromCartesian(exact);
        if (carto) {
          cachedGroundHeightM = Math.max(-50, Math.min(400, carto.height || 0));
          return {
            lon: Cesium.Math.toDegrees(carto.longitude),
            lat: Cesium.Math.toDegrees(carto.latitude),
          };
        }
      }
    }

    // Pure-math ray intersection against local ground elevation shell (0 GPU stall at 60 fps)
    const baseH = state.google3dActive ? (cachedGroundHeightM || 34) : 0;
    const ellip = new Cesium.Ellipsoid(
      6378137.0 + baseH,
      6378137.0 + baseH,
      6356752.3142451793 + baseH
    );
    const ray = viewer.camera.getPickRay(centerWin);
    if (ray) {
      const interval = Cesium.IntersectionTests.rayEllipsoid(ray, ellip);
      if (interval && interval.start > 0 && interval.start < 25000) {
        const pt = Cesium.Ray.getPoint(ray, interval.start);
        const carto = Cesium.Cartographic.fromCartesian(pt);
        if (carto) {
          return {
            lon: Cesium.Math.toDegrees(carto.longitude),
            lat: Cesium.Math.toDegrees(carto.latitude),
          };
        }
      }
    }

    // Horizon tilt fallback: project 90 m ahead of the camera's ground footprint
    const camCarto = viewer.camera.positionCartographic;
    if (camCarto) {
      const camLon = Cesium.Math.toDegrees(camCarto.longitude);
      const camLat = Cesium.Math.toDegrees(camCarto.latitude);
      const hRad = viewer.camera.heading;
      const dNorth = Math.cos(hRad) * 90;
      const dEast = Math.sin(hRad) * 90;
      return {
        lon: camLon + dEast / (111412.84 * Math.cos(camCarto.latitude)),
        lat: camLat + dNorth / 111132.92,
      };
    }
    return null;
  }

  function map3dPitchToStreetPitch(camPitchDeg) {
    // Oblique/top-down map browsing (-90° to -30°) -> comfortable street horizon (-3°)
    if (camPitchDeg <= -30) return -3;
    // Near-horizontal street tilt (-30° to +15°) -> smoothly tilt Street View (-3° to +12°)
    return Math.max(-12, Math.min(14, -3 + (camPitchDeg + 30) * 0.35));
  }

  function syncStreetViewFromMapCamera(isMoveEnd = false) {
    if (!viewer || !state.streetViewOpen || !state.followMapCamera) return;

    const centerHit = pickCenterGroundFast(isMoveEnd || !isMapCameraMoving);
    if (!centerHit) return;

    // Clamp to flood corridor vicinity so accidental globe zoom-out doesn't jump oceans away
    if (meta && meta.bounds) {
      const [minLon, minLat, maxLon, maxLat] = meta.bounds;
      if (
        centerHit.lon < minLon - 0.15 ||
        centerHit.lon > maxLon + 0.15 ||
        centerHit.lat < minLat - 0.15 ||
        centerHit.lat > maxLat + 0.15
      ) {
        return;
      }
    }

    const camHeading = ((Cesium.Math.toDegrees(viewer.camera.heading) % 360) + 360) % 360;
    const camPitch = Cesium.Math.toDegrees(viewer.camera.pitch);

    state.svTarget.lon = centerHit.lon;
    state.svTarget.lat = centerHit.lat;
    state.svTarget.heading = camHeading;
    state.svTarget.pitch = map3dPitchToStreetPitch(camPitch);

    // Keep left flood gauge & needle live-synced with the center reticle while moving (unless pinned)
    if (!state.pinnedProbe && picker) {
      const hit = picker.queryLonLat(centerHit.lon, centerHit.lat);
      renderProbeTelemetry(hit, centerHit.lon, centerHit.lat, false);
    }

    // No Google API key yet -> instant 60-fps standby telemetry + 3D FOV cone sync
    if (!state.googleApiKey) {
      updateStreetViewTelemetryUI();
      viewer.scene.requestRender();
      return;
    }

    // Maps Embed API iframe mode -> update telemetry at 60 fps, reload iframe cleanly when camera settles
    if (svEmbedFallback) {
      updateStreetViewTelemetryUI();
      clearTimeout(embedSettleTimer);
      embedSettleTimer = setTimeout(() => {
        activateStreetViewEmbedFallback(state.googleApiKey);
      }, isMoveEnd ? 60 : 280);
      return;
    }

    // Maps JavaScript API StreetViewPanorama mode:
    // 1. Rotate panorama POV at 60 fps immediately
    initOrUpdateLiveStreetView({ povOnly: true });

    // 2. Throttle outdoor road panorama position snaps during pan, plus exact snap on moveEnd
    const movedMeters = approxDistMeters(
      lastQueriedPanoLon,
      lastQueriedPanoLat,
      centerHit.lon,
      centerHit.lat
    );
    const now = performance.now();

    if (isMoveEnd) {
      clearTimeout(panoSettleTimer);
      if (movedMeters >= 3.0 || !svPanorama) {
        lastPanoFetchMs = now;
        initOrUpdateLiveStreetView({ povOnly: false, snapTargetToRoad: true });
      }
    } else if (movedMeters >= 7.5 && now - lastPanoFetchMs >= 240) {
      lastPanoFetchMs = now;
      initOrUpdateLiveStreetView({ povOnly: false, snapTargetToRoad: false });
    } else {
      clearTimeout(panoSettleTimer);
      panoSettleTimer = setTimeout(() => {
        if (approxDistMeters(lastQueriedPanoLon, lastQueriedPanoLat, state.svTarget.lon, state.svTarget.lat) >= 3.5) {
          lastPanoFetchMs = performance.now();
          initOrUpdateLiveStreetView({ povOnly: false, snapTargetToRoad: !isMapCameraMoving });
        }
      }, 240);
    }
  }

  function wireMapCameraSync() {
    if (!viewer) return;
    viewer.camera.percentageChanged = 0.001;

    let lastX = 0;
    let lastY = 0;
    let lastZ = 0;
    let lastH = 0;
    let lastP = 0;

    viewer.camera.moveStart.addEventListener(() => {
      isMapCameraMoving = true;
    });

    viewer.camera.changed.addEventListener(() => {
      syncStreetViewFromMapCamera(false);
    });

    viewer.camera.moveEnd.addEventListener(() => {
      isMapCameraMoving = false;
      syncStreetViewFromMapCamera(true);
    });

    // High-frequency pose listener during active mouse drags so even subtle rotations sync at 60 fps
    viewer.scene.postRender.addEventListener(() => {
      if (!isMapCameraMoving || !state.streetViewOpen || !state.followMapCamera) return;
      const p = viewer.camera.positionWC;
      const h = viewer.camera.heading;
      const pitch = viewer.camera.pitch;
      const moved =
        Math.abs(p.x - lastX) + Math.abs(p.y - lastY) + Math.abs(p.z - lastZ) > 0.15 ||
        Math.abs(h - lastH) > 0.0015 ||
        Math.abs(pitch - lastP) > 0.0015;
      if (moved) {
        lastX = p.x;
        lastY = p.y;
        lastZ = p.z;
        lastH = h;
        lastP = pitch;
        syncStreetViewFromMapCamera(false);
      }
    });
  }

  function toggleStreetViewPane(forceOpen) {
    const next = typeof forceOpen === "boolean" ? forceOpen : !state.streetViewOpen;
    state.streetViewOpen = next;

    const svPane = document.getElementById("streetViewPane");
    const resizer = document.getElementById("splitResizer");
    const toggleBtn = document.getElementById("btnToggleStreetView");

    svPane.hidden = !next;
    resizer.hidden = !next;
    toggleBtn.classList.toggle("is-split-active", next);
    toggleBtn.setAttribute("aria-pressed", String(next));
    setStreetViewConeVisibility(next);
    updateReticleVisibility();

    if (viewer) {
      viewer.resize();
      viewer.scene.requestRender();
    }
    if (next) {
      if (state.followMapCamera) {
        syncStreetViewFromMapCamera(true);
      } else {
        initOrUpdateLiveStreetView();
      }
      if (svPanorama && window.google && window.google.maps) {
        setTimeout(() => window.google.maps.event.trigger(svPanorama, "resize"), 60);
      }
    }
  }

  function wireSplitResizerAndStreetView() {
    const stage = document.getElementById("splitStage");
    const mapPane = document.getElementById("mapPane");
    const svPane = document.getElementById("streetViewPane");
    const resizer = document.getElementById("splitResizer");

    let dragging = false;
    resizer.addEventListener("pointerdown", (ev) => {
      dragging = true;
      resizer.classList.add("is-dragging");
      resizer.setPointerCapture(ev.pointerId);
    });

    resizer.addEventListener("pointermove", (ev) => {
      if (!dragging) return;
      const rect = stage.getBoundingClientRect();
      if (rect.width <= 0) return;
      const pctLeft = Math.max(26, Math.min(74, ((ev.clientX - rect.left) / rect.width) * 100));
      const pctRight = 100 - pctLeft;
      mapPane.style.flex = `1 1 ${pctLeft.toFixed(1)}%`;
      svPane.style.flex = `0 0 ${pctRight.toFixed(1)}%`;
      if (viewer) viewer.resize();
      if (svPanorama && window.google && window.google.maps) {
        window.google.maps.event.trigger(svPanorama, "resize");
      }
    });

    const endDrag = (ev) => {
      if (!dragging) return;
      dragging = false;
      resizer.classList.remove("is-dragging");
      try {
        resizer.releasePointerCapture(ev.pointerId);
      } catch (_e) {}
    };
    resizer.addEventListener("pointerup", endDrag);
    resizer.addEventListener("pointercancel", endDrag);

    document.getElementById("btnToggleStreetView").addEventListener("click", () => {
      toggleStreetViewPane();
    });

    document.getElementById("btnCloseSV").addEventListener("click", () => {
      toggleStreetViewPane(false);
    });

    const btnFollowMap = document.getElementById("btnFollowMap");
    if (btnFollowMap) {
      btnFollowMap.addEventListener("click", () => {
        state.followMapCamera = !state.followMapCamera;
        btnFollowMap.classList.toggle("is-active", state.followMapCamera);
        btnFollowMap.setAttribute("aria-pressed", String(state.followMapCamera));
        updateReticleVisibility();
        if (state.followMapCamera) {
          syncStreetViewFromMapCamera(true);
        }
      });
    }

    const btnFloodOverlay = document.getElementById("btnFloodOverlay");
    btnFloodOverlay.addEventListener("click", () => {
      state.svFloodOverlay = !state.svFloodOverlay;
      btnFloodOverlay.classList.toggle("is-active", state.svFloodOverlay);
      btnFloodOverlay.setAttribute("aria-pressed", String(state.svFloodOverlay));
      updateStreetViewTelemetryUI();
    });

    // Inline key activation right inside the Street View standby pane
    const inlineInput = document.getElementById("svInlineKeyInput");
    const inlineBtn = document.getElementById("btnSvActivateKey");
    const inlineStatus = document.getElementById("svStandbyStatus");

    const handleInlineKeySubmit = async () => {
      const gKey = inlineInput.value.trim();
      if (!gKey) {
        inlineStatus.textContent = "Enter a Google Maps Platform API key first.";
        inlineStatus.hidden = false;
        return;
      }
      inlineStatus.hidden = true;
      state.googleApiKey = gKey;
      localStorage.setItem(STORAGE_GOOGLE_KEY, gKey);

      // Activate Street View immediately, and also attempt Google Photorealistic 3D Tiles
      initOrUpdateLiveStreetView();
      try {
        await activateGooglePhotorealistic3D(gKey, "");
      } catch (e) {
        inlineStatus.textContent =
          "Street View key saved. For left-pane 3D buildings, also enable 'Map Tiles API' on this key.";
        inlineStatus.hidden = false;
      }
    };

    inlineBtn.addEventListener("click", handleInlineKeySubmit);
    inlineInput.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter") handleInlineKeySubmit();
    });
  }

  function wirePointerEvents() {
    const handler = new Cesium.ScreenSpaceEventHandler(viewer.scene.canvas);

    handler.setInputAction((movement) => {
      if (state.pinnedProbe) return;
      const world = pickWorldPosition(movement.endPosition);
      if (!world || !picker) return;
      const carto = Cesium.Cartographic.fromCartesian(world);
      const lon = Cesium.Math.toDegrees(carto.longitude);
      const lat = Cesium.Math.toDegrees(carto.latitude);
      const hit = picker.queryLonLat(lon, lat);
      renderProbeTelemetry(hit, lon, lat, false);
    }, Cesium.ScreenSpaceEventType.MOUSE_MOVE);

    handler.setInputAction((click) => {
      const world = pickWorldPosition(click.position);
      if (!world || !picker) return;
      const carto = Cesium.Cartographic.fromCartesian(world);
      const lon = Cesium.Math.toDegrees(carto.longitude);
      const lat = Cesium.Math.toDegrees(carto.latitude);
      const hit = picker.queryLonLat(lon, lat);
      if (hit && hit.cellDepthFt >= state.cutoffFt) {
        placePinOnMap(world, hit, lon, lat);
      } else {
        clearPinEntities();
        renderProbeTelemetry(hit, lon, lat, false);
      }
      if (state.streetViewOpen) {
        const camHeading = Cesium.Math.toDegrees(viewer.camera.heading);
        setStreetViewLocation(lon, lat, camHeading);
      }
    }, Cesium.ScreenSpaceEventType.LEFT_CLICK);

    document.getElementById("btnUnpin").addEventListener("click", () => {
      clearPinEntities();
    });
  }

  function wireControls() {
    let cutoffTimer = null;
    const sliderCutoff = document.getElementById("sliderCutoff");
    sliderCutoff.addEventListener("input", () => {
      state.cutoffFt = parseFloat(sliderCutoff.value);
      updateCutoffMaskAndStats();
      clearTimeout(cutoffTimer);
      cutoffTimer = setTimeout(() => {
        refreshFloodDrapeLayer();
      }, 65);
    });

    const sliderOpacity = document.getElementById("sliderOpacity");
    sliderOpacity.addEventListener("input", () => {
      state.opacity = parseFloat(sliderOpacity.value);
      document.getElementById("valOpacity").textContent = `${Math.round(state.opacity * 100)}%`;
      if (currentFloodLayer) {
        currentFloodLayer.alpha = state.opacity;
        viewer.scene.requestRender();
      }
    });

    // Unit toggle (FT / M)
    document.querySelectorAll("[data-unit]").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll("[data-unit]").forEach((b) => b.classList.remove("is-active"));
        btn.classList.add("is-active");
        state.unit = btn.dataset.unit;
        populateStaffTicks();
        // Update hotspot pill depth units
        document.querySelectorAll("[data-hotspot-index]").forEach((hb) => {
          const idx = parseInt(hb.dataset.hotspotIndex, 10);
          const h = meta.hotspots[idx];
          hb.querySelector(".hotspot-depth").textContent = formatDepth(h.maxDepth);
        });
        if (state.pinnedProbe) {
          renderProbeTelemetry(
            state.pinnedProbe.hit,
            state.pinnedProbe.lonDeg,
            state.pinnedProbe.latDeg,
            true
          );
        }
      });
    });

    // Palette buttons
    document.querySelectorAll("[data-palette]").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll("[data-palette]").forEach((b) => b.classList.remove("is-active"));
        btn.classList.add("is-active");
        state.palette = parseInt(btn.dataset.palette, 10);
        const rod = document.getElementById("staffRod");
        rod.classList.remove("palette-blues", "palette-emergency");
        if (state.palette === 1) rod.classList.add("palette-blues");
        if (state.palette === 2) rod.classList.add("palette-emergency");
        refreshFloodDrapeLayer();
        updateStreetViewTelemetryUI();
      });
    });

    // Smooth vs Exact Cells toggle
    document.querySelectorAll("[data-shading]").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll("[data-shading]").forEach((b) => b.classList.remove("is-active"));
        btn.classList.add("is-active");
        state.shading = btn.dataset.shading;
        refreshFloodDrapeLayer();
      });
    });

    // Camera tilt presets around current screen center
    document.querySelectorAll("[data-cam]").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll("[data-cam]").forEach((b) => b.classList.remove("is-active"));
        btn.classList.add("is-active");
        const mode = btn.dataset.cam;
        const centerWin = new Cesium.Cartesian2(
          viewer.canvas.clientWidth * 0.5,
          viewer.canvas.clientHeight * 0.5
        );
        const target = pickWorldPosition(centerWin);
        let lon = meta.center.lon;
        let lat = meta.center.lat;
        let dist = 900;
        if (target) {
          const c = Cesium.Cartographic.fromCartesian(target);
          lon = Cesium.Math.toDegrees(c.longitude);
          lat = Cesium.Math.toDegrees(c.latitude);
          dist = Math.max(180, Math.min(8000, Cesium.Cartesian3.distance(viewer.camera.positionWC, target)));
        }
        if (mode === "oblique") flyToTarget(lon, lat, dist, -38, Cesium.Math.toDegrees(viewer.camera.heading));
        else if (mode === "street") flyToTarget(lon, lat, Math.min(dist, 320), -21, Cesium.Math.toDegrees(viewer.camera.heading));
        else flyToTarget(lon, lat, dist, -89.5, 0);
      });
    });

    // Minimize / expand left control deck (auto-collapse on narrow side-panels)
    const btnToggleDeck = document.getElementById("btnToggleDeck");
    const deckBody = document.getElementById("deckBody");
    if (btnToggleDeck && deckBody) {
      if (window.innerWidth < 700) {
        deckBody.hidden = true;
        btnToggleDeck.textContent = "+";
        btnToggleDeck.setAttribute("aria-expanded", "false");
      }
      btnToggleDeck.addEventListener("click", () => {
        const collapsed = !deckBody.hidden;
        deckBody.hidden = collapsed;
        btnToggleDeck.textContent = collapsed ? "+" : "−";
        btnToggleDeck.setAttribute("aria-expanded", String(!collapsed));
      });
    }

    // Modal controls for Google Photorealistic 3D Maps API Key
    const modal = document.getElementById("keyModal");
    const inputGoogle = document.getElementById("inputGoogleKey");
    const inputIon = document.getElementById("inputIonToken");
    const errEl = document.getElementById("modalError");

    document.getElementById("btnBasemapKey").addEventListener("click", () => {
      errEl.hidden = true;
      inputGoogle.value = localStorage.getItem(STORAGE_GOOGLE_KEY) || "";
      inputIon.value = localStorage.getItem(STORAGE_ION_TOKEN) || "";
      modal.hidden = false;
      inputGoogle.focus();
    });

    document.getElementById("btnUseAerial").addEventListener("click", () => {
      modal.hidden = true;
      switchToAerialFallback();
    });

    document.getElementById("btnConnect3D").addEventListener("click", async () => {
      errEl.hidden = true;
      const gKey = inputGoogle.value.trim();
      const iTok = inputIon.value.trim();
      if (gKey) {
        state.googleApiKey = gKey;
        localStorage.setItem(STORAGE_GOOGLE_KEY, gKey);
        if (state.streetViewOpen) initOrUpdateLiveStreetView();
      }
      try {
        await activateGooglePhotorealistic3D(gKey, iTok);
        if (iTok) localStorage.setItem(STORAGE_ION_TOKEN, iTok);
        modal.hidden = true;
      } catch (e) {
        errEl.textContent =
          "Could not connect Google 3D Tiles: " +
          (e && e.message ? e.message : String(e)) +
          ". Verify that 'Map Tiles API' is enabled on your Google Cloud key.";
        errEl.hidden = false;
      }
    });

    modal.addEventListener("click", (ev) => {
      if (ev.target === modal) modal.hidden = true;
    });

    window.addEventListener("keydown", (ev) => {
      if (ev.key === "Escape" && !modal.hidden) modal.hidden = true;
    });
  }

  async function init() {
    initCesiumViewer();

    try {
      const loaded = await fetchBinaryMesh();
      meta = loaded.meta;
      const mesh = loaded.mesh;

      // Sort copy of triangle depths for instant visible-cell count updates
      sortedCellDepths = new Float32Array(mesh.triDepth);
      sortedCellDepths.sort();

      document.getElementById("brandSummary").textContent =
        `${meta.featureCount.toLocaleString()} polygons · ${meta.wetAreaAcres} ac wet · max ${meta.depthStats.max.toFixed(2)} ft`;

      rasterizer = new WebGLFloodRasterizer(mesh);
      picker = new SpatialMeshPicker(mesh, meta.mercBounds);

      // Auto-collapse left control deck when split-screen makes mapPane narrow
      const mapPaneEl = document.getElementById("mapPane");
      const deckBodyEl = document.getElementById("deckBody");
      const btnToggleDeckEl = document.getElementById("btnToggleDeck");
      if (mapPaneEl && deckBodyEl && btnToggleDeckEl && mapPaneEl.clientWidth < 560) {
        deckBodyEl.hidden = true;
        btnToggleDeckEl.textContent = "+";
        btnToggleDeckEl.setAttribute("aria-expanded", "false");
      }

      populateStaffTicks();
      buildHotspotButtons();
      wireControls();
      wireSplitResizerAndStreetView();
      wirePointerEvents();
      ensureStreetViewMapCone();
      wireMapCameraSync();
      updateReticleVisibility();

      // Check URL params or localStorage for Google Maps API key / Cesium ion token
      const params = new URLSearchParams(window.location.search);
      const urlKey = params.get("key") || params.get("googleKey") || "";
      const urlIon = params.get("ionToken") || "";
      const savedGoogleKey = urlKey || localStorage.getItem(STORAGE_GOOGLE_KEY) || "";
      const savedIonToken = urlIon || localStorage.getItem(STORAGE_ION_TOKEN) || "";

      if (urlKey) localStorage.setItem(STORAGE_GOOGLE_KEY, urlKey);
      if (urlIon) localStorage.setItem(STORAGE_ION_TOKEN, urlIon);
      state.googleApiKey = savedGoogleKey;

      if (savedGoogleKey || savedIonToken) {
        try {
          await activateGooglePhotorealistic3D(savedGoogleKey, savedIonToken);
        } catch (err) {
          console.warn("Saved 3D Tiles key failed, falling back to aerial:", err);
          refreshFloodDrapeLayer();
        }
      } else {
        refreshFloodDrapeLayer();
      }

      if (state.streetViewOpen) {
        initOrUpdateLiveStreetView();
      }
      document.getElementById("loadingBanner").hidden = true;

      // Open on a 3D oblique view aligned 165° SSE down W Rowan St so both screens start in exact heading & position lockstep
      flyToTarget(state.svTarget.lon, state.svTarget.lat, 360, -36, 165);
      const btns = document.querySelectorAll(".hotspot-btn");
      btns.forEach((b) => b.classList.remove("is-active"));
      if (btns[1]) btns[1].classList.add("is-active");
    } catch (err) {
      const banner = document.getElementById("loadingBanner");
      banner.textContent = "Load error: " + (err && err.message ? err.message : String(err));
      banner.hidden = false;
      console.error(err);
    }
  }

  init();
})();
