# Fayetteville 2-Yr Flood Inundation · Google Photorealistic 3D Maps + Street View

Interactive 3D flood-depth visualization draping 205,297 2D stormwater mesh polygons (`2yr.json`, `DEPTH2D` up to `10.50 ft`) directly onto **Google Photorealistic 3D Tiles** in Fayetteville, NC, paired with a synchronized side-by-side **Google Street View** inspector.

## Live Demo (GitHub Pages)

**[https://qiaoxin136.github.io/fayetteville-3d-flood-viewer/](https://qiaoxin136.github.io/fayetteville-3d-flood-viewer/)**

## Features

- **GPU-Accelerated Ground Draping**: Preprocesses `2yr.json` (51.2 MB, 205,297 polygons) into a compact `4.71 MB` indexed binary mesh (`flood_mesh.bin`, 213,075 triangles) rendered dynamically up to zoom 22 via a custom WebGL2 `Cesium.ImageryProvider`.
- **Google Photorealistic 3D Tiles**: Connects via your Google Maps Platform API key (`Map Tiles API`) or Cesium ion token, with automatic Esri World Imagery fallback before a key is entered.
- **Graduated Survey Staff Gauge (`DEPTH2D`)**: Live cursor depth probing (`<0.1 ms` uniform spatial-grid triangle lookup), interactive dry-ground cutoff slider, opacity control, and 3 hydrologic color ramps.
- **Side-by-Side Synchronized Google Street View**: Closed by default; click **Street View** in the top bar to open a resizable right-side pane that continuously tracks 3D map movement and rotation (`Sync Move`), projects a simulated 2-yr flood crest water line, and highlights curb/tire/door/hood submergence benchmarks.

## Local Usage

```bat
start_server.bat
```

Then open `http://127.0.0.1:8765/`.

To regenerate `flood_mesh.bin` and `flood_meta.json` from `2yr.json`:

```bash
python preprocess_mesh.py
```
