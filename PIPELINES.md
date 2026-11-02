# Fish Monitoring Pipelines

This guide shows how to run the project’s image, video, and single-fish classification workflows. Run commands from the repository root so paths such as `src/detection/...` resolve correctly.

## Choose a workflow

| Workflow | Input | What it produces |
|---|---|---|
| [Image pipeline](#image-pipeline) | Image containing multiple fish | Fish and species counts, density estimate, heat map, integrated report, and GIS map |
| [Video pipeline](#video-pipeline) | Video containing fish | Tracking results, density estimate, integrated report, and GIS map |
| [Classification only](#classification-only) | Image of one fish | Top five species predictions with confidence percentages |

Replace example paths and values with the ones for your run. The values shown below are examples; use measurements and location details appropriate to your pond.

## Image pipeline

Use this workflow to analyze one image of a pond or other area containing multiple fish.

### 1. Detect fish and identify species

```bash
python src/detection/count_fish.py --image "path/to/image.jpg"
```

**Output:** species counts and total fish detected.

### 2. Estimate fish density

```bash
python src/detection/density.py \
  --image "path/to/image.jpg" \
  --depth 2 \
  --fov 90 \
  --pond_area 2000
```

**Output:** fish per square meter and estimated total fish in the pond.

### 3. Generate a heat map

```bash
python src/detection/heatmap.py --image "path/to/image.jpg"
```

**Output:** `models/heatmap/heatmap_<image>.jpg`.

### 4. Generate an integrated report

```bash
python src/integrated_report.py \
  --image "path/to/image.jpg" \
  --turbidity 3.9 \
  --hardness 196 \
  --tds 500 \
  --depth 2 \
  --fov 90 \
  --pond_area 2000
```

**Output:** fish count, density, and water quality report.

### 5. Create a GIS map

Use the fish count, density, and water quality values for this image analysis.

```bash
python src/gis_map.py \
  --lat 23.4120 \
  --lon 85.3094 \
  --location "Kanke Dam, Ranchi" \
  --total 27 \
  --density 2.25 \
  --ph 7.59 \
  --water_safe 1
```

**Output:** `models/gis/fish_gis_map.html`.

To view the map in a browser, run:

```bash
cd models/gis
python -m http.server 8080
```

Then open `http://localhost:8080`.

## Video pipeline

Use this workflow to track fish in a video. For density estimation and the integrated report, choose a representative frame from the video and use its image path in both commands.

### 1. Track fish

```bash
python src/detection/track_fish.py --video "path/to/video.mp4"
```

**Output:** unique fish count, average fish per frame, and a tracked video.

### 2. Estimate density from a representative frame

```bash
python src/detection/density.py \
  --image "path/to/frame.jpg" \
  --depth 2 \
  --fov 90 \
  --pond_area 2000
```

**Output:** fish per square meter and estimated total fish in the pond.

### 3. Generate an integrated report

```bash
python src/integrated_report.py \
  --image "path/to/frame.jpg" \
  --turbidity 3.9 \
  --hardness 196 \
  --tds 500 \
  --depth 2 \
  --fov 90 \
  --pond_area 2000
```

**Output:** fish count, density, and water quality report for the representative frame.

### 4. Create a GIS map

Use the total fish and density values from the video analysis, along with the pond’s location and water quality values.

```bash
python src/gis_map.py \
  --lat 23.4120 \
  --lon 85.3094 \
  --location "Pond name" \
  --total 368 \
  --density 1.76 \
  --ph 7.59 \
  --water_safe 1
```

**Output:** `models/gis/fish_gis_map.html`.

To view the map in a browser, run:

```bash
cd models/gis
python -m http.server 8080
```

Then open `http://localhost:8080`.

## Classification only

Use this workflow to classify one fish without running detection, density estimation, or reporting.

```bash
python src/classification/predict.py --image "path/to/fish.jpg"
```

**Output:** the top five species predictions with confidence percentages.

## Parameters reference

Values in the **Default** column are the documented defaults. A value marked **required** must be supplied for that command.

| Parameter | Description | Default |
|---|---|---:|
| `--image` | Path to the input image | Required for image-based commands |
| `--video` | Path to the input video | Required for video tracking |
| `--depth` | Camera depth, in meters | `2.0` |
| `--fov` | Camera field of view, in degrees | `90.0` |
| `--pond_area` | Total pond area, in m² | `2000.0` |
| `--turbidity` | Water turbidity, in NTU | `3.9` |
| `--hardness` | Water hardness, in mg/L | `196.0` |
| `--tds` | Total dissolved solids, in mg/L | `500.0` |
| `--lat` | Location latitude | `23.4120` |
| `--lon` | Location longitude | `85.3094` |
| `--location` | Human-readable location name | `Kanke Dam, Ranchi` |
| `--total` | Total fish detected | `27` |
| `--density` | Fish density, in fish/m² | `2.25` |
| `--ph` | Predicted water pH | `7.59` |
| `--water_safe` | Water safety (`1` = safe, `0` = unsafe) | `1` |

## Output files

| Result | Output location |
|---|---|
| Species classification | Terminal output |
| Fish detection | `models/combined/<image>` |
| Heat map | `models/heatmap/heatmap_<image>` |
| Tracking video | `models/tracking/tracked_<video>` |
| GIS map | `models/gis/fish_gis_map.html` |
| Integrated report | Terminal output |

## Sample results

These are example results from previous runs.

### Image pipeline: `frame1008`

| Metric | Result |
|---|---:|
| Fish detected | 27 |
| Species | Glass Perchlet: 4; Goby: 2; Gold Fish: 1; Unknown: 20 |
| Camera area | 12.00 m² |
| Areal density | 2.2500 fish/m² |
| Estimated total fish | 4,500 in pond |
| Water pH | 7.59 |
| Water safety | Safe ✅ |

### Video pipeline: `video_test.mp4`

| Metric | Result |
|---|---:|
| Frames processed | 454 |
| Average fish per frame | 21.1 |
| Maximum fish per frame | 31 |
| Unique fish | 368 |
| Areal density | 1.76 fish/m² |
| Water pH | 7.59 |
| Water safety | Safe ✅ |
