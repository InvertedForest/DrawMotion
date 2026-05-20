# DrawMotion Studio

Public alpha web demo for DrawMotion. This release exposes text plus trajectory control only.
It does not require downloading the HumanML3D or KIT-ML datasets; the backend creates
an empty motion template and applies the text prompt and browser trajectory at generation time.
Stickman conditioning is intentionally disabled because the current IFG path optimizes trajectory
alignment and can overpower inconsistent stickman guidance.

Run on `#compshare` from the repository root:

```bash
DRAWMOTION_CKPT=logs/human_ml3d/last.ckpt \
DRAWMOTION_GPU=0 \
/usr/local/miniconda3/bin/conda run -n mogen \
uvicorn demo.drawmotion_studio.app:app --host 0.0.0.0 --port 12008
```

If `DRAWMOTION_CKPT` is not set, the backend uses
`logs/human_ml3d/last.ckpt`. The default HumanML3D demo also
expects `mid_feat/t2m/mid_feat.pt` and `stickman/weight/real_init/t2m/stickman_encoder.ckpt`.

Open `http://<server>:12008`. The current public tunnel maps Aliyun port `11028`
to `127.0.0.1:12008`.

## API

- `GET /api/status`: reports whether the model has been loaded.
- `POST /api/generate`: accepts `text`, `trajectory`, `length`, `density`,
  `trajectory_scale`, `ifg_repeat`, and `ifg_scale`.

The backend validates public inputs, serializes generation through a single lock,
and ignores any `stickmen` field in the request.

## Parameters

- `text`: global motion semantics.
- `trajectory`: browser-drawn pelvis/root path.
- `length`: generated frame count, from `2` to `196`.
- `density`: UI `Alpha`; `0` is uniform by arc length, larger values preserve more drawing speed.
- `ifg_repeat`: IFG refinement steps.
- `ifg_scale`: IFG guidance strength.

## Run Artifacts

Each generation writes artifacts under `demo/drawmotion_studio/runs/<timestamp>/`.
The server keeps the newest 100 run directories.

- `request.json`: validated frontend payload.
- `poly_traj.npy`: processed trajectory.
- `draw_input.npz`: processed trajectory archive.
- `all_infor.pkl`: internal Blender visualization payload.
- `result.json`: browser-friendly skeleton and trajectory data.
