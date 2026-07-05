"""
Manual guide-line annotator for CQ500 train MPR PNGs.

For each case you place 2 clicks on each of 3 projections (6 clicks total):
  axial | coronal | sagittal  ->  one guide segment (yellow line) per panel.

Controls:
  left click     add point on active panel (2 points -> line)
  u              undo last point (current case)
  r              reset points on screen (does not delete saved JSON)
  d / Delete     delete saved JSON for current case + clear points (re-annotate)
  s / Enter      save JSON and go to next case
  w              save JSON, stay on current case
  p / n / ← / →  previous / next case (loads saved annotation from disk)
  q              quit

Bottom buttons: Prev | Next | Save+Next | Delete all | Reset

Output JSON per case: data/cq500_train/cq500_train_guides/{case_id}.json

Usage:
    python -u scripts/guide_line_annotator.py              # all cases, browse + annotate
    python -u scripts/guide_line_annotator.py --only-missing
    python -u scripts/guide_line_annotator.py --start-case CQ500CT100
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import AxesImage
from matplotlib.widgets import Button

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paths import TRAIN_GUIDES_DIR, TRAIN_MPR_DIR, TRAIN_MPR_MANIFEST

DEFAULT_IMAGES = TRAIN_MPR_DIR
DEFAULT_OUT = TRAIN_GUIDES_DIR
PLANES = ("axial", "coronal", "sagittal")


def _disable_mpl_default_keys() -> None:
    """Matplotlib binds ``s`` to save-figure dialog — breaks annotator shortcuts."""
    for name in ("save", "quit", "fullscreen", "home", "back", "forward"):
        plt.rcParams[f"keymap.{name}"] = []


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Annotate 3 guide segments per CQ500 case")
    p.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--manifest", type=Path, default=TRAIN_MPR_MANIFEST)
    p.add_argument("--start-case", type=str, default=None)
    p.add_argument("--only-missing", action="store_true", help="Skip cases with existing JSON")
    return p.parse_args()


def load_manifest(path: Path) -> dict:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def case_ids_from_images(images_dir: Path) -> list[str]:
    ids: set[str] = set()
    for p in images_dir.glob("*_axial.png"):
        ids.add(p.name.replace("_axial.png", ""))
    return sorted(ids)


def annotation_path(out_dir: Path, case_id: str) -> Path:
    return out_dir / f"{case_id}.json"


def load_annotation(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_annotation(
    out_dir: Path,
    case_id: str,
    *,
    images_dir: Path,
    points: dict[str, list[list[float]]],
    manifest: dict,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    planes_out: dict[str, dict] = {}
    case_meta = manifest.get("cases", {}).get(case_id, {})
    for plane in PLANES:
        pts = points.get(plane, [])
        entry: dict = {
            "png": f"{case_id}_{plane}.png",
            "p0": pts[0] if len(pts) > 0 else None,
            "p1": pts[1] if len(pts) > 1 else None,
        }
        if plane in case_meta.get("planes", {}):
            entry.update({k: v for k, v in case_meta["planes"][plane].items() if k != "png"})
        planes_out[plane] = entry

    payload = {
        "case_id": case_id,
        "images_dir": str(images_dir),
        "spacing_mm": manifest.get("spacing_mm", 1.0),
        "planes": planes_out,
        "complete": all(len(points.get(p, [])) == 2 for p in PLANES),
        "annotated_at": datetime.now(timezone.utc).isoformat(),
    }
    path = annotation_path(out_dir, case_id)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


class GuideAnnotator:
    def __init__(
        self,
        case_ids: list[str],
        images_dir: Path,
        out_dir: Path,
        manifest: dict,
    ) -> None:
        self.case_ids = case_ids
        self.images_dir = images_dir
        self.out_dir = out_dir
        self.manifest = manifest
        self.idx = 0
        self.fig: plt.Figure | None = None
        self.axes: list[plt.Axes] = []
        self.images: list[AxesImage] = []
        self.points: dict[str, list[list[float]]] = {p: [] for p in PLANES}
        self.markers: dict[str, list] = {p: [] for p in PLANES}
        self.lines: dict[str, object | None] = {p: None for p in PLANES}
        self.dirty = False
        self.status_text = None
        self._button_axes: set[plt.Axes] = set()

    def _current_case_id(self) -> str:
        return self.case_ids[self.idx]

    def _has_saved_annotation(self, case_id: str | None = None) -> bool:
        return annotation_path(self.out_dir, case_id or self._current_case_id()).is_file()

    def _saved_is_complete(self, case_id: str | None = None) -> bool:
        ann = load_annotation(annotation_path(self.out_dir, case_id or self._current_case_id()))
        return bool(ann and ann.get("complete"))

    def _annotation_label(self) -> str:
        case_id = self._current_case_id()
        if self.dirty:
            return "unsaved edits"
        if self._saved_is_complete(case_id):
            return "saved complete"
        if self._has_saved_annotation(case_id):
            return "saved partial"
        return "no annotation"

    @staticmethod
    def _safe_remove_artist(artist) -> None:
        if artist is None:
            return
        try:
            artist.remove()
        except (NotImplementedError, ValueError):
            pass

    def _forget_overlays(self) -> None:
        """Drop overlay refs without remove() — axes may already be cleared."""
        self.markers = {p: [] for p in PLANES}
        self.lines = {p: None for p in PLANES}

    def _plane_png(self, case_id: str, plane: str) -> Path:
        return self.images_dir / f"{case_id}_{plane}.png"

    def _load_points_from_json(self, case_id: str) -> None:
        ann = load_annotation(annotation_path(self.out_dir, case_id))
        self.points = {p: [] for p in PLANES}
        if ann is None:
            return
        for plane in PLANES:
            pl = ann.get("planes", {}).get(plane, {})
            if pl.get("p0") is not None and pl.get("p1") is not None:
                self.points[plane] = [list(pl["p0"]), list(pl["p1"])]

    def _clear_drawings(self) -> None:
        for plane in PLANES:
            for m in self.markers[plane]:
                self._safe_remove_artist(m)
            if self.lines[plane] is not None:
                self._safe_remove_artist(self.lines[plane])
        self._forget_overlays()

    def _redraw_plane(self, plane: str) -> None:
        ax = self.axes[PLANES.index(plane)]
        pts = self.points[plane]
        for m in self.markers[plane]:
            self._safe_remove_artist(m)
        self.markers[plane] = []
        if self.lines[plane] is not None:
            self._safe_remove_artist(self.lines[plane])
            self.lines[plane] = None
        for pt in pts:
            (m,) = ax.plot(pt[0], pt[1], "o", color="#ffe66d", markersize=6, markeredgecolor="k")
            self.markers[plane].append(m)
        if len(pts) == 2:
            (ln,) = ax.plot(
                [pts[0][0], pts[1][0]],
                [pts[0][1], pts[1][1]],
                color="#ffe66d",
                linewidth=2.0,
            )
            self.lines[plane] = ln

    def _update_status(self) -> None:
        case_id = self._current_case_id()
        done = sum(1 for p in PLANES if len(self.points[p]) == 2)
        self.status_text.set_text(
            f"{case_id}  [{self.idx + 1}/{len(self.case_ids)}]  "
            f"guides {done}/3  |  {self._annotation_label()}  |  "
            f"click x2/panel  s=save+next  d=delete all  ←/→ or p/n browse"
        )
        self.fig.canvas.draw_idle()

    def _show_case(self) -> None:
        case_id = self.case_ids[self.idx]
        for ax, plane in zip(self.axes, PLANES):
            png = self._plane_png(case_id, plane)
            if not png.is_file():
                raise FileNotFoundError(png)
            img = plt.imread(str(png))
            ax.clear()
            ax.imshow(img, cmap="gray", origin="upper", aspect="equal")
            ax.axis("off")
            ax.set_title(plane, fontsize=10, color="#cccccc")

        self._load_points_from_json(case_id)
        self._forget_overlays()
        for plane in PLANES:
            self._redraw_plane(plane)
        self.dirty = False
        self._update_status()

    def _plane_from_axes(self, ax: plt.Axes) -> str | None:
        for i, candidate in enumerate(self.axes):
            if candidate is ax:
                return PLANES[i]
        return None

    def _on_click(self, event) -> None:
        if event.inaxes is None or event.inaxes in self._button_axes:
            return
        if event.xdata is None or event.ydata is None:
            return
        plane = self._plane_from_axes(event.inaxes)
        if plane is None:
            return
        if len(self.points[plane]) >= 2:
            return
        self.points[plane].append([float(event.xdata), float(event.ydata)])
        self.dirty = True
        self._redraw_plane(plane)
        self._update_status()

    def _undo(self) -> None:
        for plane in reversed(PLANES):
            if self.points[plane]:
                self.points[plane].pop()
                self.dirty = True
                self._redraw_plane(plane)
                self._update_status()
                return

    def _reset(self) -> None:
        self.points = {p: [] for p in PLANES}
        self.dirty = True
        for plane in PLANES:
            self._redraw_plane(plane)
        self._update_status()

    def _delete_all(self) -> None:
        """Remove saved JSON for current case and clear on-screen points."""
        case_id = self._current_case_id()
        path = annotation_path(self.out_dir, case_id)
        if path.is_file():
            path.unlink()
            print(f"deleted -> {path}")
        self.points = {p: [] for p in PLANES}
        self.dirty = False
        for plane in PLANES:
            self._redraw_plane(plane)
        self._update_status()

    def _refocus_canvas(self) -> None:
        canvas = self.fig.canvas
        try:
            canvas.get_tk_widget().focus_force()
        except AttributeError:
            pass
        try:
            canvas.widget().setFocus()
        except AttributeError:
            pass

    def _hook_keyboard(self) -> None:
        manager = self.fig.canvas.manager
        if manager is not None:
            handler_id = getattr(manager, "key_press_handler_id", None)
            if handler_id is not None:
                self.fig.canvas.mpl_disconnect(handler_id)
        self._key_cid = self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _save_and_next(self) -> None:
        self._save_current()
        if self.idx < len(self.case_ids) - 1:
            self.idx += 1
            self._show_case()
        else:
            self._update_status()
            print("last case — saved, no next")

    def _save_current(self) -> Path:
        case_id = self._current_case_id()
        path = save_annotation(
            self.out_dir,
            case_id,
            images_dir=self.images_dir,
            points=self.points,
            manifest=self.manifest,
        )
        self.dirty = False
        print(f"saved -> {path}")
        return path

    def _save_only(self) -> None:
        self._save_current()
        self._update_status()

    def _next(self, *, save: bool = False) -> None:
        if save:
            self._save_and_next()
            return
        if self.idx < len(self.case_ids) - 1:
            self.idx += 1
            self._show_case()

    def _prev(self) -> None:
        if self.idx > 0:
            self.idx -= 1
            self._show_case()
        self._refocus_canvas()

    def _on_key(self, event) -> None:
        key = (event.key or "").lower()
        if key in ("s", "enter", "return"):
            self._save_and_next()
        elif key == "w":
            self._save_only()
        elif key in ("n", "right"):
            self._next(save=False)
        elif key in ("p", "left"):
            self._prev()
        elif key == "u":
            self._undo()
        elif key == "r":
            self._reset()
        elif key in ("d", "delete"):
            self._delete_all()
        elif key == "q":
            plt.close(self.fig)
        self._refocus_canvas()

    def _add_button(self, label: str, x: float, width: float, callback) -> None:
        ax_btn = self.fig.add_axes([x, 0.01, width, 0.055])
        ax_btn.set_facecolor("#2a2a2a")
        btn = Button(ax_btn, label, color="#3a3a3a", hovercolor="#555555")
        btn.label.set_color("#eeeeee")

        def _wrapped(_event) -> None:
            callback()
            self._refocus_canvas()

        btn.on_clicked(_wrapped)
        self._button_axes.add(ax_btn)

    def run(self) -> None:
        _disable_mpl_default_keys()
        self.fig = plt.figure(figsize=(14, 5.9))
        gs = self.fig.add_gridspec(1, 3, left=0.02, right=0.98, top=0.92, bottom=0.12, wspace=0.03)
        self.axes = [self.fig.add_subplot(gs[0, i]) for i in range(3)]
        self.fig.patch.set_facecolor("#1e1e1e")
        self.status_text = self.fig.text(
            0.5,
            0.075,
            "",
            ha="center",
            va="center",
            fontsize=10,
            color="#eeeeee",
        )
        self._add_button("◀ Prev", 0.04, 0.11, self._prev)
        self._add_button("Next ▶", 0.17, 0.11, lambda: self._next(save=False))
        self._add_button("Save+Next", 0.32, 0.13, self._save_and_next)
        self._add_button("Delete all", 0.48, 0.13, self._delete_all)
        self._add_button("Reset", 0.64, 0.10, self._reset)
        self._add_button("Save", 0.77, 0.10, self._save_only)
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self._hook_keyboard()
        self._show_case()
        self._refocus_canvas()
        plt.show()


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    case_ids = case_ids_from_images(args.images_dir)
    if not case_ids:
        raise SystemExit(
            f"No PNG in {args.images_dir}. Run scripts/render_guide_annotation_mpr.py first."
        )

    if args.only_missing:
        case_ids = [c for c in case_ids if not annotation_path(args.out_dir, c).is_file()]

    if args.start_case:
        if args.start_case not in case_ids:
            raise SystemExit(f"Case {args.start_case!r} not in image list")
        start = case_ids.index(args.start_case)
        case_ids = case_ids[start:]

    print(f"Annotator: {len(case_ids)} cases")
    print(f"Images: {args.images_dir}")
    print(f"Output: {args.out_dir}")
    GuideAnnotator(case_ids, args.images_dir, args.out_dir, manifest).run()


if __name__ == "__main__":
    main()
