"""
solvers/sw_verification.py
SolidWorksVerification — queue manager for flagging designs for human
SolidWorks / 3DEXPERIENCE review before manufacturing sign-off.

Integrates with:
  - CAD/freecad_bridge.py  (geometry export)
  - sw_verification/pending_export/  (file drop zone)

Usage:
    from solvers.sw_verification import SolidWorksVerification
    sw = SolidWorksVerification(watch_folder="sw_verification", mode="manual")
    hash = sw.flag_for_verification(geometry, pipeline_results, priority="normal")
    readiness = sw.check_cutover_readiness()
"""

from __future__ import annotations

import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

from solvers.base_solver import GeometrySpec

# Optional FreeCAD bridge
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from CAD.freecad_bridge import FreeCADBridge  # type: ignore
    FREECAD_AVAILABLE = True
except ImportError:
    FREECAD_AVAILABLE = False


# ---------------------------------------------------------------------------
# SolidWorksVerification
# ---------------------------------------------------------------------------

class SolidWorksVerification:
    """
    Manages a file-based queue of designs awaiting SolidWorks verification.

    Folder layout created automatically:
        <watch_folder>/
            pending_export/     ← pipeline writes here; SW reads here
            in_review/          ← moved here when SW session opens
            approved/           ← moved here after sign-off
            rejected/           ← flagged issues go here
            queue.json          ← master queue manifest
    """

    PRIORITIES = {"high": 0, "normal": 1, "low": 2}

    def __init__(
        self,
        watch_folder: str = "sw_verification",
        mode: str = "manual",       # "manual" | "auto" | "watcher"
        verbose: bool = True,
        sw_exe_path: Optional[str] = None,   # path to SLDWORKS.exe if auto-launch
    ):
        self.watch_folder = Path(watch_folder)
        self.mode = mode
        self.verbose = verbose
        self.sw_exe_path = sw_exe_path

        # Create directory structure
        for sub in ("pending_export", "in_review", "approved", "rejected"):
            (self.watch_folder / sub).mkdir(parents=True, exist_ok=True)

        self.queue_file = self.watch_folder / "queue.json"
        if not self.queue_file.exists():
            self._write_queue([])

        if self.verbose:
            print(f"[SWVerification] Queue root : {self.watch_folder.resolve()}")
            print(f"[SWVerification] Mode       : {self.mode}")
            print(f"[SWVerification] FreeCAD    : {'✓' if FREECAD_AVAILABLE else '✗ (JSON export only)'}")

    # ------------------------------------------------------------------
    # Queue I/O
    # ------------------------------------------------------------------

    def _read_queue(self) -> List[dict]:
        try:
            with open(self.queue_file) as fh:
                return json.load(fh)
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _write_queue(self, entries: List[dict]) -> None:
        with open(self.queue_file, "w") as fh:
            json.dump(entries, fh, indent=2)

    def _append_queue(self, entry: dict) -> None:
        entries = self._read_queue()
        # Remove any existing entry for same hash
        entries = [e for e in entries if e.get("design_hash") != entry["design_hash"]]
        entries.append(entry)
        # Sort by priority then timestamp
        entries.sort(key=lambda e: (self.PRIORITIES.get(e.get("priority", "normal"), 1), e.get("queued_at", 0)))
        self._write_queue(entries)

    # ------------------------------------------------------------------
    # flag_for_verification
    # ------------------------------------------------------------------

    def flag_for_verification(
        self,
        geometry: GeometrySpec,
        pipeline_results: Dict[str, Any],
        priority: str = "normal",
        notes: str = "",
    ) -> str:
        """
        Write design spec + results to pending_export/ and add to queue.

        Returns:
            design_hash (str)
        """
        design_hash = geometry.design_hash()
        timestamp   = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        folder_name = f"{timestamp}_{design_hash}"
        export_dir  = self.watch_folder / "pending_export" / folder_name
        export_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save geometry spec JSON
        geo_path = export_dir / "geometry_spec.json"
        geometry.save_json(str(geo_path))

        # 2. Save pipeline results JSON
        results_path = export_dir / "pipeline_results.json"
        with open(results_path, "w") as fh:
            json.dump(pipeline_results, fh, indent=2, default=str)

        # 3. Save human-readable summary
        summary_path = export_dir / "REVIEW_SUMMARY.md"
        self._write_summary(summary_path, geometry, pipeline_results, notes)

        # 4. Try FreeCAD STL export
        stl_path = None
        if FREECAD_AVAILABLE:
            try:
                bridge = FreeCADBridge()
                stl_path = str(export_dir / f"{design_hash}.stl")
                bridge.export(geometry.to_dict(), stl_path, fmt="stl")
                if self.verbose:
                    print(f"[SWVerification] STL exported: {stl_path}")
            except Exception as e:
                if self.verbose:
                    print(f"[SWVerification] FreeCAD export failed ({e}) — geometry JSON saved instead")
        else:
            # Fallback: save Bézier control points as CSV for manual reconstruction
            import csv
            csv_path = export_dir / "bezier_control_points.csv"
            with open(csv_path, "w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(["curve", "point_index", "value"])
                for ci, curve in enumerate([geometry.bezier_curve1, geometry.bezier_curve2, geometry.bezier_curve3], 1):
                    for pi, val in enumerate(curve):
                        writer.writerow([f"curve{ci}", pi, val])

        # 5. Queue entry
        entry = {
            "design_hash":    design_hash,
            "folder":         str(export_dir.relative_to(self.watch_folder)),
            "priority":       priority,
            "status":         "pending",
            "notes":          notes,
            "queued_at":      time.time(),
            "timestamp":      timestamp,
            "has_stl":        stl_path is not None,
            "targets_met": {
                "thermal":    pipeline_results.get("thermal", {}).get("passed", None),
                "em":         pipeline_results.get("electromagnetic", {}).get("passed", None),
                "structural": pipeline_results.get("structural", {}).get("passed", None),
            },
        }
        self._append_queue(entry)

        if self.verbose:
            print(f"\n[SWVerification] ✓ Design queued: {design_hash}")
            print(f"  Priority  : {priority}")
            print(f"  Export dir: {export_dir}")
            print(f"  Queue size: {len(self._read_queue())}")

        return design_hash

    # ------------------------------------------------------------------
    # Summary markdown
    # ------------------------------------------------------------------

    def _write_summary(
        self,
        path: Path,
        geometry: GeometrySpec,
        results: Dict[str, Any],
        notes: str,
    ) -> None:
        em  = results.get("electromagnetic", {})
        th  = results.get("thermal", {})
        st  = results.get("structural", {})

        lines = [
            f"# SolidWorks Verification — Design `{geometry.design_hash()}`",
            f"",
            f"Generated: {datetime.utcnow().isoformat()}Z",
            f"",
            f"## Notes",
            f"{notes or '(none)'}",
            f"",
            f"## Design Parameters",
            f"| Parameter | Value |",
            f"|-----------|-------|",
            f"| Rated Power | {geometry.rated_power_w/1000:.1f} kW |",
            f"| Rated Speed | {geometry.rated_speed_rpm:.0f} RPM |",
            f"| Pole Pairs  | {geometry.pole_pairs} |",
            f"| Stator Slots| {geometry.stator_slots} |",
            f"| Magnet Grade| {geometry.magnet_grade} |",
            f"| Remanence   | {geometry.remanence_T:.2f} T |",
            f"",
            f"## Physics Results",
            f"| Domain | Key Metric | Value | Target | Pass? |",
            f"|--------|-----------|-------|--------|-------|",
            f"| EM     | Efficiency | {em.get('efficiency_pct', '—'):.1f}% | ≥95% | {'✓' if em.get('passed') else '✗'} |",
            f"| EM     | Cogging    | {em.get('cogging_Nm', '—'):.1f} N·m | <25 N·m | — |",
            f"| EM     | Br_min     | {em.get('Br_min_T', '—'):.3f} T | >0.3 T | — |",
            f"| Thermal| T_winding  | {th.get('T_winding_C', '—'):.1f}°C | <180°C | {'✓' if th.get('passed') else '✗'} |",
            f"| Thermal| T_magnet   | {th.get('T_magnet_C', '—'):.1f}°C | <60°C | — |",
            f"| Struct.| Safety f.  | {st.get('safety_factor', '—')} | ≥2.0 | {'✓' if st.get('passed') else '✗'} |",
            f"",
            f"## Bézier Curves (Control Points)",
            f"**Curve 1:** {geometry.bezier_curve1}",
            f"",
            f"**Curve 2:** {geometry.bezier_curve2}",
            f"",
            f"**Curve 3:** {geometry.bezier_curve3}",
            f"",
            f"## Next Steps",
            f"1. Open SolidWorks / 3DEXPERIENCE",
            f"2. Import `geometry_spec.json` or STL if available",
            f"3. Run full FEA confirmation",
            f"4. Move folder to `approved/` or `rejected/` and update `queue.json`",
        ]
        path.write_text("\n".join(lines))

    # ------------------------------------------------------------------
    # Cutover readiness check
    # ------------------------------------------------------------------

    def check_cutover_readiness(self) -> dict:
        """
        Check whether the SW verification pipeline is ready for automated cutover.
        Returns readiness dict with a human-readable message.
        """
        queue = self._read_queue()
        pending  = [e for e in queue if e.get("status") == "pending"]
        approved = [e for e in queue if e.get("status") == "approved"]
        rejected = [e for e in queue if e.get("status") == "rejected"]

        has_freecad = FREECAD_AVAILABLE
        has_sw_exe  = self.sw_exe_path is not None and Path(self.sw_exe_path).exists() if self.sw_exe_path else False

        all_passed = all(
            all(v for v in e.get("targets_met", {}).values() if v is not None)
            for e in approved
        ) if approved else False

        ready = has_freecad and len(pending) == 0 and len(approved) > 0 and all_passed

        return {
            "ready":          ready,
            "pending_count":  len(pending),
            "approved_count": len(approved),
            "rejected_count": len(rejected),
            "freecad_available": has_freecad,
            "sw_exe_found":   has_sw_exe,
            "message": (
                "✓ Ready for automated SW cutover"
                if ready else
                f"Not ready — {len(pending)} pending review(s), FreeCAD={'✓' if has_freecad else '✗ not installed'}"
            ),
        }

    # ------------------------------------------------------------------
    # Queue management helpers
    # ------------------------------------------------------------------

    def list_pending(self) -> List[dict]:
        return [e for e in self._read_queue() if e.get("status") == "pending"]

    def mark_approved(self, design_hash: str, reviewer: str = "manual") -> bool:
        return self._update_status(design_hash, "approved", reviewer)

    def mark_rejected(self, design_hash: str, reason: str = "", reviewer: str = "manual") -> bool:
        return self._update_status(design_hash, "rejected", reviewer, extra={"rejection_reason": reason})

    def _update_status(self, design_hash: str, status: str, reviewer: str, extra: dict = None) -> bool:
        entries = self._read_queue()
        for e in entries:
            if e.get("design_hash") == design_hash:
                e["status"]      = status
                e["reviewed_by"] = reviewer
                e["reviewed_at"] = time.time()
                if extra:
                    e.update(extra)
                # Move folder
                src = self.watch_folder / "pending_export" / e["folder"].replace("pending_export/", "")
                if not src.exists():
                    src = self.watch_folder / e["folder"]
                dst = self.watch_folder / status / src.name
                if src.exists():
                    shutil.move(str(src), str(dst))
                    e["folder"] = str(Path(status) / src.name)
                self._write_queue(entries)
                if self.verbose:
                    print(f"[SWVerification] {design_hash} → {status}")
                return True
        return False
