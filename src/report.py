"""Stable coverage reports plus static and interactive voxel diagnostics."""

import csv
import json
from pathlib import Path

import numpy as np


def _percent(value):
    return f"{100.0 * value:.2f}%"


def _percentage_points(value):
    return f"{100.0 * value:.2f} pp"


def _highlight(text, colour):
    return f'<span style="color:{colour}"><strong>{text}</strong></span>'


def _recall_text(value, bold=False):
    text = _percent(value)
    if value < 0.8:
        return _highlight(text, "#dc2626")
    return f"**{text}**" if bold else text


def _status_text(status):
    if status == "incomplete":
        return _highlight(status, "#c2410c")
    if status == "skipped":
        return _highlight(status, "#dc2626")
    return status


def _grid_status(mission, agent_id, grid_id):
    return (
        mission.get("grid_outcomes", {})
        .get(str(int(agent_id)), {})
        .get(str(int(grid_id)), {})
        .get("status", "skipped")
    )


def _total_status(statuses):
    if "local_complete" in statuses:
        return "local_complete"
    if "incomplete" in statuses:
        return "incomplete"
    return "skipped"


def _metric_cell(metrics, status):
    if metrics is None:
        return f"—<br>{_status_text(status)}"
    return (
        f"R {_recall_text(metrics['recall'])}<br>"
        f"P {_percent(metrics['precision'])}<br>"
        f"F1 {_percent(metrics['f1'])}<br>{_status_text(status)}"
    )


def _outcome_counts(statuses):
    counts = {
        status: statuses.count(status)
        for status in ("local_complete", "incomplete", "skipped")
    }
    parts = [f"{counts['local_complete']} local_complete"]
    for status in ("incomplete", "skipped"):
        text = f"{counts[status]} {status}"
        colour = "#c2410c" if status == "incomplete" else "#dc2626"
        parts.append(_highlight(text, colour) if counts[status] else text)
    return " / ".join(parts)


def _mission_report_lines(summary, region_agents):
    mission = summary.get("mission")
    if not mission:
        return []
    dynamic_ids = {int(value) for value in mission["dynamic_grid_ids"]}
    agent_metrics = {}
    for region, rows in region_agents.items():
        agent_metrics[region] = {int(row["agent_id"]): row for row in rows}

    lines = [
        "", "## Per-grid statistics", "",
        (
            "Each table follows one team's configured grid sequence. Coverage "
            "totals use the evaluated maps from that team. "
            "The result status is `local_complete` when any configured agent "
            "reported it, `incomplete` when reports exist but none completed, "
            "and `skipped` when no configured agent reported the grid."
        ), "",
        (
            '<span style="color:#dc2626;font-weight:700">Red recall</span> is below '
            "80%. Non-complete results are highlighted in amber or red."
        ), "",
    ]
    for team in mission["teams"]:
        team_id = int(team["team_id"])
        agent_ids = [int(value) for value in team["agent_ids"]]
        lines.extend([
            f"### Team {team_id}", "",
            "| Grid | Type | Recall | Precision | F1 | Result | Agent outcomes |",
            "|---:|---|---:|---:|---:|---|---|",
        ])
        for raw_grid_id in team["frontier_grid_order"]:
            grid_id = int(raw_grid_id)
            region = f"grid_{grid_id}"
            metrics = (
                mission.get("team_grid_metrics", {})
                .get(str(team_id), {})
                .get(region)
            )
            if metrics is None:
                continue
            statuses = [
                _grid_status(mission, agent_id, grid_id)
                for agent_id in agent_ids
            ]
            lines.append(
                f"| {grid_id} | {'dynamic' if grid_id in dynamic_ids else 'static'} | "
                f"{_recall_text(metrics['recall'])} | "
                f"{_percent(metrics['precision'])} | {_percent(metrics['f1'])} | "
                f"{_status_text(_total_status(statuses))} | "
                f"{_outcome_counts(statuses)} |"
            )
        lines.append("")

    lines.extend(["", "## Per-agent grid statistics", ""])
    for team in mission["teams"]:
        team_id = int(team["team_id"])
        agent_ids = [int(value) for value in team["agent_ids"]]
        lines.extend([
            f"### Team {team_id}", "",
            "| Grid | Type | "
            + " | ".join(f"Agent {agent_id}" for agent_id in agent_ids)
            + " | Total |",
            "|---:|---|" + "---|" * (len(agent_ids) + 1),
        ])
        for raw_grid_id in team["frontier_grid_order"]:
            grid_id = int(raw_grid_id)
            region = f"grid_{grid_id}"
            if region not in summary["regions"]:
                continue
            statuses = [_grid_status(mission, agent_id, grid_id)
                        for agent_id in agent_ids]
            cells = [
                _metric_cell(
                    agent_metrics.get(region, {}).get(agent_id),
                    status,
                )
                for agent_id, status in zip(agent_ids, statuses)
            ]
            total_metrics = (
                mission.get("team_grid_metrics", {})
                .get(str(team_id), {})
                .get(region)
            )
            cells.append(_metric_cell(total_metrics, _total_status(statuses)))
            lines.append(
                f"| {grid_id} | {'dynamic' if grid_id in dynamic_ids else 'static'} | "
                + " | ".join(cells) + " |"
            )
        lines.extend(["", "`Total` is the coverage union of evaluated maps from this team. ",
                      "`—` means that agent had no saved map in this evaluation.", ""])
    return lines


def _voxel_centres(keys, origin, voxel_size):
    """Convert integer voxel keys to metric voxel-centre coordinates."""
    if not keys:
        return np.empty((0, 3), dtype=np.float64)
    return (
        np.asarray(tuple(keys), dtype=np.float64) + 0.5
    ) * voxel_size + np.asarray(origin, dtype=np.float64)


def _voxel_marker_area(axis, figure, voxel_size, bounds_min, bounds_max):
    """Return scatter area whose diameter projects to one voxel edge."""
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    from mpl_toolkits.mplot3d import proj3d

    lower = np.asarray(bounds_min, dtype=np.float64)
    upper = np.asarray(bounds_max, dtype=np.float64)
    centre = (lower + upper) / 2.0
    projection = axis.get_proj()

    projected_centre = proj3d.proj_transform(*centre, projection)[:2]
    centre_pixels = axis.transData.transform(projected_centre)
    projected_lengths = []
    for dimension in range(3):
        endpoint = centre.copy()
        endpoint[dimension] += voxel_size
        projected_endpoint = proj3d.proj_transform(*endpoint, projection)[:2]
        endpoint_pixels = axis.transData.transform(projected_endpoint)
        projected_lengths.append(np.linalg.norm(endpoint_pixels - centre_pixels))

    diameter_pixels = max(projected_lengths)
    if not np.isfinite(diameter_pixels) or diameter_pixels <= 0:
        raise RuntimeError("cannot project voxel size into the 3D viewer")
    diameter_points = diameter_pixels * 72.0 / figure.dpi
    return diameter_points * diameter_points


def _plot_grid_regions(axis, grid_regions):
    """Overlay rectangular grid bounds and IDs on a top-down plot."""
    for name in sorted(grid_regions, key=str):
        label = str(name)
        if label.startswith("grid_"):
            label = f"Grid {label[5:]}"
        for lower, upper in grid_regions[name]:
            lower = np.asarray(lower, dtype=np.float64)
            upper = np.asarray(upper, dtype=np.float64)
            corners_x = [lower[0], upper[0], upper[0], lower[0], lower[0]]
            corners_y = [lower[1], lower[1], upper[1], upper[1], lower[1]]
            axis.plot(
                corners_x,
                corners_y,
                color="#111827",
                linestyle="--",
                linewidth=0.9,
                alpha=0.85,
                zorder=3,
            )
            axis.text(
                (lower[0] + upper[0]) / 2.0,
                (lower[1] + upper[1]) / 2.0,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color="#111827",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "#111827",
                    "alpha": 0.78,
                    "pad": 2.0,
                },
                zorder=4,
            )


def _pyplot(interactive):
    import matplotlib

    if not interactive:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def write_reports(
    output_dir, summary, region_agents, plot_data=None, interactive_backend=False
):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "coverage_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")

    agent_fields = [
        "region", "agent_id", "map", "ground_truth_voxels", "occupied_voxels",
        "known_voxels", "free_voxels", "matched_ground_truth_voxels",
        "matched_occupied_voxels", "recall", "precision", "f1",
        "unique_contribution_voxels", "overlap_occupied_voxels",
    ]
    with (output / "agent_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=agent_fields)
        writer.writeheader()
        for region, rows in region_agents.items():
            for row in rows:
                writer.writerow({"region": region, **{key: row[key] for key in agent_fields[1:]}})

    region_fields = ["region"] + list(next(iter(summary["regions"].values())).keys())
    with (output / "region_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=region_fields)
        writer.writeheader()
        for name, metrics in summary["regions"].items():
            writer.writerow({"region": name, **metrics})

    primary = summary["regions"][summary["primary_region"]]
    lines = [
        f"# Coverage report — `{summary['run_id']}`", "",
        "| Measurement | Result |", "|---|---:|",
        f"| Agents evaluated | {summary['agents_evaluated']} |",
        f"| GT reference voxels | {primary['ground_truth_voxels']:,} |",
        f"| 🟢 Occupied-voxel recall | {_recall_text(primary['recall'], bold=True)} |",
        f"| 🔵 Occupied-voxel precision | **{_percent(primary['precision'])}** |",
        f"| Occupied-voxel F1 | **{_percent(primary['f1'])}** |",
        f"| Improvement over best agent | {_percentage_points(primary['improvement_over_best_agent'])} |",
        f"| 🟠 Classification conflicts | {primary['classification_conflicts']:,} |",
    ]
    lines.extend(_mission_report_lines(summary, region_agents))
    lines.extend([
        "", "## Reproducibility", "",
        f"- Voxel size: `{summary['evaluation']['voxel_size_m']:.6g} m`",
        f"- Match tolerance: `{summary['evaluation']['match_tolerance_m']:.6g} m`",
        f"- Frame: `{summary['evaluation']['frame_id']}`",
        f"- Manifest: `{summary['manifest']}`",
    ])
    (output / "coverage_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    if plot_data is not None:
        plt = _pyplot(interactive_backend)
        gt = plot_data["ground_truth"]
        matched = plot_data["matched_ground_truth"]
        ordered_gt = tuple(gt)
        points = _voxel_centres(
            ordered_gt, plot_data["origin"], plot_data["voxel_size"]
        )
        colours = [
            "#16a34a" if key in matched else "#dc2626"
            for key in ordered_gt
        ]
        plots = output / "plots"
        plots.mkdir(exist_ok=True)
        figure, axis = plt.subplots(figsize=(10, 8))
        axis.scatter(
            points[:, 0], points[:, 1], c=colours, s=1, linewidths=0, zorder=1
        )
        _plot_grid_regions(axis, plot_data.get("grid_regions", {}))
        bounds_min = np.asarray(plot_data["bounds_min"], dtype=np.float64)
        bounds_max = np.asarray(plot_data["bounds_max"], dtype=np.float64)
        axis.set_xlim(bounds_min[0], bounds_max[0])
        axis.set_ylim(bounds_min[1], bounds_max[1])
        axis.set_aspect("equal")
        axis.set_title(
            "GT reference voxels: matched (green), missed (red); grid IDs shown"
        )
        axis.set_xlabel("map x (m)")
        axis.set_ylabel("map y (m)")
        figure.tight_layout()
        figure.savefig(plots / "coverage_top_down.png", dpi=180)
        plt.close(figure)


def show_interactive_3d(plot_data, run_id, region_name):
    """Open a blocking 3D viewer with independently toggleable voxel layers."""
    plt = _pyplot(interactive=True)
    from matplotlib.widgets import CheckButtons

    figure = plt.figure(figsize=(12, 8))
    required_framework = getattr(
        figure.canvas, "required_interactive_framework", None
    )
    if required_framework is None:
        backend = plt.get_backend()
        plt.close(figure)
        raise RuntimeError(
            f"vis=true requires an interactive Matplotlib backend; got {backend!r}. "
            "Configure a GUI backend/display or run with --no-vis"
        )

    axis = figure.add_subplot(111, projection="3d")
    figure.subplots_adjust(left=0.04, right=0.76, bottom=0.08, top=0.92)

    origin = plot_data["origin"]
    voxel_size = plot_data["voxel_size"]
    ground_truth = plot_data["ground_truth"]
    matched = plot_data["matched_ground_truth"]
    missed = ground_truth - matched
    layers = (
        (
            "Agent occupied union",
            _voxel_centres(plot_data["agent_union"], origin, voxel_size),
            "#2563eb",
            0.55,
        ),
        (
            "GT reference voxels",
            _voxel_centres(ground_truth, origin, voxel_size),
            "#16a34a",
            0.18,
        ),
        (
            "Missed GT voxels",
            _voxel_centres(missed, origin, voxel_size),
            "#dc2626",
            0.90,
        ),
    )
    artists = {}
    for label, points, colour, alpha in layers:
        artists[label] = axis.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=colour,
            s=1.0,
            alpha=alpha,
            linewidths=0,
            depthshade=False,
            label=label,
        )

    lower = np.asarray(plot_data["bounds_min"], dtype=np.float64)
    upper = np.asarray(plot_data["bounds_max"], dtype=np.float64)
    axis.set_xlim(lower[0], upper[0])
    axis.set_ylim(lower[1], upper[1])
    axis.set_zlim(lower[2], upper[2])
    axis.set_box_aspect(upper - lower)
    axis.set_xlabel("map x (m)")
    axis.set_ylabel("map y (m)")
    axis.set_zlabel("map z (m)")
    axis.set_title(f"Occupied-voxel coverage — {run_id} — {region_name}")

    def _sync_marker_size(event=None):
        area = _voxel_marker_area(
            axis, figure, voxel_size, lower, upper
        )
        for artist in artists.values():
            artist.set_sizes([area])
        if event is not None:
            figure.canvas.draw_idle()

    _sync_marker_size()
    figure.canvas.mpl_connect("button_release_event", _sync_marker_size)
    figure.canvas.mpl_connect("scroll_event", _sync_marker_size)
    figure.canvas.mpl_connect("resize_event", _sync_marker_size)

    controls = figure.add_axes((0.79, 0.68, 0.20, 0.18))
    labels = tuple(artists)
    checkboxes = CheckButtons(controls, labels, (True,) * len(labels))
    controls.set_title("Voxel layers", loc="left")

    def _toggle(label):
        artist = artists[label]
        artist.set_visible(not artist.get_visible())
        figure.canvas.draw_idle()

    checkboxes.on_clicked(_toggle)
    manager = figure.canvas.manager
    if hasattr(manager, "set_window_title"):
        manager.set_window_title(f"Coverage viewer — {run_id}")
    plt.show(block=True)
