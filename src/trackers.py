import json
import pickle
import shutil
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from uuid import uuid4

from omegaconf import OmegaConf
from pathlib import Path
from typing import Optional, Any

class ExperimentTracker(ABC):

    @abstractmethod
    def log_config(self, config): pass

    @abstractmethod
    def log_scalar(self, name: str, value): pass

    @abstractmethod
    def log_scalars(self, values: dict): pass

    @abstractmethod
    def log_artifact(self, name: str, data): pass

    @abstractmethod
    def log_table(self, df, title: str, series=None): pass

    @abstractmethod
    def log_figure(self, fig, title: str, series=None): pass

    @abstractmethod
    def finalize(self): pass

    @property
    @abstractmethod
    def experiment_id(self) -> str: ...


class DummyTracker(ExperimentTracker):

    def __init__(self, *args, **kwargs):
        self._experiment_id = uuid4().hex

    def log_config(self, config): pass
    def log_scalar(self, name, value): pass
    def log_scalars(self, values): pass
    def log_artifact(self, name, data): pass
    def log_table(self, df, title, series): pass
    def log_figure(self, fig, title, series): pass
    def finalize(self): pass

    @property
    def experiment_id(self) -> str:
        return self._experiment_id


class ClearmlTracker(ExperimentTracker):

    def __init__(self, project_name: str, task_name: str, tags: list[str]):
        from clearml import Task
        self._task = Task.init(project_name=project_name, task_name=task_name,
                               tags=tags, reuse_last_task_id=False)

    def log_config(self, config):
        self._task.connect(OmegaConf.to_container(config, resolve=True))

    def log_scalar(self, name: str, value):
        self._task.get_logger().report_single_value(name, value)

    def log_scalars(self, values: dict):
        logger = self._task.get_logger()
        for name, value in values.items():
            logger.report_single_value(name, value)

    def log_artifact(self, name: str, data):
        self._task.upload_artifact(name, data)

    def log_table(self, df, title: str, series: str):
        self._task.get_logger().report_table(title=title, series=series, table_plot=df)

    def log_figure(self, fig, title: str, series: str):
        self._task.get_logger().report_matplotlib_figure(
            title=title, series=series, figure=fig)

    def finalize(self):
        self._task.close()

    @property
    def experiment_id(self) -> str:
        return self._task.id


class DiskTracker(ExperimentTracker):
    def __init__(self, root_dir: str | Path, experiment_name: Optional[str] = None):
        self._root_dir = Path(root_dir)
        self._root_dir.mkdir(parents=True, exist_ok=True)
        base = self._sanitize_name(experiment_name)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        exp_dir = self._root_dir / f"{base}_{ts}"
        
        idx = 1
        while exp_dir.exists():
            exp_dir = self._root_dir / f"{base}_{ts}_{idx}"
            idx += 1
        self._exp_dir = exp_dir
        self._exp_dir.mkdir(parents=True, exist_ok=False)
        self._experiment_id = self._exp_dir.name
        self._artifacts_dir = self._exp_dir / "artifacts"
        self._tables_dir = self._exp_dir / "tables"
        self._figures_dir = self._exp_dir / "figures"
        self._artifacts_dir.mkdir(exist_ok=True)
        self._tables_dir.mkdir(exist_ok=True)
        self._figures_dir.mkdir(exist_ok=True)
        self._scalars_path = self._exp_dir / "scalars.jsonl"
        self._epoch_metrics_path = self._exp_dir / "epoch_metrics.jsonl"
        self._meta_path = self._exp_dir / "meta.json"
        self._finalized = False
        self._write_meta({
            "experiment_id": self._experiment_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "root_dir": str(self._root_dir),
            "finished_at_utc": None,
        })

        
    def _write_meta(self, meta: dict) -> None:
        self._meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        
    @staticmethod
    def _sanitize_name(name: str) -> str:
        return (
            name.strip()
            .replace("/", "_")
            .replace("\\", "_")
            .replace(" ", "_")
        )

        
    def _check_not_finalized(self) -> None:
        if self._finalized:
            raise RuntimeError("Tracker already finalized")

            
    @staticmethod
    def _json_default(obj: Any):
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return str(obj)

        
    def log_config(self, config):
        self._check_not_finalized()

        try:
            from omegaconf import OmegaConf
            if config is not None and not isinstance(
                config, (dict, list, str, int, float, bool)
            ):
                config = OmegaConf.to_container(config, resolve=True)
        except Exception:
            pass
        (self._exp_dir / "config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2, default=self._json_default),
            encoding="utf-8",
        )

        
    def log_scalar(self, name: str, value):
        self._check_not_finalized()
        rec = {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "name": name,
            "value": value,
        }
        with self._scalars_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=self._json_default) + "\n")

            
    def log_scalars(self, values: dict):
        self._check_not_finalized()
        ts = datetime.now(timezone.utc).isoformat()
        with self._scalars_path.open("a", encoding="utf-8") as f:
            for name, value in values.items():
                rec = {"ts_utc": ts, "name": name, "value": value}
                f.write(json.dumps(rec, ensure_ascii=False, default=self._json_default) + "\n")
        if "epoch" in values:
            epoch_rec = {
                "ts_utc": ts,
                "epoch": values["epoch"],
                "metrics": {k: v for k, v in values.items() if k != "epoch"},
            }
            with self._epoch_metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(epoch_rec, ensure_ascii=False, default=self._json_default) + "\n")

                
    def log_artifact(self, name: str, data):
        self._check_not_finalized()
        safe_name = self._sanitize_name(name)
        dst_base = self._artifacts_dir / safe_name

        if isinstance(data, (str, Path)) and Path(data).exists():
            src = Path(data)
            if src.is_file():
                target = dst_base if dst_base.suffix else dst_base.with_suffix(src.suffix)
                shutil.copy2(src, target)
            elif src.is_dir():
                target_dir = dst_base
                if target_dir.exists():
                    raise FileExistsError(f"Artifact directory already exists: {target_dir}")
                shutil.copytree(src, target_dir)
            return
        # bytes -> .bin
        if isinstance(data, bytes):
            dst_base.with_suffix(".bin").write_bytes(data)
            return
        # str -> .txt
        if isinstance(data, str):
            dst_base.with_suffix(".txt").write_text(data, encoding="utf-8")
            return
        # fallback -> pickle
        with dst_base.with_suffix(".pkl").open("wb") as f:
            pickle.dump(data, f)

            
    def log_table(self, df, title: str, series=None):
        self._check_not_finalized()
        stem = self._sanitize_name(f"{title}__{series}" if series else title)
        out_path = self._tables_dir / f"{stem}.csv"

        if hasattr(df, "to_csv"):
            df.to_csv(out_path, index=False)
            return
        # fallback
        out_path.write_text(str(df), encoding="utf-8")

        
    def log_figure(self, fig, title: str, series=None):
        self._check_not_finalized()
        stem = self._sanitize_name(f"{title}__{series}" if series else title)
        out_path = self._figures_dir / f"{stem}.png"
        if hasattr(fig, "savefig"):
            fig.savefig(out_path, bbox_inches="tight")
        else:
            raise TypeError("Unsupported figure object: expected object with savefig()")

            
    def finalize(self):
        """Mark the run complete.

        The `finished_at_utc` stamp is what separates a finished run from one that died
        mid-training: both leave a populated directory behind, and only this stamp says
        which is which when a sweep is aggregated later.
        """
        meta = json.loads(self._meta_path.read_text(encoding="utf-8"))
        meta["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        self._write_meta(meta)
        self._finalized = True

        
    @property
    def experiment_id(self) -> str:
        return self._experiment_id