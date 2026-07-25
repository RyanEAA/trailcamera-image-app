from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, Signal

from .config import PipelineConfig

@dataclass(frozen=True)
class Stage:
    name: str
    script: Path
    args: List[str]

class PipelineRunner(QObject):
    log = Signal(str)
    stage_started = Signal(str)
    stage_finished = Signal(str, int)
    pipeline_finished = Signal(bool)
    running_changed = Signal(bool)

    def __init__(self, engine_dir: Path, parent=None):
        super().__init__(parent)
        self.engine_dir = engine_dir
        self._process: QProcess | None = None
        self._stages: List[Stage] = []
        self._stage_index = -1
        self._cancelled = False

    @property
    def is_running(self) -> bool:
        return self._process is not None

    def start(self, config: PipelineConfig) -> None:
        if self.is_running:
            raise RuntimeError("A pipeline is already running.")

        config.output_dir.mkdir(parents=True, exist_ok=True)
        self._cancelled = False
        self._stages = self._build_stages(config)
        self._stage_index = -1

        if not self._stages:
            self.log.emit("No processing stages were selected.")
            self.pipeline_finished.emit(True)
            return

        self._config = config
        self.running_changed.emit(True)
        self._start_next_stage()

    def cancel(self) -> None:
        self._cancelled = True
        if self._process is not None:
            self.log.emit("\nCancelling current stage...")
            self._process.terminate()
            if not self._process.waitForFinished(3000):
                self._process.kill()

    def _build_stages(self, config: PipelineConfig) -> List[Stage]:
        stages: List[Stage] = []

        if config.run_url_scan:
            stages.append(
                Stage(
                    "Collect Box image records",
                    self.engine_dir / "box-get-urls.py",
                    [
                        "--output-file", str(config.url_file),
                    ],
                )
            )

        if config.run_speciesnet:
            stages.append(
                Stage(
                    "Run SpeciesNet",
                    self.engine_dir / "batch-box-run-speciesnet.py",
                    [
                        "--input-file", str(config.url_file),
                        "--results-file", str(config.speciesnet_file),
                        "--batch-size", str(config.speciesnet_batch_size),
                        "--download-workers", str(config.download_workers),
                    ],
                )
            )

        if config.run_ocr:
            stages.append(
                Stage(
                    "Run PaddleOCR",
                    self.engine_dir / "batch-box-paddle-ocr-gpu.py",
                    [
                        "--input-file", str(config.url_file),
                        "--results-file", str(config.ocr_file),
                        "--batch-size", str(config.ocr_batch_size),
                        "--download-workers", str(config.download_workers),
                        "--crop-percent", str(config.ocr_crop_percent),
                        "--device", "auto",
                    ],
                )
            )

        return stages

    def _start_next_stage(self) -> None:
        self._stage_index += 1

        if self._cancelled:
            self._cleanup_process()
            self.running_changed.emit(False)
            self.pipeline_finished.emit(False)
            return

        if self._stage_index >= len(self._stages):
            self._cleanup_process()
            self.log.emit("\nPipeline completed.")
            self.running_changed.emit(False)
            self.pipeline_finished.emit(True)
            return

        stage = self._stages[self._stage_index]
        self.stage_started.emit(stage.name)
        self.log.emit(f"\n=== {stage.name} ===")
        self.log.emit(f"$ {sys.executable} {stage.script.name} {' '.join(stage.args)}")

        process = QProcess(self)
        process.setWorkingDirectory(str(self.engine_dir))
        process.setProgram(sys.executable)
        process.setArguments([str(stage.script), *stage.args])
        process.setProcessChannelMode(QProcess.MergedChannels)

        env = QProcessEnvironment.systemEnvironment()
        env.insert("CLIENT_ID", self._config.client_id)
        env.insert("CLIENT_SECRET", self._config.client_secret)
        env.insert("ACCESS_TOKEN", self._config.access_token)
        env.insert("REFRESH_TOKEN", self._config.refresh_token)
        env.insert("ROOT_FOLDER_ID", self._config.folder_id)
        env.insert("OUTPUT_FILE", str(self._config.url_file))
        env.insert("PYTHONUNBUFFERED", "1")
        process.setProcessEnvironment(env)

        process.readyReadStandardOutput.connect(self._read_output)
        process.finished.connect(self._process_finished)
        process.errorOccurred.connect(self._process_error)

        self._process = process
        process.start()

    def _read_output(self) -> None:
        if self._process is None:
            return
        data = bytes(self._process.readAllStandardOutput()).decode(
            "utf-8", errors="replace"
        )
        if data:
            self.log.emit(data.rstrip("\n"))

    def _process_finished(self, exit_code: int, _exit_status) -> None:
        stage = self._stages[self._stage_index]
        self._read_output()
        self.stage_finished.emit(stage.name, exit_code)

        process = self._process
        self._process = None
        if process is not None:
            process.deleteLater()

        if self._cancelled:
            self.running_changed.emit(False)
            self.pipeline_finished.emit(False)
            return

        if exit_code != 0:
            self.log.emit(
                f"\n{stage.name} failed with exit code {exit_code}. "
                "The remaining stages were not started."
            )
            self.running_changed.emit(False)
            self.pipeline_finished.emit(False)
            return

        self._start_next_stage()

    def _process_error(self, error) -> None:
        self.log.emit(f"Process error: {error}")

    def _cleanup_process(self) -> None:
        if self._process is not None:
            self._process.deleteLater()
            self._process = None
