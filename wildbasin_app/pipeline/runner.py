from __future__ import annotations

import traceback

from PySide6.QtCore import QObject, QThread, Signal, Slot

from wildbasin_app.engines.box_urls import collect_box_image_records
from wildbasin_app.engines.common import PipelineCancelled
from wildbasin_app.engines.paddle_ocr_engine import run_paddle_ocr
from wildbasin_app.engines.speciesnet_engine import run_speciesnet

from .config import PipelineConfig


class _PipelineWorker(QObject):
    log = Signal(str)
    stage_started = Signal(str)
    stage_finished = Signal(str, bool)
    tokens_refreshed = Signal(str, str)
    finished = Signal(bool)

    def __init__(self, config: PipelineConfig):
        super().__init__()
        self.config = config
        self._cancel_requested = False

        self._access_token = config.access_token
        self._refresh_token = config.refresh_token

    def request_cancel(self) -> None:
        self._cancel_requested = True

    def _should_cancel(self) -> bool:
        return self._cancel_requested

    def _token_callback(
        self,
        access_token: str,
        refresh_token: str,
    ) -> None:
        self._access_token = access_token
        if refresh_token:
            self._refresh_token = refresh_token

        self.tokens_refreshed.emit(
            self._access_token,
            self._refresh_token,
        )

    def _common_credentials(self) -> dict:
        return {
            "client_id": self.config.client_id,
            "client_secret": self.config.client_secret,
            "access_token": self._access_token,
            "refresh_token": self._refresh_token,
            "token_callback": self._token_callback,
        }

    @Slot()
    def run(self) -> None:
        try:
            self.config.output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            if self.config.run_url_scan:
                self._run_stage(
                    "Collect Box image records",
                    self._run_box_urls,
                )

            if self.config.run_speciesnet:
                self._run_stage(
                    "Run SpeciesNet",
                    self._run_speciesnet,
                )

            if self.config.run_ocr:
                self._run_stage(
                    "Run PaddleOCR",
                    self._run_paddle_ocr,
                )

            self.log.emit("\nPipeline completed.")
            self.finished.emit(True)

        except PipelineCancelled:
            self.log.emit("\nPipeline cancelled.")
            self.finished.emit(False)

        except Exception as exc:
            self.log.emit(
                "\nPipeline failed:\n"
                f"{type(exc).__name__}: {exc}\n"
                f"{traceback.format_exc()}"
            )
            self.finished.emit(False)

    def _run_stage(self, name: str, callback) -> None:
        if self._should_cancel():
            raise PipelineCancelled()

        self.stage_started.emit(name)
        self.log.emit(f"\n=== {name} ===")

        callback()

        self.stage_finished.emit(name, True)

    def _run_box_urls(self) -> None:
        collect_box_image_records(
            **self._common_credentials(),
            folder_id=self.config.folder_id,
            output_file=str(self.config.url_file),
            batch_size=100,
            log_callback=self.log.emit,
            should_cancel=self._should_cancel,
        )

    def _run_speciesnet(self) -> None:
        run_speciesnet(
            **self._common_credentials(),
            input_file=str(self.config.url_file),
            results_file=str(self.config.speciesnet_file),
            batch_size=self.config.speciesnet_batch_size,
            download_workers=self.config.download_workers,
            log_callback=self.log.emit,
            should_cancel=self._should_cancel,
        )

    def _run_paddle_ocr(self) -> None:
        run_paddle_ocr(
            **self._common_credentials(),
            input_file=str(self.config.url_file),
            results_file=str(self.config.ocr_file),
            batch_size=self.config.ocr_batch_size,
            download_workers=self.config.download_workers,
            crop_percent=self.config.ocr_crop_percent,
            device="auto",
            log_callback=self.log.emit,
            should_cancel=self._should_cancel,
        )


class PipelineRunner(QObject):
    log = Signal(str)
    stage_started = Signal(str)
    stage_finished = Signal(str, bool)
    tokens_refreshed = Signal(str, str)
    pipeline_finished = Signal(bool)
    running_changed = Signal(bool)

    def __init__(self, _engine_dir=None, parent=None):
        # _engine_dir is accepted only for compatibility with the current GUI.
        super().__init__(parent)
        self._thread: QThread | None = None
        self._worker: _PipelineWorker | None = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.isRunning()

    def start(self, config: PipelineConfig) -> None:
        if self.is_running:
            raise RuntimeError("A pipeline is already running.")

        thread = QThread(self)
        worker = _PipelineWorker(config)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)

        worker.log.connect(self.log)
        worker.stage_started.connect(self.stage_started)
        worker.stage_finished.connect(self.stage_finished)
        worker.tokens_refreshed.connect(self.tokens_refreshed)
        worker.finished.connect(self._worker_finished)

        worker.finished.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(self._thread_finished)

        self._thread = thread
        self._worker = worker

        self.running_changed.emit(True)
        thread.start()

    def cancel(self) -> None:
        if self._worker is not None:
            self.log.emit(
                "\nCancellation requested. "
                "The current image/model batch will finish safely first."
            )
            self._worker.request_cancel()

    @Slot(bool)
    def _worker_finished(self, success: bool) -> None:
        self.pipeline_finished.emit(success)
        self.running_changed.emit(False)

    @Slot()
    def _thread_finished(self) -> None:
        if self._thread is not None:
            self._thread.deleteLater()

        self._thread = None
        self._worker = None
