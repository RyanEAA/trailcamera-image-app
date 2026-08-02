from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSettings, QStandardPaths, QUrl, Qt
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from wildbasin_app.auth.box_oauth import (
    BoxAuthResult,
    build_authorize_url,
    complete_oauth,
)
from wildbasin_app.pipeline.config import PipelineConfig, normalize_box_folder_id
from wildbasin_app.pipeline.runner import PipelineRunner


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Wild Basin Image Processor")
        self.setMinimumSize(600, 500)
        self.resize(900, 850)

        self.settings = QSettings()
        self.engine_dir = Path(__file__).resolve().parents[2] / "engine"
        self.runner = PipelineRunner(self.engine_dir, self)

        # OAuth tokens deliberately live only in memory for now.
        # We can move them to macOS Keychain in a later milestone.
        self.box_auth: BoxAuthResult | None = None

        self._build_ui()
        self._load_settings()
        self._connect_runner()
        self._set_auth_required()

    def _copy_access_token(self):
        token = self.access_token_display.text()

        if not token:
            return

        QApplication.clipboard().setText(token)
        self.status.setText("Access token copied to clipboard.")


    def _copy_refresh_token(self):
        token = self.refresh_token_display.text()

        if not token:
            return

        QApplication.clipboard().setText(token)
        self.status.setText("Refresh token copied to clipboard.")


    def _build_ui(self):
        root = QWidget()
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarAsNeeded
        )
        scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarAsNeeded
        )

        content = QWidget()
        layout = QVBoxLayout(content)

        scroll_area.setWidget(content)

        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.addWidget(scroll_area)

        heading = QLabel("<h1>Wild Basin Image Processor</h1>")
        subtitle = QLabel(
            "Collect Box image records, run SpeciesNet, and extract OCR text."
        )
        layout.addWidget(heading)
        layout.addWidget(subtitle)

        # ------------------------------------------------------------------
        # Box OAuth
        # ------------------------------------------------------------------
        box_group = QGroupBox("Box configuration")
        box_layout = QVBoxLayout(box_group)

        self.auth_status = QLabel()
        self.auth_status.setStyleSheet(
            "font-size: 16px; font-weight: 600; padding: 6px 0;"
        )
        box_layout.addWidget(self.auth_status)

        box_form = QFormLayout()

        self.client_id = QLineEdit()

        self.client_secret = QLineEdit()
        self.client_secret.setEchoMode(QLineEdit.Password)

        self.redirect_uri = QLineEdit()
        self.redirect_uri.setPlaceholderText("http://localhost:8080")

        self.box_developer_button = QPushButton("1. Open Box Developer Portal")
        self.box_developer_button.clicked.connect(
            lambda: QDesktopServices.openUrl(
                QUrl("https://developer.box.com/")
            )
        )

        developer_button_row = QHBoxLayout()
        developer_button_row.addWidget(self.box_developer_button)
        developer_button_row.addStretch()

        box_layout.addLayout(developer_button_row)

        box_form = QFormLayout()
        
        box_form.addRow("Client ID", self.client_id)
        box_form.addRow("Client Secret", self.client_secret)
        box_form.addRow("Redirect URI", self.redirect_uri)

        box_layout.addLayout(box_form)

        auth_buttons = QHBoxLayout()
        self.open_box_auth_button = QPushButton("2. Open Box Authentication")
        self.open_box_auth_button.clicked.connect(self._open_box_auth)

        self.disconnect_box_button = QPushButton("Disconnect")
        self.disconnect_box_button.clicked.connect(self._disconnect_box)
        self.disconnect_box_button.setEnabled(False)

        auth_buttons.addWidget(self.open_box_auth_button)
        auth_buttons.addWidget(self.disconnect_box_button)
        auth_buttons.addStretch()
        box_layout.addLayout(auth_buttons)

        redirect_help = QLabel(
            "After approving access in Box, copy the redirected localhost URL "
            "from the browser address bar and paste it below."
        )
        redirect_help.setWordWrap(True)
        box_layout.addWidget(redirect_help)

        redirect_row = QHBoxLayout()
        self.redirect_result = QLineEdit()
        self.redirect_result.setPlaceholderText(
            "Paste localhost redirect URL here, e.g. "
            "http://localhost:8080/?code=..."
        )

        self.complete_auth_button = QPushButton("3. Complete Authentication")
        self.complete_auth_button.clicked.connect(self._complete_box_auth)

        redirect_row.addWidget(self.redirect_result, 1)
        redirect_row.addWidget(self.complete_auth_button)
        box_layout.addLayout(redirect_row)

        layout.addWidget(box_group)

        # ------------------------------------------------------------------
        # Box Tokens
        # ------------------------------------------------------------------
        
        
        tokens_group = QGroupBox("Box Tokens")
        tokens_layout = QGridLayout(tokens_group)

        tokens_help = QLabel(
            "These tokens are generated after successful Box authentication. "
            "They can be copied into the Wild Basin website configuration."
        )
        tokens_help.setWordWrap(True)

        self.access_token_display = QLineEdit()
        self.access_token_display.setReadOnly(True)
        self.access_token_display.setPlaceholderText(
            "Authenticate with Box to generate an access token"
        )

        self.refresh_token_display = QLineEdit()
        self.refresh_token_display.setReadOnly(True)
        self.refresh_token_display.setPlaceholderText(
            "Authenticate with Box to generate a refresh token"
        )

        self.copy_access_token_button = QPushButton("Copy")
        self.copy_refresh_token_button = QPushButton("Copy")

        self.copy_access_token_button.clicked.connect(
            self._copy_access_token
        )
        self.copy_refresh_token_button.clicked.connect(
            self._copy_refresh_token
        )

        self.copy_access_token_button.setEnabled(False)
        self.copy_refresh_token_button.setEnabled(False)

        tokens_layout.addWidget(tokens_help, 0, 0, 1, 3)

        tokens_layout.addWidget(QLabel("Access Token"), 1, 0)
        tokens_layout.addWidget(self.access_token_display, 1, 1)
        tokens_layout.addWidget(self.copy_access_token_button, 1, 2)

        tokens_layout.addWidget(QLabel("Refresh Token"), 2, 0)
        tokens_layout.addWidget(self.refresh_token_display, 2, 1)
        tokens_layout.addWidget(self.copy_refresh_token_button, 2, 2)

        layout.addWidget(tokens_group)

        # ------------------------------------------------------------------
        # Box folder Input
        # ------------------------------------------------------------------
        folder_group = QGroupBox("Box Folder")
        folder_layout = QFormLayout(folder_group)

        self.folder = QLineEdit()
        self.folder.setPlaceholderText(
            "Folder ID or https://app.box.com/folder/123456789"
        )

        folder_help = QLabel(
            "Enter the Box folder containing the trail camera images. "
            "You can paste either the folder ID or the full Box folder URL."
        )
        folder_help.setWordWrap(True)

        folder_layout.addRow("Image Folder", self.folder)
        folder_layout.addRow("", folder_help)

        layout.addWidget(folder_group)

        # ------------------------------------------------------------------
        # Output files
        # ------------------------------------------------------------------
        outputs_group = QGroupBox("Output files")
        outputs = QGridLayout(outputs_group)

        self.output_dir = QLineEdit()
        browse = QPushButton("4. Browse…")
        browse.clicked.connect(self._browse_output)

        self.url_filename = QLineEdit("box-images.json")
        self.species_filename = QLineEdit("speciesnet-results.jsonl")
        self.ocr_filename = QLineEdit("paddle-ocr-results.jsonl")

        outputs.addWidget(QLabel("Results folder"), 0, 0)
        outputs.addWidget(self.output_dir, 0, 1)
        outputs.addWidget(browse, 0, 2)
        outputs.addWidget(QLabel("Image URL file"), 1, 0)
        outputs.addWidget(self.url_filename, 1, 1, 1, 2)
        outputs.addWidget(QLabel("SpeciesNet results"), 2, 0)
        outputs.addWidget(self.species_filename, 2, 1, 1, 2)
        outputs.addWidget(QLabel("OCR results"), 3, 0)
        outputs.addWidget(self.ocr_filename, 3, 1, 1, 2)
        layout.addWidget(outputs_group)

        # ------------------------------------------------------------------
        # Processing
        # ------------------------------------------------------------------
        stages_group = QGroupBox("Processing")
        stages = QGridLayout(stages_group)

        self.run_urls = QCheckBox("Collect Box image records")
        self.run_species = QCheckBox("Run SpeciesNet")
        self.run_ocr = QCheckBox("Run PaddleOCR")
        self.run_urls.setChecked(True)
        self.run_species.setChecked(True)
        self.run_ocr.setChecked(True)

        self.species_batch = QSpinBox()
        self.species_batch.setRange(1, 4096)
        self.species_batch.setValue(128)

        self.ocr_batch = QSpinBox()
        self.ocr_batch.setRange(1, 4096)
        self.ocr_batch.setValue(128)

        self.download_workers = QSpinBox()
        self.download_workers.setRange(1, 256)
        self.download_workers.setValue(32)

        self.crop_percent = QDoubleSpinBox()
        self.crop_percent.setRange(0.001, 1.0)
        self.crop_percent.setDecimals(3)
        self.crop_percent.setSingleStep(0.005)
        self.crop_percent.setValue(0.065)

        stages.addWidget(self.run_urls, 0, 0, 1, 2)
        stages.addWidget(self.run_species, 1, 0, 1, 2)
        stages.addWidget(self.run_ocr, 2, 0, 1, 2)
        stages.addWidget(QLabel("SpeciesNet batch size"), 0, 2)
        stages.addWidget(self.species_batch, 0, 3)
        stages.addWidget(QLabel("OCR batch size"), 1, 2)
        stages.addWidget(self.ocr_batch, 1, 3)
        stages.addWidget(QLabel("Download workers"), 2, 2)
        stages.addWidget(self.download_workers, 2, 3)
        stages.addWidget(QLabel("OCR bottom crop"), 3, 2)
        stages.addWidget(self.crop_percent, 3, 3)

        layout.addWidget(stages_group)

        controls = QHBoxLayout()
        self.start_button = QPushButton("5. Start Processing")
        self.cancel_button = QPushButton("Cancel")
        self.open_folder_button = QPushButton("Open Results Folder")
        self.cancel_button.setEnabled(False)

        self.start_button.clicked.connect(self._start)
        self.cancel_button.clicked.connect(self.runner.cancel)
        self.open_folder_button.clicked.connect(self._open_results_folder)

        controls.addWidget(self.start_button)
        controls.addWidget(self.cancel_button)
        controls.addStretch()
        controls.addWidget(self.open_folder_button)
        layout.addLayout(controls)

        self.status = QLabel("Ready")
        layout.addWidget(self.status)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("Pipeline output will appear here…")
        layout.addWidget(self.log, 1)

        self.setCentralWidget(root)

    # ----------------------------------------------------------------------
    # Box OAuth
    # ----------------------------------------------------------------------
    def _set_auth_required(self):
        self.auth_status.setText("⚠ Need Proper Authentication")
        self.auth_status.setStyleSheet(
            "font-size: 16px; "
            "font-weight: 600; "
            "color: #b45309; "
            "padding: 6px 0;"
        )

        self.disconnect_box_button.setEnabled(False)

        if hasattr(self, "access_token_display"):
            self.access_token_display.clear()
            self.refresh_token_display.clear()

            self.copy_access_token_button.setEnabled(False)
            self.copy_refresh_token_button.setEnabled(False)

    def _set_authenticated(self, auth: BoxAuthResult):
        details = f"Welcome, {auth.user_name}"

        if auth.user_login:
            details += f" ({auth.user_login})"

        self.auth_status.setText(f"✓ {details}")
        self.auth_status.setStyleSheet(
            "font-size: 16px; "
            "font-weight: 600; "
            "color: #15803d; "
            "padding: 6px 0;"
        )

        self.disconnect_box_button.setEnabled(True)

        # Show generated Box tokens.
        self.access_token_display.setText(auth.access_token)
        self.refresh_token_display.setText(auth.refresh_token)

        self.copy_access_token_button.setEnabled(
            bool(auth.access_token)
        )
        self.copy_refresh_token_button.setEnabled(
            bool(auth.refresh_token)
        )

    def _open_box_auth(self):
        try:
            client_id = self.client_id.text().strip()
            client_secret = self.client_secret.text()
            redirect_uri = self.redirect_uri.text().strip()

            if not client_secret:
                raise ValueError("Client Secret is required.")

            auth_url = build_authorize_url(client_id, redirect_uri)

            self._save_non_secret_settings()

            opened = QDesktopServices.openUrl(QUrl(auth_url))
            if not opened:
                raise RuntimeError(
                    "The browser could not be opened automatically."
                )

            self.status.setText(
                "Box authentication opened in your browser. "
                "Approve access, then paste the localhost redirect URL."
            )
            self.redirect_result.setFocus()

        except Exception as exc:
            self._set_auth_required()
            QMessageBox.critical(self, "Box Authentication", str(exc))

    def _complete_box_auth(self):
        try:
            self.status.setText("Verifying Box authentication…")
            self.complete_auth_button.setEnabled(False)

            auth = complete_oauth(
                client_id=self.client_id.text().strip(),
                client_secret=self.client_secret.text(),
                redirect_uri=self.redirect_uri.text().strip(),
                redirected_url_or_code=self.redirect_result.text(),
            )

            self.box_auth = auth
            self._set_authenticated(auth)
            self.redirect_result.clear()
            self.status.setText("Box authentication successful.")

            QMessageBox.information(
                self,
                "Box Authentication",
                f"Successfully authenticated with Box as {auth.user_name}.",
            )

        except Exception as exc:
            self.box_auth = None
            self._set_auth_required()
            self.status.setText("Box authentication failed.")
            QMessageBox.critical(self, "Box Authentication Failed", str(exc))

        finally:
            self.complete_auth_button.setEnabled(True)

    def _disconnect_box(self):
        self.box_auth = None
        self.redirect_result.clear()

        self.access_token_display.clear()
        self.refresh_token_display.clear()

        self.copy_access_token_button.setEnabled(False)
        self.copy_refresh_token_button.setEnabled(False)

        self._set_auth_required()
        self.status.setText("Disconnected from Box.")

    # ----------------------------------------------------------------------
    # Pipeline
    # ----------------------------------------------------------------------
    def _connect_runner(self):
        self.runner.log.connect(self._append_log)
        self.runner.stage_started.connect(
            lambda name: self.status.setText(f"Running: {name}")
        )
        self.runner.pipeline_finished.connect(self._pipeline_finished)
        self.runner.running_changed.connect(self._running_changed)
        self.runner.tokens_refreshed.connect(
            self._box_tokens_refreshed
        )


    def _box_tokens_refreshed(
        self,
        access_token: str,
        refresh_token: str,
    ):
        if self.box_auth is None:
            return

        self.box_auth = BoxAuthResult(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=self.box_auth.expires_in,
            user_id=self.box_auth.user_id,
            user_name=self.box_auth.user_name,
            user_login=self.box_auth.user_login,
        )

        self.access_token_display.setText(access_token)
        self.refresh_token_display.setText(refresh_token)

        self.copy_access_token_button.setEnabled(
            bool(access_token)
        )
        self.copy_refresh_token_button.setEnabled(
            bool(refresh_token)
        )

        self.status.setText(
            "Box authentication tokens were automatically refreshed."
        )

    def _default_output_dir(self) -> str:
        documents = QStandardPaths.writableLocation(
            QStandardPaths.DocumentsLocation
        )
        return str(Path(documents) / "Wild Basin Results")

    def _load_settings(self):
        self.client_id.clear()

        self.redirect_uri.setText("http://localhost:8080")

        self.folder.clear()

        self.output_dir.setText(self._default_output_dir())

    def _save_non_secret_settings(self):
        # Deliberately do not persist OAuth credentials or user-specific paths.
        # Each launch starts with fresh values.
        pass

    def _browse_output(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "Choose results folder",
            self.output_dir.text() or self._default_output_dir(),
        )
        if folder:
            self.output_dir.setText(folder)

    def _build_config(self) -> PipelineConfig:
        if self.box_auth is None:
            raise ValueError(
                "Need Proper Authentication. Authenticate with Box before "
                "starting the pipeline."
            )

        folder_id = normalize_box_folder_id(self.folder.text())
        output_dir = Path(self.output_dir.text().strip()).expanduser()

        if not self.client_id.text().strip():
            raise ValueError("Client ID is required.")
        if not self.client_secret.text():
            raise ValueError("Client Secret is required.")
        if not self.url_filename.text().strip():
            raise ValueError("Image URL filename is required.")
        if not self.species_filename.text().strip():
            raise ValueError("SpeciesNet filename is required.")
        if not self.ocr_filename.text().strip():
            raise ValueError("OCR filename is required.")

        return PipelineConfig(
            client_id=self.client_id.text().strip(),
            client_secret=self.client_secret.text(),
            access_token=self.box_auth.access_token,
            refresh_token=self.box_auth.refresh_token,
            folder_id=folder_id,
            output_dir=output_dir,
            url_filename=self.url_filename.text().strip(),
            speciesnet_filename=self.species_filename.text().strip(),
            ocr_filename=self.ocr_filename.text().strip(),
            run_url_scan=self.run_urls.isChecked(),
            run_speciesnet=self.run_species.isChecked(),
            run_ocr=self.run_ocr.isChecked(),
            speciesnet_batch_size=self.species_batch.value(),
            ocr_batch_size=self.ocr_batch.value(),
            download_workers=self.download_workers.value(),
            ocr_crop_percent=self.crop_percent.value(),
        )

    def _start(self):
        try:
            config = self._build_config()
            self._save_non_secret_settings()
            self.log.clear()
            self.runner.start(config)
        except Exception as exc:
            QMessageBox.critical(self, "Cannot start", str(exc))

    def _running_changed(self, running: bool):
        self.start_button.setEnabled(not running)
        self.cancel_button.setEnabled(running)
        self.open_box_auth_button.setEnabled(not running)
        self.complete_auth_button.setEnabled(not running)

    def _pipeline_finished(self, success: bool):
        self.status.setText("Completed" if success else "Stopped / failed")
        if success:
            self._append_log("\nResults are in: " + self.output_dir.text())

    def _append_log(self, text: str):
        self.log.appendPlainText(text)
        bar = self.log.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _open_results_folder(self):
        path = Path(self.output_dir.text().strip()).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _copy_access_token(self):
        token = self.access_token_display.text()

        if not token:
            return

        QApplication.clipboard().setText(token)
        self.status.setText("Access token copied to clipboard.")


    def _copy_refresh_token(self):
        token = self.refresh_token_display.text()

        if not token:
            return

        QApplication.clipboard().setText(token)
        self.status.setText("Refresh token copied to clipboard.")
