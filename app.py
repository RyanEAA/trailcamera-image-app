import sys
from PySide6.QtWidgets import QApplication
from wildbasin_app.gui.main_window import MainWindow

def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Wild Basin Image Processor")
    app.setOrganizationName("Wild Basin Wilderness Preserve")

    window = MainWindow()
    window.show()
    return app.exec()

if __name__ == "__main__":
    raise SystemExit(main())
