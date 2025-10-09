import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QFileDialog, QSpinBox
)
from analyze import search

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Signal Analyzer")
        self.resize(400, 250)

        layout = QVBoxLayout()

        self.message_label = QLabel("")
        self.message_label.setStyleSheet("color: red;")
        self.message_label.hide()
        layout.addWidget(self.message_label)

        # --- вибір файлу ---
        file_layout = QHBoxLayout()
        self.file_label = QLabel("Файл: не вибрано")
        file_btn = QPushButton("Вибрати файл")
        file_btn.clicked.connect(self.choose_file)
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(file_btn)
        layout.addLayout(file_layout)

        # --- початок ---
        start_layout = QHBoxLayout()
        start_layout.addWidget(QLabel("Початок (сек):"))
        self.start_input = QLineEdit()
        self.start_input.setText("0.0")
        start_layout.addWidget(self.start_input)
        layout.addLayout(start_layout)

        # --- тривалість ---
        duration_layout = QHBoxLayout()
        duration_layout.addWidget(QLabel("Тривалість (сек):"))
        self.duration_input = QLineEdit()
        self.duration_input.setText("5.0")
        duration_layout.addWidget(self.duration_input)
        layout.addLayout(duration_layout)

        band_layout = QHBoxLayout()
        band_layout.addWidget(QLabel("Діапазон пошуку піку (Гц):"))

        self.band_min = QSpinBox()
        self.band_min.setRange(1, 500)
        self.band_min.setValue(1)  # початкове значення
        band_layout.addWidget(self.band_min)

        self.band_max = QSpinBox()
        self.band_max.setRange(1, 500)
        self.band_max.setValue(200)
        band_layout.addWidget(self.band_max)

        layout.addLayout(band_layout)

        # --- к-сть періодів ---
        periods_layout = QHBoxLayout()
        periods_layout.addWidget(QLabel("Мінімальна к-сть періодів:"))
        self.periods_input = QLineEdit()
        self.periods_input.setText("10")
        periods_layout.addWidget(self.periods_input)
        layout.addLayout(periods_layout)

        # --- дискретність ---
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("У скільки разів пік має бути вищим за середнє:"))
        self.threshold_input = QLineEdit()
        self.threshold_input.setText("5")
        threshold_layout.addWidget(self.threshold_input)
        layout.addLayout(threshold_layout)

        # --- кнопка запуску ---
        run_btn = QPushButton("Запустити аналіз")
        run_btn.clicked.connect(self.search_gui)
        layout.addWidget(run_btn)

        self.setLayout(layout)

    def choose_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Вибери аудіофайл", "", "Audio Files (*.wav *.mp3 *.flac)"
        )
        if file_path:
            self.file_label.setText(file_path)

    def search_gui(self):
        self.message_label.hide()
        file = self.file_label.text()
        start = float(self.start_input.text())
        duration = float(self.duration_input.text())
        band = (int(self.band_min.value()), int(self.band_max.value()))
        period = int(self.periods_input.text())
        threshold = int(self.threshold_input.text())
        if not file or "не вибрано" in file:
            self.message_label.setStyleSheet("color: red;")
            self.message_label.setText("Спочатку виберіть файл!")
            self.message_label.show()
            return
        try:
            search(file, start, duration, band, period, threshold)
            self.message_label.setStyleSheet("color: green;")
            self.message_label.setText("Аналіз виконано успішно!")
            self.message_label.show()
        except Exception as e:
            self.message_label.setStyleSheet("color: red;")
            self.message_label.setText(f"Помилка: {str(e)}")
            self.message_label.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
