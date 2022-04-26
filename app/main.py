from PyQt5 import QtWidgets, QtCore, QtGui
import design
import sys
from author_attribution import process, preprocess
from PyQt5.QtCore import QThread, pyqtSignal

class Thread(QThread):
    _signal = pyqtSignal(str)

    def __init__(self):
        super(Thread, self).__init__()

    def __del__(self):
        self.wait()

    def run(self):
        window.progress_bar.show()
        window.result_label.show()
        test_text = window.text_input.toPlainText()
        df, authors = preprocess.create_dataset(window.authors_dir, app=window, thread=self)
        result = authors[process.predict(df, test_text, thread=self)]
        window.result_label.setText("Результат: " + result)
        window.progress_bar.hide()

class App(QtWidgets.QMainWindow, design.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.pick_file_btn.clicked.connect(self.pick_folder)
        self.go_btn.clicked.connect(self.go)
        self.authors_dir = ''
        self.progress_bar.hide()
        self.result_label.hide()

    def pick_folder(self, event):
        dir_name = QtWidgets.QFileDialog.getExistingDirectory(self, 'Open file', '', 
        QtWidgets.QFileDialog.ShowDirsOnly)
        if dir_name != '':
            self.pick_file_btn.setText(dir_name)
            self.authors_dir = dir_name

    def go(self, event):
        if self.authors_dir != '':
            self.thread = Thread()
            self.thread._signal.connect(self.signal_accept)
            self.thread.start()

    def set_progress(self, progress, msg=''):
        self.progress_bar.setValue(progress)
        self.result_label.setText(msg)

    def signal_accept(self, msg):
        if msg.isdigit():
            self.progress_bar.setValue(int(msg))
        else:
            self.result_label.setText(str(msg))


app = QtWidgets.QApplication(sys.argv)
window = App()
window.show()
app.exec_()

