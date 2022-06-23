from PyQt5 import QtWidgets, uic
# from PyQt5 import QtGui
import sys
# import os


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        # Load the UI Page
        uic.loadUi('GroupIR_Layout.ui', self)
        self.pushButton.clicked.connect(self.pushButtonClicked)
        self.actionExit_2.triggered.connect(self.closeApplication)

        self.radioButton.setChecked(True)  # Wasvani is selected by default
        print(dir(self))

    def pushButtonClicked(self):
        query_text = self.plainTextEdit.toPlainText()
        selected_dataset = "None"

        if self.radioButton.isChecked():
            selected_dataset = "wasvani"
        elif self.radioButton_2.isChecked():
            selected_dataset = "mmarco"
        elif self.radioButton_3.isChecked():
            selected_dataset = "cord19"

        print("Query text: "+query_text)
        print("Selected dataset: "+selected_dataset)
        print("Button clicked...")
        text_ = "1-deneme\n2-1234\n3-5678"
        self.label_2.setText(text_)

    def closeApplication(self):
        QtWidgets.QApplication.quit()


app = QtWidgets.QApplication(sys.argv)
main = MainWindow()
main.show()
sys.exit(app.exec_())
