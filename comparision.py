# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'comparision.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_comparision(object):
    def setupUi(self, comparision):
        comparision.setObjectName("comparision")
        comparision.resize(605, 464)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("icon2.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        comparision.setWindowIcon(icon)
        comparision.setStyleSheet("background-color: rgb(61, 61, 61);")
        self.gridLayoutWidget = QtWidgets.QWidget(comparision)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(10, 20, 581, 431))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.c4 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.c4.setStyleSheet("font: 18pt \"Calibri\";\n"
"color: rgb(255, 255, 255);")
        self.c4.setText("")
        self.c4.setIndent(5)
        self.c4.setObjectName("c4")
        self.gridLayout.addWidget(self.c4, 4, 1, 1, 2)
        self.c2 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.c2.setStyleSheet("font: 18pt \"Calibri\";\n"
"color: rgb(255, 255, 255);")
        self.c2.setText("")
        self.c2.setIndent(5)
        self.c2.setObjectName("c2")
        self.gridLayout.addWidget(self.c2, 2, 1, 1, 2)
        self.label_31 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_31.setText("")
        self.label_31.setObjectName("label_31")
        self.gridLayout.addWidget(self.label_31, 6, 1, 1, 1)
        self.label_32 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_32.setText("")
        self.label_32.setObjectName("label_32")
        self.gridLayout.addWidget(self.label_32, 6, 2, 1, 1)
        self.clas5 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.clas5.setStyleSheet("font: 18pt \"Times New Roman\";\n"
"color: rgb(85, 255, 255);")
        self.clas5.setText("")
        self.clas5.setAlignment(QtCore.Qt.AlignCenter)
        self.clas5.setObjectName("clas5")
        self.gridLayout.addWidget(self.clas5, 5, 4, 1, 1)
        self.a5 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.a5.setStyleSheet("font: 18pt \"Times New Roman\";\n"
"color: rgb(85, 255, 255);")
        self.a5.setText("")
        self.a5.setAlignment(QtCore.Qt.AlignCenter)
        self.a5.setObjectName("a5")
        self.gridLayout.addWidget(self.a5, 5, 3, 1, 1)
        self.c5 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.c5.setStyleSheet("font: 18pt \"Calibri\";\n"
"color: rgb(255, 255, 255);")
        self.c5.setText("")
        self.c5.setIndent(5)
        self.c5.setObjectName("c5")
        self.gridLayout.addWidget(self.c5, 5, 1, 1, 2)
        self.label_5 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_5.setStyleSheet("color: rgb(255, 255, 255);\n"
"font: italic 18pt \"Calibri\";")
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 0, 1, 1, 2)
        self.label_3 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_3.setStyleSheet("color: rgb(255, 255, 255);\n"
"font: italic 18pt \"Calibri\";")
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 0, 4, 1, 1)
        self.c1 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.c1.setStyleSheet("font: 18pt \"Calibri\";\n"
"color: rgb(255, 255, 255);")
        self.c1.setText("")
        self.c1.setIndent(5)
        self.c1.setObjectName("c1")
        self.gridLayout.addWidget(self.c1, 1, 1, 1, 2)
        self.clas1 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.clas1.setStyleSheet("font: 18pt \"Times New Roman\";\n"
"color: rgb(85, 255, 255);")
        self.clas1.setText("")
        self.clas1.setAlignment(QtCore.Qt.AlignCenter)
        self.clas1.setObjectName("clas1")
        self.gridLayout.addWidget(self.clas1, 1, 4, 1, 1)
        self.a1 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.a1.setStyleSheet("\n"
"font: 18pt \"Times New Roman\";\n"
"color: rgb(85, 255, 255);")
        self.a1.setText("")
        self.a1.setAlignment(QtCore.Qt.AlignCenter)
        self.a1.setObjectName("a1")
        self.gridLayout.addWidget(self.a1, 1, 3, 1, 1)
        self.a2 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.a2.setStyleSheet("font: 18pt \"Times New Roman\";\n"
"color: rgb(85, 255, 255);")
        self.a2.setText("")
        self.a2.setAlignment(QtCore.Qt.AlignCenter)
        self.a2.setObjectName("a2")
        self.gridLayout.addWidget(self.a2, 2, 3, 1, 1)
        self.clas2 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.clas2.setStyleSheet("font: 18pt \"Times New Roman\";\n"
"color: rgb(85, 255, 255);")
        self.clas2.setText("")
        self.clas2.setAlignment(QtCore.Qt.AlignCenter)
        self.clas2.setObjectName("clas2")
        self.gridLayout.addWidget(self.clas2, 2, 4, 1, 1)
        self.c3 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.c3.setStyleSheet("font: 18pt \"Calibri\";\n"
"color: rgb(255, 255, 255);")
        self.c3.setText("")
        self.c3.setIndent(5)
        self.c3.setObjectName("c3")
        self.gridLayout.addWidget(self.c3, 3, 1, 1, 2)
        self.clas3 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.clas3.setStyleSheet("font: 18pt \"Times New Roman\";\n"
"color: rgb(85, 255, 255);")
        self.clas3.setText("")
        self.clas3.setAlignment(QtCore.Qt.AlignCenter)
        self.clas3.setObjectName("clas3")
        self.gridLayout.addWidget(self.clas3, 3, 4, 1, 1)
        self.a3 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.a3.setStyleSheet("font: 18pt \"Times New Roman\";\n"
"color: rgb(85, 255, 255);")
        self.a3.setText("")
        self.a3.setAlignment(QtCore.Qt.AlignCenter)
        self.a3.setObjectName("a3")
        self.gridLayout.addWidget(self.a3, 3, 3, 1, 1)
        self.a4 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.a4.setStyleSheet("font: 18pt \"Times New Roman\";\n"
"color: rgb(85, 255, 255);")
        self.a4.setText("")
        self.a4.setAlignment(QtCore.Qt.AlignCenter)
        self.a4.setObjectName("a4")
        self.gridLayout.addWidget(self.a4, 4, 3, 1, 1)
        self.clas4 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.clas4.setStyleSheet("font: 18pt \"Times New Roman\";\n"
"color: rgb(85, 255, 255);")
        self.clas4.setText("")
        self.clas4.setAlignment(QtCore.Qt.AlignCenter)
        self.clas4.setObjectName("clas4")
        self.gridLayout.addWidget(self.clas4, 4, 4, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_4.setStyleSheet("color: rgb(255, 255, 255);\n"
"font: italic 18pt \"Calibri\";")
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 0, 3, 1, 1)

        self.retranslateUi(comparision)
        QtCore.QMetaObject.connectSlotsByName(comparision)

    def retranslateUi(self, comparision):
        _translate = QtCore.QCoreApplication.translate
        comparision.setWindowTitle(_translate("comparision", "Comparision Table"))
        self.label_5.setText(_translate("comparision", "Classifier"))
        self.label_3.setText(_translate("comparision", "Class"))
        self.label_4.setText(_translate("comparision", "Confidence Level"))


# if __name__ == "__main__":
#     import sys
#     app = QtWidgets.QApplication(sys.argv)
#     comparision = QtWidgets.QWidget()
#     ui = Ui_comparision()
#     ui.setupUi(comparision)
#     comparision.show()
#     sys.exit(app.exec_())
