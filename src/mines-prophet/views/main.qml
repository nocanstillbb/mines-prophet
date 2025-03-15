import QtQml 2.15
import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import QtQuick.Window 2.15
import QtQuick.Controls 1.4 as Q1
import QtQuick.Controls.Styles 1.4
import QtQml.Models 2.12
import Qt.labs.platform 1.1 as QtPlatform
import QtQuick.Dialogs 1.3

import prismCpp 1.0
import prism_qt_ui 1.0
import viewmodels 1.0

BorderlessWindow_mac{
    //ApplicationWindow {
    width: 480
    height: 640
    visible: true
    title: qsTr("minsweeper")
    ColumnLayout{
        anchors.fill: parent
        spacing: 0
        Rectangle{
            color: "lightblue"
            Layout.preferredHeight: 30
            Layout.fillWidth: true
        }
        Item {
            Layout.fillWidth: true
            Layout.fillHeight: true
            LiveLoader{
                source: CppUtility.transUrl("qrc:/mines-prophet/views/page1.qml")
            }
        }
    }
}
