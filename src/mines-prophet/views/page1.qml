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

Rectangle {
    anchors.fill: parent
    property var vm:GameVeiwmodel
    Item{
        Row{

            Q1.Button{
                text: "restart"
                onClicked: {
                    vm.regen()
                }
            }

            Q1.Button{
                id:btn_train
                text: "train_dqn"
                onClicked: {
                    JsEx.delay(btn_train,100,function(){
                        vm.trainDnq();
                    })
                }
            }
            Q1.Button{
                id:btn_saveData
                text: "save data"
                onClicked: {
                    vm.trainDnq();
                    vm.save_data_flag = true
                    JsEx.delay(btn_train,1000,function(){
                        vm.save_data_flag = false
                    })
                }
            }


        }


        anchors.fill: parent
        anchors.margins: 10
        GridLayout{
            id:layout
            Layout.preferredWidth: parent.width
            Layout.preferredHeight: parent.width*layout.rows/layout.columns
            anchors.centerIn: parent
            columnSpacing: 0
            rowSpacing: 0
            columns: vm.cols
            rows: vm.rows
            Repeater{
                model:vm.cells
                delegate: Rectangle{
                    Layout.preferredWidth: parent.Layout.preferredWidth/layout.columns
                    Layout.preferredHeight:  parent.Layout.preferredHeight/layout.rows
                    property var rvm: vm.cells.getRowData(index)
                    property int idx: index
                    border.color: "white"
                    border.width: 1
                    property int value: Bind.create(rvm,"visual_value")
                    color: {
                        var v = Bind.create(rvm,"visual_value")
                        if(vm.predictIndex === index)
                            return Style.blue40
                        //if(vm.clickedIndex === index)
                        //    return Style.red40

                        if(v === 10)
                            return "red"
                        else if(ma.hoveredIndex === idx)
                            return Style.gray20
                        else if(v === -1)
                            return Style.gray40
                        else
                            return Style.gray20
                    }
                    Text {
                        id:tb
                        anchors.fill: parent
                        verticalAlignment: Text.AlignVCenter
                        horizontalAlignment: Text.AlignHCenter
                        property int v: Bind.create(rvm,"visual_value")
                        color: {
                            var v = Bind.create(rvm,"visual_value")

                            if(v === 1)
                                return "#0100fe"
                            else if(v === 2)
                                return "#017f01"
                            else if(v === 3)
                                return "#fe0000"
                            else if(v === 4)
                                return "#010080"
                            else if(v === 5)
                                return "#810102"
                            else if(v === 6)
                                return "#008081"
                            else if(v === 7)
                                return "#000000"
                            else if(v === 8)
                                return "#808080"
                            else
                                return "black"
                        }
                        text:{
                            if(v === 9 || v===10)
                                return "B"
                            else if(v === -1 || v === 0)
                                return ""
                            else if(v === 11)
                                return "F"
                            else if(v ===12)
                                return "?"
                            else
                                return v
                        }
                        font.bold: true
                        font.pixelSize: width*2/3
                    }
                }
            }
        }
        MouseArea{
            id:ma
            property int hoveredIndex: -1
            anchors.fill: layout
            anchors.centerIn: parent
            hoverEnabled: true
            onPositionChanged: {
                if(pressed)
                {
                    var index = Math.floor(mouseY/(height/layout.rows)) *layout.columns + Math.floor(mouseX/ (width/layout.columns))
                    hoveredIndex = index
                }
                else
                {
                    hoveredIndex = -1
                }
            }
            onReleased: function(e) {
                var index = Math.floor(mouseY/(height/layout.rows)) *layout.columns + Math.floor(mouseX/ (width/layout.columns))
                if(e.button === Qt.LeftButton)
                {
                    vm.open(index);
                }
            }
        }
    }

}
