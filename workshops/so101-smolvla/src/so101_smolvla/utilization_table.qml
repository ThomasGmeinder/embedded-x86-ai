// The utilization and power table as one Qt Quick scene: three rows (CPU, iGPU, NPU), two columns
// (utilization, power). No shader effect, so Qt Quick's graphics-processor renderer and its software
// renderer draw the same picture. Look as approved from the mock-up of 2026-11-22.
//
// ``rows`` holds per row: label, colour, percent (null when missing), power (null when missing),
// columns (the values the segments show; one for CPU and iGPU, eight for the NPU), reason_util and
// reason_power (why a value is missing, shown while the mouse rests on the cell).
import QtQuick
import QtQuick.Window

Window {
    id: win
    property string caption: ""
    property var rows: []
    readonly property int pad: 20            // room for the shadow around the card
    readonly property color bright: Qt.rgba(244 / 255, 244 / 255, 248 / 255, 1)
    readonly property color dim: Qt.rgba(1, 1, 1, 110 / 255)
    readonly property string face: "Ubuntu Sans"

    width: 320 + 2 * pad
    height: 160 + 2 * pad
    visible: false
    color: "transparent"
    flags: Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.X11BypassWindowManagerHint | Qt.WindowDoesNotAcceptFocus

    function percentText(value) { return value === null || value === undefined ? "n/a" : Math.round(value) + " %" }
    function powerText(value) { return value === null || value === undefined ? "n/a" : value.toFixed(1) + " W" }

    Repeater {                               // soft shadow: stacked translucent rounded rectangles
        model: 14
        Rectangle {
            required property int index
            readonly property int grow: 14 - index
            x: win.pad - grow
            y: win.pad - grow + 5
            width: 320 + 2 * grow
            height: 160 + 2 * grow
            radius: 16 + grow
            color: Qt.rgba(0, 0, 0, 5 / 255)
        }
    }

    Rectangle {
        id: card
        x: win.pad
        y: win.pad
        width: 320
        height: 160
        radius: 16
        antialiasing: true
        color: Qt.rgba(20 / 255, 21 / 255, 28 / 255, 226 / 255)
        border.color: Qt.rgba(1, 1, 1, 26 / 255)
        border.width: 1

        MouseArea {                          // a drag anywhere on the card moves the window
            anchors.fill: parent
            property point pressedAt
            property point windowAt
            onPressed: (mouse) => {
                pressedAt = mapToGlobal(mouse.x, mouse.y)
                windowAt = Qt.point(win.x, win.y)
            }
            onPositionChanged: (mouse) => {
                const now = mapToGlobal(mouse.x, mouse.y)
                win.x = windowAt.x + now.x - pressedAt.x
                win.y = windowAt.y + now.y - pressedAt.y
            }
        }

        Text {
            objectName: "caption"
            x: 18; y: 11; height: 16
            verticalAlignment: Text.AlignVCenter
            text: win.caption
            color: win.dim
            font { family: win.face; pixelSize: 10; weight: Font.Medium; letterSpacing: 0.3 }
        }
        Text {
            objectName: "close"
            x: 288; y: 9; width: 20; height: 20
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
            text: "×"
            color: win.dim
            font { family: win.face; pixelSize: 14 }
            MouseArea { anchors.fill: parent; onClicked: Qt.quit() }
        }
        Text {
            x: 66; y: 36; width: 172; height: 12
            horizontalAlignment: Text.AlignRight
            verticalAlignment: Text.AlignVCenter
            text: "UTILIZATION"
            color: win.dim
            font { family: win.face; pixelSize: 8; weight: Font.DemiBold; letterSpacing: 0.9 }
        }
        Text {
            x: 244; y: 36; width: 58; height: 12
            horizontalAlignment: Text.AlignRight
            verticalAlignment: Text.AlignVCenter
            text: "POWER"
            color: win.dim
            font { family: win.face; pixelSize: 8; weight: Font.DemiBold; letterSpacing: 0.9 }
        }

        Repeater {
            model: win.rows.length
            Item {
                id: line
                required property int index
                readonly property var row: win.rows[index]
                readonly property var columns: row.columns === null || row.columns === undefined ? [] : row.columns
                readonly property real gap: 2
                readonly property int pieces: Math.max(columns.length, 1)
                readonly property real piece: (118 - gap * (pieces - 1)) / pieces
                y: 56 + index * 32
                width: 320
                height: 24

                Text {
                    x: 18; height: 24
                    verticalAlignment: Text.AlignVCenter
                    text: line.row.label
                    color: win.bright
                    font { family: win.face; pixelSize: 13; weight: Font.DemiBold }
                }
                Repeater {
                    model: line.columns.length
                    Rectangle {
                        id: track
                        objectName: "segment_" + line.row.label + "_" + index
                        required property int index
                        x: 66 + index * (line.piece + line.gap)
                        y: 9
                        width: line.piece
                        height: 6
                        radius: 3
                        color: Qt.rgba(1, 1, 1, 24 / 255)
                        Rectangle {
                            width: track.width * line.columns[track.index] / 100
                            height: 6
                            radius: 3
                            gradient: Gradient {
                                orientation: Gradient.Horizontal
                                GradientStop { position: 0; color: Qt.darker(line.row.colour, 1.25) }
                                GradientStop { position: 1; color: line.row.colour }
                            }
                        }
                    }
                }
                Text {
                    id: utilCell
                    objectName: "util_" + line.row.label
                    x: 190; width: 48; height: 24
                    horizontalAlignment: Text.AlignRight
                    verticalAlignment: Text.AlignVCenter
                    text: win.percentText(line.row.percent)
                    color: win.bright
                    font { family: win.face; pixelSize: 13; weight: Font.Medium }
                    HoverHandler { id: utilHover }
                }
                Text {
                    id: powerCell
                    objectName: "power_" + line.row.label
                    x: 244; width: 58; height: 24
                    horizontalAlignment: Text.AlignRight
                    verticalAlignment: Text.AlignVCenter
                    text: win.powerText(line.row.power)
                    color: win.bright
                    font { family: win.face; pixelSize: 13; weight: Font.Medium }
                    HoverHandler { id: powerHover }
                }
                Rectangle {                  // the reason of a missing value, while the mouse rests on its cell
                    readonly property string reason: utilHover.hovered ? (line.row.reason_util || "")
                                                    : powerHover.hovered ? (line.row.reason_power || "") : ""
                    visible: reason !== ""
                    x: 18; y: -14; height: 18
                    width: reasonText.implicitWidth + 12
                    radius: 4
                    color: Qt.rgba(40 / 255, 42 / 255, 54 / 255, 1)
                    z: 10
                    Text {
                        id: reasonText
                        anchors.centerIn: parent
                        text: parent.reason
                        color: win.bright
                        font { family: win.face; pixelSize: 9 }
                    }
                }
            }
        }
    }
}
