#include "define.h"

MichDefine E180() {
  MichDefine mich;
  auto &polygon = mich.polygon;
  polygon.edges = 24;
  polygon.detectorPerEdge = 1;
  polygon.detectorLen = 0;
  polygon.radius = 106.5;
  polygon.angleOf1stPerp = 0;
  polygon.detectorRings = 2;
  polygon.ringDistance = 26.5 * 4 + 2;
  auto &detector = mich.detector;
  detector.blockNumU = 4;
  detector.blockNumV = 1;
  detector.blockSizeU = 26.5;
  detector.blockSizeV = 26.5;
  detector.crystalNumU = 13;
  detector.crystalNumV = 13;
  detector.crystalSizeU = 2.0;
  detector.crystalSizeV = 2.0;
  return mich;
}

MichDefine D80() {
  MichDefine mich;
  auto &polygon = mich.polygon;
  polygon.edges = 12;
  polygon.detectorPerEdge = 1;
  polygon.detectorLen = 0;
  polygon.radius = 52.9;
  polygon.angleOf1stPerp = 0;
  polygon.detectorRings = 1;
  polygon.ringDistance = 26.5 * 4 + 2;
  auto &detector = mich.detector;
  detector.blockNumU = 4;
  detector.blockNumV = 1;
  detector.blockSizeU = 26.5;
  detector.blockSizeV = 26.5;
  detector.crystalNumU = 13;
  detector.crystalNumV = 13;
  detector.crystalSizeU = 2.0;
  detector.crystalSizeV = 2.0;
  return mich;
}
