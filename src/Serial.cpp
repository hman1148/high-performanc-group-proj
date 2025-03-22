//
// Created by Joe Brijs on 3/21/25.
//

struct Point {
  double x, y;     // coordinates
  int cluster;     // no default cluster
  double minDist;  // default infinite dist to nearest cluster

  Point() :
      x(0.0),
      y(0.0),
      cluster(-1),
      minDist(__DBL_MAX__) {}

  Point(double x, double y) :
      x(x),
      y(y),
      cluster(-1),
      minDist(__DBL_MAX__) {}

  double distance(Point p) {
    return (p.x - x) * (p.x - x) + (p.y - y) * (p.y - y);
  }
};