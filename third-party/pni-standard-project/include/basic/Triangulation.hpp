#pragma once

#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <algorithm>
#include <cmath>

namespace openpni::basic
{
  typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
  typedef CGAL::Delaunay_triangulation_2<K> Delaunay_triangulation;
  typedef K::Point_2 Point_2;

  template <typename T>
  struct Triangle
  {
    typedef T value_type;

    Point_2 p1, p2, p3;
    T v1, v2, v3;

    template <typename TT = T>
    auto Area() const -> decltype(T(0) + TT(0))
    {
      using ReturnType = decltype(T(0) + TT(0));
      return std::abs(static_cast<ReturnType>(p1.x()) * (static_cast<ReturnType>(p2.y()) - static_cast<ReturnType>(p3.y())) +
                      static_cast<ReturnType>(p2.x()) * (static_cast<ReturnType>(p3.y()) - static_cast<ReturnType>(p1.y())) +
                      static_cast<ReturnType>(p3.x()) * (static_cast<ReturnType>(p1.y()) - static_cast<ReturnType>(p2.y()))) /
             ReturnType(2);
    }

    template <typename TT = T>
    auto BarycentricInterpolationValue(
        const Point_2 &P) const -> decltype(v1 + TT(0))
    {
      using ReturnType = decltype(v1 + TT(0));

      auto totalArea = Area<TT>();
      auto area1 = Triangle{P, p2, p3, TT(0), TT(0), TT(0)}.template Area<TT>();
      auto area2 = Triangle{p1, P, p3, TT(0), TT(0), TT(0)}.template Area<TT>();
      auto area3 = Triangle{p1, p2, P, TT(0), TT(0), TT(0)}.template Area<TT>();

      auto alpha = area1 / totalArea;
      auto beta = area2 / totalArea;
      auto gamma = area3 / totalArea;

      return static_cast<ReturnType>(alpha * v1 + beta * v2 + gamma * v3);
    }

    template <typename TT = T>
    bool IsInside(
        const Point_2 &P) const
    {
      auto totalArea = Area<TT>();
      auto area1 = Triangle{P, p2, p3, TT(0), TT(0), TT(0)}.template Area<TT>();
      auto area2 = Triangle{p1, P, p3, TT(0), TT(0), TT(0)}.template Area<TT>();
      auto area3 = Triangle{p1, p2, P, TT(0), TT(0), TT(0)}.template Area<TT>();

      using CompareType = decltype(totalArea);
      return std::abs(totalArea - (area1 + area2 + area3)) < CompareType(1e-3);
    }

    template <typename TT = T>
    auto minX() const -> decltype(T(0) + TT(0))
    {
      using ReturnType = decltype(T(0) + TT(0));
      return std::min({static_cast<ReturnType>(p1.x()), static_cast<ReturnType>(p2.x()), static_cast<ReturnType>(p3.x())});
    }

    template <typename TT>
    Triangle<T> &setValues(
        const TT &val1, const TT &val2, const TT &val3)
    {
      v1 = val1;
      v2 = val2;
      v3 = val3;
      return *this;
    }

    template <typename TT = T>
    auto maxValue() const -> decltype(v1 + TT(0))
    {
      return std::max({v1, v2, v3});
    }

    template <typename TT = T>
    auto minValue() const -> decltype(v1 + TT(0))
    {
      return std::min({v1, v2, v3});
    }
  };

  template <typename T>
  static Triangle<T> make_triangle(
      Point_2 p1, Point_2 p2, Point_2 p3, T v1, T v2, T v3)
  {
    return Triangle<T>{p1, p2, p3, v1, v2, v3};
  }

  template <typename TT, typename T>
  static Triangle<TT> make_triangle(
      const Triangle<T> &tri)
  {
    if constexpr (std::is_same_v<T, TT>)
      return tri;

    return Triangle<TT>{tri.p1, tri.p2, tri.p3, static_cast<TT>(tri.v1), static_cast<TT>(tri.v2), static_cast<TT>(tri.v3)};
  }

  template <typename T>
  std::unique_ptr<Triangle<T>[]> GetDelaunayTriangles(
      const std::unique_ptr<Point_2[]> &points, std::size_t pointCount, std::size_t &triangleCount)
  {
    Delaunay_triangulation dt;
    for (std::size_t i = 0; i < pointCount; ++i)
      dt.insert(points[i]);
    triangleCount = 0;
    for (auto face = dt.finite_faces_begin(); face != dt.finite_faces_end(); ++face)
      ++triangleCount;
    auto triangles = std::make_unique<Triangle<T>[]>(triangleCount);
    std::size_t index = 0;
    for (auto face = dt.finite_faces_begin(); face != dt.finite_faces_end(); ++face)
    {
      Point_2 p1 = face->vertex(0)->point();
      Point_2 p2 = face->vertex(1)->point();
      Point_2 p3 = face->vertex(2)->point();
      triangles[index] = Triangle<T>{p1, p2, p3, T(0), T(0), T(0)};
      ++index;
    }
    return triangles;
  }

} // namespace openpni::basic
