#ifndef HEADERALL_H
#define HEADERALL_H
#include <opencv2/opencv.hpp>
#include <opencv2/ml/ml.hpp>
//#include <opencv2/nonfree/nonfree.hpp>
//#include <opencv2/legacy/legacy.hpp>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_statistics_double.h>
#include <gsl/gsl_statistics_int.h>
#include <gsl/gsl_statistics.h>
#include<ginac/ginac.h>
#include<boost/lexical_cast.hpp>
#include<QDebug>
#include<QStack>
#include<QVector>
#include <QTimer>
#include<QPoint>
//#include<QPainter>
#include<QDateTime>
//#include<QColor>
//#include<QImage>
//#include<QPainter>
#include<QRect>
#include<stdexcept>
#include<vector>
#include<numeric>
#include<cmath>
#include<QFile>
#include <fstream>
#include <math.h>
#include <stdlib.h>
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <stdio.h>
#include <boost/function.hpp>
#include <boost/thread/thread.hpp>
#include <boost/bind.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/function.hpp>
#include <boost/algorithm/minmax_element.hpp>
#include <omp.h>
#include <time.h>
#include <CGAL/Simple_cartesian.h>
#include <mlpack/core.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>
#include <mlpack/methods/linear_regression/linear_regression.hpp>
#include <mlpack/methods/logistic_regression/logistic_regression.hpp>
#include <mlpack/methods/pca/pca.hpp>
#include <mlpack/methods/softmax_regression/softmax_regression.hpp>
#include <mlpack/methods/adaboost/adaboost.hpp>
#include <mlpack/methods/naive_bayes/naive_bayes_classifier.hpp>
#include <mlpack/methods/mean_shift/mean_shift.hpp>
#include <mlpack/methods/kmeans/kmeans.hpp>
#include <mlpack/methods/kernel_pca/kernel_pca.hpp>
#include <armadillo>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polygon_2_algorithms.h>
//using namespace arma;
using namespace mlpack;
using namespace mlpack::neighbor; // NeighborSearch and NearestNeighborSort
using namespace mlpack::metric; // ManhattanDistance
using namespace mlpack::regression;
using namespace mlpack::pca;
using namespace mlpack::adaboost;
using namespace mlpack::naive_bayes;
using namespace mlpack::kmeans;
using namespace mlpack::kpca;
using namespace mlpack::meanshift;
using namespace cv;
using namespace std;
using namespace GiNaC;
using boost::lexical_cast;
#endif // HEADERALL_H
