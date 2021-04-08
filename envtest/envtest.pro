TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

# include
##algorithm lib + boost
#INCLUDEPATH += /opt/hmi_depends/algorithm_depends/include
#opencv3.4.1
INCLUDEPATH   += /opt/hmi_depends/opencv3.4.1/include
INCLUDEPATH   += /opt/hmi_depends/opencv3.4.1/include/opencv
INCLUDEPATH   += /opt/hmi_depends/opencv3.4.1/include/opencv2

#libs
LIBS += /opt/hmi_depends/opencv3.4.1/libs/libopencv*.so.3.4.1
#LIBS += /opt/hmi_depends/algorithm_depends/libs/boost/libboost_*.so.1.58.0

#LIBS += -L/opt/hmi_depends/algorithm_depends/libs/glog -lglog
#LIBS += -L/opt/hmi_depends/algorithm_depends/libs/gflags -lgflags
#LIBS += -L/opt/hmi_depends/algorithm_depends/libs/mlpack -lmlpack
#LIBS += -L/opt/hmi_depends/algorithm_depends/libs/cholmod -lcholmod
#LIBS += -L/opt/hmi_depends/algorithm_depends/libs/cln/ -lcln
#LIBS += -L/opt/hmi_depends/algorithm_depends/libs/ginac/ -lginac
#LIBS += -L/opt/hmi_depends/algorithm_depends/libs/cgal
#LIBS += -L/opt/hmi_depends/algorithm_depends/libs/gomp -lgomp
#LIBS += -L/opt/hmi_depends/algorithm_depends/libs/gsl/ -lgsl -lgslcblas

SOURCES += main.cpp
