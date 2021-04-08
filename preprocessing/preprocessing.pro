#TEMPLATE = app
#CONFIG += console c++11
#CONFIG -= app_bundle
#CONFIG -= qt

##opencv3.4.1
#INCLUDEPATH   += /opt/hmi_depends/opencv3.4.1/include
#INCLUDEPATH   += /opt/hmi_depends/opencv3.4.1/include/opencv
#INCLUDEPATH   += /opt/hmi_depends/opencv3.4.1/include/opencv2

##libs
#LIBS += /opt/hmi_depends/opencv3.4.1/libs/libopencv*.so.3.4.1

QT -= gui

CONFIG += c++11 console
CONFIG -= app_bundle

# The following define makes your compiler emit warnings if you use
# any Qt feature that has been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

INCLUDEPATH   += /opt/hmi_depends/opencv3.4.1/include
INCLUDEPATH   += /opt/hmi_depends/opencv3.4.1/include/opencv
INCLUDEPATH   += /opt/hmi_depends/opencv3.4.1/include/opencv2
#LIBS += -L/opt/hmi_depends/opencv3.4.1/libs
#LIBS += -lopencv_highgui
#LIBS += -lopencv_core
#LIBS += -lopencv_imgproc
#LIBS += -lopencv_imgcodecs
LIBS += /opt/hmi_depends/opencv3.4.1/libs/libopencv*.so.3.4.1

#INCLUDEPATH += /usr/local/include \
#               /usr/local/include/opencv \
#               /usr/local/include/opencv2 \

#LIBS += /usr/local/lib/libopencv*.so.2.4.9

INCLUDEPATH += /opt/hmi_depends/algorithm_depends/include
#INCLUDEPATH += /opt/hmi_depends/boost/include



LIBS += -L/opt/hmi_depends/algorithm_depends/libs/cln/ -lcln
LIBS += -L/opt/hmi_depends/algorithm_depends/libs/ginac/ -lginac

LIBS += /opt/hmi_depends/algorithm_depends/libs/armadillo/libopenblas.a
LIBS += /opt/hmi_depends/algorithm_depends/libs/armadillo/liblapack.a
LIBS += -L/opt/hmi_depends/algorithm_depends/libs/armadillo/ -lsuperlu
LIBS += -L/opt/hmi_depends/algorithm_depends/libs/armadillo/ -larmadillo
LIBS += -L/opt/hmi_depends/algorithm_depends/libs/glog
LIBS += -L/opt/hmi_depends/algorithm_depends/libs/gflags
LIBS += -L/opt/hmi_depends/algorithm_depends/libs/mlpack
LIBS += -L/opt/hmi_depends/algorithm_depends/libs/cholmod
LIBS += -L/opt/hmi_depends/algorithm_depends/libs/ceres
LIBS += -L/opt/hmi_depends/algorithm_depends/libs/cgal
LIBS += -L/opt/hmi_depends/algorithm_depends/libs/gsl

LIBS += -lglog
LIBS += -lgflags
LIBS += -lmlpack
LIBS += -lcholmod
LIBS += -lpthread
LIBS += -lceres
LIBS += -lCGAL_Core
LIBS += -lCGAL
LIBS += -lgmp
LIBS += -lgomp
LIBS += -lgsl
LIBS += -lgslcblas
LIBS += -lm

LIBS += /opt/hmi_depends/algorithm_depends/libs/boost/libboost_*.so.1.58.0
#LIBS += /opt/hmi_depends/algorithm_depends/libs/boost/libboost_*.so.1.69.0

QMAKE_CXXFLAGS += -fopenmp

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

SOURCES += main.cpp \
    preprocess.cpp \
    imagescreen.cpp \
    NumeralCalculations.cpp

HEADERS += \
    preprocess.h \
    imagescreen.h \
    NumeralCalculations.h \
    headerall.h
