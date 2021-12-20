#include "trajectory_generator_waypoint.h"
#include <stdio.h>
#include <ros/ros.h>
#include <ros/console.h>
#include <iostream>
#include <fstream>
#include <string>

using namespace std;
using namespace Eigen;

TrajectoryGeneratorWaypoint::TrajectoryGeneratorWaypoint() {}
TrajectoryGeneratorWaypoint::~TrajectoryGeneratorWaypoint() {}

// define factorial function, input i, output i!
int TrajectoryGeneratorWaypoint::Factorial(int x) {
    int fac = 1;
    for (int i = x; i > 0; i--) fac = fac * i;
    return fac;
}
/*

    STEP 2: Learn the "Closed-form solution to minimum snap" in L5, then finish this PolyQPGeneration function

    variable declaration: input       const int d_order,                    // the order of derivative
                                      const Eigen::MatrixXd &Path,          // waypoints coordinates (3d)
                                      const Eigen::MatrixXd &Vel,           // boundary velocity
                                      const Eigen::MatrixXd &Acc,           // boundary acceleration
                                      const Eigen::VectorXd &Time)          // time allocation in each segment
                          output      MatrixXd PolyCoeff(m, 3 * p_num1d);   // position(x,y,z), so we need (3 * p_num1d)
   coefficients

*/

Eigen::MatrixXd TrajectoryGeneratorWaypoint::PolyQPGeneration(
    const int d_order,           // the order of derivative
    const Eigen::MatrixXd &Path, // waypoints coordinates (3d)
    const Eigen::MatrixXd &Vel,  // boundary velocity
    const Eigen::MatrixXd &Acc,  // boundary acceleration
    const Eigen::VectorXd &Time) // time allocation in each segment
{
    // enforce initial and final velocity and accleration, for higher order derivatives, just assume them be 0;
    int p_order = 2 * d_order - 1; // the order of polynomial
    int p_num1d = p_order + 1;     // the number of variables in each segment

    int n_seg = Time.size();                                 // the number of segments
    MatrixXd PolyCoeff = MatrixXd::Zero(3, n_seg * p_num1d); // position(x,y,z), so we need (3 * p_num1d) coefficients
    VectorXd Px(p_num1d * n_seg), Py(p_num1d * n_seg), Pz(p_num1d * n_seg);

    for (int dim = 0; dim < 3; dim++) { // x,y,z三维
        // 计算Q矩阵
        MatrixXd Q;
        for (int seg = 0; seg < n_seg; seg++) {
            MatrixXd Q_k = MatrixXd::Zero(p_num1d, p_num1d);
            for (int row = 0; row < p_num1d; row++) {
                int i = row;
                for (int col = 0; col < p_num1d; col++) {
                    int l = col;
                    if (i >= 4 && l >= 4) {
                        Q_k(row, col) = i * (i - 1) * (i - 2) * (i - 3) * l * (l - 1) * (l - 2) * (l - 3) /
                                        (i + l - 7) * pow(Time(seg), i + l - 7);
                    }
                }
            }
            int rows = Q.rows(), cols = Q.cols();
            Q.conservativeResizeLike(MatrixXd::Zero(Q.rows() + Q_k.rows(), Q.cols() + Q_k.cols()));
            Q.block(rows, cols, Q_k.rows(), Q_k.cols()) = Q_k;
        }
        cout << "Q:" << endl << Q << endl;
        // 构建转换矩阵M
        MatrixXd M;
        for (int seg = 0; seg < n_seg; seg++) {
            MatrixXd M_k = MatrixXd::Zero(8, p_num1d); // 6行代表该段起点、终点的p,v,a,j
            for (int row = 0; row < 4; row++) {
                int k = row; //当前阶次
                for (int col = 0; col < p_num1d; col++) {
                    int i = col; //当前多项式的系数
                    if (i >= k) {
                        M_k(row, col) = Factorial(i) / Factorial(i - k) * pow(0, i - k);
                    }
                }
            }
            for (int row = 4; row < 8; row++) {
                int k = row - 4;
                for (int col = 0; col < p_num1d; col++) {
                    int i = col;
                    if (i >= k) {
                        M_k(row, col) = Factorial(i) / Factorial(i - k) * pow(Time(seg), i - k);
                    }
                }
            }
            int rows = M.rows(), cols = M.cols();
            M.conservativeResizeLike(MatrixXd::Zero(M.rows() + M_k.rows(), M.cols() + M_k.cols()));
            M.block(rows, cols, M_k.rows(), M_k.cols()) = M_k;
        }
        cout << "M:" << endl << M << endl;
        // 构造Ct
        MatrixXd Ct = MatrixXd::Zero(8 * n_seg, 4 * (n_seg + 1));
        for (int row = 0; row < 4; row++) { //起始点的p,v,a,j
            int col = row;
            Ct(row, col) = 1;
        }
        for (int row = 4; row <= 8 * (n_seg - 2) + 4; row = row + 8) { //中间点的p
            int col = (row - 4) / 8 + 4;
            cout << "col:" << col << endl;
            Ct(row, col) = 1;
            Ct(row + 4, col) = 1;
        }
        for (int row = 5; row <= 8 * (n_seg - 2) + 5; row = row + 8) { //中间点的v, a, j
            int col = n_seg + 6 + 3 * (row - 5) / 8;
            // v
            Ct(row, col) = 1;
            Ct(row + 4, col) = 1;
            // a
            Ct(row + 1, col + 1) = 1;
            Ct(row + 5, col + 1) = 1;
            // j
            Ct(row + 2, col + 2) = 1;
            Ct(row + 6, col + 2) = 1;
        }
        int row = 4 + 8 * (n_seg - 1); // 末尾点的p,v,a,j
        int col = n_seg + 3;
        Ct(row, col) = 1;
        Ct(row + 1, col + 1) = 1;
        Ct(row + 2, col + 2) = 1;
        // 末尾的j单独算，因为是优化变量，在dp中
        Ct(row + 3, 4 * (n_seg + 1) - 1) = 1;
        cout << "Ct:" << endl << Ct << endl;
        // 计算R_pp和R_fp
        MatrixXd R = Ct.transpose() * M.inverse().transpose() * Q * M.inverse() * Ct;
        MatrixXd Rpp = R.block(n_seg + 6, n_seg + 6, 3 * (n_seg - 1) + 1, 3 * (n_seg - 1) + 1);
        MatrixXd Rfp = R.block(0, n_seg + 6, n_seg + 6, 3 * (n_seg - 1) + 1);
        // 计算dF
        MatrixXd dF = MatrixXd::Zero(4, 1);
        dF(0, 0) = Path(0, dim);
        for (int wp = 1; wp <= n_seg; wp++) {
            dF.conservativeResizeLike(MatrixXd::Zero(dF.rows() + 1, 1));
            dF(dF.rows() - 1, 0) = Path(wp, dim);
        }
        dF.conservativeResizeLike(MatrixXd::Zero(dF.rows() + 2, 1));
        cout << "dF:" << endl << dF << endl;
        MatrixXd dP = -Rpp.inverse() * Rfp.transpose() * dF;
        MatrixXd dFdP = MatrixXd::Zero(dP.rows() + dF.rows(), 1);
        dFdP << dF, //
            dP;
        cout << "dFdP:" << endl << dFdP << endl;
        MatrixXd poly_coef = M.inverse() * Ct * dFdP; // 是列向量
        cout << "poly_coef:" << poly_coef << endl;

        PolyCoeff.block(dim, 0, 1, poly_coef.rows()) = poly_coef.transpose();
    }
    cout << "PolyCoeff:" << PolyCoeff << endl;

    return PolyCoeff;
}
