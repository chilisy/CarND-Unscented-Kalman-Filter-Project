#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
  TODO:
    * Calculate the RMSE here.
  */
    VectorXd rmse(4);
    rmse << 0,0,0,0;
    
    
    if (estimations.size() == 0)
    {
        cout << "CalculateRMSE - Error - Estimation should not be zero!" << endl;
        return rmse;
    }
    
    if (estimations.size() != ground_truth.size())
    {
        cout << "CalculateRMSE - Error - Estimation dimension should match Ground Truth dimension" << endl;
        return rmse;
    }
    
    
    //accumulate squared residuals
    for(int i=0; i < estimations.size(); ++i){
        
        VectorXd residual = estimations[i] - ground_truth[i];
        
        residual = residual.array()*residual.array();
        rmse += residual;
        
    }
    
    //calculate the mean
    rmse /= estimations.size();
    
    //calculate the squared root
    rmse = rmse.array().sqrt();
    
    return rmse;

}
