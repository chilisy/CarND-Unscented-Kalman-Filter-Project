#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
    // if this is false, laser measurements will be ignored (except during init)
    use_laser_ = true;
    
    // if this is false, radar measurements will be ignored (except during init)
    use_radar_ = true;
    
    // Process noise standard deviation longitudinal acceleration in m/s^2
    std_a_ = 3;
    
    // Process noise standard deviation yaw acceleration in rad/s^2
    std_yawdd_ = 0.2;
    
    // Laser measurement noise standard deviation position1 in m
    std_laspx_ = 0.15;
    
    // Laser measurement noise standard deviation position2 in m
    std_laspy_ = 0.15;
    
    // Radar measurement noise standard deviation radius in m
    std_radr_ = 0.3;
    
    // Radar measurement noise standard deviation angle in rad
    std_radphi_ = 0.03;
    
    // Radar measurement noise standard deviation radius change in m/s
    std_radrd_ = 0.3;
    
    /**
     TODO:
     
     Complete the initialization. See ukf.h for other member properties.
     
     Hint: one or more values initialized above might be wildly off...
     */
    is_initialized_ = false;
    
    n_x_ = 5;
    
    n_aug_ = 7;
    
    // initial state vector
    x_ = VectorXd(n_x_);
    
    // initial covariance matrix
    P_ = MatrixXd(n_x_, n_x_);
    
    lambda_ = 3 - n_aug_;
    
    Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
    
    previous_timestamp_ = 0;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
    /**
     TODO:
     
     Complete this function! Make sure you switch between lidar and radar
     measurements.
     */
    if (!is_initialized_) {
        if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
            float rho = meas_package.raw_measurements_[0];
            float phi = meas_package.raw_measurements_[1];
            float rho_dot = meas_package.raw_measurements_[2];
            
            x_ << rho*sin(phi), rho*cos(phi), rho_dot, 0, 0;
            
        }
        else if (meas_package.sensor_type_ == MeasurementPackage::LASER){
            float p_x = meas_package.raw_measurements_[0];
            float p_y = meas_package.raw_measurements_[1];
            
            x_ << p_x, p_y, 0, 0, 0;
        }
        
        // set first timestamp
        previous_timestamp_ = meas_package.timestamp_;
        
        is_initialized_ = true;
        return;
    }
    
    // compute the time elapsed between the current and previous measurements
    double delta_t = (meas_package.timestamp_ - previous_timestamp_) / 1000000.0;	//dt - expressed in seconds
    previous_timestamp_ = meas_package.timestamp_;
    
    // prediction step
    if ((use_laser_ && meas_package.sensor_type_ == MeasurementPackage::LASER)
        || (use_radar_ && meas_package.sensor_type_ == MeasurementPackage::RADAR))
    {
        Prediction(delta_t);
    }
    
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
    /**
     TODO:
     
     Complete this function! Estimate the object's location. Modify the state
     vector, x_. Predict sigma points, the state, and the state covariance matrix.
     */
    
    MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
    
    AugmentedSigmaPoints( &Xsig_aug );
    SigmaPointPrediction( Xsig_aug, delta_t );
    PredictMeanAndCovariance();
    
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
    /**
     TODO:
     
     Complete this function! Use lidar data to update the belief about the object's
     position. Modify the state vector, x_, and covariance, P_.
     
     You'll also need to calculate the lidar NIS.
     */
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
    /**
     TODO:
     
     Complete this function! Use radar data to update the belief about the object's
     position. Modify the state vector, x_, and covariance, P_.
     
     You'll also need to calculate the radar NIS.
     */
}


void UKF::AugmentedSigmaPoints(MatrixXd *Xsig_out){
    
    //create augmented mean vector
    VectorXd x_aug = VectorXd(7);
    
    //create augmented state covariance
    MatrixXd P_aug = MatrixXd(7, 7);
    
    //create sigma point matrix
    MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
    
    //create augmented mean state
    x_aug.head(5) = x_;
    x_aug(5) = 0;
    x_aug(6) = 0;
    
    //create augmented covariance matrix
    P_aug.fill(0.0);
    P_aug.topLeftCorner(5, 5) = P_;
    P_aug(5, 5) = pow(std_a_, 2);
    P_aug(6, 6) = pow(std_yawdd_, 2);
    
    //create square root matrix
    MatrixXd A = P_aug.llt().matrixL();
    
    //create augmented sigma points
    Xsig_aug.col(0) = x_aug;
    
    for (int i = 0; i < n_aug_; i++)
    {
        Xsig_aug.col(i+1) = x_aug + sqrt(lambda_ + n_aug_)*A.col(i);
        Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_ + n_aug_)*A.col(i);
    }
    
    *Xsig_out = Xsig_aug;
    
}


void UKF::SigmaPointPrediction( MatrixXd Xsig_aug, double delta_t ){

    double tol = 1e-6;
    
    for (int i=0; i<(2*n_aug_ + 1); i++)
    {
        //predict sigma points
        double px = Xsig_aug(0, i);
        double py = Xsig_aug(1, i);
        double v = Xsig_aug(2, i);
        double phi = Xsig_aug(3, i);
        double phi_dot = Xsig_aug(4, i);
        double nu_a = Xsig_aug(5, i);
        double nu_phi_ddot = Xsig_aug(6, i);
        
        double px_p = px + 0.5 * pow(delta_t, 2) * cos(phi) * nu_a;
        double py_p = py + 0.5 * pow(delta_t, 2) * sin(phi) * nu_a;
        
        //avoid division by zero
        if (phi_dot > -tol && phi_dot < tol)
        {
            px_p += v * cos(phi) * delta_t;
            py_p += v * sin(phi) * delta_t;
        }
        else
        {
            px_p += v/phi_dot * (sin(phi + phi_dot*delta_t)-sin(phi));
            py_p += v/phi_dot * (-cos(phi + phi_dot*delta_t)+cos(phi));
        }
        
        double v_p = v + delta_t * nu_a;
        double phi_p = phi + delta_t * phi_dot + 0.5 * pow(delta_t, 2) * nu_phi_ddot;
        double phi_dot_p = phi_dot + delta_t * nu_phi_ddot;
        
        //write predicted sigma points into right column
        Xsig_pred_(0, i) = px_p;
        Xsig_pred_(1, i) = py_p;
        Xsig_pred_(2, i) = v_p;
        Xsig_pred_(3, i) = phi_p;
        Xsig_pred_(4, i) = phi_dot_p;
    }

}

void UKF::PredictMeanAndCovariance(){
    
    // create vector for weights
    VectorXd weights = VectorXd(2*n_aug_+1);
    
    // diff vector
    VectorXd diff = VectorXd::Zero(n_x_);
    
    //set weights
    weights(0) = lambda_/(lambda_+n_aug_);
    for (int i=1; i<(2*n_aug_+1); i++)
        weights(i) = 1.0/2.0/(lambda_+n_aug_);
    
    //predict state mean
    x_.fill(0.0);
    for (int i=0; i<(2*n_aug_+1); i++)
    {
        x_ += weights(i) * Xsig_pred_.col(i);
    }
    
    //predict state covariance matrix
    P_.fill(0.0);
    for (int i=0; i<(2*n_aug_+1); i++)
    {
        diff = Xsig_pred_.col(i) - x_;
        
        //angle normalization
        while (diff(3)> M_PI) diff(3)-=2.0*M_PI;
        while (diff(3)< -M_PI) diff(3)+=2.0*M_PI;
        
        P_ += weights(i) * diff * diff.transpose();
    }

}
