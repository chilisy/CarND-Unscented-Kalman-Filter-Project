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
    std_a_ = 5.;
    
    // Process noise standard deviation yaw acceleration in rad/s^2
    std_yawdd_ = M_PI/180.0*120.0;
    
    // Laser measurement noise standard deviation position1 in m
    std_laspx_ = 0.15;
    
    // Laser measurement noise standard deviation position2 in m
    std_laspy_ = 0.15;
    
    // Radar measurement noise standard deviation radius in m
    std_radr_ = 0.3;
    
    // Radar measurement noise standard deviation angle in rad
    std_radphi_ = 0.3;
    
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
    P_ = MatrixXd::Identity(n_x_, n_x_);
    
    lambda_ = 3 - n_aug_;
    
    Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
    
    previous_timestamp_ = 0;
    
    //set weights
    weights_ = VectorXd(2*n_aug_+1);

    weights_(0) = lambda_/(lambda_+n_aug_);
    for (int i=1; i<(2*n_aug_+1); i++)
        weights_(i) = 0.5/(lambda_+n_aug_);
    
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
        previous_timestamp_ = meas_package.timestamp_- 5e5;
        counter_ = 0;
        
        is_initialized_ = true;
        return;
    }
    
    // compute the time elapsed between the current and previous measurements
    double delta_t = (meas_package.timestamp_ - previous_timestamp_) / 1000000.0;	//dt - expressed in seconds
    previous_timestamp_ = meas_package.timestamp_;
    
    // prediction step
    if ((use_laser_ && meas_package.sensor_type_ == MeasurementPackage::LASER)
        || (use_radar_ && meas_package.sensor_type_ == MeasurementPackage::RADAR)) {
        Prediction(delta_t);
    }
    
    if (use_radar_ && meas_package.sensor_type_ == MeasurementPackage::RADAR){
        
        UpdateRadar(meas_package);
    }
    else if (use_laser_ && meas_package.sensor_type_ == MeasurementPackage::LASER){
        
        UpdateLidar(meas_package);
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
    int n_z = 2;
    VectorXd z_pred = VectorXd(n_z);
    MatrixXd S = MatrixXd(n_z, n_z);
    MatrixXd H = MatrixXd(n_z, n_x_);
    
    PredictLidarMeasurement(&z_pred, &S, H, n_z);
    
    double nis;
    nis = CalculateNIS(meas_package, z_pred, S);
    nis_laser_.push_back(nis);
    counter_ ++;
    
    UpdataState4Lidar(meas_package, z_pred, S, H);
    if (counter_>490){
        counter_ = 0;
        WriteNIS();
    }
    
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
    int n_z = 3;
    VectorXd z_pred = VectorXd(n_z);
    MatrixXd S = MatrixXd(n_z, n_z);
    MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
    
    PredictRadarMeasurement(&z_pred, &S, &Zsig, n_z);
    
    double nis;
    nis = CalculateNIS(meas_package, z_pred, S);
    nis_radar_.push_back(nis);
    counter_ ++;
    
    UpdateState4Radar(meas_package, z_pred, S, n_z, Zsig);
    if (counter_>490){
        counter_ = 0;
        WriteNIS();
    }
    
   
    
}

void UKF::AugmentedSigmaPoints(MatrixXd *Xsig_out){
    
    //create augmented mean vector
    VectorXd x_aug = VectorXd(n_aug_);
    
    //create augmented state covariance
    MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
    
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

    double tol = 1e-9;
    
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
    
    // diff vector
    VectorXd diff = VectorXd::Zero(n_x_);
    
    //predict state mean
    x_.fill(0.0);
    for (int i=0; i<(2*n_aug_+1); i++)
    {
        x_ += weights_(i) * Xsig_pred_.col(i);
    }
    
    //predict state covariance matrix
    P_.fill(0.0);
    for (int i=0; i<(2*n_aug_+1); i++)
    {
        diff = Xsig_pred_.col(i) - x_;
        
        //angle normalization
        while (diff(3) > M_PI  || diff(3) < -M_PI)
        {
            if (diff(3) > M_PI)
                diff(3) -= 2*M_PI;
            else
                diff(3) += 2*M_PI;
        }
        
        P_ += weights_(i) * diff * diff.transpose();
    }

}

void UKF::PredictRadarMeasurement(VectorXd* z_out, MatrixXd* S_out, MatrixXd* Zsig_out, int n_z){
    
    //create matrix for sigma points in measurement space
    MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
    
    //mean predicted measurement
    VectorXd z_pred = VectorXd(n_z);
    
    //measurement covariance matrix S
    MatrixXd S = MatrixXd(n_z,n_z);
    
    double px, py, v, phi;
    double rho_r, phi_r, rho_dot_r;
    double tol = 1e-6;
    VectorXd diff;
    
    //transform sigma points into measurement space
    Zsig.fill(0.0);
    for (int i=0; i<(2*n_aug_+1); i++)
    {
        px = Xsig_pred_(0, i);
        py = Xsig_pred_(1, i);
        v = Xsig_pred_(2, i);
        phi = Xsig_pred_(3, i);
        
        rho_r = sqrt(pow(px, 2) + pow(py, 2));
        phi_r = atan2(py, px);
        
        if (rho_r < tol && rho_r > -tol)
            rho_dot_r = 0;
        else
            rho_dot_r = (px*cos(phi)*v + py*sin(phi)*v)/rho_r;
        
        Zsig.col(i) << rho_r, phi_r, rho_dot_r;
    }
    
    //calculate mean predicted measurement
    z_pred.fill(0.0);
    for (int i=0; i<(2*n_aug_+1); i++)
    {
        z_pred += weights_(i) * Zsig.col(i);
    }
    
    //calculate measurement covariance matrix S
    S.fill(0.0);
    for (int i=0; i<(2*n_aug_+1); i++)
    {
        diff =  Zsig.col(i) - z_pred;
        
        S += weights_(i) * diff * diff.transpose();
    }
    MatrixXd R = MatrixXd::Zero(n_z, n_z);
    R(0, 0) = pow(std_radr_, 2);
    R(1, 1) = pow(std_radphi_, 2);
    R(2, 2) = pow(std_radrd_, 2);
    S += R;
    
    *z_out = z_pred;
    *S_out = S;
    *Zsig_out = Zsig;

}

void UKF::PredictLidarMeasurement(VectorXd* z_out, MatrixXd* S_out, MatrixXd &H, int n_z){
    
    // measurment matrix
    H << 1, 0, 0, 0, 0,
    0, 1, 0, 0, 0;
    
    VectorXd z_pred = H * x_;
    
    // measurement covariance matrix S
    MatrixXd S = H * P_ * H.transpose();
    
    // add measurement noise
    MatrixXd R = MatrixXd::Zero(n_z, n_z);
    R(0, 0) = pow(std_laspx_, 2);
    R(1, 1) = pow(std_laspy_, 2);
    S += R;
    
    *z_out = z_pred;
    *S_out = S;
    
}

void UKF::UpdateState4Radar(MeasurementPackage meas_package, VectorXd z_pred, MatrixXd S, int n_z, MatrixXd Zsig){
    
    // measurement vector
    VectorXd z = meas_package.raw_measurements_;
    
    //create matrix for cross correlation Tc
    MatrixXd Tc = MatrixXd(n_x_, n_z);
    
    //calculate cross correlation matrix
    VectorXd x_diff = VectorXd(n_x_);
    VectorXd z_diff = VectorXd(n_z);
    
    Tc.fill(0.0);
    for (int i=0; i<(2*n_aug_+1); i++)
    {
        x_diff = Xsig_pred_.col(i) - x_;
        
        while (x_diff(3) > M_PI  || x_diff(3) < -M_PI)
        {
            if (x_diff(3) > M_PI)
                x_diff(3) -= 2.*M_PI;
            else
                x_diff(3) += 2.*M_PI;
        }
        
        z_diff = Zsig.col(i) - z_pred;
        
        while (z_diff(1) > M_PI  || z_diff(1) < -M_PI)
        {
            if (z_diff(1) > M_PI)
                z_diff(1) -= 2.*M_PI;
            else
                z_diff(1) += 2.*M_PI;
        }
        
        Tc += weights_(i) * x_diff * z_diff.transpose();
    }
    
    //calculate Kalman gain K;
    MatrixXd K = Tc * S.inverse();
    
    VectorXd y = z - z_pred;
    
    // phi-diff should be a value between -pi and pi
    while (y(1) > M_PI  || y(1) < -M_PI)
    {
        if (y(1) > M_PI)
            y(1) -= 2.*M_PI;
        else
            y(1) += 2.*M_PI;
    }
    
    //update state mean and covariance matrix
    x_ = x_ + K * y;
    P_ = P_ - K * S * K.transpose();
    
    //cout << "x = " << endl << x_ << endl;
    //cout << "P = " << endl << P_ << endl;
    
}

void UKF::UpdataState4Lidar(MeasurementPackage meas_package, VectorXd z_pred, MatrixXd S, MatrixXd H) {
    
    // measurment vector
    VectorXd z = meas_package.raw_measurements_;
    
    MatrixXd Si = S.inverse();
    MatrixXd PHt = P_ * H.transpose();
    MatrixXd K = PHt * Si;
    
    //new estimate
    x_ = x_ + K * (z-z_pred);

    MatrixXd I = MatrixXd::Identity(n_x_, n_x_);
    P_ = (I - K * H) * P_;
    
}

double UKF::CalculateNIS(MeasurementPackage meas_package, VectorXd z_pred, MatrixXd S){
    
    double nis;
    long n_z = z_pred.size();
    VectorXd y = VectorXd(n_z);
    y = meas_package.raw_measurements_ - z_pred;
    
    nis = y.transpose() * S.inverse() * y;
    
    //cout << "nis = " << endl << nis << endl;
    
    return nis;
}

void UKF::WriteNIS(){
    
    ofstream nis_file;
    int size_nis;
    nis_file.open("NIS.csv");
    
    if (nis_radar_.size() >= nis_laser_.size())
        size_nis = nis_laser_.size();
    else
        size_nis = nis_radar_.size();
    
    nis_file << "NIS radar measurements, NIS laser measurements" << "\n";
    
    for (int i=0; i<size_nis; i++){
        
        nis_file << nis_radar_[i] << ", " << nis_laser_[i] << "\n";
        
    }
    nis_file.close();
    
}

