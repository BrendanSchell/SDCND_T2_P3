/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 *  Updated on: Apr 30, 2017
 *      Author: Brendan Schell
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <cmath>
#include "particle_filter.h"
#include "helper_functions.h"
#include "map"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// takes x, y, theta and their uncertainties from GPS and initializes 
	//   all weights to 1. 

	// set number of particles
	num_particles = 100;

	
	default_random_engine gen;

	// Gaussian noise distributions 
	normal_distribution<double> N_x_init(x, std[0]);
	normal_distribution<double> N_y_init(y, std[1]);
	normal_distribution<double> N_theta_init(theta, std[2]);

	//initialize all particles to first position plus gaussian noise
	for (int i = 0; i < num_particles; i++){
		Particle this_particle = {i, N_x_init(gen), N_y_init(gen), N_theta_init(gen), 1.};
		particles.push_back(this_particle);
	}
	// initialize weights to 1
	weights.resize(num_particles,1.);

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// Add measurements to each particle and add random Gaussian noise.

	default_random_engine gen;
	
	// Gaussian noise distributions
	normal_distribution<double> N_x(0.,std_pos[0]);
	normal_distribution<double> N_y(0.,std_pos[1]);
	normal_distribution<double> N_theta(0.,std_pos[2]);

	// calculate predicted x, y, and theta values for all particles
	for (int i = 0; i < num_particles; i++){
		// if yaw rate close to 0, use alternative equations
		if (fabs(yaw_rate) < 0.001){
			particles[i].x += velocity * cos(particles[i].theta) * delta_t;
			particles[i].y += velocity * sin(particles[i].theta) * delta_t; 
		}else{
		
			particles[i].x += ((velocity/yaw_rate) * (sin(particles[i].theta+yaw_rate*delta_t)-sin(particles[i].theta)));
			particles[i].y += ((velocity/yaw_rate) * (cos(particles[i].theta)-cos(particles[i].theta+yaw_rate*delta_t)));
		}

		particles[i].theta += (yaw_rate * delta_t );

		// add noise to x 
		particles[i].x += N_x(gen);
		// add noise to y
		particles[i].y += N_y(gen);
		// add noise to theta
		particles[i].theta += N_theta(gen);
	}


}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// Find the predicted measurement that is closest to each landmark and 
	//   assign observed measurement to this particular landmark.
	for (int i = 0; i < observations.size(); i++){
        // initialize smallest distance to max limit for double type
		double smallest_distance = numeric_limits<double>::max();

		// determine the landmark that minimizes the distance and set
		//   the observation's id to this index for future reference
		for (int j = 0; j < predicted.size();j++){
			double this_dist = dist(observations[i].x,observations[i].y,predicted[j].x,predicted[j].y);
			if (this_dist < smallest_distance){
				smallest_distance = this_dist;
				observations[i].id = predicted[j].id;
			} 
		}
	}	

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// Update the weights of each particle using a multi-variate 
	//   Gaussian distribution. 
	
	
	// update each particle 
	for (int p = 0; p < num_particles; p++){

		// predicted vector to store landmarks within range of particle
		// transformed_obs vector to store the observations with
		//   transformed coordinates
		std::vector<LandmarkObs> predicted;
		std::vector<LandmarkObs> transformed_obs;

		// store particle's values for readability
		double ptheta = particles[p].theta;
		double px = particles[p].x;
		double py = particles[p].y;

		// loop through each observation
		for (int i = 0; i < observations.size();i++){

			// store observation values for readability
			double ox = observations[i].x;
			double oy = observations[i].y;

			// convert observations from vehicle coordinate system to map 
			//   coordinate system
			ox =  px + (observations[i].x*cos(ptheta) - observations[i].y*sin(ptheta));
			oy = py + (observations[i].x*sin(ptheta) + observations[i].y*cos(ptheta));
            
			// add transformed observation to vector
			transformed_obs.push_back(LandmarkObs {observations[i].id, ox, oy});
		
		// determine nearest landmarks within range of sensor, add to 
		//   predicted vector
		}
		for (int l =0; l < map_landmarks.landmark_list.size();l++){
            auto lmx = map_landmarks.landmark_list[l].x_f;
			auto lmy = map_landmarks.landmark_list[l].y_f;
			auto lmid = map_landmarks.landmark_list[l].id_i;
			if (dist(px, py,lmx,lmy) <= sensor_range){
				predicted.push_back(LandmarkObs {l, lmx, lmy});
			}
		}
        
		// perform data association
		this->dataAssociation(predicted,transformed_obs);
        
		// initialize new weight to 1
		long double this_weight = 1.0;
		
        // update weights based on all observations
		for (int i = 0; i < observations.size();i++){

			// stored terms in variables for readability	
			long double landmarkx = map_landmarks.landmark_list[transformed_obs[i].id].x_f;
			long double landmarky = map_landmarks.landmark_list[transformed_obs[i].id].y_f;
			long double stdx = std_landmark[0];
			long double stdy = std_landmark[1];
			long double stdx2 = stdx * stdx;
			long double stdy2 = stdy * stdy;
			long double tox = transformed_obs[i].x;
			long double toy = transformed_obs[i].y;
			long double xdist = tox-landmarkx;
			long double ydist = toy-landmarky;

			// update weight based on probability for this observation
			long double denom = 1.0/(2.0 * M_PI * stdx * stdy);
			long double num= exp(-0.5*((xdist * xdist / stdx2)+(ydist * ydist / stdy2)));
			long double prob = num * denom;	
			this_weight *= prob;
		}
		
		// update particle weight to updated weight value
		particles[p].weight = this_weight;
		weights[p] = this_weight;
	}
}

void ParticleFilter::resample() {
	// Resample particles with replacement with probability proportional 
	//  to their weight. 
    
    // Setup the weights (in this case linearly weighted)
	std::vector<Particle> particles_new;
    default_random_engine gen;
    std::discrete_distribution<> d(weights.begin(), weights.end());
    
	// choose new particles based off their weighted distribution
    for(int n=0; n<num_particles; n++) {
        particles_new.push_back(particles[d(gen)]);

    }
	// update particle with new particles
	particles = particles_new;
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
