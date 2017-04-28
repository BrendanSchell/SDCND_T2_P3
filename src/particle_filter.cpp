/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
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
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// set number of particles
	num_particles = 100;
	//initialize all particles to first position
	
	for (int i = 0; i < num_particles; i++){
		Particle this_particle = {i, x, y, theta, 1.};
		particles.push_back(this_particle);
	}
	default_random_engine gen;

	// Add random Gaussian noise to each particle.
	normal_distribution<double> N_x_init(0., std[0]);
	normal_distribution<double> N_y_init(0., std[1]);
	normal_distribution<double> N_theta_init(0., std[2]);

	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	for (int i = 0; i < num_particles; i++){
		particles[i].x += N_x_init(gen);
		particles[i].y += N_y_init(gen);
		particles[i].theta += N_theta_init(gen);
	}	
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	default_random_engine gen;
	normal_distribution<double> N_x(0.,std_pos[0]);
	normal_distribution<double> N_y(0.,std_pos[1]);
	normal_distribution<double> N_theta(0.,std_pos[2]);

	// if yaw rate not equal to 0
	if (fabs(yaw_rate) < 0.0001){
		yaw_rate = 0.0001;
	}
	
	for (int i = 0; i < num_particles; i++){
		particles[i].x += (velocity/yaw_rate) * (sin(particles[i].theta+yaw_rate*delta_t)-sin(particles[i].theta));
		//add noise to x 
		particles[i].x += N_x(gen); 
		particles[i].y += (velocity/yaw_rate) * (cos(particles[i].theta)-cos(particles[i].theta+yaw_rate*delta_t));
		//add noise to y
		particles[i].y += N_y(gen);
		particles[i].theta += yaw_rate * delta_t;
		particles[i].theta += N_theta(gen);
	}
	


}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for (int i = 0; i < observations.size(); i++){
		double smallest_distance = numeric_limits<double>::max();
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
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html
	
	// loop through each particle
	
	for (int o = 0; o < num_particles; o++){
		std::vector<LandmarkObs> predicted;
		double ptheta = particles[o].theta;
		for (int i = 0; i < observations.size();i++){
			double ox = observations[i].x;
			double oy = observations[i].y;
		
			//convert observations from vehicle coordinate system to map coordinate system
			observations[i].x = particles[o].x + (ox*cos(ptheta) - oy*sin(ptheta));
			observations[i].y = particles[o].y +  (ox*sin(ptheta) + oy*cos(ptheta));
			//determine nearest landmarks
		}
		for (int l =0; l < map_landmarks.landmark_list.size();l++){
			predicted.push_back(LandmarkObs {map_landmarks.landmark_list[l].id_i, map_landmarks.landmark_list[l].x_f,map_landmarks.landmark_list[l].y_f});
		}
		this->dataAssociation(predicted,observations);
		// update weights
		double this_weight = 1.;
		
		
		for (int i = 0; i < observations.size();i++){
			
			double landmarkx = map_landmarks.landmark_list[observations[i].id].x_f;
			double landmarky = map_landmarks.landmark_list[observations[i].id].y_f;
			double stdx = std_landmark[0];
			double stdy = std_landmark[1];
			double stdx2 = pow(stdx,2);
			double stdy2 = pow(stdy,2);
			double px = observations[i].x;
			double py = observations[i].y;
			double xdist = px-landmarkx;
			double ydist = py-landmarky;

			//update weight for each observation
			if (dist(px,py,landmarkx,landmarky) < sensor_range){

				this_weight *= exp(-0.5*((pow(xdist,2)/stdx)+(pow(ydist,2)/stdy)))/(2*M_PI*stdx*stdy);
			}
		}
        if (this_weight == 1){
			this_weight = 0;
		}
		particles[o].weight = this_weight;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    
    // Setup the weights (in this case linearly weighted)
    std::vector<double> weights;
    for(int i=0; i<num_particles; i++) {
        weights.push_back(particles[i].weight);
    }
	std::vector<Particle> particles_new;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> d(weights.begin(), weights.end());
    for(int n=0; n<num_particles; n++) {
        particles_new.push_back(particles[d(gen)]);
    }
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
