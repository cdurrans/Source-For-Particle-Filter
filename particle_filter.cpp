/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"
#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <cmath>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;


void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */

  std::default_random_engine gen;
  num_particles = 500;  // TODO: Set the number of particles
  
  double std_x, std_y, std_theta;

  std_x = std[0];
  std_y = std[1];
  std_theta = std[2];

  normal_distribution<double> dist_x(x, std_x);
  normal_distribution<double> dist_y(y, std_y);
  normal_distribution<double> dist_theta(theta, std_theta);

  for (int i = 0; i < num_particles; ++i) 
  {
	  struct Particle p;
	  p.id = i;
	  p.x = dist_x(gen);
	  p.y = dist_y(gen);
	  p.theta = dist_theta(gen);
	  p.weight = 1.0;
	  particles.push_back(p);
  }
  is_initialized = true;
  std::cout << "Initialized Particle Filter" << std::endl;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   */

//   normal_distribution<double> dist_x(0.0, std_pos[0]);
//   normal_distribution<double> dist_y(0.0, std_pos[1]);
//   normal_distribution<double> dist_theta(0.0, std_pos[2]);

	normal_distribution<double> dist_yaw_rate(yaw_rate, std_pos[1]);
	normal_distribution<double> dist_velocity(velocity, std_pos[0]);
	double yaw_rate_noisy = dist_yaw_rate(gen);
	double velocity_noisy = dist_velocity(gen);
    double theta_next_step;

	for (int i = 0; i < num_particles; ++i) 
  {
      theta_next_step = particles[i].theta + (yaw_rate_noisy * delta_t);
      if (fabs(yaw_rate_noisy) > 0.001) 
      {
        particles[i].x = particles[i].x + (velocity_noisy / yaw_rate_noisy) * (sin(particles[i].theta + yaw_rate_noisy * delta_t) - sin(particles[i].theta));
        particles[i].y = particles[i].y + (velocity_noisy / yaw_rate_noisy) * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate_noisy * delta_t));
        particles[i].theta = theta_next_step;
      }
      else 
      {
        particles[i].x = particles[i].x + velocity_noisy * delta_t * cos(yaw_rate_noisy);
        particles[i].y = particles[i].y + velocity_noisy * delta_t * sin(yaw_rate_noisy);
        particles[i].theta = theta_next_step;
      }
//     particles[i].x += dist_x(gen);
//     particles[i].y += dist_y(gen);
//     particles[i].theta += dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   */
	double result_dist, min_dist;
	for (unsigned int i = 0; i < observations.size(); ++i) 
  {
		min_dist = INFINITY;
		for (unsigned int i2 = 0; i2 < predicted.size(); ++i2) 
    {
			result_dist = dist(observations[i].x, observations[i].y, predicted[i2].x, predicted[i2].y);
			if (min_dist > result_dist) 
      {
				observations[i].id = predicted[i2].id;
				min_dist = result_dist;
			}
		}		
	}  
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a multi-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
	vector<LandmarkObs> transformedObservations;

	double gauss_norm, probability, weight;
	gauss_norm = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);
  
  for (int p_i = 0; p_i < num_particles; ++p_i) 
  {
      for (unsigned int i = 0; i < transformedObservations.size(); ++i) 
      {
        LandmarkObs transformed_obs;
        
        transformed_obs.x = particles[p_i].x + (cos(particles[p_i].theta) * observations[i].x) - (sin(particles[p_i].theta) * observations[i].y);
        transformed_obs.y = particles[p_i].y + (sin(particles[p_i].theta) * observations[i].x) + (cos(particles[p_i].theta) * observations[i].y);
        transformed_obs.id = observations[i].id;

        transformedObservations.push_back(transformed_obs);
      }
      
      vector<LandmarkObs> pred_landmarks;

      for (unsigned int ld_index = 0; ld_index < map_landmarks.landmark_list.size(); ++ld_index) 
      {
        int ld_id = map_landmarks.landmark_list[ld_index].id_i;
        double ld_x = map_landmarks.landmark_list[ld_index].x_f;
        double ld_y = map_landmarks.landmark_list[ld_index].y_f;

        double landMarkDistFromParticle = dist(particles[p_i].x, particles[p_i].y, ld_x, ld_y);

        if (landMarkDistFromParticle < sensor_range) 
        {
          LandmarkObs l_is_within_range;
          l_is_within_range.id = ld_id;
          l_is_within_range.x = ld_x;
          l_is_within_range.y = ld_y;
          pred_landmarks.push_back(l_is_within_range);
        }
      }
    
    dataAssociation(pred_landmarks, transformedObservations);
	  
    weight = 1.0;
    
    double landmark_x, landmark_y;
    
    for (unsigned int i = 0; i < transformedObservations.size(); ++i) 
    {
      for (unsigned int ld_indx = 0; ld_indx < pred_landmarks.size(); ++ld_indx) 
      {
        if (transformedObservations[i].id == pred_landmarks[ld_indx].id) 
        {
           landmark_x = pred_landmarks[ld_indx].x;
           landmark_y = pred_landmarks[ld_indx].y;
           probability = (pow(transformedObservations[i].x - landmark_x, 2) / (2 * pow(std_landmark[0], 2))) + (pow(transformedObservations[i].y - landmark_y, 2) / (2 * pow(std_landmark[1], 2)));
           weight = weight * exp(-probability) * gauss_norm;
           break;
        }
      }
	}
  particles[p_i].weight = weight;
  }
  double sumOfWeights = 0.0;
  for (int p_i = 0; p_i < num_particles; ++p_i) 
  {
    sumOfWeights += particles[p_i].weight;
  }
  
  for (int p_i = 0; p_i < num_particles; ++p_i) 
  {
    particles[p_i].weight /= sumOfWeights;
  }

  for (int p_i = 0; p_i < num_particles; ++p_i) {
    sumOfWeights += particles[p_i].weight;
  }
  
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

  vector<double> weights;
  for (int pi = 0; pi < num_particles; ++pi) 
   {
	weights.push_back(particles[pi].weight);
   }
  std::discrete_distribution<int> d_weights(weights.begin(), weights.end());
  vector<Particle> particle_temp;
  
  for (int pi = 0; pi < num_particles; ++pi) 
  {
    int p_z = d_weights(gen);
    particle_temp.push_back(particles[p_z]);
  }
	particles = particle_temp;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}